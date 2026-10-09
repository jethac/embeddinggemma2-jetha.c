"""A stalled probe must expire without blocking uncached text inference."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
for name in ('binary', 'model', 'mmproj', 'stub'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--backend', default='cpu')
p.add_argument('--port', type=int, default=42678)
a = p.parse_args()
url = f'http://127.0.0.1:{a.port}'

def owned_child(pid, path, kill=False, parent=None):
    if os.name != 'nt':
        try:
            # A killed but unreaped child has no executable link.
            state = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
            if state[0] == 'Z': return parent is not None and int(state[1]) == parent
            if Path(f'/proc/{pid}/exe').resolve() != path.resolve(): return False
        except OSError: return False
        if kill: os.kill(pid, signal.SIGKILL)
        return True
    import ctypes
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.restype = ctypes.c_void_p
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    k.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
    k.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    k.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    handle = k.OpenProcess(0x1000 | 1, False, pid)
    if not handle: return False
    try:
        code = ctypes.c_ulong()
        if not k.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259: return False
        buf = ctypes.create_unicode_buffer(32768); size = ctypes.c_ulong(len(buf))
        if not k.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)): return False
        if Path(buf.value).resolve() != path.resolve(): return False
        if kill: assert k.TerminateProcess(handle, 99)
        return True
    finally: k.CloseHandle(handle)

with tempfile.TemporaryDirectory(prefix='decoder-deadline-') as tmp:
    root = Path(tmp)
    suffix = '.exe' if os.name == 'nt' else ''
    for tool in ('ffprobe', 'ffmpeg'):
        shutil.copy2(a.stub.resolve(), root / (tool + suffix))
    pid_file = root / 'probe.pid'
    env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ.get('PATH', ''),
               EI_TEST_DECODER_MODE='probe', EI_TEST_DECODER_PID_FILE=str(pid_file))
    with (root / 'service.log').open('w') as log:
        service = subprocess.Popen([str(a.binary.resolve()), '--bind', '127.0.0.1', '--port', str(a.port),
            '--model', str(a.model.resolve()), '--mmproj', str(a.mmproj.resolve()), '--media-encoders', 'vision',
            '--backend', a.backend, '--cache-entries', '0', '--response-cache-mb', '0'], env=env,
            stdout=subprocess.DEVNULL, stderr=log)
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            deadline = time.monotonic() + 180
            while True:
                assert service.poll() is None, 'service exited'
                try:
                    with urllib.request.urlopen(url + '/healthz', timeout=1): break
                except urllib.error.URLError:
                    assert time.monotonic() < deadline, 'service readiness timeout'
                    time.sleep(.2)
            def request(body):
                start = time.perf_counter()
                try:
                    with urllib.request.urlopen(urllib.request.Request(url + '/api/embed', json.dumps(body).encode(),
                        headers={'Content-Type': 'application/json'}), timeout=15) as r:
                        return r.status, json.load(r), (time.perf_counter()-start)*1000
                except urllib.error.HTTPError as e:
                    return e.code, json.load(e), (time.perf_counter()-start)*1000
            video = pool.submit(request, {'input': {'content': [{'type': 'video', 'data': base64.b64encode(b'clip').decode()}]}})
            deadline = time.monotonic() + 5
            while not pid_file.exists():
                assert time.monotonic() < deadline, 'probe did not start'
                time.sleep(.01)
            child = int(pid_file.read_text())
            text = pool.submit(request, {'input': 'task: search result | query: decoder lock probe ' + str(time.time_ns())})
            t = text.result(timeout=16)
            print('Uncached text during stalled probe:', t[0], round(t[2], 1), 'ms', flush=True)
            assert owned_child(child, root / ('ffprobe' + suffix)), 'text waited for the stalled decoder'
            v = video.result(timeout=16)
            print('Stalled probe video/text status and ms:', v[0], round(v[2], 1), t[0], round(t[2], 1), flush=True)
            assert v[0] == 400 and v[2] < 12000
            assert t[0] == 200 and t[2] < 14000
            assert not owned_child(child, root / ('ffprobe' + suffix), parent=service.pid), 'timed-out child was not reaped'
        finally:
            if pid_file.exists(): owned_child(int(pid_file.read_text()), root / ('ffprobe' + suffix), kill=True)
            service.terminate()
            try: service.wait(timeout=15)
            except subprocess.TimeoutExpired: service.kill(); service.wait()
            pool.shutdown(wait=True)
