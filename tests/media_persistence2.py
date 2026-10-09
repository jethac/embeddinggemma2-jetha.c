"""Restart journey: media responses persist without recomputing or crossing APIs."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--mmproj', type=Path, required=True)
p.add_argument('--backend', default='cpu')
p.add_argument('--port', type=int, default=42674)
a = p.parse_args()
if os.name == 'nt':
    import ctypes
    import signal
    # Own a hidden console only if this runner has none; Ctrl-Break targets
    # the child process group, never another service or the parent shell.
    console_processes = (ctypes.c_ulong * 1)()
    if not ctypes.windll.kernel32.GetConsoleProcessList(console_processes, 1):
        assert ctypes.windll.kernel32.AllocConsole()
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
url = f'http://127.0.0.1:{a.port}'
image = b'P6\n160 160\n255\n' + bytes((255, 0, 0)) * (160 * 160)
body = json.dumps({'model': 'embeddinggemma-2', 'dimensions': 128,
    'input': {'content': [{'type': 'image', 'data': base64.b64encode(image).decode()}]}}).encode()

def request(route):
    start = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(url + route, body,
        headers={'Content-Type': 'application/json'}), timeout=180) as r:
        out = json.load(r)
    return out, (time.perf_counter() - start) * 1000

with tempfile.TemporaryDirectory(prefix='media-restart-') as tmp:
    cache = Path(tmp) / 'cache'
    def start(label, encoders='all'):
        log = Path(tmp) / (label + '.log')
        with log.open('w') as err:
            process = subprocess.Popen([str(a.binary.resolve()), '--bind', '127.0.0.1',
                '--port', str(a.port), '--backend', a.backend, '--model', str(a.model.resolve()),
                '--mmproj', str(a.mmproj.resolve()), '--media-encoders', encoders,
                '--persistent-cache-path', str(cache), '--cache-entries', '0',
                '--response-cache-mb', '16'], stdout=subprocess.DEVNULL, stderr=err,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        deadline = time.monotonic() + 180
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(log.read_text(errors='replace'))
                try:
                    with urllib.request.urlopen(url + '/healthz', timeout=1):
                        return process, log
                except urllib.error.URLError:
                    time.sleep(.2)
            raise AssertionError('service readiness timeout')
        except BaseException:
            process.terminate(); process.wait(timeout=30)
            raise
    def stop(process):
        if os.name == 'nt':
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        try:
            assert process.wait(timeout=30) == 0, 'graceful cache shutdown failed'
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
            raise
    process, log = start('before')
    try:
        before = [request(route) for route in ('/api/embed', '/v1/embeddings')]
    finally:
        stop(process)
    assert log.read_text(errors='replace').count('multimodal request:') == 2
    process, log = start('after')
    try:
        after = [request(route) for route in ('/api/embed', '/v1/embeddings')]
    finally:
        stop(process)
    calls = log.read_text(errors='replace').count('multimodal request:')
    print('Media restart inference calls:', calls, 'before/after ms:',
          [[round(x[1], 2) for x in run] for run in (before, after)], flush=True)
    assert calls == 0, 'media results were recomputed after restart'
    assert [x[0] for x in before] == [x[0] for x in after]
    assert 'embeddings' in after[0][0] and 'data' in after[1][0]
    process, log = start('audio-only', 'audio')
    try:
        try:
            request('/api/embed')
            raise AssertionError('cached response bypassed disabled vision encoder')
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        stop(process)
    print('Restart cache preserves response formats and encoder configuration', flush=True)
