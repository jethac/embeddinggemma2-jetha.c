"""Numeric variants must not reuse persistent responses from the original path."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import signal
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', required=True, type=Path)
p.add_argument('--model', required=True, type=Path)
p.add_argument('--backend', default='cpu')
p.add_argument('--mode', choices=['packed-qkv', 'cuda-global-attn'], default='packed-qkv')
a = p.parse_args()
flag, marker = ('EI_QKV2', 'packed QKV:') if a.mode == 'packed-qkv' else (
    'EI_CUDA_GLOBAL_ATTN2', 'CUDA global attention fallback:')
with tempfile.TemporaryDirectory(prefix='qkv-cache2-') as directory:
    work = Path(directory)
    cache = work / 'cache'
    with socket.socket() as check:
        check.bind(('127.0.0.1', 0))
        port = check.getsockname()[1]
    command = [str(a.binary.resolve()), '--backend', a.backend,
        '--model', str(a.model.resolve()), '--bind', '127.0.0.1', '--port', str(port),
        '--cache-entries', '0', '--response-cache-mb', '1', '--persistent-cache-path', str(cache)]
    @contextmanager
    def server(log_path):
        with log_path.open('wb') as log:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
            try:
                deadline = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(log_path.read_text(errors='replace'))
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=1): break
                    except OSError:
                        if time.monotonic() > deadline: raise RuntimeError('QKV cache service readiness timeout')
                        time.sleep(.2)
                yield
            finally:
                if process.poll() is None:
                    if os.name == 'nt': process.send_signal(signal.CTRL_BREAK_EVENT)
                    else: process.terminate()
                    try:
                        assert process.wait(timeout=30) == 0, 'graceful cache shutdown failed'
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=10)
                        raise
    def request(probe=False):
        body = {'input': 'q0' + ' x' * 28}
        if probe: body['qkv_cache_probe'] = True
        with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/embed',
            json.dumps(body).encode(), headers={'Content-Type':'application/json'}), timeout=180) as response:
            return json.load(response)
    os.environ['EI_QKV2'] = '0'
    os.environ['EI_CUDA_GLOBAL_ATTN2'] = '0'
    with server(work/'off.log'):
        separate = request()
    identity_before = struct.unpack('<Q', Path(str(cache)+'.responses').read_bytes()[8:16])[0]
    os.environ[flag] = '1'
    with server(work/'on.log'):
        cached = request()
        fresh = request(True)
        assert marker in (work/'on.log').read_text(errors='replace')
        print(a.mode, 'cached/fresh identical:', cached == fresh,
              'original/variant identical:', separate == fresh, flush=True)
        assert cached == fresh, f'{a.mode} reused a response from the original path'
    identity_after = struct.unpack('<Q', Path(str(cache)+'.responses').read_bytes()[8:16])[0]
    assert identity_before != identity_after, f'persistent cache identity did not distinguish {a.mode}'
print(a.mode, 'persistent cache isolation passed')
