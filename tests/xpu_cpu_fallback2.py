"""Regression: XPU plugin discovery must allow native CPU without a SYCL GPU."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', required=True, type=Path)
parser.add_argument('--model', required=True, type=Path)
args = parser.parse_args()
binary, model = args.binary.resolve(), args.model.resolve()
assert binary.is_file() and model.is_file()
env = {key: value for key, value in os.environ.items() if not key.startswith('EI_')}
env.update(EI_THREADS='2', ONEAPI_DEVICE_SELECTOR='!*:*')
env.pop('GGML_BACKEND_PATH', None)
env.pop('LD_LIBRARY_PATH', None)
with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
with tempfile.TemporaryDirectory(prefix='xpu-cpu-fallback-') as directory:
    log_path = Path(directory) / 'service.log'
    with log_path.open('wb') as log:
        process = subprocess.Popen(
            [str(binary), '--backend', 'cpu', '--bind', '127.0.0.1', '--port', str(port),
             '--model', str(model), '--cache-entries', '0', '--response-cache-mb', '0'],
            env=env, cwd=directory, stdout=subprocess.DEVNULL, stderr=log,
        )
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text(errors='replace'))
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=1) as reply:
                        assert json.load(reply)['status'] == 'ok'
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('XPU package CPU fallback did not become ready')
                    time.sleep(.1)
            assert 'no visible SYCL GPU devices; skipping backend' in log_path.read_text()
            body = json.dumps({'input': 'task: search result | query: what powers the cell'}).encode()
            request = urllib.request.Request(f'http://127.0.0.1:{port}/api/embed', body,
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=120) as reply:
                vector = json.load(reply)['embeddings'][0]
            assert len(vector) == 768 and all(math.isfinite(value) for value in vector)
            assert abs(math.fsum(value * value for value in vector) - 1) < 1e-5
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
print('XPU package CPU fallback with no visible SYCL GPU: passed')
