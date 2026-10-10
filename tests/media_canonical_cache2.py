"""Regression: equivalent media JSON/data URLs reuse a validated response."""
import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import struct
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--mmproj', type=Path, required=True)
p.add_argument('--mode', choices=('off', 'on', 'both'), default='both')
a = p.parse_args()


def post(url, wire):
    request = urllib.request.Request(url + '/api/embed', wire.encode(),
                                    headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


frequency = 610 + (uuid.uuid4().int % 1000000) / 1000
buffer = io.BytesIO()
with wave.open(buffer, 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(b''.join(struct.pack('<h', round(4000 * math.sin(
        2 * math.pi * frequency * i / 16000))) for i in range(16000)))
payload = base64.b64encode(buffer.getvalue()).decode()
body = {'input': {'content': [{'type': 'audio', 'data': payload}]}, 'dimensions': 128}
compact = json.dumps(body, separators=(',', ':'))
previous = None
with tempfile.TemporaryDirectory(prefix='media-canonical-cache2-') as directory:
    for flag in (('0', '1') if a.mode == 'both' else ('1' if a.mode == 'on' else '0',)):
        with socket.socket() as check:
            check.bind(('127.0.0.1', 0))
            port = check.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        log_path = Path(directory) / (flag + '.log')
        command = [str(a.binary.resolve()), '--backend', 'cpu', '--bind', '127.0.0.1',
                   '--port', str(port), '--model', str(a.model.resolve()),
                   '--mmproj', str(a.mmproj.resolve()), '--cache-entries', '0',
                   '--response-cache-mb', '64']
        with log_path.open('wb') as log:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log,
                env=dict(os.environ, EI_MEDIA_CANONICAL_CACHE2=flag),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                deadline = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(log_path.read_text(errors='replace'))
                    try:
                        with urllib.request.urlopen(url + '/healthz', timeout=1): break
                    except OSError:
                        if time.monotonic() >= deadline: raise RuntimeError('readiness timeout')
                        time.sleep(.2)
                offset = log_path.stat().st_size
                expected = post(url, compact)
                if previous is not None: assert expected == previous, 'cache option changed response'
                previous = expected
                vector = expected['embeddings'][0]
                assert len(vector) == 128 and all(math.isfinite(x) for x in vector)
                assert abs(math.fsum(x*x for x in vector) - 1) < 1e-5
                assert post(url, compact) == expected, 'raw repeat changed response'
                body['input']['content'][0]['data'] = 'data:audio/wav;base64,' + payload
                pretty = json.dumps(body, indent=2)
                assert post(url, pretty) == expected, 'equivalent media changed response'
                body['input']['content'][0]['data'] = 'data:audio/wav;base64,' + payload[:-1] + '!'
                try:
                    post(url, json.dumps(body))
                    raise AssertionError('malformed base64 accepted')
                except urllib.error.HTTPError as error:
                    assert error.code == 400
                    error.read()
                assert post(url, pretty) == expected, 'cache did not recover after malformed input'
                lines = log_path.read_bytes()[offset:].decode(errors='replace').splitlines()
                inferences = sum(line.startswith('multimodal request:') for line in lines)
                assert inferences == (1 if flag == '1' else 2), (
                    'equivalent media inference count', inferences, flag)
                print(f'Canonical media cache={flag}: response reuse and malformed recovery passed')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
