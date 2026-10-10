"""Regression: sizing an audio graph must not corrupt its first execution."""
import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import urllib.request
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perf'))
from compare_llamacpp import Endpoint, ManagedServer

p = argparse.ArgumentParser(description=__doc__)
for name in ('binary', 'model', 'mmproj'):
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
audio = io.BytesIO()
with wave.open(audio, 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(b''.join(struct.pack('<h', round(4000 * math.sin(
        2 * math.pi * 330 * i / 16000))) for i in range(4000)))
body = json.dumps({'input': {'content': [
    {'type': 'text', 'text': 'Input 0.'},
    {'type': 'audio', 'data': base64.b64encode(audio.getvalue()).decode()}]},
    'dimensions': 768}).encode()

results = []
with tempfile.TemporaryDirectory(prefix='audio-graph-cache2-') as directory:
    for slots in (1, 2):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        endpoint = Endpoint('127.0.0.1', port, '/api/embed', 'embeddinggemma')
        command = [str(a.binary.resolve()), '--backend', 'cuda', '--bind',
                   '127.0.0.1', '--port', str(port), '--model', str(a.model.resolve()),
                   '--mmproj', str(a.mmproj.resolve()), '--media-encoders', 'audio',
                   '--cache-entries', '0', '--response-cache-mb', '0']
        env = dict(os.environ, EI_AUDIO_GRAPH_CACHE2='1',
                   EI_AUDIO_GRAPH_CACHE_SLOTS2=str(slots))
        log = Path(directory) / f'slots{slots}.log'
        with ManagedServer(command, endpoint, '/healthz', log, env=env):
            request = urllib.request.Request(
                f'http://127.0.0.1:{port}/api/embed', body,
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
            vector = result['embeddings'][0]
            assert len(vector) == 768 and all(math.isfinite(x) for x in vector)
            assert abs(math.fsum(x*x for x in vector) - 1) < 1e-5
            assert 'CLIP using CUDA0 backend' in log.read_text(errors='replace')
            if slots == 2:
                trace = log.read_text(errors='replace')
                assert 'audio graph cache: active slots=2' in trace
                assert 'audio graph cache: retained nx=24 ny=128' in trace
            results.append((vector, result['usage']))

assert results[0][1] == results[1][1], 'audio token counts differ'
cosine = math.fsum(x*y for x, y in zip(results[0][0], results[1][0]))
assert cosine >= .999, f'first audio graph execution: cosine {cosine:.8f}'
print(f'First audio graph execution: cosine {cosine:.8f}')
