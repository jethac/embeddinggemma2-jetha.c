"""Protect raw-input batch isolation, group splitting and API output ordering."""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perf'))
from compare_llamacpp import Endpoint, ManagedServer

p = argparse.ArgumentParser(description=__doc__)
for name in ('binary', 'model', 'mmproj'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--backend', default='cpu')
a = p.parse_args()
os.environ.update(EI_MEDIA_BATCH2='1', EI_REUSE_INPUTS2='1', EI_PROFILE_BACKBONE2='1')
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]


def item(index, word, count):
    return {'content': [{'type': 'text', 'text': 'q' + str(index) + (' ' + word) * count}]}


def post(values, openai=False):
    body = {'input': values}
    if openai:
        body.update(model='embeddinggemma-2', dimensions=256, encoding_format='base64')
    route = '/v1/embeddings' if openai else '/api/embed'
    request = urllib.request.Request(f'http://127.0.0.1:{port}' + route, json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    if openai:
        assert [row['index'] for row in result['data']] == list(range(len(values)))
        vectors = [struct.unpack('<256f', base64.b64decode(row['embedding'])) for row in result['data']]
    else:
        vectors = result['embeddings']
    assert len(vectors) == len(values)
    for vector in vectors:
        assert all(math.isfinite(x) for x in vector)
        assert abs(math.fsum(x*x for x in vector) - 1) < 1e-5
    return vectors, result['usage']['prompt_tokens']


with tempfile.TemporaryDirectory(prefix='media-batch2-') as directory:
    log = Path(directory) / 'service.log'
    command = [str(a.binary.resolve()), '--backend', a.backend, '--bind', '127.0.0.1',
               '--port', str(port), '--model', str(a.model.resolve()), '--mmproj', str(a.mmproj.resolve()),
               '--cache-entries', '0', '--response-cache-mb', '0']
    with ManagedServer(command, Endpoint('127.0.0.1', port, '/api/embed', 'embeddinggemma'), '/healthz', log):
        cases = [
            [item(0, 'x', 27), item(1, 'y', 128), item(2, 'z', 252)],
            [item(3, 'w', 252), item(4, 'a', 27), item(5, 'b', 128)],
            [item(0, 'x', 508), item(1, 'y', 508), item(2, 'z', 508)],
        ]
        # Consecutive equal-total shapes must refresh positions/masks when the
        # sequence boundaries change, without relying on a graph rebuild.
        grouped = [post(values) for values in cases]
        for values, (actual, total) in zip(cases, grouped):
            encoded, encoded_total = post(values, openai=True)
            assert total == encoded_total
            separate_total = 0
            for value, vector, truncated in zip(values, actual, encoded):
                expected, tokens = post([value])
                separate_total += tokens
                cosine = math.fsum(x*y for x, y in zip(vector, expected[0]))
                assert cosine > .999, ('batch isolation', cosine)
                norm = math.sqrt(math.fsum(x*x for x in vector[:256]))
                assert math.fsum(x*y/norm for x, y in zip(vector[:256], truncated)) > .99999
            assert total == separate_total
        trace = log.read_text(errors='replace')
        assert 'Multimodal backbone batching:' in trace
        assert 'tokens=419 batch=3 raw=1' in trace, 'unequal inputs did not use the raw batch graph'
        assert 'tokens=419 batch=3 raw=1 rebuilt=0 layout_reused=0' in trace, 'changed boundaries were not exercised'
        assert 'tokens=1024 batch=2 raw=1' in trace, '1024-token group was not exercised'
        assert 'tokens=512 batch=1 raw=1' in trace, 'split remainder was not exercised'
print('Raw media batch isolation, changed boundaries, group splitting and OpenAI ordering: passed')
