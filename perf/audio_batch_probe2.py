"""Inspect the observed warm ARM audio discrepancy using real HTTP batches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import tempfile

from compare_llamacpp import Endpoint, ManagedServer, cosine_similarity, post_json
from compare_media_llamacpp2 import fixtures

p = argparse.ArgumentParser(description=__doc__)
for name in ('embeddinggemma-bin', 'llama-server', 'model', 'mmproj'):
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
os.environ['EI_THREADS'] = '2'
golden = json.loads(Path('testdata/audio-reference2.json').read_text())
assert golden['revision'] == '914f7f89142e33e77833254d9c9b90c3cef7303b'
assert (golden['sample_rate'], golden['frequency'], golden['amplitude']) == (16000, 660, 4000)
reference = next(s['embedding'] for s in golden['fixtures'] if s['seconds'] == 1)
ours = Endpoint('127.0.0.1', 42674, '/v1/embeddings', 'embeddinggemma')
llama = Endpoint('127.0.0.1', 42675, '/v1/embeddings', 'llamacpp')
ours_cmd = [str(a.embeddinggemma_bin.resolve()), '--backend', 'cpu', '--bind', ours.host,
            '--port', str(ours.port), '--model', str(a.model.resolve()), '--mmproj', str(a.mmproj.resolve()),
            '--cache-entries', '0', '--response-cache-mb', '0']
llama_cmd = [str(a.llama_server.resolve()), '-m', str(a.model.resolve()), '--mmproj', str(a.mmproj.resolve()),
             '--host', llama.host, '--port', str(llama.port), '--embedding', '--pooling', 'mean',
             '-ngl', '0', '--threads', '2', '--threads-batch', '2', '--parallel', '32',
             '--ctx-size', '16384', '--kv-unified', '--batch-size', '16384', '--ubatch-size', '16384',
             '--flash-attn', 'on', '--no-cache-prompt', '--cache-ram', '0', '--no-cache-idle-slots',
             '--no-webui', '--device', 'none', '--no-op-offload', '--no-kv-offload', '--no-mmproj-offload',
             '--log-verbosity', '5']
llama_env = {k: v for k, v in os.environ.items() if not k.startswith('EI_')}
llama_env.update(EI_CPU_AUDIO_F16_2='1', EI_ARM_FP16_ACC_F32='1', LLAMA_BATCH_DEBUG='1')

def request(endpoint, body):
    connection = endpoint.connect()
    try:
        status, payload = post_json(connection, endpoint.path, body)
    finally:
        connection.close()
    if status != 200:
        raise RuntimeError(f'HTTP {status}: {payload[:1000]!r}')
    result = json.loads(payload)
    vectors = [row['embedding'] for row in sorted(result['data'], key=lambda row: row['index'])]
    for vector in vectors:
        assert len(vector) == 768 and all(math.isfinite(v) for v in vector)
        assert abs(sum(v*v for v in vector) - 1) < 1e-5
    return vectors

with tempfile.TemporaryDirectory(prefix='audio-batch-probe-') as directory:
    root = Path(directory)
    cases = fixtures(root, 4)['audio']
    with ManagedServer(ours_cmd, ours, '/healthz', root / 'ours.log') as op, \
         ManagedServer(llama_cmd, llama, '/health', root / 'llama.log', env=llama_env) as lp:
        modules = []
        for process in (op, lp):
            assert 'CPU audio F16 active: 132 Conformer matrices' in process.log_path.read_text()
            maps = Path(f'/proc/{process.process.pid}/maps').read_text()
            loaded = {Path(line.split()[-1]).resolve() for line in maps.splitlines()
                      if 'libggml-cpu-armv8_dotprod_fp16.so' in line}
            assert len(loaded) == 1
            modules.append(loaded.pop())
        assert modules[0] == modules[1]
        print(json.dumps({'shared_module': str(modules[0])}), flush=True)
        for engine, endpoint, process in ((0, ours, op), (1, llama, lp)):
            bodies = [pair[engine] for pair in cases]
            serial = request(endpoint, bodies[3])[0]
            previous = {}
            def run(label, order, array):
                offset = process.log_path.stat().st_size
                if array:
                    body = dict(bodies[0], input=[bodies[i]['input'] for i in order])
                    vectors = request(endpoint, body)
                else:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        vectors = [result[0] for result in pool.map(lambda i: request(endpoint, bodies[i]), order)]
                assert len(vectors) == 4
                vector = vectors[order.index(3)]
                print(json.dumps({'engine': engine, 'scenario': label, 'order': order,
                      'HF_660_cosine': cosine_similarity(vector, reference),
                      'serial_660_cosine': cosine_similarity(vector, serial),
                      'repeat_660_cosine': cosine_similarity(vector, previous[label]) if label in previous else None}), flush=True)
                previous[label] = vector
                # Keep the actual batch geometry, positions and sequence IDs.
                print(process.log_path.read_bytes()[offset:].decode(errors='replace'), flush=True)
            print(json.dumps({'engine': engine, 'serial_HF_660_cosine': cosine_similarity(serial, reference)}), flush=True)
            run('array', [0, 1, 2, 3], True)
            for _ in range(2):
                run('wave', [0, 1, 2, 3], False)
            run('array', [0, 1, 2, 3], True)
            run('rotated_array', [3, 0, 1, 2], True)
            run('array', [0, 1, 2, 3], True)
