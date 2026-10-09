#!/usr/bin/env python3
"""Matched, uncached EmbeddingGemma 2 HTTP comparison; retain wins AND losses.

Run on Linux/macOS with independently built native servers. Prints measurements
to stdout; startup, fixture generation and warmup are outside the timed region.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import statistics
import subprocess
import tempfile
import threading
import time

from compare_llamacpp import (Endpoint, ManagedServer, cosine_similarity,
                             generate_exact_prompts, parse_csv_ints,
                             percentile, post_json, run_requests, sample_host, send)


def windows_host_cpu():
    """WSL's process list omits host contention; sample actual Windows CPU time.

    Return aggregate percent in the same units as ps (100 = one logical core).
    Require a working Windows Python rather than silently qualifying an idle VM.
    """
    if 'microsoft' not in platform.release().lower():
        return None
    python = shutil.which('python.exe')
    if not python:
        raise RuntimeError('WSL comparison requires Windows python.exe on PATH for host CPU sampling')
    code = '''import ctypes, os, time
from ctypes import wintypes
get_times = ctypes.WinDLL('kernel32', use_last_error=True).GetSystemTimes
get_times.argtypes = [ctypes.POINTER(wintypes.FILETIME)] * 3
get_times.restype = wintypes.BOOL
def sample():
    times = [wintypes.FILETIME() for _ in range(3)]
    if not get_times(*(ctypes.byref(t) for t in times)):
        raise ctypes.WinError(ctypes.get_last_error())
    return [t.dwLowDateTime | (t.dwHighDateTime << 32) for t in times]
before = sample()
time.sleep(1)
idle, kernel, user = [b - a for a, b in zip(before, sample())]
total = kernel + user
if total <= 0:
    raise RuntimeError('Windows CPU sample interval is empty')
print(100 * (1 - idle / total) * os.cpu_count())
'''
    result = subprocess.run([python, '-c', code], check=True, capture_output=True,
                            text=True, timeout=15)
    load = float(result.stdout.strip())
    if not math.isfinite(load) or load < 0:
        raise RuntimeError('invalid Windows host CPU sample')
    return load


def wait_quiet(ignored, threshold):
    deadline = time.monotonic() + 300
    next_notice = time.monotonic() + 30
    samples = []
    while time.monotonic() < deadline:
        _, load, _ = sample_host(ignored, float('inf'), float('inf'))
        host_load = windows_host_cpu()
        if host_load is not None:
            load = max(load, host_load)
        samples = (samples + [load])[-5:]
        if len(samples) == 5 and statistics.mean(samples) < threshold:
            return statistics.mean(samples)
        if time.monotonic() >= next_notice:
            print(f'waiting for CPU mean below {threshold:.0f}%; '
                  f'current mean {statistics.mean(samples):.1f}%', flush=True)
            next_notice = time.monotonic() + 30
        time.sleep(1)
    raise RuntimeError('host remained busy for five minutes; no comparison published')


def measure(endpoint, bodies, minimum_rounds, seconds, tokens):
    started = []
    barrier = threading.Barrier(len(bodies), action=lambda: started.append(time.perf_counter()))

    def worker(body):
        connection = endpoint.connect()
        latencies = []
        barrier.wait()
        try:
            while len(latencies) < minimum_rounds or time.perf_counter() < started[0] + seconds:
                begin = time.perf_counter()
                send(connection, endpoint, body)
                latencies.append((time.perf_counter() - begin) * 1000)
        finally:
            connection.close()
        return latencies

    with ThreadPoolExecutor(max_workers=len(bodies)) as executor:
        samples = list(executor.map(worker, bodies))
    elapsed = time.perf_counter() - started[0]
    latencies = [x for worker_samples in samples for x in worker_samples]
    return {'requests': len(latencies), 'elapsed_s': elapsed,
            'embeddings_per_s': len(latencies) / elapsed,
            'tokens_per_s': len(latencies) * tokens / elapsed,
            'p50_ms': statistics.median(latencies), 'p95_ms': percentile(latencies, .95)}


def validate(endpoint, text, tokens):
    connection = endpoint.connect()
    try:
        status, payload = post_json(connection, endpoint.path,
                                    {'input': text, 'encoding_format': 'float', 'dimensions': 768})
    finally:
        connection.close()
    if status != 200:
        raise RuntimeError(f'HTTP {status}: {payload[:1000]!r}')
    result = json.loads(payload)
    if endpoint.api == 'llamacpp':
        vector = result['data'][0]['embedding']
        if result['usage']['prompt_tokens'] != tokens:
            raise RuntimeError('llama.cpp token count differs from the fixture')
    else:
        vector = result['embeddings'][0]
    norm = math.sqrt(sum(x * x for x in vector))
    if len(vector) != 768 or not math.isfinite(norm) or abs(norm - 1) > .002:
        raise RuntimeError('invalid embedding dimensions or norm')
    return vector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--embeddinggemma-bin', type=Path, required=True)
    p.add_argument('--llama-server', type=Path, required=True)
    p.add_argument('--backend', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--token-counts', type=parse_csv_ints, default=[32, 256, 1024])
    p.add_argument('--concurrency', type=parse_csv_ints, default=[1, 4])
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--target-seconds', type=float, default=5,
                   help='minimum measured duration per engine and order')
    p.add_argument('--quiet-total-cpu-percent', type=float, default=150,
                   help='five-sample mean non-benchmark CPU limit; 100 is one logical core')
    p.add_argument('--cooldown', type=float, default=2)
    p.add_argument('--port', type=int, default=42674)
    a = p.parse_args()
    if (a.threads < 1 or a.rounds < 3 or a.target_seconds <= 0 or a.cooldown < 0 or
            a.quiet_total_cpu_percent <= 0 or
            max(a.token_counts) > 8192 or max(a.concurrency) > 32):
        p.error('invalid threads, rounds, cooldown, token count or concurrency')
    for path in (a.model, a.embeddinggemma_bin, a.llama_server):
        if not path.is_file():
            p.error(f'not found: {path}')
    for port in (a.port, a.port + 1):
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                p.error(f'comparison port {port} is unavailable')
    os.environ['EI_THREADS'] = str(a.threads)
    ours = Endpoint('127.0.0.1', a.port, '/api/embed', 'embeddinggemma')
    llama = Endpoint('127.0.0.1', a.port + 1, '/v1/embeddings', 'llamacpp')
    batch_tokens = 16384
    ours_cmd = [str(a.embeddinggemma_bin.resolve()), '--model', str(a.model.resolve()),
                '--backend', a.backend, '--bind', '127.0.0.1', '--port', str(ours.port),
                '--cache-entries', '0', '--response-cache-mb', '0', '--workers', '64',
                '--max-batch-tokens', str(batch_tokens), '--max-batch-requests', '32']
    llama_cmd = [str(a.llama_server.resolve()), '-m', str(a.model.resolve()),
                 '--host', '127.0.0.1', '--port', str(llama.port),
                 '--embedding', '--pooling', 'mean', '-ngl',
                 '0' if a.backend == 'cpu' else 'all',
                 '--threads', str(a.threads), '--threads-batch', str(a.threads),
                 '--parallel', '32', '--ctx-size', '16384', '--kv-unified',
                 '--batch-size', str(batch_tokens), '--ubatch-size', str(batch_tokens),
                 '--flash-attn', 'on', '--no-cache-prompt', '--cache-ram', '0',
                 '--no-cache-idle-slots', '--no-webui', '--log-disable']
    if a.backend == 'cpu':
        # Zero offloaded weight layers still permits GPU host-op offload.
        llama_cmd += ['--device', 'none', '--no-op-offload', '--no-kv-offload']
    print(json.dumps({'backend': a.backend, 'threads': a.threads,
                      'packed_qkv': os.getenv('EI_QKV2') == '1',
                      'cuda_global_attn': os.getenv('EI_CUDA_GLOBAL_ATTN2') == '1',
                      'cuda_graph_cache': os.getenv('EI_GRAPH_CACHE2') == '1',
                      'minimum_rounds': a.rounds, 'target_seconds': a.target_seconds,
                      'quiet_total_cpu_percent': a.quiet_total_cpu_percent,
                      'orders': ['ours/llama', 'llama/ours'],
                      'ours_command': ours_cmd, 'llama_command': llama_cmd}), flush=True)
    results = []
    with tempfile.TemporaryDirectory(prefix='embeddinggemma2-comparison-') as tmp:
        root = Path(tmp)
        with ManagedServer(llama_cmd, llama, '/health', root / 'fixtures.log'):
            prompts = generate_exact_prompts(llama, a.token_counts, max(a.concurrency))
        for tokens in a.token_counts:
            for concurrency in a.concurrency:
                print(f'measuring {a.backend}: {tokens} tokens x {concurrency} clients', flush=True)
                # Each cell starts clean: no accumulated state from earlier shapes.
                with ManagedServer(ours_cmd, ours, '/healthz', root / 'ours.log') as op, \
                        ManagedServer(llama_cmd, llama, '/health', root / 'llama.log') as lp:
                    packed_qkv = 'packed QKV:' in op.log_path.read_text(errors='replace')
                    if packed_qkv != (os.getenv('EI_QKV2') == '1'):
                        raise RuntimeError('native server did not select the requested QKV mode')
                    global_attn = 'CUDA global attention fallback:' in op.log_path.read_text(errors='replace')
                    if global_attn != (os.getenv('EI_CUDA_GLOBAL_ATTN2') == '1'):
                        raise RuntimeError('native server did not select the requested global attention mode')
                    graph_cache = 'CUDA short-text graph cache:' in op.log_path.read_text(errors='replace')
                    if graph_cache != (os.getenv('EI_GRAPH_CACHE2') == '1'):
                        raise RuntimeError('native server did not select the requested graph cache mode')
                    vectors = []
                    for endpoint in (ours, llama):
                        with ThreadPoolExecutor(max_workers=concurrency) as executor:
                            vectors.append(list(executor.map(
                                lambda text: validate(endpoint, text, tokens),
                                prompts[tokens][:concurrency])))
                    minimum_cosine = min(cosine_similarity(x, y)
                                         for x, y in zip(*vectors))
                    if minimum_cosine < .999:
                        raise RuntimeError(f'output mismatch: cosine {minimum_cosine:.8f}')
                    bodies = [json.dumps({'input': text, 'encoding_format': 'float',
                                          'dimensions': 768}).encode()
                              for text in prompts[tokens][:concurrency]]
                    for name, endpoint in (('ours', ours), ('llama', llama)):
                        run_requests(endpoint, bodies, 2, tokens)
                    rows = {'ours': [], 'llama': []}
                    loads = []
                    ignored = {os.getpid(), op.process.pid, lp.process.pid}
                    for order in ((('ours', ours), ('llama', llama)),
                                  (('llama', llama), ('ours', ours))):
                        for name, endpoint in order:
                            time.sleep(a.cooldown)
                            load = wait_quiet(ignored, a.quiet_total_cpu_percent)
                            loads.append(load)
                            rows[name].append(measure(endpoint, bodies, a.rounds,
                                                      a.target_seconds, tokens))
                            print(f"  {name}: {rows[name][-1]['embeddings_per_s']:.2f} emb/s "
                                  f"in {rows[name][-1]['elapsed_s']:.2f}s", flush=True)
                    rates = {name: statistics.mean(float(row['embeddings_per_s'])
                                                   for row in runs)
                             for name, runs in rows.items()}
                    row = {'tokens': tokens, 'concurrency': concurrency,
                           'ours_emb_s': rates['ours'], 'llama_emb_s': rates['llama'],
                           'speedup': rates['ours'] / rates['llama'],
                           'minimum_cosine': minimum_cosine,
                           'other_cpu_percent': loads, 'passes': rows}
                    results.append(row)
                    print(json.dumps(row), flush=True)
    print('\n| Tokens | Concurrency | Ours emb/s | llama.cpp emb/s | Speedup |')
    print('| ---: | ---: | ---: | ---: | ---: |')
    for row in results:
        print(f"| {row['tokens']} | {row['concurrency']} | {row['ours_emb_s']:.2f} | "
              f"{row['llama_emb_s']:.2f} | {row['speedup']:.2f}x |")
    print(f"Geometric mean: {math.exp(statistics.mean(math.log(r['speedup']) for r in results)):.2f}x")


if __name__ == '__main__':
    main()
