#!/usr/bin/env python3
"""Compare uncached multimodal HTTP journeys with pinned llama.cpp."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
import re
from pathlib import Path
import socket
import statistics
import struct
import subprocess
import tempfile
import time
import wave

from compare_llamacpp import Endpoint, ManagedServer, cosine_similarity, parse_csv_ints, post_json, run_requests
from compare_llamacpp2 import measure, wait_quiet


def fixtures(directory, count):
    cases = {kind: [] for kind in ('image', 'audio', 'video', 'mixed')}
    for i in range(count):
        color = ((220 + 37 * i) % 256, (30 + 67 * i) % 256, (30 + 109 * i) % 256)
        image = b'P6\n96 96\n255\n' + bytes(color) * (96 * 96)
        audio = io.BytesIO()
        with wave.open(audio, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(b''.join(struct.pack('<h', round(4000 * math.sin(
                2 * math.pi * (330 + 110 * i) * t / 16000))) for t in range(16000)))
        clip = directory / f'video-{i}.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
            '-i', f'color=0x{bytes(color).hex()}:size=96x96:rate=1:duration=2',
            '-c:v', 'mpeg4', str(clip)], check=True, timeout=60)
        media = {'image': ('image/x-portable-pixmap', image),
                 'audio': ('audio/wav', audio.getvalue()), 'video': ('video/mp4', clip.read_bytes())}
        parts = {}
        for kind, (mime, data) in media.items():
            url = f'data:{mime};base64,' + base64.b64encode(data).decode()
            ours = {'type': kind, 'data': url}
            if kind == 'image':
                llama = {'type': 'image_url', 'image_url': {'url': url}}
            else:
                llama = {'type': 'input_' + kind, 'input_' + kind:
                         {'data': url, 'format': 'wav' if kind == 'audio' else 'mp4'}}
            parts[kind] = (ours, llama)
        text = {'type': 'text', 'text': f'q{i} A colored square and a tone.'}
        for kind in cases:
            contents = ([parts[kind]] if kind != 'mixed' else
                        [(text, text), parts['image'], parts['audio']])
            if kind == 'video':
                # llama.cpp's video helper inserts this prose automatically;
                # native visual-only video does not. Match the model input.
                contents.insert(0, ({'type': 'text', 'text': 'Video:'}, None))
            cases[kind].append(tuple({'model': 'embeddinggemma-2',
                                     'input': {'content': [p[engine] for p in contents if p[engine] is not None]},
                                     'dimensions': 768, 'encoding_format': 'float'}
                                    for engine in (0, 1)))
    return cases


def validate(endpoint, body):
    connection = endpoint.connect()
    try:
        status, payload = post_json(connection, endpoint.path, body)
    finally:
        connection.close()
    if status != 200:
        raise RuntimeError(f'HTTP {status}: {payload[:1000]!r}')
    result = json.loads(payload)
    vector = result['data'][0]['embedding']
    if len(vector) != 768 or not all(math.isfinite(v) for v in vector):
        raise RuntimeError('invalid multimodal embedding')
    if abs(sum(v * v for v in vector) - 1) > .002:
        raise RuntimeError('multimodal embedding is not normalized')
    return vector, result['usage']['prompt_tokens']


def diagnose_quality(kind, inputs, endpoints, concurrent):
    # A failed concurrent gate must distinguish serving/batch drift from a
    # model discrepancy. These calls are outside timing and never relax it.
    for i, pair in enumerate(inputs):
        serial = [validate(endpoint, pair[engine])
                  for engine, endpoint in enumerate(endpoints)]
        print(json.dumps({'quality_failure': kind, 'client': i,
                          'tokens': [concurrent[0][i][1], concurrent[1][i][1],
                                     serial[0][1], serial[1][1]],
                          'concurrent_cross_cosine': cosine_similarity(concurrent[0][i][0], concurrent[1][i][0]),
                          'serial_cross_cosine': cosine_similarity(serial[0][0], serial[1][0]),
                          'ours_concurrent_vs_serial': cosine_similarity(concurrent[0][i][0], serial[0][0]),
                          'llama_concurrent_vs_serial': cosine_similarity(concurrent[1][i][0], serial[1][0])}),
              flush=True)


def measure_native_pair(on, off, bodies, ignored, options):
    passes = {'on': [], 'off': []}
    loads = []
    for endpoint in (on, off):
        run_requests(endpoint, bodies, 2, 0)
    for order in ((('on', on), ('off', off)), (('off', off), ('on', on))):
        for name, endpoint in order:
            time.sleep(options.cooldown)
            loads.append(wait_quiet(ignored, options.quiet_total_cpu_percent))
            measured = measure(endpoint, bodies, options.rounds, options.target_seconds, 0)
            measured.pop('tokens_per_s')
            passes[name].append(measured)
    rates = {name: statistics.mean(run['embeddings_per_s'] for run in runs)
             for name, runs in passes.items()}
    return {'on_emb_s': rates['on'], 'off_emb_s': rates['off'],
            'speedup': rates['on'] / rates['off'], 'passes': passes,
            'other_cpu_percent': loads}


def phase_costs(path, offset):
    with path.open('rb') as log:
        log.seek(offset)
        lines = log.read().decode(errors='replace').splitlines()
    result = {}
    for name, marker, fields in (
            ('encoder', 'encoder phases:', ('build', 'alloc', 'inputs', 'compute')),
            ('backbone', 'backbone:', ('build', 'prep', 'compute', 'output'))):
        samples = [tuple(float(re.search(r'\b' + field + r'=([0-9.]+)', line).group(1))
                         for field in fields) for line in lines if marker in line]
        if samples:
            result[name] = {'samples': len(samples),
                            'median_ms': dict(zip(fields, map(statistics.median, zip(*samples))))}
    if 'encoder' not in result:
        raise RuntimeError(f'encoder phase profiling produced no samples: {path}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--mmproj', type=Path, required=True)
    p.add_argument('--embeddinggemma-bin', type=Path, required=True)
    p.add_argument('--llama-server', type=Path, required=True)
    p.add_argument('--backend', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--profile-phases', action='store_true',
                   help='Diagnose encoder/backbone costs from actual server logs')
    p.add_argument('--concurrency', type=parse_csv_ints, default=[1, 4])
    p.add_argument('--modalities', default='image,audio,video,mixed')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--target-seconds', type=float, default=8)
    p.add_argument('--quiet-total-cpu-percent', type=float, default=150)
    p.add_argument('--cooldown', type=float, default=2)
    p.add_argument('--port', type=int, default=42674)
    p.add_argument('--validate-only', action='store_true', help='Check journeys and quality without timing')
    a = p.parse_args()
    modalities = a.modalities.split(',')
    if not modalities or any(k not in ('image', 'audio', 'video', 'mixed') for k in modalities):
        p.error('modalities must be image,audio,video,mixed or a subset')
    if (a.threads < 1 or min(a.concurrency) < 1 or max(a.concurrency) > 32 or
            a.rounds < 3 or a.target_seconds <= 0 or a.cooldown < 0 or a.quiet_total_cpu_percent <= 0):
        p.error('invalid threads, concurrency, rounds or measurement duration')
    if not a.validate_only and any(k in os.environ for k in ('EI_PROFILE_BACKBONE2', 'EI_PROFILE_MEDIA2')):
        p.error('unset profiling flags before timing')
    for path in (a.model, a.mmproj, a.embeddinggemma_bin, a.llama_server):
        if not path.is_file(): p.error(f'not found: {path}')
    vnni_probe = os.getenv('EI_CPU_Q8_PAIR_VNNI2') == '1'
    if vnni_probe:
        cpuinfo = Path('/proc/cpuinfo')
        if a.backend != 'cpu' or not cpuinfo.is_file() or 'avx512_vnni' not in cpuinfo.read_text().split():
            p.error('VNNI experiment requires a Linux AVX-512 VNNI CPU')
    for port in ((a.port, a.port + 1, a.port + 2) if vnni_probe else (a.port, a.port + 1)):
        with socket.socket() as check: check.bind(('127.0.0.1', port))
    os.environ['EI_THREADS'] = str(a.threads)
    if a.profile_phases:
        os.environ['EI_PROFILE_MEDIA2'] = '1'
        os.environ['EI_PROFILE_BACKBONE2'] = '1'
    ours = Endpoint('127.0.0.1', a.port, '/v1/embeddings', 'embeddinggemma')
    llama = Endpoint('127.0.0.1', a.port + 1, '/v1/embeddings', 'llamacpp')
    ours_cmd = [str(a.embeddinggemma_bin.resolve()), '--model', str(a.model.resolve()),
                '--mmproj', str(a.mmproj.resolve()), '--backend', a.backend, '--bind', ours.host,
                '--port', str(ours.port), '--cache-entries', '0', '--response-cache-mb', '0']
    llama_cmd = [str(a.llama_server.resolve()), '-m', str(a.model.resolve()),
                 '--mmproj', str(a.mmproj.resolve()), '--host', llama.host, '--port', str(llama.port),
                 '--embedding', '--pooling', 'mean', '-ngl', '0' if a.backend == 'cpu' else 'all',
                 '--threads', str(a.threads), '--threads-batch', str(a.threads), '--parallel', '32',
                 '--ctx-size', '16384', '--kv-unified', '--batch-size', '16384', '--ubatch-size', '16384',
                 '--flash-attn', 'on', '--no-cache-prompt', '--cache-ram', '0',
                 '--no-cache-idle-slots', '--no-webui', '--log-disable']
    if a.backend == 'cpu':
        llama_cmd += ['--device', 'none', '--no-op-offload', '--no-kv-offload', '--no-mmproj-offload']
    if a.profile_phases:
        llama_cmd.remove('--log-disable')
        llama_cmd += ['--log-verbosity', '4']
    print(json.dumps({'backend': a.backend, 'threads': a.threads,
        'flags': {k: v for k, v in os.environ.items() if k.startswith('EI_')},
        'validate_only': a.validate_only, 'orders': ['ours/llama', 'llama/ours'],
        'ours_command': ours_cmd, 'llama_command': llama_cmd,
        'fixtures': '96x96 RGB PPM; 1s 16kHz mono PCM WAV; 2s 1fps 96x96 MPEG4; text+image+audio',
        'native_video_prefix': 'Video: (matches llama.cpp helper insertion)',
        'rounds': a.rounds, 'target_seconds': a.target_seconds,
        'quiet_total_cpu_percent': a.quiet_total_cpu_percent}), flush=True)
    results = []
    with tempfile.TemporaryDirectory(prefix='embeddinggemma2-media-comparison-') as tmp:
        root = Path(tmp); cases = fixtures(root, max(a.concurrency))
        for kind in modalities:
            inputs = cases[kind]
            for concurrency in a.concurrency:
                print(f'{kind}: {concurrency} distinct clients', flush=True)
                # Native patch budgets floor the square grid: 280 -> 16x16,
                # 140 -> 11x11 per video frame. Match those actual grids.
                grid_tokens = 121 if kind == 'video' else 256
                matched_llama_cmd = llama_cmd + ['--image-min-tokens', str(grid_tokens),
                    '--image-max-tokens', str(grid_tokens), '--video-fps', '1',
                    '--video-timestamp-interval', '0']
                print(json.dumps({'llama_command': matched_llama_cmd}), flush=True)
                with ManagedServer(ours_cmd, ours, '/healthz', root / 'ours.log') as op, \
                     ManagedServer(matched_llama_cmd, llama, '/health', root / 'llama.log',
                                   env={k: v for k, v in os.environ.items() if k not in ('EI_CPU_Q8_PAIR2', 'EI_CPU_Q8_PAIR_VNNI2')}) as lp:
                    log_offsets = (op.log_path.stat().st_size, lp.log_path.stat().st_size)
                    if vnni_probe:
                        startup = op.log_path.read_text(errors='replace')
                        if not any('libggml-cpu-' + name + '.so' in startup for name in
                                   ('cascadelake', 'cooperlake', 'icelake', 'sapphirerapids', 'zen4')):
                            raise RuntimeError('VNNI experiment did not load a VNNI CPU plugin')
                    rows = {'ours': [], 'llama': []}; quality = []
                    for engine, endpoint in enumerate((ours, llama)):
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            quality.append(list(pool.map(lambda pair: validate(endpoint, pair[engine]),
                                                         inputs[:concurrency])))
                    tokens = [v[1] for v in quality[0]]
                    if tokens != [v[1] for v in quality[1]]:
                        raise RuntimeError(f'{kind} token counts differ: {tokens} vs {[v[1] for v in quality[1]]}')
                    minimum = min(cosine_similarity(x[0], y[0]) for x, y in zip(*quality))
                    if minimum < .999:
                        diagnose_quality(kind, inputs[:concurrency], (ours, llama), quality)
                        raise RuntimeError(f'{kind} output mismatch: cosine {minimum:.8f}')
                    native_pair = None
                    if vnni_probe:
                        off = Endpoint(ours.host, ours.port + 2, ours.path, ours.api)
                        command = ours_cmd.copy()
                        command[command.index('--port') + 1] = str(off.port)
                        with ManagedServer(command, off, '/healthz', root / 'native-off.log',
                                env={k: v for k, v in os.environ.items()
                                     if k not in ('EI_CPU_Q8_PAIR2', 'EI_CPU_Q8_PAIR_VNNI2')}) as off_process:
                            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                                unchanged = list(pool.map(lambda pair: validate(off, pair[0]),
                                                          inputs[:concurrency]))
                            if unchanged != quality[0]:
                                raise RuntimeError(f'{kind}: VNNI paired rows changed native outputs')
                            print('VNNI paired rows: exact native on/off outputs', flush=True)
                            if not a.validate_only:
                                native_bodies = [json.dumps(pair[0], separators=(',', ':')).encode()
                                                 for pair in inputs[:concurrency]]
                                native_pair = measure_native_pair(ours, off, native_bodies,
                                    {os.getpid(), op.process.pid, lp.process.pid, off_process.process.pid}, a)
                                print(json.dumps({'modality': kind, 'concurrency': concurrency,
                                                  'native_pair_on_off': native_pair}), flush=True)
                    row = {'modality': kind, 'concurrency': concurrency, 'tokens': tokens,
                           'minimum_cosine': minimum}
                    if a.profile_phases:
                        row['phase_costs'] = {name: phase_costs(process.log_path, offset)
                            for name, process, offset in zip(('ours', 'llama'), (op, lp), log_offsets)}
                    if native_pair is not None:
                        row['native_pair_on_off'] = native_pair
                    print(json.dumps(row), flush=True)
                    if a.validate_only: continue
                    bodies = [[json.dumps(pair[engine], separators=(',', ':')).encode()
                               for pair in inputs[:concurrency]] for engine in (0, 1)]
                    for engine, endpoint in enumerate((ours, llama)):
                        run_requests(endpoint, bodies[engine], 2, 0)
                    ignored = {os.getpid(), op.process.pid, lp.process.pid}; loads = []
                    for order in (((0, 'ours', ours), (1, 'llama', llama)),
                                  ((1, 'llama', llama), (0, 'ours', ours))):
                        for engine, name, endpoint in order:
                            time.sleep(a.cooldown)
                            loads.append(wait_quiet(ignored, a.quiet_total_cpu_percent))
                            measured = measure(endpoint, bodies[engine], a.rounds, a.target_seconds, 0)
                            measured.pop('tokens_per_s')  # Mixed inputs need no synthetic text-token rate.
                            rows[name].append(measured)
                    rates = {name: statistics.mean(r['embeddings_per_s'] for r in runs)
                             for name, runs in rows.items()}
                    row.update(ours_emb_s=rates['ours'], llama_emb_s=rates['llama'],
                               speedup=rates['ours']/rates['llama'], passes=rows, other_cpu_percent=loads)
                    if a.profile_phases:
                        row['phase_costs'] = {name: phase_costs(process.log_path, offset)
                            for name, process, offset in zip(('ours', 'llama'), (op, lp), log_offsets)}
                    results.append(row); print(json.dumps(row), flush=True)
    if results:
        print('\n| Modality | Clients | Ours emb/s | llama.cpp emb/s | Ratio |')
        print('| --- | ---: | ---: | ---: | ---: |')
        for r in results:
            print(f"| {r['modality']} | {r['concurrency']} | {r['ours_emb_s']:.2f} | {r['llama_emb_s']:.2f} | {r['speedup']:.3f}x |")
        print(f"Geometric mean: {math.exp(statistics.mean(math.log(r['speedup']) for r in results)):.3f}x")


if __name__ == '__main__': main()
