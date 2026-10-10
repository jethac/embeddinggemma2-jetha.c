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
    cases = {kind: [] for kind in ('text', 'image', 'audio', 'video', 'mixed')}
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
            if kind == 'text':
                body = {'model': 'embeddinggemma-2',
                        'input': 'task: search result | query: ' + text['text'],
                        'dimensions': 768, 'encoding_format': 'float'}
                cases[kind].append((body, body))
                continue
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


def phase_costs(path, offset, require_encoder=True):
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
    if require_encoder and 'encoder' not in result:
        raise RuntimeError(f'encoder phase profiling produced no samples: {path}')
    result['backbone_messages'] = [line for line in lines if 'backbone:' in line]
    result['cpu_graph_messages'] = [line for line in lines if any(marker in line for marker in
        ('cpu graph profile:', 'cpu op profile:', 'cpu node profile:', 'cpu worker profile:'))]
    result['llama_eval_messages'] = [line for line in lines if
                                   'prompt eval time' in line or 'eval time =' in line]
    # AUTO can disable unsupported flash attention while the reference forces
    # it on. Preserve the actual encoder decisions before choosing kernel work.
    # llama.cpp warms the encoder during startup, before the phase offset.
    backend_lines = path.read_text(errors='replace').splitlines()
    result['encoder_backend_messages'] = [line for line in backend_lines if any(marker in line for marker in
        ('flash attention is ', 'flash attention not supported by ', 'unsupported operators by the backend'))]
    return result


def check_available_port(port):
    with socket.socket() as check:
        # Match the Unix servers: closed keep-alive connections may still be
        # in TIME_WAIT. A live listener must continue to reject this bind.
        if os.name != 'nt':
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(('127.0.0.1', port))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--mmproj', type=Path, required=True)
    p.add_argument('--embeddinggemma-bin', type=Path, required=True)
    p.add_argument('--llama-server', type=Path, required=True)
    p.add_argument('--backend', choices=['cpu', 'cuda', 'metal'], required=True)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--profile-phases', action='store_true',
                   help='Diagnose encoder/backbone costs from actual server logs')
    p.add_argument('--vision-clip-metadata', action='store_true',
                   help='Check and measure native vision metadata clamping on/off')
    p.add_argument('--metal-media-flash-attn', action='store_true',
                   help='Check and measure forced encoder flash attention against AUTO on Metal')
    p.add_argument('--concurrency', type=parse_csv_ints, default=[1, 4])
    p.add_argument('--modalities', default='image,audio,video,mixed')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--target-seconds', type=float, default=8)
    p.add_argument('--quiet-total-cpu-percent', type=float, default=150)
    p.add_argument('--cooldown', type=float, default=2)
    p.add_argument('--port', type=int, default=42674)
    p.add_argument('--audio-reference', type=Path, action='append', default=[],
                   help='Check both CPU engines against original FP32 SDPA audio before timing')
    p.add_argument('--validate-only', action='store_true', help='Check journeys and quality without timing')
    a = p.parse_args()
    modalities = a.modalities.split(',')
    if not modalities or any(k not in ('text', 'image', 'audio', 'video', 'mixed') for k in modalities):
        p.error('modalities must be text,image,audio,video,mixed or a subset')
    if a.backend == 'cpu' and any(k in modalities for k in ('audio', 'mixed')) and os.getenv('EI_CPU_AUDIO_F16_2') != '1':
        p.error('CPU audio/mixed comparisons require EI_CPU_AUDIO_F16_2=1 for both encoders')
    if (a.threads < 1 or min(a.concurrency) < 1 or max(a.concurrency) > 32 or
            a.rounds < 3 or a.target_seconds <= 0 or a.cooldown < 0 or a.quiet_total_cpu_percent <= 0):
        p.error('invalid threads, concurrency, rounds or measurement duration')
    if not a.validate_only and any(k in os.environ for k in ('EI_PROFILE_BACKBONE2', 'EI_PROFILE_MEDIA2')):
        p.error('unset profiling flags before timing')
    for path in (a.model, a.mmproj, a.embeddinggemma_bin, a.llama_server):
        if not path.is_file(): p.error(f'not found: {path}')
    vnni_probe = os.getenv('EI_CPU_Q8_PAIR_VNNI2') == '1'
    if sum((vnni_probe, a.vision_clip_metadata, a.metal_media_flash_attn)) > 1:
        p.error('select one native on/off experiment at a time')
    native_probe = vnni_probe or a.vision_clip_metadata or a.metal_media_flash_attn
    probe_name = ('VNNI paired rows' if vnni_probe else 'Metal encoder flash attention'
                  if a.metal_media_flash_attn else 'vision metadata clipping')
    off_flags = (('EI_CPU_Q8_PAIR2', 'EI_CPU_Q8_PAIR_VNNI2') if vnni_probe
                 else ('EI_METAL_MEDIA_FLASH_ATTN2',) if a.metal_media_flash_attn
                 else ('EI_VISION_CLIP_METADATA2',))
    if a.vision_clip_metadata:
        os.environ['EI_VISION_CLIP_METADATA2'] = '1'
    if a.metal_media_flash_attn:
        if a.backend != 'metal':
            p.error('encoder flash attention experiment requires Metal')
        os.environ['EI_METAL_MEDIA_FLASH_ATTN2'] = '1'
    if vnni_probe:
        cpuinfo = Path('/proc/cpuinfo')
        if a.backend != 'cpu' or not cpuinfo.is_file() or 'avx512_vnni' not in cpuinfo.read_text().split():
            p.error('VNNI experiment requires a Linux AVX-512 VNNI CPU')
    for port in ((a.port, a.port + 1, a.port + 2) if native_probe else (a.port, a.port + 1)):
        check_available_port(port)
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
    # Keep native experiments isolated. The CPU audio quality fix belongs
    # to the shared encoder loader and must be enabled in both engines.
    llama_env = {k: v for k, v in os.environ.items() if not k.startswith('EI_')}
    cpu_audio_f16 = a.backend == 'cpu' and os.getenv('EI_CPU_AUDIO_F16_2') == '1'
    if cpu_audio_f16:
        llama_env['EI_CPU_AUDIO_F16_2'] = '1'
    arm_precise = a.backend == 'cpu' and os.getenv('EI_ARM_FP16_ACC_F32') == '1'
    if arm_precise:
        llama_env['EI_ARM_FP16_ACC_F32'] = '1'
    if a.audio_reference and not cpu_audio_f16:
        p.error('audio references require the matched CPU audio F16 loader')
    if a.profile_phases:
        # The patched encoder's diagnostic timer is shared by both engines.
        # Keep optimization flags isolated even during a profiled comparison.
        llama_env['EI_PROFILE_MEDIA2'] = '1'
        if os.getenv('EI_CPU_GRAPH_PROFILE2') == '1':
            llama_env['EI_CPU_GRAPH_PROFILE2'] = '1'
    if a.backend == 'cpu':
        llama_cmd += ['--device', 'none', '--no-op-offload', '--no-kv-offload', '--no-mmproj-offload']
    if a.profile_phases or cpu_audio_f16 or arm_precise or a.backend in ('metal', 'cuda'):
        llama_cmd.remove('--log-disable')
        llama_cmd += ['--log-verbosity', '4']
    print(json.dumps({'backend': a.backend, 'threads': a.threads,
        'flags': {k: v for k, v in os.environ.items() if k.startswith('EI_')},
        'llama_flags': {k: v for k, v in llama_env.items() if k.startswith('EI_')},
        'validate_only': a.validate_only, 'orders': ['ours/llama', 'llama/ours'],
        'ours_command': ours_cmd, 'llama_command': llama_cmd,
        'fixtures': '96x96 RGB PPM; 1s 16kHz mono PCM WAV; 2s 1fps 96x96 MPEG4; text+image+audio',
        'native_video_prefix': 'Video: (matches llama.cpp helper insertion)',
        'rounds': a.rounds, 'target_seconds': a.target_seconds,
        'quiet_total_cpu_percent': a.quiet_total_cpu_percent}), flush=True)
    results = []
    references_checked = False
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
                                   env=llama_env) as lp:
                    if cpu_audio_f16:
                        marker = 'CPU audio F16 active: 132 Conformer matrices'
                        for name, process in (('native', op), ('llama.cpp', lp)):
                            if marker not in process.log_path.read_text(errors='replace'):
                                raise RuntimeError(f'{name} did not enable the matched CPU audio F16 loader')
                    if arm_precise:
                        native_log = op.log_path.read_text(errors='replace')
                        llama_log = lp.log_path.read_text(errors='replace')
                        if 'ARM CPU numeric variant: dotprod-fp16-acc-f32-v2' not in native_log:
                            raise RuntimeError('native ARM precise numeric variant did not activate')
                        for feature in ('DOTPROD', 'FP16_VA', 'ARM_FP16_ACC_F32'):
                            if not re.search(rf'\b{feature}\s*=\s*1\b', llama_log):
                                raise RuntimeError(f'llama.cpp ARM feature did not activate: {feature}')
                        modules = []
                        for process in (op, lp):
                            maps = Path(f'/proc/{process.process.pid}/maps').read_text()
                            loaded = {Path(line.split()[-1]).resolve() for line in maps.splitlines()
                                      if 'libggml-cpu-armv8_dotprod_fp16.so' in line}
                            if len(loaded) != 1:
                                raise RuntimeError('ARM comparison did not load exactly one optimized module')
                            modules.append(loaded.pop())
                        if modules[0] != modules[1]:
                            raise RuntimeError(f'ARM comparison loaded different GGML modules: {modules}')
                        print(json.dumps({'matched_arm_precision': 'dotprod-fp16-acc-f32-v2',
                                          'shared_module': str(modules[0])}), flush=True)
                    if a.audio_reference and not references_checked:
                        for reference_path in a.audio_reference:
                            golden = json.loads(reference_path.read_text())
                            assert golden['revision'] == '914f7f89142e33e77833254d9c9b90c3cef7303b'
                            assert (golden['reference_dtype'], golden['attention'], golden['audio_mask']) == ('float32', 'sdpa', 'boolean')
                            reference = golden['embedding']
                            assert len(reference) == 768 and all(math.isfinite(x) for x in reference)
                            assert abs(sum(x*x for x in reference) - 1) < 1e-5
                            audio = io.BytesIO()
                            with wave.open(audio, 'wb') as wav:
                                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(golden['sample_rate'])
                                wav.writeframes(b''.join(struct.pack('<h', round(golden['amplitude'] * math.sin(
                                    2 * math.pi * golden['frequency_hz'] * i / golden['sample_rate'])))
                                    for i in range(golden['seconds'] * golden['sample_rate'])))
                            url = 'data:audio/wav;base64,' + base64.b64encode(audio.getvalue()).decode()
                            parts = ({'type': 'audio', 'data': url},
                                     {'type': 'input_audio', 'input_audio': {'data': url, 'format': 'wav'}})
                            checked = []
                            for name, endpoint, part in zip(('ours', 'llama'), (ours, llama), parts):
                                vector, tokens = validate(endpoint, {'model': 'embeddinggemma-2',
                                    'input': {'content': [part]}, 'dimensions': 768, 'encoding_format': 'float'})
                                cosine = cosine_similarity(vector, reference)
                                print(json.dumps({'original_hf_audio': str(reference_path), 'engine': name,
                                                  'tokens': tokens, 'cosine': cosine}), flush=True)
                                assert tokens == golden['tokens'] and cosine > .999
                                checked.append(vector)
                            assert cosine_similarity(*checked) > .999
                        references_checked = True
                    if a.backend in ('metal', 'cuda'):
                        device = 'MTL' if a.backend == 'metal' else 'CUDA0'
                        native_log = op.log_path.read_text(errors='replace')
                        llama_log = lp.log_path.read_text(errors='replace')
                        if f'EmbeddingGemma 2: {device}' not in native_log or f'CLIP using {device}' not in native_log:
                            raise RuntimeError(f'native {a.backend} comparison fell back to CPU')
                        if not re.search(r'offloaded [1-9][0-9]*/[0-9]+ layers to GPU', llama_log):
                            raise RuntimeError(f'llama.cpp {a.backend} comparison did not offload model layers')
                        if f'CLIP using {device}' not in llama_log:
                            raise RuntimeError(f'llama.cpp {a.backend} comparison did not select GPU encoders')
                    if vnni_probe:
                        startup = op.log_path.read_text(errors='replace')
                        if not any('libggml-cpu-' + name + '.so' in startup for name in
                                   ('cascadelake', 'cooperlake', 'icelake', 'sapphirerapids', 'zen4')):
                            raise RuntimeError('VNNI experiment did not load a VNNI CPU plugin')
                    if a.vision_clip_metadata and 'Vision clipping: explicit metadata only' not in op.log_path.read_text(errors='replace'):
                        raise RuntimeError('vision metadata experiment was not enabled')
                    if a.metal_media_flash_attn and 'Metal media flash attention: forced on;' not in op.log_path.read_text(errors='replace'):
                        raise RuntimeError('Metal encoder flash attention experiment was not enabled')
                    if a.validate_only and a.profile_phases:
                        for engine, endpoint in enumerate((ours, llama)):
                            warm_bodies = [json.dumps(pair[engine], separators=(',', ':')).encode()
                                           for pair in inputs[:concurrency]]
                            run_requests(endpoint, warm_bodies, 2, 0)
                    log_offsets = (op.log_path.stat().st_size, lp.log_path.stat().st_size)
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
                    if native_probe:
                        off = Endpoint(ours.host, ours.port + 2, ours.path, ours.api)
                        command = ours_cmd.copy()
                        command[command.index('--port') + 1] = str(off.port)
                        with ManagedServer(command, off, '/healthz', root / 'native-off.log',
                                env={k: v for k, v in os.environ.items()
                                     if k not in off_flags}) as off_process:
                            if a.backend in ('metal', 'cuda'):
                                off_log = off_process.log_path.read_text(errors='replace')
                                if f'EmbeddingGemma 2: {device}' not in off_log or f'CLIP using {device}' not in off_log:
                                    raise RuntimeError(f'native {a.backend} off comparison fell back to CPU')
                            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                                unchanged = list(pool.map(lambda pair: validate(off, pair[0]),
                                                          inputs[:concurrency]))
                            if a.metal_media_flash_attn:
                                minimum_pair = min(cosine_similarity(on[0], auto[0])
                                                   for on, auto in zip(quality[0], unchanged))
                                minimum_auto_reference = min(cosine_similarity(auto[0], reference[0])
                                                             for auto, reference in zip(unchanged, quality[1]))
                                if [v[1] for v in unchanged] != tokens or min(minimum_pair, minimum_auto_reference) < .999:
                                    raise RuntimeError(f'{kind}: encoder flash attention quality mismatch: '
                                                       f'on/AUTO {minimum_pair:.8f}, AUTO/reference {minimum_auto_reference:.8f}')
                                if kind != 'text' and 'warmup: flash attention is enabled' not in op.log_path.read_text(errors='replace'):
                                    raise RuntimeError('forced encoder flash attention did not activate')
                                print(json.dumps({'modality': kind, 'concurrency': concurrency,
                                    'native_on_auto_minimum_cosine': minimum_pair,
                                    'native_auto_reference_minimum_cosine': minimum_auto_reference,
                                    'encoder_flash_attention': {name: [line for line in process.log_path.read_text(errors='replace').splitlines()
                                        if 'flash attention' in line.lower()]
                                        for name, process in (('on', op), ('auto', off_process))}}), flush=True)
                            elif unchanged != quality[0]:
                                raise RuntimeError(f'{kind}: {probe_name} changed native outputs')
                            else:
                                print(f'{probe_name}: exact native on/off outputs', flush=True)
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
                        row['phase_costs'] = {name: phase_costs(process.log_path, offset, kind != 'text')
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
                        row['phase_costs'] = {name: phase_costs(process.log_path, offset, kind != 'text')
                            for name, process, offset in zip(('ours', 'llama'), (op, lp), log_offsets)}
                    results.append(row); print(json.dumps(row), flush=True)
    if results:
        print('\n| Modality | Clients | Ours emb/s | llama.cpp emb/s | Ratio |')
        print('| --- | ---: | ---: | ---: | ---: |')
        for r in results:
            print(f"| {r['modality']} | {r['concurrency']} | {r['ours_emb_s']:.2f} | {r['llama_emb_s']:.2f} | {r['speedup']:.3f}x |")
        print(f"Geometric mean: {math.exp(statistics.mean(math.log(r['speedup']) for r in results)):.3f}x")


if __name__ == '__main__': main()
