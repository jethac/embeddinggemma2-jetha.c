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
p.add_argument('--mode', choices=['packed-qkv', 'cuda-global-attn', 'cuda-local-attn', 'cuda-local-range', 'geglu', 'media-batch', 'jpeg-turbo', 'text-buckets', 'text-batch-buckets', 'vision-clip-metadata', 'cpu-audio-f16'], default='packed-qkv')
p.add_argument('--mmproj', type=Path)
p.add_argument('--tokens', type=int, default=32)
a = p.parse_args()
if not 4 <= a.tokens <= 8192: p.error('--tokens must be 4..8192')
if a.mode in ('media-batch', 'vision-clip-metadata', 'cpu-audio-f16') and not a.mmproj: p.error(f'{a.mode} requires --mmproj')
flag, marker = {
    'packed-qkv': ('EI_QKV2', 'packed QKV:'),
    'cuda-global-attn': ('EI_CUDA_GLOBAL_ATTN2', 'CUDA global attention fallback:'),
    'cuda-local-attn': ('EI_CUDA_LOCAL_ATTN2', 'CUDA local attention:'),
    'cuda-local-range': ('EI_CUDA_LOCAL_RANGE2', 'CUDA local mask range:'),
    'geglu': ('EI_GEGLU2', 'Fused GeGLU:'),
    'media-batch': ('EI_MEDIA_BATCH2', 'Multimodal backbone batching:'),
    'jpeg-turbo': ('EI_JPEG_TURBO2', 'JPEG decoding:'),
    'text-buckets': ('EI_TEXT_BUCKETS2', 'CUDA text buckets:'),
    'text-batch-buckets': ('EI_TEXT_BATCH_BUCKETS2', 'CUDA text batch buckets:'),
    'vision-clip-metadata': ('EI_VISION_CLIP_METADATA2', 'Vision clipping: explicit metadata only'),
    'cpu-audio-f16': ('EI_CPU_AUDIO_F16_2', 'CPU audio F16 active: 132 Conformer matrices'),
}[a.mode]
with tempfile.TemporaryDirectory(prefix='qkv-cache2-') as directory:
    work = Path(directory)
    cache = work / 'cache'
    with socket.socket() as check:
        check.bind(('127.0.0.1', 0))
        port = check.getsockname()[1]
    command = [str(a.binary.resolve()), '--backend', a.backend,
        '--model', str(a.model.resolve()), '--bind', '127.0.0.1', '--port', str(port),
        '--cache-entries', '0', '--response-cache-mb', '1', '--persistent-cache-path', str(cache)]
    if a.mmproj: command += ['--mmproj', str(a.mmproj.resolve())]
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
        body = {'input': 'q0' + ' x' * (a.tokens - 4)}
        if a.mode == 'cpu-audio-f16':
            import base64, io, math, wave
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
                wav.writeframes(b''.join(struct.pack('<h', round(4000 * math.sin(
                    2 * math.pi * 660 * i / 16000))) for i in range(16000)))
            body = {'input': {'content': [{'type': 'audio', 'data':
                    base64.b64encode(buffer.getvalue()).decode()}]}}
        if a.mode == 'text-batch-buckets':
            body = {'input': ['q'+str(i)+(' '+word)*(a.tokens-4+i)
                              for i, word in enumerate(('x', 'y', 'z'))]}
        if a.mode == 'media-batch':
            body = {'input': [{'content': [{'type': 'text', 'text': 'q'+str(i)+(' '+word)*(a.tokens-4+i)}]}
                              for i, word in enumerate(('x', 'y', 'z'))]}
        if probe: body['qkv_cache_probe'] = True
        with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/embed',
            json.dumps(body).encode(), headers={'Content-Type':'application/json'}), timeout=180) as response:
            return json.load(response)
    os.environ['EI_QKV2'] = '0'
    os.environ['EI_CUDA_GLOBAL_ATTN2'] = '0'
    os.environ[flag] = '0'
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
    if a.mode == 'cpu-audio-f16':
        with server(work/'restart.log'):
            assert request() == fresh, 'F16 restart response changed'
            assert marker in (work/'restart.log').read_text(errors='replace')
            assert 'multimodal request:' not in (work/'restart.log').read_text(errors='replace')
        print('CPU audio F16 restart response is exact with no inference', flush=True)
print(a.mode, 'persistent cache isolation passed')
