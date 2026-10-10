"""Check and measure the combined CPU weight options on the installed Mac service."""
import argparse
import base64
from contextlib import ExitStack
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from types import SimpleNamespace
import wave

from compare_llamacpp import Endpoint, ManagedServer, cosine_similarity
from compare_media_llamacpp2 import check_available_port, fixtures, measure_native_pair, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'model', 'mmproj'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith('EI_')}
    for key in ('DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH', 'GGML_BACKEND_PATH'):
        env.pop(key, None)
    env.update(EI_THREADS='3', EI_CPU_AUDIO_F16_2='1')
    endpoints = {name: Endpoint('127.0.0.1', port, '/v1/embeddings', 'openai')
                 for name, port in (('off', 42667), ('on', 42668))}
    for endpoint in endpoints.values():
        check_available_port(endpoint.port)
    with tempfile.TemporaryDirectory(prefix='mac-cpu-weights-') as directory:
        root = Path(directory)
        cases = fixtures(root, 1)

        def start(stack, name, label):
            endpoint = endpoints[name]
            child_env = dict(env)
            if name == 'on':
                child_env.update(EI_CPU_REPACK2='1', EI_CPU_BF16_F32_2='1')
            command = [str(args.binary.resolve()), '--backend', 'cpu', '--bind', endpoint.host,
                       '--port', str(endpoint.port), '--model', str(args.model.resolve()),
                       '--mmproj', str(args.mmproj.resolve()), '--cache-entries', '0',
                       '--response-cache-mb', '0']
            server = stack.enter_context(ManagedServer(command, endpoint, '/healthz',
                                                       root / (label + '-' + name + '.log'), env=child_env))
            log = server.log_path.read_text(errors='replace')
            assert 'CPU audio F16 active: 132 Conformer matrices' in log, log
            assert 'ggml_metal_library_init' not in log, log
            repack = None
            if name == 'on':
                match = re.search(r'CPU Q8 repack: (\d+) matrices, ([0-9.]+) MiB', log)
                assert match or 'CPU Q8 repack: unavailable' in log, log
                repack = {'matrices': int(match[1]), 'MiB': float(match[2])} if match else 'unavailable'
                assert 'CPU projection F32 active: per_layer_model_proj.weight' in log, log
            else:
                assert 'CPU Q8 repack:' not in log and 'CPU projection F32 active:' not in log, log
            print(json.dumps({'startup': label, 'mode': name,
                              'flags': {k: v for k, v in child_env.items() if k.startswith('EI_')},
                              'actual_repack': repack,
                              'numeric_messages': [line for line in log.splitlines() if any(
                                  marker in line for marker in ('CPU Q8 repack:', 'CPU projection F32',
                                                              'CPU audio F16 active:', 'ARM CPU numeric variant:'))]}),
                  flush=True)
            return server

        # These gates must pass before any performance measurement.
        with ExitStack() as stack:
            for name in endpoints:
                start(stack, name, 'quality')
            for seconds, frequency in ((1, 330), (1, 660), (5, 660)):
                golden = json.loads((repo / f'tests/fixtures/audio-{seconds}s{frequency}-hf-f32.json').read_text())
                assert golden['revision'] == '914f7f89142e33e77833254d9c9b90c3cef7303b'
                assert (golden['reference_dtype'], golden['attention'], golden['audio_mask']) == ('float32', 'sdpa', 'boolean')
                audio = io.BytesIO()
                with wave.open(audio, 'wb') as wav:
                    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
                    wav.writeframes(b''.join(struct.pack('<h', round(4000 * math.sin(
                        2 * math.pi * frequency * i / 16000))) for i in range(seconds * 16000)))
                body = dict(cases['audio'][0][0], input={'content': [{'type': 'audio',
                    'data': 'data:audio/wav;base64,' + base64.b64encode(audio.getvalue()).decode()}]})
                scores = {}
                for name, endpoint in endpoints.items():
                    vector, tokens = validate(endpoint, body)
                    assert tokens == golden['tokens']
                    scores[name] = cosine_similarity(vector, golden['embedding'])
                print(json.dumps({'original_HF_audio': [seconds, frequency], 'cosine': scores}), flush=True)
                assert min(scores.values()) > .999, scores
            for kind in ('text', 'image', 'audio', 'video', 'mixed'):
                vectors = [validate(endpoint, cases[kind][0][0]) for endpoint in endpoints.values()]
                assert vectors[0][1] == vectors[1][1], (kind, vectors)
                cosine = cosine_similarity(vectors[0][0], vectors[1][0])
                print(json.dumps({'quality': kind, 'tokens': vectors[0][1], 'ON_OFF_cosine': cosine}), flush=True)
                assert cosine > .999, (kind, cosine)

        options = SimpleNamespace(cooldown=2, quiet_total_cpu_percent=10, rounds=5, target_seconds=8)
        for kind in ('text', 'audio'):
            # Fresh servers per cell avoid cross-modality carryover.
            with ExitStack() as stack:
                servers = [start(stack, name, kind) for name in endpoints]
                ignored = {server.process.pid for server in servers}
                bodies = [json.dumps(cases[kind][0][0]).encode()]
                result = measure_native_pair(endpoints['on'], endpoints['off'], bodies, ignored, options)
                print(json.dumps({'performance': kind, 'concurrency': 1, 'threads': 3,
                                  'cache_entries': 0, 'response_cache_mb': 0, **result}), flush=True)


if __name__ == '__main__':
    main()
