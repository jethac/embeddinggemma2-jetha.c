"""Check the observed 660 Hz audio error against original FP32 SDPA vectors.

The fixture uses the pinned Hugging Face model with no quantization changes.
PCM samples are round(4000 * sin(2 * pi * 660 * i / 16000)), signed 16-bit.
The OFF result is diagnostic. The F16 result must pass the reference gate.
"""
import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import time
import urllib.request
import wave


def cosine(left, right):
    return sum(x*y for x, y in zip(left, right)) / math.sqrt(
        sum(x*x for x in left) * sum(x*x for x in right))


def check(args):
    fixture = json.loads((Path(__file__).resolve().parent.parent /
                         'testdata/audio-reference2.json').read_text())
    assert fixture['revision'] == '914f7f89142e33e77833254d9c9b90c3cef7303b'
    assert (fixture['sample_rate'], fixture['frequency'], fixture['amplitude']) == (16000, 660, 4000)
    assert [(sample['seconds'], sample['tokens']) for sample in fixture['fixtures']] == [(1, 29), (5, 129)]
    for sample in fixture['fixtures']:
        reference = sample['embedding']
        assert len(reference) == 768 and all(math.isfinite(x) for x in reference)
        assert abs(sum(x*x for x in reference) - 1) < 1e-5
    results = {}
    for enabled in (0, 1):
        with socket.socket() as port_check:
            port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port_check.bind(('127.0.0.1', args.port))
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith('EI_')}
        environment.update(EI_THREADS=str(args.threads), EI_CPU_AUDIO_F16_2=str(enabled))
        log_path = Path(args.log_dir) / f'audio-f16-{enabled}.stderr.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w') as log:
            child = subprocess.Popen([
                str(Path(args.binary).resolve()), '--backend', 'cpu',
                '--bind', '127.0.0.1', '--port', str(args.port),
                '--model', str(Path(args.model).resolve()),
                '--mmproj', str(Path(args.mmproj).resolve()),
                '--cache-entries', '0', '--response-cache-mb', '0'],
                env=environment, stdout=subprocess.DEVNULL, stderr=log)
            try:
                base = f'http://127.0.0.1:{args.port}'
                deadline = time.monotonic() + 180
                while True:
                    assert child.poll() is None, 'CPU service exited during startup'
                    try:
                        with urllib.request.urlopen(base + '/healthz', timeout=2):
                            break
                    except OSError:
                        assert time.monotonic() < deadline, 'CPU startup timeout'
                        time.sleep(0.2)
                for sample in fixture['fixtures']:
                    pcm = b''.join(struct.pack('<h', round(fixture['amplitude'] *
                        math.sin(2 * math.pi * fixture['frequency'] * i /
                                 fixture['sample_rate'])))
                        for i in range(sample['seconds'] * fixture['sample_rate']))
                    buffer = io.BytesIO()
                    with wave.open(buffer, 'wb') as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(fixture['sample_rate'])
                        wav.writeframes(pcm)
                    body = {'input': {'content': [{'type': 'audio', 'data':
                            base64.b64encode(buffer.getvalue()).decode()}]}}
                    request = urllib.request.Request(base + '/api/embed',
                        json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(request, timeout=180) as response:
                        output = json.load(response)
                    assert len(output['embeddings']) == 1
                    vector = output['embeddings'][0]
                    assert len(vector) == 768 and all(math.isfinite(x) for x in vector)
                    assert abs(sum(x*x for x in vector) - 1) < 1e-5
                    for field in ('prompt_tokens', 'total_tokens'):
                        assert output['usage'][field] == sample['tokens'], output['usage']
                    score = cosine(vector, sample['embedding'])
                    print(f"F16={enabled} seconds={sample['seconds']} tokens={sample['tokens']} "
                          f"original_FP32_SDPA_cosine={score:.9f}", flush=True)
                    results[enabled, sample['seconds']] = vector
                    if enabled:
                        print('ON/OFF cosine=' + str(cosine(
                            vector, results[0, sample['seconds']])), flush=True)
                        assert score > .999, 'F16 audio differs from original FP32 SDPA'
            except Exception:
                print(log_path.read_text()[-12000:], flush=True)
                raise
            finally:
                if child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=30)
        text = log_path.read_text()
        assert ('CPU audio F16 active: 132 Conformer matrices' in text) == bool(enabled)
        assert 'ggml_metal_library_init' not in text, 'CPU service initialized Metal'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--mmproj', required=True)
    parser.add_argument('--port', type=int, default=42670)
    parser.add_argument('--log-dir', default='build-cmake')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    check(args)
