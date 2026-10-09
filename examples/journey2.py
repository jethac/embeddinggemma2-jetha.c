"""Run synthetic text, image, audio, video and mixed requests on a dev service."""
import argparse
import base64
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import urllib.request
import wave

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', default='http://127.0.0.1:42667')
p.add_argument('--compare-url', help='Compare each embedding with another running service')
a = p.parse_args()

def part(kind, data):
    return {'type': kind, 'data': base64.b64encode(data).decode()}

image = part('image', b'P6\n96 96\n255\n' + bytes((220, 30, 30)) * (96 * 96))
audio_bytes = io.BytesIO()
with wave.open(audio_bytes, 'wb') as wav:
    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
    wav.writeframes(b''.join(struct.pack('<h', round(4000 * math.sin(2 * math.pi * 440 * i / 16000)))
                             for i in range(16000)))
audio = part('audio', audio_bytes.getvalue())
with tempfile.TemporaryDirectory(prefix='embeddinggemma2-example-') as tmp:
    clip = Path(tmp) / 'red.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                    'color=red:size=96x96:rate=1:duration=2', '-c:v', 'mpeg4', str(clip)],
                   check=True, timeout=30)
    video = part('video', clip.read_bytes())
    cases = {
        'text': 'task: search result | query: what powers the cell',
        'image': {'content': [image]},
        'audio': {'content': [audio]},
        'video': {'content': [video]},
        'mixed': {'content': [{'type': 'text', 'text': 'A red square and a tone.'}, image, audio]},
    }
    for name, value in cases.items():
        start = time.perf_counter()
        request = urllib.request.Request(a.url.rstrip('/') + '/api/embed',
            json.dumps({'input': value}).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
        vector = result['embeddings'][0]
        assert len(vector) == 768 and all(math.isfinite(x) for x in vector), 'invalid embedding: ' + name
        norm = sum(x * x for x in vector)
        assert abs(norm - 1) < 1e-5, 'unnormalized embedding: ' + name
        tokens = '' if name == 'text' else f"{result['usage']['total_tokens']} tokens; "
        print(name, f'{(time.perf_counter() - start) * 1000:.1f} ms;',
              tokens + '768 finite normalized dimensions', flush=True)
        if a.compare_url:
            reference_request = urllib.request.Request(a.compare_url.rstrip('/') + '/api/embed',
                request.data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(reference_request, timeout=300) as response:
                reference = json.load(response)['embeddings'][0]
            assert len(reference) == 768 and all(math.isfinite(x) for x in reference)
            reference_norm = sum(x*x for x in reference)
            assert abs(reference_norm - 1) < 1e-5
            cosine = sum(x*y for x,y in zip(vector, reference)) / math.sqrt(norm * reference_norm)
            print(name, f'comparison cosine: {cosine:.8f}', flush=True)
            assert cosine > 0.999, 'embedding differs from comparison service: ' + name
