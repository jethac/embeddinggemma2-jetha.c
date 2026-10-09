"""Regression: WebM/Matroska duration belongs to the container, not the stream."""
import argparse
import base64
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request
import urllib.error

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', default='http://127.0.0.1:42667')
a = p.parse_args()
def ffmpeg(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args], check=True, timeout=30)
def request(path):
    body = json.dumps({'input': {'content': [{'type': 'video',
        'data': base64.b64encode(path.read_bytes()).decode()}]},
        'container_probe': time.time_ns()}).encode()
    start = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(a.url + '/api/embed', body,
        headers={'Content-Type': 'application/json'}), timeout=180) as r:
        out = json.load(r)
    vector = out['embeddings'][0]
    assert len(vector) == 768 and all(math.isfinite(x) for x in vector)
    assert abs(sum(x*x for x in vector) - 1) < 1e-5
    assert out['usage']['total_tokens'] == 248, out['usage']
    print(path.suffix, 'two-frame video:', round((time.perf_counter()-start)*1000, 1), 'ms', flush=True)
    return vector
with tempfile.TemporaryDirectory(prefix='video-container-') as tmp:
    root = Path(tmp)
    mp4, webm, mkv = [root / ('red' + suffix) for suffix in ('.mp4', '.webm', '.mkv')]
    ffmpeg('-f', 'lavfi', '-i', 'color=red:size=96x96:rate=1:duration=2', '-c:v', 'mpeg4', str(mp4))
    ffmpeg('-i', str(mp4), '-c:v', 'libvpx-vp9', '-lossless', '1', str(webm))
    ffmpeg('-i', str(mp4), '-c', 'copy', str(mkv))
    reference = request(mp4)
    for path in (webm, mkv):
        vector = request(path)
        cosine = sum(x*y for x,y in zip(vector, reference))
        assert cosine > .99999, cosine
        print(path.suffix, 'MP4 cosine:', cosine, flush=True)
    long_audio = root / 'short-video-long-audio.mp4'
    ffmpeg('-i', str(mp4), '-f', 'lavfi', '-i', 'sine=frequency=440:duration=37',
           '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', str(long_audio))
    assert sum(x*y for x,y in zip(request(long_audio), reference)) > .99999
    print('Video stream duration takes precedence over longer audio', flush=True)
    too_long = root / 'too-long.webm'
    ffmpeg('-f', 'lavfi', '-i', 'color=red:size=96x96:rate=1:duration=33',
           '-c:v', 'libvpx-vp9', '-lossless', '1', str(too_long))
    try:
        request(too_long)
        raise AssertionError('container duration bypassed the 32-frame bound')
    except urllib.error.HTTPError as e:
        assert e.code == 400
        assert '32 video frames' in e.read().decode()
        print('33-frame WebM rejected before encoding', flush=True)
