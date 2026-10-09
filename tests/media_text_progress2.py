"""Regression: short text must progress while the CPU vision encoder is busy."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
import urllib.request
import uuid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:42667')
args = parser.parse_args()


def request(value):
    start = time.perf_counter()
    req = urllib.request.Request(args.url + '/api/embed', json.dumps({'input': value}).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        data = json.load(response)
    vector = data['embeddings'][0]
    assert len(vector) == 768 and all(math.isfinite(x) for x in vector)
    assert abs(sum(x*x for x in vector) - 1) < .001
    return (time.perf_counter() - start) * 1000, vector


nonce = uuid.uuid4().hex
media = {'content': [
    {'type': 'image', 'data': base64.b64encode(
        b'P6\n96 96\n255\n' + bytes((220, 30+i, 30)) * 9216).decode()}
    for i in range(4)] + [{'type': 'text', 'text': nonce}]}
text = 'task: search result | query: cell ' + nonce
with ThreadPoolExecutor(max_workers=2) as pool:
    future = pool.submit(request, media)
    time.sleep(.5)
    assert not future.done(), 'media finished before the CPU contention check'
    text_ms, vector = request(text)
    completed_before_media = not future.done()
    media_ms, media_vector = future.result()
print(f'Text during CPU media encoding: {text_ms:.1f} ms; media: {media_ms:.1f} ms')
assert completed_before_media, 'short text waited for the entire vision encoder'
_, text_serial = request(text)
_, media_serial = request(media)
assert vector == text_serial, 'concurrent text differs from standalone text'
assert media_vector == media_serial, 'concurrent media differs from standalone media'
print('Concurrent and standalone CPU text/media embeddings are identical')
