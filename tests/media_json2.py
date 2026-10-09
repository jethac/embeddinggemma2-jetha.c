"""Protect media JSON decoding and validation when changing parsers."""
import argparse
import base64
import json
import math
import struct
import urllib.error
import urllib.request
import uuid

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', default='http://127.0.0.1:42667')
a = p.parse_args()


def post(body):
    request = urllib.request.Request(a.url + '/v1/embeddings', body.encode(),
                                    headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


item = {'content': [{'type': 'text', 'text': 'café 日本語 😀 / " \\'}]}
canonical = {'model': 'embeddinggemma-2', 'input': [item, item],
             'dimensions': 256, 'encoding_format': 'base64'}
expected = post(json.dumps(canonical))
# Duplicate fields retain their final value, escaped Unicode is decoded, and
# unused integers beyond uint64 retain the original parser's accepted behavior.
body = ('{"dimensions":128,"model":"embeddinggemma-2","dimensions":256,'
        '"encoding_format":"base64","input":' + json.dumps([item, item]) +
        ',"extra":18446744073709551616,"probe":"' + uuid.uuid4().hex + '"}')
actual = post(body)
assert actual == expected, 'parser changed input, dimensions, ordering or usage'
assert [row['index'] for row in actual['data']] == [0, 1]
for row in actual['data']:
    vector = struct.unpack('<256f', base64.b64decode(row['embedding']))
    assert all(math.isfinite(x) for x in vector)
    assert abs(math.fsum(x*x for x in vector) - 1) < 1e-5

for suffix in (' trailing', ','):
    try:
        post(body + suffix)
        raise AssertionError('malformed JSON accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 400
        error.read()
for bad in ('"\\uD800"', 'NaN'):
    wire = ('{"model":"embeddinggemma-2","input":' + json.dumps(item) +
            ',"invalid":' + bad + '}')
    try:
        post(wire)
        raise AssertionError('invalid Unicode/number accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 400
        error.read()
print('Media JSON Unicode, duplicate keys, large integers and strict rejection: passed')
