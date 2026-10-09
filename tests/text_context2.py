"""Regression: oversized text must report the current model's context limit."""
import argparse
import json
import urllib.error
import urllib.request


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:42667')
    args = parser.parse_args()
    for route in ('/api/embed', '/v1/embeddings'):
        request = urllib.request.Request(args.url + route,
            json.dumps({'model': 'embeddinggemma-2', 'input': ' x' * 8192}).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response.read()
            raise AssertionError('oversized text accepted')
        except urllib.error.HTTPError as error:
            result = json.load(error)
            assert error.code == 400, result
            assert '1..8192' in json.dumps(result), result
    print('text context limit: passed')
