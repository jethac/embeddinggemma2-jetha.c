"""Regress delayed request bodies through WSL's localhost TCP relay.

Run from Windows against the Linux service. Invalid requests isolate transport
from model time; the large body used to add about 40 ms on a reused connection.
"""
import argparse
import http.client
import json
import socket
import statistics
import time
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:42667')
    args = parser.parse_args()
    url = urlsplit(args.url)
    if url.scheme != 'http' or not url.hostname:
        parser.error('--url must be an http service URL')
    bodies = {
        'small': json.dumps({'input': None}).encode(),
        'large': json.dumps({'input': None, 'padding': 'x' * 43000}).encode(),
        'upload': json.dumps({'input': None, 'padding': 'x' * 2097152}).encode(),
    }
    for order in (('small', 'large', 'upload'), ('upload', 'large', 'small')):
        samples = {name: [] for name in order}
        connection = socket.create_connection((url.hostname, url.port or 80),
                                               timeout=30)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            for _ in range(10):
                for name in order:
                    start = time.perf_counter()
                    body = bodies[name]
                    header = (
                        f'POST /api/embed HTTP/1.1\r\nHost: {url.netloc}\r\n'
                        f'Content-Type: application/json\r\nContent-Length: {len(body)}\r\n'
                        'Connection: keep-alive\r\n\r\n').encode()
                    connection.sendall(header + body)
                    response = http.client.HTTPResponse(connection)
                    response.begin()
                    result = json.loads(response.read())
                    assert response.status == 400, (response.status, result)
                    assert not response.will_close, 'connection was not reusable'
                    samples[name].append((time.perf_counter() - start) * 1000)
            medians = {name: statistics.median(values[2:])
                       for name, values in samples.items()}
            print(f'{order}: ' + '; '.join(f'{name} {value:.3f} ms'
                  for name, value in medians.items()), flush=True)
            for name in ('large', 'upload'):
                assert medians[name] - medians['small'] < 25, (
                    name + ' request body incurred a delayed-ACK stall')
        finally:
            connection.close()


if __name__ == '__main__':
    main()
