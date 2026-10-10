"""Regression: repeat comparisons can reuse closed Unix server ports."""
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perf'))
from compare_media_llamacpp2 import check_available_port

if os.name == 'nt':
    raise SystemExit('Unix TIME_WAIT regression: run on Linux/macOS')
with socket.socket() as server, socket.socket() as client:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', 0))
    port = server.getsockname()[1]
    server.listen()
    try:
        check_available_port(port)
    except OSError:
        pass
    else:
        raise AssertionError('precheck accepted a live listener')
    client.connect(('127.0.0.1', port))
    accepted, _ = server.accept()
    accepted.shutdown(socket.SHUT_WR)
    assert client.recv(1) == b''
    accepted.close()
    client.close()
check_available_port(port)
print('comparison port precheck rejects live listeners and permits closed ports: passed')
