"""Protect decoded-media allocation limits through the deployed HTTP service."""
import argparse
import base64
import binascii
import io
import json
import struct
import urllib.error
import urllib.request
import wave
import zlib


def solid_png(side):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xffffffff)

    compressor = zlib.compressobj()
    row = b"\0" + bytes([220, 30, 30]) * side
    compressed = b"".join(compressor.compress(row) for _ in range(side)) + compressor.flush()
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", compressed) + chunk(b"IEND", b""))


def reject(url, parts):
    body = {"input": {"content": [{"type": kind, "data": base64.b64encode(data).decode()} for kind, data in parts]}}
    request = urllib.request.Request(url + "/api/embed", json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response.read()
        raise AssertionError("decoded input over budget was accepted")
    except urllib.error.HTTPError as error:
        message = json.load(error)["error"]
        assert error.code == 400 and ("decoded size limit" in message or "decoded input budget" in message), message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:42667")
    args = parser.parse_args()
    # Observed bug: this tiny compressed PNG allocated >16M pixels and ran
    # inference instead of rejecting the decoded size before allocation.
    reject(args.url, [("image", solid_png(4100))])
    print("image pixel limit: rejected")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(328 * 16000 * 2))
    reject(args.url, [("audio", buffer.getvalue())])
    print("audio sample limit: rejected before preprocessing")
    image = solid_png(4000)
    reject(args.url, [("image", image)] * 3)
    print("aggregate decoded byte limit: rejected")
    with urllib.request.urlopen(args.url + "/healthz", timeout=10) as response:
        assert json.load(response)["status"] == "ok"
