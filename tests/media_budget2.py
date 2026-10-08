"""Regression for the observed image/video patch-budget mismatch on the service."""
import argparse
import base64
import json
from pathlib import Path
import urllib.request


def request(url, kind, data):
    body = {"input": {"content": [{"type": kind, "data": base64.b64encode(data).decode()}]}}
    req = urllib.request.Request(url + "/api/embed", json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:42667")
    parser.add_argument("--two-frame-video", type=Path, help="square two-frame clip sampled at 1 fps")
    args = parser.parse_args()
    image = b"P6\n96 96\n255\n" + bytes([220, 30, 30]) * (96 * 96)
    result = request(args.url, "image", image)
    # 280-token budget, floor-aligned 48px sides: 768px = 16x16 soft
    # tokens, plus BOS, image start/end, EOS.
    count = result["usage"]["prompt_tokens"]
    assert count == 260, f"image budget mismatch: expected 260 tokens, got {count}"
    print("image patch budget: 260 tokens")
    if args.two_frame_video:
        result = request(args.url, "video", args.two_frame_video.read_bytes())
        count = result["usage"]["prompt_tokens"]
        assert count == 248, f"video budget/prefix mismatch: expected 248 tokens, got {count}"
        print("video patch budget: 248 tokens")
