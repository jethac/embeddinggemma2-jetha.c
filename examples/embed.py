"""Embed text and/or local media through the running development service."""
import argparse
import base64
import json
from pathlib import Path
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:42667")
parser.add_argument("--text")
parser.add_argument("--image", type=Path, action="append", default=[])
parser.add_argument("--audio", type=Path, action="append", default=[])
parser.add_argument("--video", type=Path, action="append", default=[])
parser.add_argument("--dimensions", type=int, choices=[128, 256, 512, 768], default=768)
args = parser.parse_args()
content = []
if args.text is not None:
    content.append({"type": "text", "text": args.text})
for kind in ["image", "audio", "video"]:
    for path in getattr(args, kind):
        content.append({"type": kind, "data": base64.b64encode(path.read_bytes()).decode()})
if not content:
    parser.error("provide text or a media file")
input_value = args.text if len(content) == 1 and content[0]["type"] == "text" else {"content": content}
request = urllib.request.Request(
    args.url.rstrip("/") + "/api/embed",
    data=json.dumps({"input": input_value, "dimensions": args.dimensions}).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    print(urllib.request.urlopen(request, timeout=300).read().decode())
except urllib.error.HTTPError as error:
    parser.exit(1, error.read().decode() + "\n")
