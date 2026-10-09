"""Regression: simultaneous identical media requests must run inference once."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import threading
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--url", default="http://127.0.0.1:42667")
p.add_argument("--log", type=Path, required=True)
a = p.parse_args()
image = b"P6\n160 160\n255\n" + bytes((255, 0, 0)) * (160 * 160)
body = json.dumps({"input": {"content": [{"type": "image", "data":
    base64.b64encode(image).decode()}]}, "dimensions": 128,
    "model": "embeddinggemma-2", "singleflight_probe": time.time_ns()}).encode()
barrier = threading.Barrier(4)
before = a.log.read_text(errors="replace").count("multimodal request:")

def submit(_):
    barrier.wait(timeout=10)
    start = time.perf_counter()
    request = urllib.request.Request(a.url + "/api/embed", body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as r:
        response = json.load(r)
    vector = response["embeddings"][0]
    assert len(vector) == 128 and all(math.isfinite(v) for v in vector)
    assert abs(sum(v * v for v in vector) - 1) < 1e-5
    return response, (time.perf_counter() - start) * 1000

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(submit, range(4)))
assert all(result[0] == results[0][0] for result in results)
executions = a.log.read_text(errors="replace").count("multimodal request:") - before
print("Four identical requests:", executions, "inference calls; latencies ms",
      [round(result[1], 1) for result in results], flush=True)
assert executions == 1, "duplicate concurrent media inference was not coalesced"
