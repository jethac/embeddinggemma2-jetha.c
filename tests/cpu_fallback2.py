"""Run the deployed text journey with only the baseline CPU plugin available."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--binary", required=True, type=Path)
p.add_argument("--model", required=True, type=Path)
p.add_argument("--emulator", type=Path,
               help="Use qemu-x86_64 with qemu64 (no AVX), with all plugins present")
a = p.parse_args()
binary, model = a.binary.resolve(), a.model.resolve()
root = binary.parent.parent
work = Path(tempfile.mkdtemp(prefix="cpu-fallback-", dir=root)).resolve()
assert work.parent == root.resolve()
process = None
try:
    plugins = 0
    for source in binary.parent.iterdir():
        name = source.name
        if not source.is_file():
            continue
        if name.startswith(("ggml-cpu-", "libggml-cpu-")):
            baseline = name.startswith(("ggml-cpu-x64.", "libggml-cpu-x64."))
            if not a.emulator and not baseline:
                continue
            plugins += int(baseline)
        if source == binary or name.endswith((".dll", ".dylib")) or ".so" in name:
            shutil.copy2(source, work / name)
    assert plugins > 0, "baseline CPU plugin is missing"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, EI_THREADS="2")
    env.pop("GGML_BACKEND_PATH", None)
    runner = [str(a.emulator.resolve()), "-cpu", "qemu64"] if a.emulator else []
    with (work / "service.log").open("wb") as log:
        process = subprocess.Popen(
            runner + [str(work / binary.name), "--bind", "127.0.0.1", "--port", str(port),
             "--backend", "cpu", "--model", str(model), "--cache-entries", "0",
             "--response-cache-mb", "0"],
            cwd=work, env=env, stdout=subprocess.DEVNULL, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 120
        while True:
            if process.poll() is not None:
                raise RuntimeError((work / "service.log").read_text(errors="replace"))
            try:
                with urllib.request.urlopen(url + "/healthz", timeout=1) as r:
                    assert json.load(r)["status"] == "ok"
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("baseline service did not become ready")
                time.sleep(.2)
        body = json.dumps({"input": "task: search result | query: what powers the cell"}).encode()
        request = urllib.request.Request(url + "/api/embed", body,
                                        headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as r:
            vector = json.load(r)["embeddings"][0]
        assert len(vector) == 768 and all(math.isfinite(v) for v in vector)
        assert abs(sum(v * v for v in vector) - 1) < 1e-5
        text = (work / "service.log").read_text(errors="replace")
        assert "ggml-cpu-x64" in text, "baseline plugin was not selected"
        print("Baseline CPU service journey passed (768 finite normalized dimensions)")
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    # Delete only the verified fixture directory inside this build directory.
    assert work.resolve().parent == root.resolve()
    shutil.rmtree(work)
