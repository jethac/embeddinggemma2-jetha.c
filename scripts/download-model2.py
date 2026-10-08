"""Download the reference GGUFs; model weights are not part of the repository."""
import argparse
from pathlib import Path
import urllib.request

REVISION = "bfcd298762cc34d0357ece5ebdd31791a3a374d8"
BASE = f"https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/{REVISION}/"

def download(name, directory):
    dest = directory / name
    if dest.exists():
        print(f"Using {dest}", flush=True)
        return
    temp = dest.with_suffix(".partial")
    print(f"Downloading {name}", flush=True)
    with urllib.request.urlopen(BASE + name) as response, temp.open("wb") as out:
        while block := response.read(4 * 1024 * 1024):
            out.write(block)
    temp.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1048576:.1f} MiB)", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("model"))
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    download("embeddinggemma-2-Q8_0.gguf", args.directory)
    if not args.text_only:
        download("mmproj-embeddinggemma-2-Q8_0.gguf", args.directory)
