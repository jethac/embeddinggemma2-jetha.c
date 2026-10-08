"""Regression for the observed audio reference mismatch; requires torch/transformers.

Run with a transformers build containing EmbeddingGemma 2. The first run fetches
the published reference weights. Compare an actual HTTP response to that model.
"""
import argparse
import base64
import io
import json
import urllib.request
import wave

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:42667")
    args = parser.parse_args()
    torch.set_num_threads(6)
    pcm = (np.sin(np.arange(16000) * (2 * np.pi * 440 / 16000)) * 8192).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())
    processor = AutoProcessor.from_pretrained("google/embeddinggemma-2", revision=REVISION)
    model = AutoModel.from_pretrained("google/embeddinggemma-2", revision=REVISION,
                                      dtype=torch.float32, attn_implementation="eager").eval()
    inputs = processor(audio=pcm.astype(np.float32) / 32768, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        reference = ((hidden * mask).sum(1) / mask.sum(1))[0].float().numpy()
    body = {"input": {"content": [{"type": "audio", "data": base64.b64encode(buffer.getvalue()).decode()}]}}
    request = urllib.request.Request(args.url + "/api/embed", json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        actual = np.array(json.load(response)["embeddings"][0])
    cosine = float(reference @ actual / (np.linalg.norm(reference) * np.linalg.norm(actual)))
    print(f"audio reference cosine: {cosine:.8f}")
    assert cosine > 0.999, f"audio reference regression: cosine {cosine:.8f}"
