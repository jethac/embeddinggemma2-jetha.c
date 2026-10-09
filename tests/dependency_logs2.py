"""Regression: full-model startup must not flood synchronous logs with debug tensors."""
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--log', type=Path, required=True)
parser.add_argument('--video', action='store_true', help='Also check a completed video request')
args = parser.parse_args()
log = args.log.read_text(errors='replace')
assert 'media encoders: vision=1 audio=1 video=1' in log, 'full encoder startup required'
assert 'startup: media encoder load' in log, 'startup timings must remain visible'
assert 'clip_model_loader: tensor[' not in log, 'dependency debug tensor inventory flooded startup log'
assert 'init_audio: audio input is in experimental stage' in log, 'dependency warnings must remain visible'
if args.video:
    assert 'multimodal request:' in log, 'a completed media request is required'
    assert 'read_next_frame:' not in log, 'video helper debug lines flooded the request log'
    assert 'probe: launching:' not in log, 'video probe debug lines flooded the request log'
print('Full-model startup keeps timings/warnings without debug tensor inventory: passed')
