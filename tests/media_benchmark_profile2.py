"""Regression: profiling the baseline must retain diagnostics without experiments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
for name in ('embeddinggemma-bin', 'llama-server', 'model', 'mmproj'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--backend', default='cpu')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
env = {k: v for k, v in os.environ.items() if not k.startswith('EI_')}
env['EI_VISION_CLIP_METADATA2'] = '1'
command = [sys.executable, str(root / 'perf/compare_media_llamacpp2.py'),
           '--embeddinggemma-bin', str(a.embeddinggemma_bin), '--llama-server', str(a.llama_server),
           '--model', str(a.model), '--mmproj', str(a.mmproj), '--backend', a.backend,
           '--threads', '2', '--modalities', 'image', '--concurrency', '1',
           '--profile-phases', '--validate-only']
result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=240)
assert result.returncode == 0, result.stdout + result.stderr
rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
settings = next(row for row in rows if 'llama_flags' in row)
assert settings['flags']['EI_VISION_CLIP_METADATA2'] == '1'
assert settings['llama_flags'] == {'EI_PROFILE_MEDIA2': '1'}, settings
image = next(row for row in rows if row.get('modality') == 'image')
assert image['minimum_cosine'] >= .999, image
for name in ('ours', 'llama'):
    assert image['phase_costs'][name]['encoder']['samples'] > 0, image
print('Both real encoders produced diagnostic phases; baseline optimization flags remain isolated')
