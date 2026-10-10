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
p.add_argument('--cpu-audio-f16', action='store_true',
               help='Check the shared CPU audio quality fix in both real encoders')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
env = {k: v for k, v in os.environ.items() if not k.startswith('EI_')}
env['EI_VISION_CLIP_METADATA2'] = '1'
if a.cpu_audio_f16:
    if a.backend != 'cpu': p.error('--cpu-audio-f16 requires CPU')
    env['EI_CPU_AUDIO_F16_2'] = '1'
command = [sys.executable, str(root / 'perf/compare_media_llamacpp2.py'),
           '--embeddinggemma-bin', str(a.embeddinggemma_bin), '--llama-server', str(a.llama_server),
           '--model', str(a.model), '--mmproj', str(a.mmproj), '--backend', a.backend,
           '--threads', '2', '--modalities', 'audio' if a.cpu_audio_f16 else 'image',
           '--concurrency', '4' if a.cpu_audio_f16 else '1',
           '--profile-phases', '--validate-only']
result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=240)
assert result.returncode == 0, result.stdout + result.stderr
rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
settings = next(row for row in rows if 'llama_flags' in row)
assert settings['flags']['EI_VISION_CLIP_METADATA2'] == '1'
expected = {'EI_PROFILE_MEDIA2': '1'}
if a.cpu_audio_f16: expected['EI_CPU_AUDIO_F16_2'] = '1'
assert settings['llama_flags'] == expected, settings
modality = next(row for row in rows if row.get('modality') == ('audio' if a.cpu_audio_f16 else 'image'))
assert modality['minimum_cosine'] >= .999, modality
for name in ('ours', 'llama'):
    assert modality['phase_costs'][name]['encoder']['samples'] > 0, modality
print('Both real encoders produced diagnostic phases; baseline optimization flags remain isolated')
