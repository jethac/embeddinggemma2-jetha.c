"""Install real Unix release assets; reject corruption without replacing the app."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--assets', type=Path, required=True)
parser.add_argument('--variant', choices=['cpu', 'metal'], default='cpu')
args = parser.parse_args()
assets = args.assets.resolve()
root = Path(__file__).resolve().parent.parent
with tempfile.TemporaryDirectory(prefix='installer-runtime2-') as directory:
    fixture = Path(directory)
    checksums = fixture / 'SHA256SUMS'
    checksums.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n'
                                for p in assets.iterdir() if p.is_file()))
    curl = fixture / 'curl'
    curl.write_text('''#!/usr/bin/env python3
import os, pathlib, shutil, sys
url = sys.argv[-1]
base = 'https://github.com/jethac/embeddinggemma2-jetha.c/releases/latest/download/'
assert url.startswith(base), url
name = url[len(base):]
assert '/' not in name
output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])
source = pathlib.Path(os.environ['PROBE_CHECKSUMS']) if name == 'SHA256SUMS' else pathlib.Path(os.environ['PROBE_ASSETS']) / name
shutil.copyfile(source, output)
if os.environ.get('PROBE_CORRUPT') and name.endswith('.runtime.tar.gz'):
    with output.open('ab') as stream: stream.write(b'corrupted download')
''')
    curl.chmod(0o755)
    install = fixture / 'installed app with spaces'
    env = dict(os.environ, PATH=str(fixture) + os.pathsep + os.environ['PATH'],
               PROBE_ASSETS=str(assets), PROBE_CHECKSUMS=str(checksums))
    for name in ('LD_LIBRARY_PATH', 'GGML_BACKEND_PATH', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH'):
        env.pop(name, None)
    command = ['sh', str(root / 'install.sh'), '--variant', args.variant, '--install-dir', str(install)]
    subprocess.run(command, env=env, check=True, timeout=90)
    binary = install / 'embeddinggemma2-jetha'
    before = binary.readlink()
    subprocess.run([str(binary), '--help'], env=env, cwd=fixture,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30)
    result = subprocess.run(command, env=dict(env, PROBE_CORRUPT='1'),
                            capture_output=True, text=True, timeout=90)
    assert result.returncode != 0 and 'checksum verification failed' in result.stderr, result
    assert binary.readlink() == before, 'corrupt download replaced working installation'
    subprocess.run([str(binary), '--help'], env=env, cwd=fixture,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30)
print('Unix runtime installer loads with spaces and preserves the working installation after checksum failure: passed')
