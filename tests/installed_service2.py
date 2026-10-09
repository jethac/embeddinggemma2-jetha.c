"""Regression: an installed service must load without build-tree search paths."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args()
    suffix = '.exe' if os.name == 'nt' else ''
    binary = args.prefix.resolve() / 'bin' / ('embeddinggemma2-jetha' + suffix)
    env = os.environ.copy()
    for name in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH',
                 'GGML_BACKEND_PATH'):
        env.pop(name, None)
    if os.name == 'nt':
        import ctypes
        ctypes.WinDLL('kernel32').SetErrorMode(0x8003)
        env['PATH'] = str(Path(os.environ['SystemRoot']) / 'System32')
    with tempfile.TemporaryDirectory(prefix='installed-service-') as cwd:
        result = subprocess.run([str(binary), '--help'], cwd=cwd, env=env,
                                capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert '--model' in result.stdout + result.stderr, result
    print('installed service loads outside the build tree: passed')
