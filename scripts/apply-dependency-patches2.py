"""Apply the missing suffix of a pinned patch sequence, preserving local edits.

Later patches may change earlier patch context. Compare complete target sources
against each expected prefix instead of relying on independent reverse checks.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile


def apply(source, patches, git='git', base='HEAD'):
    source = source.resolve()
    patches = [path.resolve() for path in patches]
    names = sorted({line[6:] for patch in patches
                    for line in patch.read_text(encoding='utf8').splitlines()
                    if line.startswith('+++ b/')})
    for name in names:
        if not (source / name).resolve().is_relative_to(source):
            raise RuntimeError('dependency patch target escapes source directory')
    actual = {name: (source / name).read_text(encoding='utf8') for name in names}
    # Only source files touched by these patches are copied, not the dependency
    # or its build. Nothing is stored as patch state or a content manifest.
    with tempfile.TemporaryDirectory(prefix='dependency-patches2-') as tmp:
        expected = Path(tmp)
        for name in names:
            path = expected / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output([git, '-C', str(source), 'show', base + ':' + name]))
        def matches():
            return all((expected / name).read_text(encoding='utf8') == actual[name] for name in names)
        prefix = 0 if matches() else None
        for index, patch in enumerate(patches, 1):
            subprocess.run([git, '-C', str(expected), 'apply', '--ignore-space-change', str(patch)], check=True)
            if matches(): prefix = index
        if prefix is None:
            raise RuntimeError('dependency sources do not match the pinned source or patch sequence; preserving local edits')
        for patch in patches[prefix:]:
            subprocess.run([git, '-C', str(source), 'apply', '--ignore-space-change', str(patch)], check=True)
            print('Applied dependency patch:', patch.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--git', default='git')
    parser.add_argument('--base', default='HEAD')
    parser.add_argument('patches', nargs='+', type=Path)
    args = parser.parse_args()
    try:
        apply(args.source, args.patches, args.git, args.base)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + '\n')
