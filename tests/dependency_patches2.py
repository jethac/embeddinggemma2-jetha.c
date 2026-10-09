"""Regression: overlapping patches must upgrade and reconfigure without edits."""
import difflib
import importlib.util
from pathlib import Path
import subprocess
import tempfile

spec = importlib.util.spec_from_file_location('patches2', Path(__file__).resolve().parents[1] / 'scripts/apply-dependency-patches2.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix='dependency-patch-regression-') as tmp:
    root = Path(tmp); source = root / 'source'; source.mkdir()
    original = 'struct opts {\n    int frames;\n    int fps;\n};\n'
    first = original.replace('    int fps;', '    int bytes;\n    int fps;')
    second = first.replace('    int bytes;', '    int bytes;\n    int timeout;')
    (source / 'opts.h').write_text(original)
    def git(*args): subprocess.run(['git', '-C', str(source), *args], check=True, stdout=subprocess.DEVNULL)
    git('init', '-q'); git('add', 'opts.h')
    git('-c', 'user.name=Cache test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'initial')
    patches = []
    for name, before, after in [('first', original, first), ('second', first, second)]:
        path = root / (name + '.patch')
        path.write_text(''.join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                        fromfile='a/opts.h', tofile='b/opts.h')))
        patches.append(path)
    module.apply(source, patches[:1])
    module.apply(source, patches)
    assert (source / 'opts.h').read_text() == second
    modified = (source / 'opts.h').stat().st_mtime_ns
    module.apply(source, patches)
    assert (source / 'opts.h').stat().st_mtime_ns == modified, 'reconfiguration touched already patched sources'
    edited = second + '// local change\n'; (source / 'opts.h').write_text(edited)
    try:
        module.apply(source, patches)
        raise AssertionError('local source edits were accepted as the known patch sequence')
    except RuntimeError as error:
        assert 'preserving local edits' in str(error)
    assert (source / 'opts.h').read_text() == edited
print('Overlapping patch upgrade, idempotence and local-edit preservation passed')
