"""The port's installer must never download an upstream 300M release."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
with tempfile.TemporaryDirectory(prefix="installer-identity-") as directory:
    fixture = Path(directory)
    curl = fixture / "curl"
    curl.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$INSTALLER_PROBE_LOG"\nexit 22\n')
    curl.chmod(0o755)
    log = fixture / "download.txt"
    installer = fixture / "install.sh"
    installer.write_text((root / "install.sh").read_text())
    env = dict(os.environ, PATH=str(fixture) + os.pathsep + os.environ["PATH"],
               INSTALLER_PROBE_LOG=str(log))
    result = subprocess.run(["sh", str(installer), "--variant", "cpu",
                             "--install-dir", str(fixture / "installed")],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0, "a failed download must not install a binary"
    assert log.exists(), result.stderr
    attempted = log.read_text().splitlines()[-1]
    assert attempted == 'https://github.com/jethac/embeddinggemma2-jetha.c/releases/latest/download/SHA256SUMS', attempted
    assert not (fixture / "installed").exists()
print("Installer targets this port and leaves no installation after a failed download")
