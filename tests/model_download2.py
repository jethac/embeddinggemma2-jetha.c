"""Regression: a clean HTTP EOF can still truncate a model download."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io

spec = importlib.util.spec_from_file_location('download2',
    Path(__file__).resolve().parents[1] / 'scripts' / 'download-model2.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DownloadTest(unittest.TestCase):
    def test_truncated_eof_never_installed(self):
        name = 'mmproj-embeddinggemma-2-Q8_0.gguf'
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch.object(module.urllib.request, 'urlopen',
                              side_effect=lambda *args, **kwargs: io.BytesIO(b'partial')):
                with self.assertRaises(OSError):
                    module.download(name, directory)
            self.assertFalse((directory / name).exists())


if __name__ == '__main__':
    unittest.main()
