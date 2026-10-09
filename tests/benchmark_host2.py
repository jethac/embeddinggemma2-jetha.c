"""Regression: WSL's idle guest must not qualify a saturated Windows host."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perf'))
import compare_llamacpp2 as benchmark


class HostGuard(unittest.TestCase):
    def test_busy_windows_host_rejects_idle_guest(self):
        with patch.object(benchmark, 'sample_host', return_value=([], 0, '')), \
                patch.object(benchmark, 'windows_host_cpu', return_value=1200, create=True), \
                patch.object(benchmark.time, 'sleep'), \
                patch.object(benchmark.time, 'monotonic', side_effect=range(0, 2000, 10)):
            with self.assertRaisesRegex(RuntimeError, 'host remained busy'):
                benchmark.wait_quiet(set(), 150)


if __name__ == '__main__':
    unittest.main()
