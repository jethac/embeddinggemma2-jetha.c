"""Observed regression: native Windows benchmarking tried to execute Unix ps.

Reject this sampler if one busy thread has machine-normalized units or if
ignoring its PID subtracts anything other than its measured CPU delta.
"""
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perf'))
import compare_llamacpp as benchmark


@unittest.skipUnless(os.name == 'nt', 'actual Windows API regression')
class WindowsHost(unittest.TestCase):
    def test_native_sampling_and_owned_busy_thread_units(self):
        # The original public call raised WinError 2 before a server could time.
        _, total, _ = benchmark.sample_host(set(), math.inf, math.inf)
        self.assertTrue(math.isfinite(total) and total >= 0)
        # A venv's python.exe can be a redirector with a separate zero-CPU PID.
        executable = Path(sys.base_prefix) / 'python.exe'
        worker = subprocess.Popen([str(executable), '-u', '-c',
            "import os,time; print(os.getpid(),flush=True); end=time.monotonic()+30; "
            "exec('while time.monotonic()<end: pass')"], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(int(worker.stdout.readline().strip()), worker.pid)
            snapshot = benchmark._windows_host_snapshot()
            global_cpu, entries = snapshot
            owned = next(entry for entry in entries if entry[0] == worker.pid)
            self.assertGreater(owned[1], 70, 'busy thread did not calibrate one core')
            self.assertLess(owned[1], 130, 'CPU units must be 100 per busy core')
            self.assertGreater(owned[2], 0, 'working set query failed')
            self.assertEqual(owned[3], executable.name)
            self.assertGreaterEqual(global_cpu, owned[1])
            counted = benchmark._windows_host_summary(snapshot, set(), 0, math.inf)
            ignored = benchmark._windows_host_summary(snapshot, {worker.pid}, 0, math.inf)
            self.assertAlmostEqual(counted[1] - ignored[1], owned[1], places=8)
            self.assertTrue(any(line.startswith(f'pid {worker.pid}:') for line in counted[0]))
            self.assertFalse(any(line.startswith(f'pid {worker.pid}:') for line in ignored[0]))
            print(f'Windows CPU calibration: busy thread={owned[1]:.3f}, '
                  f'global={global_cpu:.3f}, ignored={ignored[1]:.3f} '
                  '(100=one core)', flush=True)
        finally:
            if worker.poll() is None:
                worker.terminate()
            worker.wait(timeout=5)
            worker.stdout.close()


if __name__ == '__main__':
    unittest.main()
