import os
import tempfile
import unittest
from pathlib import Path

from forex_bot.runlock import AlreadyRunningError, RunLock


class TestRunLock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "locks" / "engine-demo.lock"

    def tearDown(self):
        self.tmp.cleanup()

    def test_acquire_creates_lock_with_own_pid(self):
        with RunLock(self.path):
            self.assertTrue(self.path.exists())
            self.assertEqual(self.path.read_text().strip(), str(os.getpid()))
        self.assertFalse(self.path.exists())  # released on exit

    def test_second_engine_is_refused_while_holder_alive(self):
        # PID 1 (init) is always alive and is never this test process.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("1")
        with self.assertRaises(AlreadyRunningError) as ctx:
            RunLock(self.path).acquire()
        self.assertEqual(ctx.exception.pid, 1)

    def test_stale_lock_is_taken_over(self):
        # A PID that cannot exist (way beyond pid_max) marks the lock stale.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("99999999")
        lock = RunLock(self.path).acquire()
        self.assertEqual(self.path.read_text().strip(), str(os.getpid()))
        lock.release()

    def test_garbage_lock_content_is_treated_as_stale(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("not-a-pid")
        lock = RunLock(self.path).acquire()
        self.assertEqual(self.path.read_text().strip(), str(os.getpid()))
        lock.release()

    def test_release_never_deletes_someone_elses_lock(self):
        lock = RunLock(self.path).acquire()
        self.path.write_text("1")  # another process re-took the lock
        lock.release()
        self.assertTrue(self.path.exists())  # not ours anymore -> left alone

    def test_release_without_acquire_is_noop(self):
        RunLock(self.path).release()  # must not raise
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
