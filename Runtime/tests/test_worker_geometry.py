import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
import psutil

from rebuild3d_worker.bake import valid_projection
from rebuild3d_worker.common import camera_project, full_intrinsic, raster
from rebuild3d_worker.fuse import retain_component
from rebuild3d_worker.worker import Supervisor, Cancelled, checkpoint_valid, sha


class CameraTests(unittest.TestCase):
    def test_landscape_and_portrait_map_to_full_photo_with_pixel_centers(self):
        for width, height, full_w, full_h in [(1600, 1200, 5712, 4284), (1200, 1600, 4284, 5712)]:
            with self.subTest(size=(width, height)):
                nw, nh = round(width * 518 / max(width, height)), round(height * 518 / max(width, height))
                sx, sy = nw / width, nh / height
                transform = np.array([[sx, 0, (518-nw)//2 + (sx-1)/2],
                                      [0, sy, (518-nh)//2 + (sy-1)/2], [0, 0, 1]])
                record = dict(workingWidth=width, workingHeight=height, uprightWidth=full_w, uprightHeight=full_h)
                k = np.array([[370., 0, 259], [0, 369., 259], [0, 0, 1]])
                ext = np.c_[np.eye(3), [0, 0, 0]]
                points = np.array([[.12, .08, .8], [-.2, -.1, 1.4]])
                full = full_intrinsic(record, k, {"workingToModelPixels": transform})
                pixels = camera_project(points, ext, full)[:, :2]
                work = (pixels + .5) * [width/full_w, height/full_h] - .5
                model = np.c_[work, np.ones(len(work))] @ transform.T
                np.testing.assert_allclose(model[:, :2], camera_project(points, ext, k)[:, :2], atol=4e-5)

    def test_visibility_respects_each_photos_actual_extent(self):
        camera = {"ext": np.c_[np.eye(3), [0, 0, 0]], "workK": np.eye(3)}
        points = np.array([[1400., 900, 1], [900., 1400, 1]])
        for shape, expected in [((1200, 1600), [True, False]), ((1600, 1200), [False, True])]:
            valid, _, _ = valid_projection(points, camera, np.ones(shape), np.ones(shape) * 10, .001)
            self.assertEqual(valid.tolist(), expected)

    def test_visibility_uses_nearest_surface(self):
        vertices = np.array([[2,2,2],[10,2,2],[2,10,2],[2,2,1],[10,2,1],[2,10,1]], dtype="float32")
        z, ids, bary = raster(vertices, np.array([[0,1,2],[3,4,5]]), 14, 14)
        self.assertEqual(ids[4,4], 1)
        self.assertAlmostEqual(z[4,4], 1)
        self.assertEqual(ids[13,13], -1)
        np.testing.assert_allclose(bary[4,4], [.5,.25])


class ComponentTests(unittest.TestCase):
    def test_observed_accessory_survives_a_broken_thin_connector(self):
        area = np.ones(100) * .002
        support = np.zeros((100, 7), dtype=bool)
        support[:60, 0] = True
        support[40:, 1] = True
        keep, reason, _, _ = retain_component(1, .04, .003, area, 4., 1., support)
        self.assertTrue(keep)
        self.assertEqual(reason, "substantial-multiview-part")
        # The same detached geometry with evidence from just one photograph is
        # not enough; neither is a tiny floating fragment or a distant object.
        one_view = np.zeros_like(support); one_view[:, 0] = True
        self.assertFalse(retain_component(1, .04, .003, area, 4., 1., one_view)[0])
        self.assertFalse(retain_component(1, .04, .003, area * .01, 4., 1., support)[0])
        self.assertFalse(retain_component(1, .4, .003, area, 4., 1., support)[0])
        self.assertFalse(retain_component(1, .04, .003, area, 4., 1., np.zeros_like(support))[0])


class RecoveryTests(unittest.TestCase):
    def test_killed_supervisor_does_not_leave_a_running_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pidfile = root / "algorithm.pid"
            algorithm = "import os,pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)"
            supervisor_script = (
                "from rebuild3d_worker.worker import Supervisor;import pathlib,sys,os;"
                "Supervisor({'jobID':'owner-death'},pathlib.Path(sys.argv[1])).execute("
                "'test',[sys.executable,'-c',sys.argv[2],sys.argv[3]],os.environ.copy(),1024**3)"
            )
            parent = subprocess.Popen([sys.executable, "-c", supervisor_script, str(root), algorithm, str(pidfile)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            algorithm_pid = None
            try:
                deadline = time.monotonic() + 10
                while not pidfile.exists() and parent.poll() is None and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue(pidfile.exists(), "Stage must actually start before the interruption")
                algorithm_pid = int(pidfile.read_text())
                parent.kill()
                parent.wait(timeout=5)
                deadline = time.monotonic() + 12
                while time.monotonic() < deadline:
                    try:
                        if psutil.Process(algorithm_pid).status() == psutil.STATUS_ZOMBIE:
                            break
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(.1)
                else:
                    self.fail("Heavy stage kept running after its supervisor was killed")
            finally:
                if parent.poll() is None:
                    parent.kill()
                parent.wait()
                parent.stderr.close()
                if algorithm_pid:
                    try:
                        psutil.Process(algorithm_pid).kill()
                    except psutil.NoSuchProcess:
                        pass

    def test_checkpoint_rejects_changed_bytes_missing_outputs_and_foreign_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "image.bin"; output.write_bytes(b"abc")
            checkpoint = {"signature": "expected", "artifacts": [{"path": "image.bin", "bytes": 3, "sha256": sha(output)}]}
            self.assertTrue(checkpoint_valid(checkpoint, "expected", root))
            self.assertFalse(checkpoint_valid(checkpoint, "new-input", root))
            output.write_bytes(b"xyz")
            self.assertFalse(checkpoint_valid(checkpoint, "expected", root))
            output.unlink()
            self.assertFalse(checkpoint_valid(checkpoint, "expected", root))
            checkpoint["artifacts"][0]["path"] = "../outside.bin"
            self.assertFalse(checkpoint_valid(checkpoint, "expected", root))

    def test_cancel_terminates_real_worker_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supervisor = Supervisor({"jobID": "cancel-test"}, root)
            script = ("import subprocess,sys,time,pathlib; "
                      "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
                      "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)")
            timer = threading.Timer(1.5, supervisor.stop)
            timer.start()
            try:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(Cancelled):
                    supervisor.execute("cancel", [sys.executable, "-c", script, str(root / "child.pid")], os.environ.copy(), 1024**3)
                pid = int((root / "child.pid").read_text())
                try:
                    process = psutil.Process(pid)
                    self.assertEqual(process.status(), psutil.STATUS_ZOMBIE, "Child must not continue running after cancellation")
                except psutil.NoSuchProcess:
                    pass
            finally:
                timer.cancel()


if __name__ == "__main__":
    unittest.main()
