import os
import unittest
from unittest import mock

from source.common.gpu_environment import configure_mujoco_egl_device


class GPUEnvironmentTests(unittest.TestCase):
    def test_binds_egl_to_the_single_visible_physical_gpu(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}, clear=True):
            selected = configure_mujoco_egl_device()

            self.assertEqual(selected, 3)
            self.assertEqual(os.environ["MUJOCO_EGL_DEVICE_ID"], "3")
            self.assertEqual(os.environ["MUJOCO_GL"], "egl")
            self.assertEqual(os.environ["PYOPENGL_PLATFORM"], "egl")

    def test_rejects_prohibited_gpu(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "6"}, clear=True):
            with self.assertRaisesRegex(EnvironmentError, "only GPU0-3"):
                configure_mujoco_egl_device()

    def test_rejects_multiple_visible_gpus(self):
        with mock.patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=True
        ):
            with self.assertRaisesRegex(EnvironmentError, "exactly one"):
                configure_mujoco_egl_device()

    def test_rejects_conflicting_existing_egl_binding(self):
        with mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "3", "MUJOCO_EGL_DEVICE_ID": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(EnvironmentError, "differs"):
                configure_mujoco_egl_device()


if __name__ == "__main__":
    unittest.main()
