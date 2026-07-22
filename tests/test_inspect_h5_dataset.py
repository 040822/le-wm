import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from scripts.inspect_h5_dataset import episode_lengths, find_h5_files, inspect_file


class InspectH5DatasetTests(unittest.TestCase):
    def _make_dataset(self, directory: str) -> Path:
        path = Path(directory) / "sample.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("ep_len", data=np.asarray([2, 3], dtype=np.int32))
            handle.create_dataset("ep_offset", data=np.asarray([0, 2], dtype=np.int64))
            handle.create_dataset("action", data=np.zeros((5, 2), dtype=np.float32))
            handle.create_dataset("pixels", data=np.zeros((5, 4, 4, 3), dtype=np.uint8))
        return path

    def test_finds_h5_files_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "nested"
            nested.mkdir()
            expected = self._make_dataset(str(nested))
            (Path(directory) / "ignored.txt").write_text("not hdf5")

            self.assertEqual(find_h5_files(Path(directory)), [expected])

    def test_prints_episode_lengths_and_dataset_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._make_dataset(directory)
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                inspect_file(path)

            report = output.getvalue()
            self.assertIn("Episodes: 2", report)
            self.assertIn("Total steps: 5", report)
            self.assertIn("episode      0: 2", report)
            self.assertIn("episode      1: 3", report)
            self.assertIn("/action: dataset, shape=(5, 2), dtype=float32, role=per-step", report)
            self.assertIn("/ep_len: dataset, shape=(2,), dtype=int32, role=per-episode", report)

    def test_infers_lengths_from_noncontiguous_episode_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    "episode_idx",
                    data=np.asarray([10, 10, 42, 42, 42], dtype=np.int64),
                )

            with h5py.File(path, "r") as handle:
                lengths, source = episode_lengths(handle)

            np.testing.assert_array_equal(lengths, np.asarray([2, 3]))
            self.assertEqual(source, "/episode_idx")

    def test_summary_only_omits_middle_episode_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "many_episodes.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("ep_len", data=np.arange(1, 26))
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                inspect_file(path, summary_only=True)

            report = output.getvalue()
            self.assertIn("episode      9: 10", report)
            self.assertNotIn("episode     10: 11", report)
            self.assertIn("episode     15: 16", report)
            self.assertIn("5 episodes omitted", report)


if __name__ == "__main__":
    unittest.main()
