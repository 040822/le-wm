import unittest

from tests.test_round3_phase1_core import make_dataset
from source.common.round3_variants import build_development_protocol_manifests


class VariantManifestTests(unittest.TestCase):
    def test_development_protocols_share_only_the_intended_cohort(self):
        manifests = build_development_protocol_manifests(
            make_dataset(), task="cube", seed=42, goal_offset_steps=25
        )
        self.assertEqual(set(manifests), {"legacy", "sampling_revised", "tolerance_revised", "round3_revised"})
        self.assertEqual(
            [item.row_index for item in manifests["legacy"].entries],
            [item.row_index for item in manifests["tolerance_revised"].entries],
        )
        self.assertEqual(
            [item.row_index for item in manifests["sampling_revised"].entries],
            [item.row_index for item in manifests["round3_revised"].entries],
        )
        self.assertNotEqual(manifests["legacy"].cohort_sha256, manifests["tolerance_revised"].cohort_sha256)


if __name__ == "__main__":
    unittest.main()
