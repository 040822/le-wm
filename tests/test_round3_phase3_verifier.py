import unittest

from omegaconf import OmegaConf

from scripts.verify_round3_phase3_e5 import _config_value


class Round3Phase3VerifierTests(unittest.TestCase):
    def test_scalar_hydra_values_are_read_without_to_container_error(self):
        config = OmegaConf.create(
            {
                "seed": 3072,
                "token_encoding": "physical_time_type",
                "detach_clean_action": False,
            }
        )
        self.assertEqual(_config_value(config, "seed"), 3072)
        self.assertEqual(
            _config_value(config, "token_encoding"), "physical_time_type"
        )
        self.assertIs(_config_value(config, "detach_clean_action"), False)


if __name__ == "__main__":
    unittest.main()
