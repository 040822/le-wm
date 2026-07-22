import unittest

import torch

from eval import validate_generic_eval_policy
from tests.test_fast_lewam_model import make_model


class GenericEvalEntrypointTests(unittest.TestCase):
    def test_rejects_fast_lewam_and_accepts_non_fast_models(self):
        with self.assertRaisesRegex(ValueError, "eval_fast_lewam.py"):
            validate_generic_eval_policy(make_model())

        model = torch.nn.Linear(2, 2)
        self.assertIs(validate_generic_eval_policy(model), model)


if __name__ == "__main__":
    unittest.main()
