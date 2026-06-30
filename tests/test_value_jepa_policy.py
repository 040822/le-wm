import unittest

import torch

from source.policy.value_jepa import value_jepa_forward


class FakeModel:
    def encode(self, batch):
        return {
            "emb": batch["emb"],
            "act_emb": batch["action"],
        }

    def predict(self, emb, act_emb):
        return emb + 0.25 * act_emb


class FakeModule:
    def __init__(self):
        self.model = FakeModel()
        self.logged = None

    def sigreg(self, emb):
        return emb.new_tensor(2.0)

    def log_dict(self, metrics, **kwargs):
        self.logged = metrics


def make_batch():
    emb = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
        ]
    )
    action = torch.ones_like(emb)
    return {"emb": emb, "action": action}


class ValueJEPAPolicyForwardTests(unittest.TestCase):
    def test_value_disabled_matches_lewm_loss(self):
        module = FakeModule()
        batch = make_batch()

        output = value_jepa_forward(
            module,
            batch=batch,
            stage="fit",
            history_size=2,
            num_preds=1,
            sigreg_weight=0.5,
            value_weight=0.0,
            value_gamma=0.98,
            value_tau=0.8,
        )

        emb = batch["emb"]
        expected_pred = (
            module.model.predict(emb[:, :2], batch["action"][:, :2]) - emb[:, 1:]
        ).pow(2).mean()
        expected_loss = expected_pred + 0.5 * output["sigreg_loss"]

        self.assertNotIn("value_loss", output)
        self.assertTrue(torch.allclose(output["loss"], expected_loss))

    def test_value_enabled_outputs_finite_value_metrics(self):
        torch.manual_seed(0)
        module = FakeModule()

        output = value_jepa_forward(
            module,
            batch=make_batch(),
            stage="fit",
            history_size=2,
            num_preds=1,
            sigreg_weight=0.5,
            value_weight=0.1,
            value_gamma=0.98,
            value_tau=0.8,
        )

        base_loss = output["pred_loss"] + 0.5 * output["sigreg_loss"]
        expected_loss = base_loss + 0.1 * output["value_loss"]

        self.assertTrue(torch.allclose(output["loss"], expected_loss))
        for key in ("value_loss", "td_abs", "value_terminal", "value_random"):
            self.assertIn(key, output)
            self.assertTrue(output[key].isfinite())
            self.assertIn(f"fit/{key}", module.logged)


if __name__ == "__main__":
    unittest.main()
