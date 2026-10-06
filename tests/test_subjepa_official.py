import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from eval import get_eval_policy_kind, validate_generic_eval_policy
from scripts.prepare_subjepa_official import fetch, sha256
from source.common.checkpoint import load_policy_or_model
from source.model.subjepa.official import FORMAT, OfficialSubJEPA, conversion_checks


def make_config():
    return {
        "encoder": {
            "hidden_size": 12, "intermediate_size": 24,
            "num_hidden_layers": 1, "num_attention_heads": 3,
            "image_size": 28, "patch_size": 14,
        },
        "latent_dim": 4, "history_size": 2, "action_input_dim": 2,
        "action_smoothed_dim": 2, "action_mlp_scale": 4,
        "projector_hidden_dim": 16, "pred_proj_hidden_dim": 16,
        "predictor": {
            "depth": 1, "heads": 2, "dim_head": 2, "mlp_dim": 16,
            "dropout": 0.0, "emb_dropout": 0.0,
        },
    }


def make_payload():
    config = make_config()
    model = OfficialSubJEPA(config, {"task": "tworoom", "revision": "fixture"}).eval()
    return {
        "format": FORMAT, "model_config": config,
        "state_dict": model.state_dict(), "provenance": model.provenance,
    }


class OfficialSubJEPATests(unittest.TestCase):
    def test_direct_path_does_not_require_a_writable_default_cache(self):
        payload = make_payload()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "subjepa.pt"
            torch.save(payload, path)
            with mock.patch("source.common.checkpoint._get_swm_cache_dir", side_effect=RuntimeError("unwritable")):
                model, resolved = load_policy_or_model(path)
        self.assertIsInstance(model, OfficialSubJEPA)
        self.assertEqual(resolved, path)

    def test_direct_and_cache_paths_load_weights_and_keep_method_identity(self):
        payload = make_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "checkpoints" / "subjepa_official" / "tworoom" / "subjepa.pt"
            path.parent.mkdir(parents=True)
            torch.save(payload, path)
            for name in (path, "subjepa_official/tworoom/subjepa.pt"):
                model, resolved = load_policy_or_model(name, cache_dir=root)
                self.assertIsInstance(model, OfficialSubJEPA)
                self.assertEqual(resolved, path)
                self.assertEqual(get_eval_policy_kind(model), "subjepa_official")
                self.assertIs(validate_generic_eval_policy(model), model)
                self.assertEqual(model.provenance, payload["provenance"])
                self.assertFalse(model.training)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, payload["state_dict"][key]))

    def test_loading_rejects_missing_weights(self):
        payload = make_payload()
        del payload["state_dict"]["predictor.pos_embedding"]
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            OfficialSubJEPA.from_payload(payload)

    def test_rollout_uses_configured_history_and_is_independent_of_candidate_chunking(self):
        model = OfficialSubJEPA.from_payload(make_payload())
        candidates = torch.randn(1, 3, 5, 2)
        pixels = torch.randn(1, 1, 3, 28, 28)
        goal = torch.randn_like(pixels)

        def score(actions):
            count = actions.shape[1]
            info = {
                "pixels": pixels[:, None].expand(-1, count, -1, -1, -1, -1).clone(),
                "goal": goal[:, None].expand(-1, count, -1, -1, -1, -1).clone(),
                "action": actions[:, :, :1].clone(),
            }
            return model.get_cost(info, actions)

        with torch.no_grad():
            cost = score(candidates)
            chunked = torch.cat([score(candidates[:, i:i+1]) for i in range(3)], dim=1)
        self.assertEqual(cost.shape, (1, 3))
        self.assertTrue(torch.isfinite(cost).all())
        torch.testing.assert_close(cost, chunked, atol=2e-6, rtol=2e-6)

    def test_conversion_checks_reject_changed_predictions(self):
        model = OfficialSubJEPA.from_payload(make_payload())
        pixels = torch.randn(1, 2, 3, 28, 28)
        actions = torch.randn(1, 2, 2)
        candidates = torch.randn(1, 2, 5, 2)
        info = {
            "pixels": pixels[:, None, :1].expand(-1, 2, -1, -1, -1, -1).clone(),
            "goal": pixels[:, None, 1:2].expand(-1, 2, -1, -1, -1, -1).clone(),
            "action": candidates[:, :, :1].clone(),
        }
        with torch.no_grad():
            encoded = model.encode({"pixels": pixels, "action": actions})
            prediction = model.predict(encoded["emb"], encoded["act_emb"])
            scored = {key: value.clone() for key, value in info.items()}
            cost = model.get_cost(scored, candidates)
        expected = {
            "emb": encoded["emb"], "act_emb": encoded["act_emb"],
            "prediction": prediction, "rollout": scored["predicted_emb"], "cost": cost,
        }
        fixture = {"pixels": pixels, "action": actions, "cost_info": info, "candidates": candidates}
        self.assertTrue(conversion_checks(model, fixture, expected)["ranking_equal"])
        expected["prediction"] = expected["prediction"] + 0.1
        with self.assertRaises(AssertionError):
            conversion_checks(model, fixture, expected)

    def test_offline_download_validates_existing_asset_and_preserves_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.ckpt"
            path.write_bytes(b"original")
            fetch("unused", path, sha256(path), offline=True)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                fetch("unused", path, "wrong", offline=True)
            self.assertEqual(path.read_bytes(), b"original")
            with self.assertRaisesRegex(FileNotFoundError, "Offline asset"):
                fetch("unused", path.with_name("missing"), "wrong", offline=True)


if __name__ == "__main__":
    unittest.main()
