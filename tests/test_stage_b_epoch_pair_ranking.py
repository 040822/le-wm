import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from scripts import diagnose_fast_lewam_epoch_pair_ranking as diagnostic_cli

from source.diagnostics.stage_b_epoch_pair_ranking import (
    EpochPair,
    EpochPairManifest,
    PairEpoch,
    build_cross_epoch_panel,
    build_ground_identity,
    cem_candidate_indices,
    classify_epoch_pair_failure,
    compute_ranking_metrics,
    compute_pair_physical_terminal_cost,
    cross_score_costs,
    grounded_pair_acceptance_errors,
    load_epoch_pair_manifest,
    planning_state_sha256,
    require_matching_cache_identity,
    summarize_grounded_pair,
    validate_diagnostic_device,
    validate_pair_device,
    write_atomic_json,
    write_pair_trace_artifacts,
)
from source.diagnostics.stage_b_episode_failures import CEMTraceCollector


_FINITE_METRICS = {
    "predicted_physical_spearman": 0.5,
    "predicted_true_terminal_latent_spearman": 0.5,
    "true_terminal_latent_physical_spearman": 0.5,
    "physical_topk_recall": 0.5,
    "successful_candidate_recall_at_k": 0.0,
    "top1_success": False,
    "oracle_success": False,
    "candidate_success_rate": 0.0,
    "physical_best_candidate_regret": 1.0,
    "predicted_terminal_latent_mse": 0.1,
}


def _complete_panel(state_epoch, replan, iteration):
    return {
        "state_epoch": state_epoch,
        "replan": replan,
        "iteration": iteration,
        "planning_state_sha256": "1" * 64,
        "prefix_sha256": "2" * 64,
        "sampled_candidate_sha256": "3" * 64,
        "candidate_sha256": "4" * 64,
        "candidate_count": 2,
        "candidate_successes": [False, False],
        "cem_candidate_successes": [False, False],
        "models": {
            epoch: {"metrics": dict(_FINITE_METRICS)}
            for epoch in ("e8", "e10")
        },
    }


def _complete_slot(pair, slot):
    panels = [
        _complete_panel("shared", 0, iteration)
        for iteration in (0, 5, 29)
    ]
    panels.extend(
        _complete_panel(epoch, 1, iteration)
        for epoch in ("e8", "e10")
        for iteration in (0, 5, 29)
    )
    return {
        "slot": slot,
        "category": pair.category_for(slot),
        "final_success": {"e8": False, "e10": False},
        "terminal_before_replan_1": [],
        "attributions": {"e8": [], "e10": []},
        "panels": panels,
    }


class EpochPairManifestTests(unittest.TestCase):
    def test_manifest_declares_three_valid_pairs_and_38_fixed_slots(self):
        manifest = load_epoch_pair_manifest(
            "config/diagnostics/fast_lewam_epoch_pair_ranking.yaml"
        )

        self.assertEqual(
            set(manifest.pairs),
            {"reacher_s32", "reacher_s41", "pusht_s41"},
        )
        self.assertEqual(
            sum(len(pair.slots) for pair in manifest.pairs.values()),
            38,
        )
        for pair in manifest.pairs.values():
            self.assertEqual(set(pair.epochs), {"e8", "e10"})
            self.assertEqual(pair.epochs["e8"].epoch, 8)
            self.assertEqual(pair.epochs["e10"].epoch, 10)
            self.assertTrue(pair.run_config.is_file())
            self.assertTrue(pair.epochs["e8"].checkpoint.is_file())
            self.assertTrue(pair.epochs["e10"].reference_result.is_file())

    def test_gpu_guard_requires_an_explicit_permitted_physical_device(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(EnvironmentError, "CUDA_VISIBLE_DEVICES"):
                validate_diagnostic_device("cuda:0")
        with mock.patch.dict(
            "os.environ", {"CUDA_VISIBLE_DEVICES": "4"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "GPU0-3"):
                validate_diagnostic_device("cuda:0")
        with mock.patch.dict(
            "os.environ", {"CUDA_VISIBLE_DEVICES": "1"}, clear=True
        ):
            self.assertEqual(validate_diagnostic_device("cuda:0"), (1,))

    def test_pair_guard_enforces_manifest_preferred_gpu(self):
        pair = load_epoch_pair_manifest(
            "config/diagnostics/fast_lewam_epoch_pair_ranking.yaml"
        ).pairs["pusht_s41"]
        with mock.patch.dict(
            "os.environ", {"CUDA_VISIBLE_DEVICES": "2"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "preferred physical GPU3"):
                validate_pair_device(pair, "cuda:0")
        with mock.patch.dict(
            "os.environ", {"CUDA_VISIBLE_DEVICES": "3"}, clear=True
        ):
            self.assertEqual(validate_pair_device(pair, "cuda:0"), (3,))

    def test_state_hash_ignores_render_timing_but_detects_state_changes(self):
        base = {
            "pixels": torch.zeros(1, 2),
            "render_time": np.asarray([1.0]),
        }
        changed_timing = {**base, "render_time": np.asarray([99.0])}
        changed_state = {**base, "pixels": torch.ones(1, 2)}

        self.assertEqual(
            planning_state_sha256(base),
            planning_state_sha256(changed_timing),
        )
        self.assertNotEqual(
            planning_state_sha256(base),
            planning_state_sha256(changed_state),
        )

    def test_cube_physical_metric_matches_position_only_success_geometry(self):
        cost = compute_pair_physical_terminal_cost(
            "cube",
            {
                "privileged_block_0_pos": [[0.0, 0.0, 0.0]],
                "goal_privileged_block_0_pos": [[0.03, 0.04, 0.0]],
            },
        )
        np.testing.assert_allclose(cost, [0.05])


class SharedPanelTests(unittest.TestCase):
    def test_union_contains_both_epoch_elites_and_deduplicates_action_bytes(self):
        def detail(values, offset):
            candidates = torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1)
            return {
                "candidates": candidates,
                "costs": torch.arange(len(values), dtype=torch.float32),
                "topk_indices": torch.tensor([0, 1]),
                "previous_mean": torch.tensor([[20.0 + offset]]),
                "best_ever": torch.tensor([[30.0 + offset]]),
            }

        panel = build_cross_epoch_panel(
            {"e8": detail([0, 1, 2, 3, 4, 5], 0),
             "e10": detail([0, 6, 7, 8, 9, 10], 1)},
            expert=torch.tensor([[99.0]]),
            non_elite_count=1,
            seed=42,
        )

        action_bytes = [row.contiguous().numpy().tobytes() for row in panel.actions]
        self.assertEqual(len(action_bytes), len(set(action_bytes)))
        self.assertIn("e8:elite_0", panel.sources)
        self.assertIn("e10:elite_1", panel.sources)
        self.assertIn("anchor:expert", panel.sources)
        duplicate_index = next(
            index
            for index, aliases in enumerate(panel.source_aliases)
            if "e8:elite_0" in aliases
        )
        self.assertIn("e10:elite_0", panel.source_aliases[duplicate_index])
        anchor_index = next(
            index
            for index, aliases in enumerate(panel.source_aliases)
            if aliases == ("anchor:expert",)
        )
        self.assertNotIn(anchor_index, cem_candidate_indices(panel))
        self.assertIn(duplicate_index, cem_candidate_indices(panel))
        self.assertEqual(len(panel.sha256), 64)

    def test_cross_scoring_preserves_one_shared_candidate_panel(self):
        actions = torch.tensor([[[1.0]], [[2.0]]])
        scored = cross_score_costs(
            {"e8": 1.0, "e10": 10.0},
            {"state": torch.tensor([[3.0]])},
            actions,
            scorer=lambda model, info, panel: (
                panel.reshape(-1).numpy() * model
            ),
        )

        np.testing.assert_array_equal(scored["e8"], [1.0, 2.0])
        np.testing.assert_array_equal(scored["e10"], [10.0, 20.0])
        torch.testing.assert_close(actions, torch.tensor([[[1.0]], [[2.0]]]))

    def test_replan_panel_can_use_one_epochs_candidates_and_cross_score_both(
        self,
    ):
        detail = {
            "candidates": torch.tensor([[[0.0]], [[1.0]]]),
            "costs": torch.tensor([0.0, 1.0]),
            "topk_indices": torch.tensor([0]),
            "previous_mean": torch.tensor([[0.5]]),
            "best_ever": torch.tensor([[0.0]]),
        }
        panel = build_cross_epoch_panel(
            {"e8": detail},
            expert=torch.tensor([[2.0]]),
            non_elite_count=1,
        )
        scored = cross_score_costs(
            {"e8": 1.0, "e10": 2.0},
            {},
            panel.actions,
            scorer=lambda model, info, actions: (
                actions.reshape(-1).numpy() * model
            ),
        )
        self.assertEqual(set(scored), {"e8", "e10"})
        self.assertTrue(
            all(
                source.startswith(("e8:", "anchor:"))
                for source in panel.sources
            )
        )


class RankingMetricTests(unittest.TestCase):
    def test_metrics_compare_one_model_with_shared_physical_outcomes(self):
        metrics = compute_ranking_metrics(
            predicted_cost=[0.0, 1.0, 2.0, 3.0],
            true_terminal_latent_cost=[0.0, 2.0, 1.0, 3.0],
            physical_cost=[3.0, 0.0, 1.0, 2.0],
            successes=[False, True, False, True],
            predicted_terminal_latents=torch.tensor([[0.0], [2.0], [1.0], [3.0]]),
            true_terminal_latents=torch.tensor([[0.0], [1.0], [1.0], [1.0]]),
            topk=2,
        )

        self.assertEqual(metrics["physical_topk_recall"], 0.5)
        self.assertEqual(metrics["successful_candidate_recall_at_k"], 0.5)
        self.assertFalse(metrics["top1_success"])
        self.assertTrue(metrics["oracle_success"])
        self.assertEqual(metrics["candidate_success_rate"], 0.5)
        self.assertEqual(metrics["physical_best_candidate_regret"], 3.0)
        self.assertEqual(metrics["predicted_terminal_latent_mse"], 1.25)


class CacheIdentityTests(unittest.TestCase):
    def test_cache_reuse_refuses_every_identity_change(self):
        identity = {
            "manifest_sha256": "manifest-a",
            "checkpoint_sha256": "checkpoint-a",
            "config_sha256": "config-a",
            "reference_result_sha256": "reference-a",
            "prefix_sha256": "prefix-a",
            "candidate_sha256": "candidate-a",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            write_atomic_json(path, {"identity": identity, "status": "ok"})
            self.assertTrue(require_matching_cache_identity(path, identity))
            for key in identity:
                changed = dict(identity)
                changed[key] += "-changed"
                with self.subTest(key), self.assertRaisesRegex(
                    RuntimeError, "stale cache identity"
                ):
                    require_matching_cache_identity(path, changed)
            self.assertEqual(json.loads(path.read_text())["status"], "ok")

    def test_pair_ground_identity_covers_prefix_and_all_panel_inputs(self):
        manifest = load_epoch_pair_manifest(
            "config/diagnostics/fast_lewam_epoch_pair_ranking.yaml"
        )
        pair = manifest.pairs["reacher_s32"]

        def traces(prefix=0.0, candidate=0.0, cost=0.0):
            return {
                label: {
                    "plans": [
                        {
                            "slot": 6,
                            "replan": 0,
                            "executed_prefix_before": np.asarray(
                                [[prefix]], dtype=np.float32
                            ),
                            "planning_info": {
                                "pixels": torch.tensor([[1.0]])
                            },
                        }
                    ],
                    "details": [
                        {
                            "slot": 6,
                            "replan": 0,
                            "iteration": 0,
                            "candidates": torch.tensor([[[candidate]]]),
                            "costs": torch.tensor([cost]),
                            "topk_indices": torch.tensor([0]),
                            "previous_mean": torch.tensor([[0.0]]),
                            "best_ever": torch.tensor([[0.0]]),
                        }
                    ],
                }
                for label in ("e8", "e10")
            }

        original = build_ground_identity(manifest, pair, traces())
        changed_prefix = build_ground_identity(
            manifest, pair, traces(prefix=1.0)
        )
        changed_candidate = build_ground_identity(
            manifest, pair, traces(candidate=1.0)
        )
        changed_cost = build_ground_identity(
            manifest, pair, traces(cost=1.0)
        )
        self.assertNotEqual(
            original["prefix_sha256"], changed_prefix["prefix_sha256"]
        )
        self.assertNotEqual(
            original["candidate_sha256"],
            changed_candidate["candidate_sha256"],
        )
        self.assertNotEqual(
            original["candidate_sha256"],
            changed_cost["candidate_sha256"],
        )


class AttributionTests(unittest.TestCase):
    def test_synthetic_evidence_covers_every_conservative_failure_label(self):
        cases = {
            "coverage_failure": {
                "all_candidates_grounded": True,
                "candidate_successes": [False, False],
                "selected_success": False,
            },
            "ranking_failure": {
                "all_candidates_grounded": True,
                "candidate_successes": [False, True],
                "selected_success": False,
            },
            "dynamics_error": {
                "predicted_true_terminal_latent_spearman": -0.5,
                "true_terminal_latent_physical_spearman": 0.5,
            },
            "latent_metric_error": {
                "predicted_true_terminal_latent_spearman": 0.5,
                "true_terminal_latent_physical_spearman": -0.5,
            },
            "cem_exploitation": {
                "predicted_elite_cost_0_5_29": [3.0, 2.0, 1.0],
                "physical_elite_cost_0_5_29": [1.0, 2.0, 3.0],
                "earlier_best_physical_cost": 0.5,
                "final_mean_physical_cost": 3.0,
            },
            "replan_regression": {
                "physical_distance_start_after_first_final": [4.0, 1.0, 3.0]
            },
        }
        for expected, evidence in cases.items():
            with self.subTest(expected):
                self.assertIn(expected, classify_epoch_pair_failure(evidence))
        self.assertEqual(
            classify_epoch_pair_failure({}), ["mixed_or_undetermined"]
        )


class PairedSummaryTests(unittest.TestCase):
    def test_summary_reports_e10_minus_e8_within_slot_category(self):
        pair = load_epoch_pair_manifest(
            "config/diagnostics/fast_lewam_epoch_pair_ranking.yaml"
        ).pairs["reacher_s32"]
        grounded = {
            "status": "ok",
            "slots": [
                {
                    "slot": 6,
                    "category": "regression",
                    "attributions": {
                        "e8": ["ranking_failure"],
                        "e10": ["coverage_failure"],
                    },
                    "panels": [
                        {
                            "models": {
                                "e8": {"metrics": {
                                    "physical_topk_recall": 0.75,
                                    "physical_best_candidate_regret": 1.0,
                                }},
                                "e10": {"metrics": {
                                    "physical_topk_recall": 0.25,
                                    "physical_best_candidate_regret": 3.0,
                                }},
                            }
                        }
                    ],
                }
            ],
        }

        summary = summarize_grounded_pair(pair, grounded)

        category = summary["categories"]["regression"]
        self.assertEqual(
            category["e10_minus_e8"]["physical_topk_recall"], -0.5
        )
        self.assertEqual(
            category["e10_minus_e8"]["physical_best_candidate_regret"], 2.0
        )
        self.assertEqual(category["attribution_counts"]["e10"], {
            "coverage_failure": 1
        })

    def test_acceptance_rejects_missing_panel_nonfinite_metric_and_bad_hash(self):
        pair = load_epoch_pair_manifest(
            "config/diagnostics/fast_lewam_epoch_pair_ranking.yaml"
        ).pairs["pusht_s41"]
        grounded = {
            "status": "ok",
            "slots": [_complete_slot(pair, slot) for slot in pair.slots],
        }
        self.assertEqual(grounded_pair_acceptance_errors(pair, grounded), [])

        grounded["slots"][0]["panels"].pop()
        self.assertTrue(grounded_pair_acceptance_errors(pair, grounded))
        grounded["slots"][0] = _complete_slot(pair, pair.slots[0])
        grounded["slots"][0]["panels"][0]["models"]["e8"]["metrics"][
            "predicted_physical_spearman"
        ] = None
        self.assertTrue(grounded_pair_acceptance_errors(pair, grounded))
        grounded["slots"][0] = _complete_slot(pair, pair.slots[0])
        grounded["slots"][0]["panels"][0]["candidate_sha256"] = "bad"
        self.assertTrue(grounded_pair_acceptance_errors(pair, grounded))

        complete = {
            "status": "incomplete",
            "acceptance_errors": ["score artifact is missing"],
            "slots": [
                _complete_slot(pair, slot) for slot in pair.slots
            ],
        }
        summary = summarize_grounded_pair(pair, complete)
        self.assertEqual(summary["status"], "incomplete")
        self.assertIn(
            "score artifact is missing", summary["acceptance_errors"]
        )

    def test_complete_synthetic_grounding_produces_ok_summary_artifact(self):
        pair = load_epoch_pair_manifest(
            "config/diagnostics/fast_lewam_epoch_pair_ranking.yaml"
        ).pairs["pusht_s41"]
        slots = [_complete_slot(pair, slot) for slot in pair.slots]
        summary = summarize_grounded_pair(
            pair, {"status": "ok", "slots": slots}
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            write_atomic_json(path, summary)
            restored = json.loads(path.read_text())

        self.assertEqual(restored["status"], "ok")
        self.assertEqual(restored["overall"]["slot_count"], 6)


class TraceArtifactTests(unittest.TestCase):
    def test_trace_writer_keeps_scalars_but_only_fixed_slot_details(self):
        collector = CEMTraceCollector(detailed_steps={0, 1})
        for slot in (0, 1):
            for iteration in (0, 1):
                collector.summaries.append(
                    {
                        "slot": slot,
                        "replan": 0,
                        "iteration": iteration,
                        "candidate_count": 2,
                        "cost_min": 0.0,
                    }
                )
                collector.details.append(
                    {
                        "slot": slot,
                        "replan": 0,
                        "iteration": iteration,
                        "candidates": torch.zeros(2, 1, 1) + slot,
                        "costs": torch.zeros(2),
                        "topk_indices": torch.tensor([0]),
                        "previous_mean": torch.zeros(1, 1),
                    }
                )
            collector.plans.append(
                {
                    "slot": slot,
                    "replan": 0,
                    "planning_info": {
                        "pixels": torch.zeros(1, 1, 1) + slot
                    },
                    "executed_prefix_before": np.empty(
                        (0, 1), dtype=np.float32
                    ),
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            result = write_pair_trace_artifacts(
                temporary,
                collector=collector,
                selected_slots=[1],
                actual_successes=[False, True],
                expected_successes=[False, True],
                identity={"phase": "trace"},
                protocol={"n_steps": 2, "detailed_iterations": [0, 1]},
            )
            trace = torch.load(
                Path(temporary) / "trace.pt",
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(result["status"], "reproducible")
        self.assertEqual({row["slot"] for row in trace["details"]}, {1})
        self.assertEqual(len(trace["summaries"]), 2)

    def test_reacher_single_slot_trace_ground_summarize_runtime_smoke(self):
        class FakeModel:
            def to(self, device):
                return self

            def eval(self):
                return self

        class FakeSession:
            def __init__(self, cfg, task):
                pass

            def _build_policy(self, subject, identity, device):
                return SimpleNamespace(
                    solver=SimpleNamespace(callbacks=[], batch_size=1)
                )

            def evaluate(self, policy, **kwargs):
                collector = policy.collector
                for replan in (0, 1):
                    prefix = np.asarray(
                        [[0.1]] if replan else [], dtype=np.float32
                    ).reshape(-1, 1)
                    collector.plans.append(
                        {
                            "slot": 0,
                            "replan": replan,
                            "normalized_plan": torch.zeros(1, 1),
                            "executed_prefix_before": prefix,
                            "planning_info": {
                                "pixels": torch.tensor(
                                    [[[float(replan)]]]
                                )
                            },
                        }
                    )
                    for iteration in (0, 5, 29):
                        candidates = torch.tensor(
                            [[[0.0]], [[1.0]], [[2.0]]]
                        )
                        collector.summaries.append(
                            {
                                "slot": 0,
                                "replan": replan,
                                "iteration": iteration,
                                "candidate_count": 3,
                                "cost_min": 0.0,
                            }
                        )
                        collector.details.append(
                            {
                                "slot": 0,
                                "replan": replan,
                                "iteration": iteration,
                                "candidates": candidates,
                                "costs": torch.tensor([0.0, 1.0, 2.0]),
                                "topk_indices": torch.tensor([0, 1]),
                                "previous_mean": torch.tensor([[0.5]]),
                                "previous_variance": torch.ones(1, 1),
                                "updated_mean": torch.tensor([[0.5]]),
                                "updated_variance": torch.ones(1, 1),
                                "best_ever": torch.tensor([[0.0]]),
                            }
                        )
                return SimpleNamespace(
                    episodes=[SimpleNamespace(success=False)]
                )

        class FakeDataset:
            def load_chunk(self, *args, **kwargs):
                return [{}]

        class FakeGrounder:
            def run(self, *, candidates, **kwargs):
                count = len(candidates)
                terminal = torch.arange(count, dtype=torch.float32).reshape(
                    count, 1
                )
                return {
                    "physical_cost": np.arange(count, dtype=np.float64),
                    "terminal_pixels": terminal,
                    "terminal_goal": torch.zeros_like(terminal),
                    "success": np.zeros(count, dtype=bool),
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.yaml"
            run_config = root / "run.yaml"
            manifest_path.write_text("version: 1\n", encoding="utf-8")
            run_config.write_text("task: reacher\n", encoding="utf-8")
            epochs = {}
            for label, epoch in (("e8", 8), ("e10", 10)):
                checkpoint = root / f"{label}.ckpt"
                reference = root / f"{label}.json"
                checkpoint.write_bytes(label.encode("ascii"))
                reference.write_text(
                    json.dumps(
                        {
                            "status": "ok",
                            "parameters": {
                                "episode_ids": [0],
                                "start_steps": [0],
                            },
                            "episodes": [{"success": False}],
                        }
                    ),
                    encoding="utf-8",
                )
                epochs[label] = PairEpoch(
                    label=label,
                    epoch=epoch,
                    checkpoint=checkpoint,
                    reference_result=reference,
                )
            pair = EpochPair(
                label="reacher_smoke",
                task="reacher",
                section="smoke",
                preferred_gpu=0,
                run_config=run_config,
                epochs=epochs,
                slot_categories={
                    "regression": (0,),
                    "improvement_control": (),
                    "stable_success": (),
                    "stable_failure": (),
                },
            )
            manifest = EpochPairManifest(
                path=manifest_path,
                root=root,
                output_root=root / "outputs",
                protocol={
                    "seed": 42,
                    "n_steps": 3,
                    "detailed_iterations": [0, 5, 29],
                    "fixed_non_elites": 1,
                    "topk": 2,
                    "goal_offset_steps": 1,
                },
                pairs={pair.label: pair},
            )
            cfg = SimpleNamespace(
                plan_config=SimpleNamespace(horizon=1, action_block=1),
                eval=SimpleNamespace(
                    goal_offset_steps=1,
                    eval_budget=2,
                    callables={},
                    dataset_name="smoke"
                ),
            )

            def predict_costs(model, info, actions, device):
                return np.arange(len(actions), dtype=np.float64)

            def predict_terminal(model, info, actions, device):
                return torch.arange(
                    len(actions), dtype=torch.float32
                ).reshape(-1, 1)

            def terminal_latents(model, info):
                count = len(info["pixels"])
                values = torch.arange(
                    count, dtype=torch.float32
                ).reshape(-1, 1)
                return values, np.arange(count, dtype=np.float64)

            patches = (
                mock.patch.object(
                    diagnostic_cli,
                    "load_epoch_pair_manifest",
                    return_value=manifest,
                ),
                mock.patch.object(
                    diagnostic_cli, "validate_pair_device", return_value=()
                ),
                mock.patch.object(
                    diagnostic_cli,
                    "DatasetEvaluationSession",
                    FakeSession,
                ),
                mock.patch.object(
                    diagnostic_cli, "compose_eval_config", return_value=cfg
                ),
                mock.patch.object(
                    diagnostic_cli,
                    "_load_subject",
                    return_value=FakeModel(),
                ),
                mock.patch.object(
                    diagnostic_cli,
                    "get_dataset",
                    return_value=FakeDataset(),
                ),
                mock.patch.object(
                    diagnostic_cli,
                    "fit_eval_processors",
                    return_value={"action": object()},
                ),
                mock.patch.object(
                    diagnostic_cli,
                    "PairSimulatorGrounder",
                    return_value=FakeGrounder(),
                ),
                mock.patch.object(
                    diagnostic_cli,
                    "normalize_expert_plan",
                    return_value=torch.tensor([[9.0]]),
                ),
                mock.patch.object(
                    diagnostic_cli,
                    "_initial_physical_distance",
                    return_value=2.0,
                ),
                mock.patch.object(
                    diagnostic_cli, "_predict_costs", predict_costs
                ),
                mock.patch.object(
                    diagnostic_cli,
                    "_predict_terminal_latents",
                    predict_terminal,
                ),
                mock.patch.object(
                    diagnostic_cli, "_terminal_latents", terminal_latents
                ),
            )
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                trace = diagnostic_cli.run_trace(
                    manifest_path=manifest_path,
                    pair_label=pair.label,
                    device="cpu",
                )
                grounded = diagnostic_cli.run_ground(
                    manifest_path=manifest_path,
                    pair_label=pair.label,
                    device="cpu",
                )
                summary = diagnostic_cli.run_summarize(
                    manifest_path=manifest_path,
                    pair_label=pair.label,
                )

            global_summary = json.loads(
                (manifest.output_root / "summary.json").read_text()
            )

        self.assertEqual(trace["status"], "reproducible")
        self.assertEqual(
            trace["epochs"]["e8"]["success_vector"], [False]
        )
        self.assertEqual(grounded["status"], "ok")
        self.assertEqual(grounded["acceptance_errors"], [])
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(global_summary["status"], "ok")



if __name__ == "__main__":
    unittest.main()
