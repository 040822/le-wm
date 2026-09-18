import unittest

from scripts.round4_phase45_4 import (
    DEFAULT_CONFIG,
    _cohort_section,
    _condition_name,
    _conditions,
    _load_config,
)
from source.common.round3_phase1 import CohortManifest
from source.common.round4_eval import validate_round4_protocol_variant
from source.common.round4_phase45_4 import (
    condition_key,
    phase454_condition_specs,
)


class Round4Phase454Tests(unittest.TestCase):
    def test_matrix_has_exactly_96_conditions_and_no_forbidden_axes(self):
        conditions = _conditions()

        self.assertEqual(len(conditions), 96)
        self.assertEqual(
            {task: sum(item["task"] == task for item in conditions) for task in ("cube", "pusht", "reacher", "tworoom")},
            {"cube": 19, "pusht": 19, "reacher": 33, "tworoom": 25},
        )
        self.assertTrue(
            all(
                item["action_flow_integrator"]
                == ("not_applicable" if item["action_flow_steps"] is None else "euler")
                for item in conditions
            )
        )
        self.assertTrue(
            all(
                item["action_flow_steps"] in {None, 1, 2, 5, 10, 16, 32}
                for item in conditions
            )
        )
        self.assertTrue(all("heun" not in _condition_name(item) for item in conditions))
        self.assertEqual(
            {
                item["action_flow_steps"]
                for item in conditions
                if item["action_flow_steps"] is not None
            },
            {1, 2, 5, 10, 16, 32},
        )

    def test_every_step_mode_covers_full_flow_grid(self):
        conditions = _conditions()

        for task in ("cube", "pusht", "reacher", "tworoom"):
            for mode, protocol in (
                ("P0", "not_applicable"),
                ("P2", "legacy"),
                ("P3", "not_applicable"),
            ):
                steps = {
                    item["action_flow_steps"]
                    for item in conditions
                    if item["task"] == task
                    and item["mode"] == mode
                    and item["cem_protocol"] == protocol
                }
                self.assertEqual(
                    steps, {1, 2, 5, 10, 16, 32}, (task, mode, protocol)
                )

    def test_legacy_and_dev_result_keys_are_distinct(self):
        common = {
            "task": "reacher",
            "round4_mode": "P2",
            "protocol_variant": "legacy",
            "round4_planning": {
                "cem_protocol": "cem-clip",
                "action_bound_mode": "candidate_clip",
                "action_flow_steps": 1,
                "action_flow_integrator": "euler",
            },
        }

        legacy_key = condition_key(common)
        common["protocol_variant"] = "round3_revised"
        dev_key = condition_key(common)

        self.assertNotEqual(legacy_key, dev_key)
        self.assertEqual(legacy_key[-1], "legacy")
        self.assertEqual(dev_key[-1], "round3_revised")

    def test_matrix_restricts_extra_cem_clip_to_registered_tasks(self):
        conditions = _conditions()

        clip_tasks = {
            item["task"]
            for item in conditions
            if item["cem_protocol"] == "cem-clip"
        }
        self.assertEqual(clip_tasks, {"reacher", "tworoom"})

        scale_conditions = [
            item for item in conditions if item["cem_protocol"] == "cem-scale"
        ]
        self.assertEqual({item["task"] for item in scale_conditions}, {"reacher"})
        self.assertEqual(
            {
                (item["mode"], item["action_flow_steps"])
                for item in scale_conditions
            },
            {("P1", None), *{("P2", step) for step in (1, 2, 5, 10, 16, 32)}},
        )
        self.assertTrue(
            all(
                item["action_flow_integrator"]
                == ("not_applicable" if item["action_flow_steps"] is None else "euler")
                for item in scale_conditions
            )
        )
        self.assertTrue(
            all(
                item["cem_protocol"] == "cem-clip"
                for item in conditions
                if item["task"] == "tworoom"
                and item["mode"] == "P2"
                and item["cem_protocol"] == "cem-clip"
            )
        )

    def test_protocol_variant_allowlist_keeps_default_strict_and_phase454_explicit(self):
        manifest = CohortManifest.load(
            "outputs/round3/phase1/cohorts/cube/legacy_50.json"
        )

        with self.assertRaisesRegex(ValueError, "round3_revised"):
            validate_round4_protocol_variant(manifest)
        self.assertIs(
            validate_round4_protocol_variant(
                manifest,
                allowed_protocol_variants=("legacy", "round3_revised"),
            ),
            manifest,
        )


    def test_condition_key_accepts_cem_scale_and_rejects_wrong_bound_mode(self):
        payload = {
            "task": "reacher",
            "round4_mode": "P1",
            "protocol_variant": "legacy",
            "round4_planning": {
                "cem_protocol": "cem-scale",
                "action_bound_mode": "candidate_scale",
                "action_flow_steps": None,
                "action_flow_integrator": "not_applicable",
            },
        }

        key = condition_key(payload)
        self.assertEqual(
            key, ("reacher", "P1", "cem-scale", None, "not_applicable", "legacy")
        )

        payload["round4_planning"]["action_bound_mode"] = "candidate_clip"
        with self.assertRaisesRegex(ValueError, "candidate_scale"):
            condition_key(payload)

    def test_cohort_section_resolves_protocol_variant_to_config_key(self):
        config = _load_config(DEFAULT_CONFIG)

        legacy = _cohort_section(config, "legacy")
        dev = _cohort_section(config, "round3_revised")

        self.assertEqual(legacy["protocol_variant"], "legacy")
        self.assertEqual(dev["protocol_variant"], "round3_revised")
        self.assertIn("legacy_50.json", legacy["paths"]["cube"])
        self.assertIn("dev_round3_revised.json", dev["paths"]["cube"])
        with self.assertRaisesRegex(ValueError, "round3_revised"):
            _cohort_section({"cohort": {"legacy": legacy}}, "round3_revised")

    def test_invariant_integrator_is_not_treated_as_heun(self):
        integrators = {
            spec["action_flow_integrator"] for spec in phase454_condition_specs()
        }

        self.assertEqual(integrators, {"euler", "not_applicable"})
        self.assertNotIn("heun", integrators)


if __name__ == "__main__":
    unittest.main()
