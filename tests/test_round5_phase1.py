import unittest

from source.common.round5_phase1 import (
    ROUND5_FLOW_STEPS,
    ROUND5_P3_GUIDANCE_MODES,
    ROUND5_TASKS,
    baseline_condition_specs,
    condition_key_from_spec,
    condition_name,
    guidance_condition_specs,
    p2_protocol,
)


class Round5Phase1SpecTests(unittest.TestCase):
    def test_guidance_specs_cover_expected_matrix(self):
        specs = guidance_condition_specs()
        self.assertEqual(len(specs), 4 * 6 * (2 + 2 + 3))
        per_task = {}
        for spec in specs:
            per_task.setdefault(spec["task"], []).append(spec)
        for task in ROUND5_TASKS:
            self.assertEqual(len(per_task[task]), 42)

    def test_baseline_specs_cover_reused_no_guidance_matrix(self):
        specs = baseline_condition_specs()
        self.assertEqual(len(specs), 4 * (6 + 1 + 6 + 6))
        for spec in specs:
            self.assertEqual(spec["guidance"], "none")
            self.assertIn(spec["task"], ROUND5_TASKS)

    def test_p2_protocol_uses_cem_clip_only_for_reacher(self):
        self.assertEqual(p2_protocol("reacher"), "cem-clip")
        for task in ("cube", "pusht", "tworoom"):
            self.assertEqual(p2_protocol(task), "legacy")

    def test_guidance_specs_use_expected_protocols(self):
        specs = guidance_condition_specs()
        for spec in specs:
            if spec["mode"] in {"P0", "P3"}:
                self.assertEqual(spec["cem_protocol"], "not_applicable")
            if spec["mode"] == "P2":
                self.assertEqual(spec["cem_protocol"], p2_protocol(spec["task"]))
            self.assertEqual(spec["action_flow_integrator"], "euler")

    def test_condition_name_is_stable_and_encodes_guidance(self):
        spec = {
            "task": "reacher",
            "mode": "P3",
            "cem_protocol": "not_applicable",
            "action_flow_steps": 2,
            "action_flow_integrator": "euler",
            "guidance": "post_opt_refine",
        }
        self.assertEqual(
            condition_name(spec),
            "P3/not_applicable/post_opt_refine/step_2/euler/dev",
        )
        self.assertEqual(
            condition_key_from_spec(spec),
            ("reacher", "P3", "not_applicable", 2, "euler", "post_opt_refine"),
        )

    def test_flow_steps_and_p3_guidance_contract(self):
        self.assertEqual(ROUND5_FLOW_STEPS, (1, 2, 5, 10, 16, 32))
        self.assertEqual(
            ROUND5_P3_GUIDANCE_MODES, ("guided_flow", "post_opt", "post_opt_refine")
        )


if __name__ == "__main__":
    unittest.main()
