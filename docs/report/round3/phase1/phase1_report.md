# Round 3 Phase 1 report

This report is descriptive. Stage B is the primary metric; Stage A and A-shuffled are auxiliary.
Final testing uses only the frozen `round3_revised` cohort and is not used to tune predicates or sampling.

| Task | Method | Stage | Protocol | n | Success | Wilson 95% | Status |
|---|---|---|---|---:|---:|---|---|
| cube | e0_lewm | stage_b | legacy | 50 | 78.0% | 64.8%–87.2% | ok |
| cube | e0_lewm | stage_b | sampling_revised | None | — | — | not_run |
| cube | e0_lewm | stage_b | tolerance_revised | None | — | — | not_run |
| cube | e0_lewm | stage_b | round3_revised | 50 | 54.0% | 40.4%–67.0% | ok |
| cube | e3_fast | stage_a | legacy | 50 | 100.0% | 92.9%–100.0% | ok |
| cube | e3_fast | stage_a | sampling_revised | None | — | — | not_run |
| cube | e3_fast | stage_a | tolerance_revised | None | — | — | not_run |
| cube | e3_fast | stage_a | round3_revised | 50 | 100.0% | 92.9%–100.0% | ok |
| cube | e3_fast | stage_a_shuffled_goal | legacy | 50 | 42.0% | 29.4%–55.8% | ok |
| cube | e3_fast | stage_a_shuffled_goal | sampling_revised | None | — | — | not_run |
| cube | e3_fast | stage_a_shuffled_goal | tolerance_revised | None | — | — | not_run |
| cube | e3_fast | stage_a_shuffled_goal | round3_revised | 50 | 10.0% | 4.3%–21.4% | ok |
| cube | e3_fast | stage_b | legacy | 50 | 72.0% | 58.3%–82.5% | ok |
| cube | e3_fast | stage_b | sampling_revised | None | — | — | not_run |
| cube | e3_fast | stage_b | tolerance_revised | None | — | — | not_run |
| cube | e3_fast | stage_b | round3_revised | 50 | 42.0% | 29.4%–55.8% | ok |
| cube | e5_fast | stage_a | legacy | 50 | 100.0% | 92.9%–100.0% | ok |
| cube | e5_fast | stage_a | sampling_revised | None | — | — | not_run |
| cube | e5_fast | stage_a | tolerance_revised | None | — | — | not_run |
| cube | e5_fast | stage_a | round3_revised | 50 | 100.0% | 92.9%–100.0% | ok |
| cube | e5_fast | stage_a_shuffled_goal | legacy | 50 | 42.0% | 29.4%–55.8% | ok |
| cube | e5_fast | stage_a_shuffled_goal | sampling_revised | None | — | — | not_run |
| cube | e5_fast | stage_a_shuffled_goal | tolerance_revised | None | — | — | not_run |
| cube | e5_fast | stage_a_shuffled_goal | round3_revised | 50 | 8.0% | 3.2%–18.8% | ok |
| cube | e5_fast | stage_b | legacy | 50 | 74.0% | 60.4%–84.1% | ok |
| cube | e5_fast | stage_b | sampling_revised | None | — | — | not_run |
| cube | e5_fast | stage_b | tolerance_revised | None | — | — | not_run |
| cube | e5_fast | stage_b | round3_revised | 50 | 48.0% | 34.8%–61.5% | ok |
| pusht | e0_lewm | stage_b | legacy | 50 | 98.0% | 89.5%–99.6% | ok |
| pusht | e0_lewm | stage_b | sampling_revised | None | — | — | not_run |
| pusht | e0_lewm | stage_b | tolerance_revised | None | — | — | not_run |
| pusht | e0_lewm | stage_b | round3_revised | 50 | 96.0% | 86.5%–98.9% | ok |
| pusht | e3_fast | stage_a | legacy | 50 | 96.0% | 86.5%–98.9% | ok |
| pusht | e3_fast | stage_a | sampling_revised | None | — | — | not_run |
| pusht | e3_fast | stage_a | tolerance_revised | None | — | — | not_run |
| pusht | e3_fast | stage_a | round3_revised | 50 | 98.0% | 89.5%–99.6% | ok |
| pusht | e3_fast | stage_a_shuffled_goal | legacy | 50 | 4.0% | 1.1%–13.5% | ok |
| pusht | e3_fast | stage_a_shuffled_goal | sampling_revised | None | — | — | not_run |
| pusht | e3_fast | stage_a_shuffled_goal | tolerance_revised | None | — | — | not_run |
| pusht | e3_fast | stage_a_shuffled_goal | round3_revised | 50 | 16.0% | 8.3%–28.5% | ok |
| pusht | e3_fast | stage_b | legacy | 50 | 92.0% | 81.2%–96.8% | ok |
| pusht | e3_fast | stage_b | sampling_revised | None | — | — | not_run |
| pusht | e3_fast | stage_b | tolerance_revised | None | — | — | not_run |
| pusht | e3_fast | stage_b | round3_revised | 50 | 86.0% | 73.8%–93.0% | ok |
| pusht | e5_fast | stage_a | legacy | 50 | 98.0% | 89.5%–99.6% | ok |
| pusht | e5_fast | stage_a | sampling_revised | None | — | — | not_run |
| pusht | e5_fast | stage_a | tolerance_revised | None | — | — | not_run |
| pusht | e5_fast | stage_a | round3_revised | 50 | 94.0% | 83.8%–97.9% | ok |
| pusht | e5_fast | stage_a_shuffled_goal | legacy | 50 | 10.0% | 4.3%–21.4% | ok |
| pusht | e5_fast | stage_a_shuffled_goal | sampling_revised | None | — | — | not_run |
| pusht | e5_fast | stage_a_shuffled_goal | tolerance_revised | None | — | — | not_run |
| pusht | e5_fast | stage_a_shuffled_goal | round3_revised | 50 | 12.0% | 5.6%–23.8% | ok |
| pusht | e5_fast | stage_b | legacy | 50 | 86.0% | 73.8%–93.0% | ok |
| pusht | e5_fast | stage_b | sampling_revised | None | — | — | not_run |
| pusht | e5_fast | stage_b | tolerance_revised | None | — | — | not_run |
| pusht | e5_fast | stage_b | round3_revised | 50 | 94.0% | 83.8%–97.9% | ok |
| reacher | e0_lewm | stage_b | legacy | 50 | 72.0% | 58.3%–82.5% | ok |
| reacher | e0_lewm | stage_b | sampling_revised | None | — | — | not_run |
| reacher | e0_lewm | stage_b | tolerance_revised | None | — | — | not_run |
| reacher | e0_lewm | stage_b | round3_revised | 50 | 82.0% | 69.2%–90.2% | ok |
| reacher | e3_fast | stage_a | legacy | 50 | 76.0% | 62.6%–85.7% | ok |
| reacher | e3_fast | stage_a | sampling_revised | None | — | — | not_run |
| reacher | e3_fast | stage_a | tolerance_revised | None | — | — | not_run |
| reacher | e3_fast | stage_a | round3_revised | 50 | 68.0% | 54.2%–79.2% | ok |
| reacher | e3_fast | stage_a_shuffled_goal | legacy | 50 | 4.0% | 1.1%–13.5% | ok |
| reacher | e3_fast | stage_a_shuffled_goal | sampling_revised | None | — | — | not_run |
| reacher | e3_fast | stage_a_shuffled_goal | tolerance_revised | None | — | — | not_run |
| reacher | e3_fast | stage_a_shuffled_goal | round3_revised | 50 | 4.0% | 1.1%–13.5% | ok |
| reacher | e3_fast | stage_b | legacy | 50 | 78.0% | 64.8%–87.2% | ok |
| reacher | e3_fast | stage_b | sampling_revised | None | — | — | not_run |
| reacher | e3_fast | stage_b | tolerance_revised | None | — | — | not_run |
| reacher | e3_fast | stage_b | round3_revised | 50 | 78.0% | 64.8%–87.2% | ok |
| reacher | e5_fast | stage_a | legacy | 50 | 76.0% | 62.6%–85.7% | ok |
| reacher | e5_fast | stage_a | sampling_revised | None | — | — | not_run |
| reacher | e5_fast | stage_a | tolerance_revised | None | — | — | not_run |
| reacher | e5_fast | stage_a | round3_revised | 50 | 78.0% | 64.8%–87.2% | ok |
| reacher | e5_fast | stage_a_shuffled_goal | legacy | 50 | 10.0% | 4.3%–21.4% | ok |
| reacher | e5_fast | stage_a_shuffled_goal | sampling_revised | None | — | — | not_run |
| reacher | e5_fast | stage_a_shuffled_goal | tolerance_revised | None | — | — | not_run |
| reacher | e5_fast | stage_a_shuffled_goal | round3_revised | 50 | 6.0% | 2.1%–16.2% | ok |
| reacher | e5_fast | stage_b | legacy | 50 | 94.0% | 83.8%–97.9% | ok |
| reacher | e5_fast | stage_b | sampling_revised | None | — | — | not_run |
| reacher | e5_fast | stage_b | tolerance_revised | None | — | — | not_run |
| reacher | e5_fast | stage_b | round3_revised | 50 | 78.0% | 64.8%–87.2% | ok |
| tworoom | e0_lewm | stage_b | legacy | 50 | 88.0% | 76.2%–94.4% | ok |
| tworoom | e0_lewm | stage_b | sampling_revised | None | — | — | not_run |
| tworoom | e0_lewm | stage_b | tolerance_revised | None | — | — | not_run |
| tworoom | e0_lewm | stage_b | round3_revised | 50 | 90.0% | 78.6%–95.7% | ok |
| tworoom | e3_fast | stage_a | legacy | None | — | — | missing_weight |
| tworoom | e3_fast | stage_a | sampling_revised | None | — | — | missing_weight |
| tworoom | e3_fast | stage_a | tolerance_revised | None | — | — | missing_weight |
| tworoom | e3_fast | stage_a | round3_revised | None | — | — | missing_weight |
| tworoom | e3_fast | stage_a_shuffled_goal | legacy | None | — | — | missing_weight |
| tworoom | e3_fast | stage_a_shuffled_goal | sampling_revised | None | — | — | missing_weight |
| tworoom | e3_fast | stage_a_shuffled_goal | tolerance_revised | None | — | — | missing_weight |
| tworoom | e3_fast | stage_a_shuffled_goal | round3_revised | None | — | — | missing_weight |
| tworoom | e3_fast | stage_b | legacy | None | — | — | missing_weight |
| tworoom | e3_fast | stage_b | sampling_revised | None | — | — | missing_weight |
| tworoom | e3_fast | stage_b | tolerance_revised | None | — | — | missing_weight |
| tworoom | e3_fast | stage_b | round3_revised | None | — | — | missing_weight |
| tworoom | e5_fast | stage_a | legacy | 50 | 96.0% | 86.5%–98.9% | ok |
| tworoom | e5_fast | stage_a | sampling_revised | None | — | — | not_run |
| tworoom | e5_fast | stage_a | tolerance_revised | None | — | — | not_run |
| tworoom | e5_fast | stage_a | round3_revised | 50 | 96.0% | 86.5%–98.9% | ok |
| tworoom | e5_fast | stage_a_shuffled_goal | legacy | 50 | 38.0% | 25.9%–51.8% | ok |
| tworoom | e5_fast | stage_a_shuffled_goal | sampling_revised | None | — | — | not_run |
| tworoom | e5_fast | stage_a_shuffled_goal | tolerance_revised | None | — | — | not_run |
| tworoom | e5_fast | stage_a_shuffled_goal | round3_revised | 50 | 50.0% | 36.6%–63.4% | ok |
| tworoom | e5_fast | stage_b | legacy | 50 | 98.0% | 89.5%–99.6% | ok |
| tworoom | e5_fast | stage_b | sampling_revised | None | — | — | not_run |
| tworoom | e5_fast | stage_b | tolerance_revised | None | — | — | not_run |
| tworoom | e5_fast | stage_b | round3_revised | 50 | 96.0% | 86.5%–98.9% | ok |

## Frozen final test (round3_revised)

The final cohort is independent of development cohorts and is reported after protocol freeze.

| Task | Method | Stage | n | Success | Wilson 95% | Status |
|---|---|---|---:|---:|---|---|
| cube | e0_lewm | stage_b | 200 | 48.0% | 41.2%–54.9% | ok |
| cube | e3_fast | stage_a | 200 | 99.5% | 97.2%–99.9% | ok |
| cube | e3_fast | stage_a_shuffled_goal | 200 | 12.0% | 8.2%–17.2% | ok |
| cube | e3_fast | stage_b | 200 | 46.0% | 39.2%–52.9% | ok |
| cube | e5_fast | stage_a | 200 | 99.5% | 97.2%–99.9% | ok |
| cube | e5_fast | stage_a_shuffled_goal | 200 | 9.0% | 5.8%–13.8% | ok |
| cube | e5_fast | stage_b | 200 | 49.5% | 42.6%–56.4% | ok |
| pusht | e0_lewm | stage_b | 200 | 93.0% | 88.6%–95.8% | ok |
| pusht | e3_fast | stage_a | 200 | 94.0% | 89.8%–96.5% | ok |
| pusht | e3_fast | stage_a_shuffled_goal | 200 | 7.5% | 4.6%–12.0% | ok |
| pusht | e3_fast | stage_b | 200 | 84.5% | 78.8%–88.9% | ok |
| pusht | e5_fast | stage_a | 200 | 93.5% | 89.2%–96.2% | ok |
| pusht | e5_fast | stage_a_shuffled_goal | 200 | 7.0% | 4.2%–11.4% | ok |
| pusht | e5_fast | stage_b | 200 | 89.5% | 84.5%–93.0% | ok |
| reacher | e0_lewm | stage_b | 200 | 84.5% | 78.8%–88.9% | ok |
| reacher | e3_fast | stage_a | 200 | 70.5% | 63.8%–76.4% | ok |
| reacher | e3_fast | stage_a_shuffled_goal | 200 | 6.5% | 3.8%–10.8% | ok |
| reacher | e3_fast | stage_b | 200 | 84.5% | 78.8%–88.9% | ok |
| reacher | e5_fast | stage_a | 200 | 79.5% | 73.4%–84.5% | ok |
| reacher | e5_fast | stage_a_shuffled_goal | 200 | 9.0% | 5.8%–13.8% | ok |
| reacher | e5_fast | stage_b | 200 | 84.5% | 78.8%–88.9% | ok |
| tworoom | e0_lewm | stage_b | 200 | 84.5% | 78.8%–88.9% | ok |
| tworoom | e3_fast | stage_a | None | — | — | missing_weight |
| tworoom | e3_fast | stage_a_shuffled_goal | None | — | — | missing_weight |
| tworoom | e3_fast | stage_b | None | — | — | missing_weight |
| tworoom | e5_fast | stage_a | 200 | 95.0% | 91.0%–97.3% | ok |
| tworoom | e5_fast | stage_a_shuffled_goal | 200 | 44.0% | 37.3%–50.9% | ok |
| tworoom | e5_fast | stage_b | 200 | 95.5% | 91.7%–97.6% | ok |

Final matrix entries: 28; valid results: 25; unavailable weights: 3.

Registered artifacts: 28; missing: 3; invalid: 0.
