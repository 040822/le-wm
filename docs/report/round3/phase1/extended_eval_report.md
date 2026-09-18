# Round 1/2 extended evaluation report

- Generated: 2026-09-10T07:12:02.059023+08:00
- Manifest entries: 42
- Expected stage tasks: 93
- Result rows found: 93
- Fully valid rows: 93
- Validation issues: 0
- Protocol: `round3_revised`, cohort: `final`, episodes per stage: 200, `save_video=true`.

## Summary by round/task/experiment/stage

| Round | Task | Experiment | Stage | Runs | Mean success | Min | Max | Videos |
|---|---|---:|---|---:|---:|---:|---:|---:|
| round1 | cube | E1 | stage_b | 3 | 0.378 | 0.360 | 0.395 | 600/600 |
| round1 | cube | E2 | stage_a | 1 | 1.000 | 1.000 | 1.000 | 200/200 |
| round1 | cube | E2 | stage_a_shuffled_goal | 1 | 0.110 | 0.110 | 0.110 | 200/200 |
| round1 | cube | E3 | stage_a | 3 | 0.995 | 0.995 | 0.995 | 600/600 |
| round1 | cube | E3 | stage_a_shuffled_goal | 3 | 0.102 | 0.090 | 0.120 | 600/600 |
| round1 | cube | E3 | stage_b | 3 | 0.457 | 0.430 | 0.480 | 600/600 |
| round1 | cube | E4 | stage_a | 1 | 0.995 | 0.995 | 0.995 | 200/200 |
| round1 | cube | E4 | stage_a_shuffled_goal | 1 | 0.100 | 0.100 | 0.100 | 200/200 |
| round1 | cube | E4 | stage_b | 1 | 0.535 | 0.535 | 0.535 | 200/200 |
| round1 | cube | E5 | stage_a | 1 | 0.995 | 0.995 | 0.995 | 200/200 |
| round1 | cube | E5 | stage_a_shuffled_goal | 1 | 0.090 | 0.090 | 0.090 | 200/200 |
| round1 | cube | E5 | stage_b | 1 | 0.495 | 0.495 | 0.495 | 200/200 |
| round1 | cube | E6 | stage_b_actor_warm_start | 1 | 0.975 | 0.975 | 0.975 | 200/200 |
| round1 | pusht | E1 | stage_b | 3 | 0.897 | 0.890 | 0.905 | 600/600 |
| round1 | pusht | E2 | stage_a | 1 | 0.905 | 0.905 | 0.905 | 200/200 |
| round1 | pusht | E2 | stage_a_shuffled_goal | 1 | 0.090 | 0.090 | 0.090 | 200/200 |
| round1 | pusht | E3 | stage_a | 3 | 0.922 | 0.910 | 0.940 | 600/600 |
| round1 | pusht | E3 | stage_a_shuffled_goal | 3 | 0.080 | 0.075 | 0.085 | 600/600 |
| round1 | pusht | E3 | stage_b | 3 | 0.858 | 0.845 | 0.870 | 600/600 |
| round1 | pusht | E4 | stage_a | 3 | 0.903 | 0.890 | 0.915 | 600/600 |
| round1 | pusht | E4 | stage_a_shuffled_goal | 3 | 0.095 | 0.085 | 0.110 | 600/600 |
| round1 | pusht | E4 | stage_b | 3 | 0.862 | 0.855 | 0.865 | 600/600 |
| round1 | pusht | E5 | stage_a | 3 | 0.925 | 0.920 | 0.930 | 600/600 |
| round1 | pusht | E5 | stage_a_shuffled_goal | 3 | 0.103 | 0.070 | 0.125 | 600/600 |
| round1 | pusht | E5 | stage_b | 3 | 0.877 | 0.870 | 0.890 | 600/600 |
| round1 | pusht | E6 | stage_b_actor_warm_start | 1 | 0.930 | 0.930 | 0.930 | 200/200 |
| round1 | reacher | E1 | stage_b | 1 | 0.840 | 0.840 | 0.840 | 200/200 |
| round1 | reacher | E2 | stage_a | 1 | 0.695 | 0.695 | 0.695 | 200/200 |
| round1 | reacher | E2 | stage_a_shuffled_goal | 1 | 0.090 | 0.090 | 0.090 | 200/200 |
| round1 | reacher | E3 | stage_a | 1 | 0.705 | 0.705 | 0.705 | 200/200 |
| round1 | reacher | E3 | stage_a_shuffled_goal | 1 | 0.065 | 0.065 | 0.065 | 200/200 |
| round1 | reacher | E3 | stage_b | 1 | 0.845 | 0.845 | 0.845 | 200/200 |
| round1 | reacher | E4 | stage_a | 1 | 0.730 | 0.730 | 0.730 | 200/200 |
| round1 | reacher | E4 | stage_a_shuffled_goal | 1 | 0.075 | 0.075 | 0.075 | 200/200 |
| round1 | reacher | E4 | stage_b | 1 | 0.845 | 0.845 | 0.845 | 200/200 |
| round1 | reacher | E5 | stage_a | 1 | 0.795 | 0.795 | 0.795 | 200/200 |
| round1 | reacher | E5 | stage_a_shuffled_goal | 1 | 0.090 | 0.090 | 0.090 | 200/200 |
| round1 | reacher | E5 | stage_b | 1 | 0.845 | 0.845 | 0.845 | 200/200 |
| round1 | reacher | E6 | stage_b_actor_warm_start | 1 | 0.685 | 0.685 | 0.685 | 200/200 |
| round1 | tworoom | E5 | stage_a | 1 | 0.950 | 0.950 | 0.950 | 200/200 |
| round1 | tworoom | E5 | stage_a_shuffled_goal | 1 | 0.440 | 0.440 | 0.440 | 200/200 |
| round1 | tworoom | E5 | stage_b | 1 | 0.955 | 0.955 | 0.955 | 200/200 |
| round1 | tworoom | E6 | stage_b_actor_warm_start | 1 | 0.965 | 0.965 | 0.965 | 200/200 |
| round2 | pusht | E1 | stage_b | 1 | 0.855 | 0.855 | 0.855 | 200/200 |
| round2 | pusht | E3 | stage_a | 1 | 0.910 | 0.910 | 0.910 | 200/200 |
| round2 | pusht | E3 | stage_a_shuffled_goal | 1 | 0.085 | 0.085 | 0.085 | 200/200 |
| round2 | pusht | E3 | stage_b | 1 | 0.885 | 0.885 | 0.885 | 200/200 |
| round2 | reacher | E1 | stage_b | 1 | 0.810 | 0.810 | 0.810 | 200/200 |
| round2 | reacher | E3 | stage_a | 1 | 0.735 | 0.735 | 0.735 | 200/200 |
| round2 | reacher | E3 | stage_a_shuffled_goal | 1 | 0.075 | 0.075 | 0.075 | 200/200 |
| round2 | reacher | E3 | stage_b | 1 | 0.820 | 0.820 | 0.820 | 200/200 |
| round2_followup | cube | E4 | stage_a | 1 | 0.995 | 0.995 | 0.995 | 200/200 |
| round2_followup | cube | E4 | stage_a_shuffled_goal | 1 | 0.095 | 0.095 | 0.095 | 200/200 |
| round2_followup | cube | E4 | stage_b | 1 | 0.500 | 0.500 | 0.500 | 200/200 |
| round2_followup | pusht | E1 | stage_b | 1 | 0.905 | 0.905 | 0.905 | 200/200 |
| round2_followup | pusht | E3 | stage_a | 1 | 0.935 | 0.935 | 0.935 | 200/200 |
| round2_followup | pusht | E3 | stage_a_shuffled_goal | 1 | 0.090 | 0.090 | 0.090 | 200/200 |
| round2_followup | pusht | E3 | stage_b | 1 | 0.895 | 0.895 | 0.895 | 200/200 |
| round2_followup | reacher | E1 | stage_b | 1 | 0.685 | 0.685 | 0.685 | 200/200 |
| round2_followup | reacher | E3 | stage_a | 1 | 0.730 | 0.730 | 0.730 | 200/200 |
| round2_followup | reacher | E3 | stage_a_shuffled_goal | 1 | 0.080 | 0.080 | 0.080 | 200/200 |
| round2_followup | reacher | E3 | stage_b | 1 | 0.835 | 0.835 | 0.835 | 200/200 |
| round2_followup | reacher | E4 | stage_a | 1 | 0.750 | 0.750 | 0.750 | 200/200 |
| round2_followup | reacher | E4 | stage_a_shuffled_goal | 1 | 0.095 | 0.095 | 0.095 | 200/200 |
| round2_followup | reacher | E4 | stage_b | 1 | 0.830 | 0.830 | 0.830 | 200/200 |

## Summary by task and experiment

| Task | Experiment | Runs | Mean success | Min | Max |
|---|---:|---:|---:|---:|---:|
| cube | E1 | 3 | 0.378 | 0.360 | 0.395 |
| cube | E2 | 2 | 0.555 | 0.110 | 1.000 |
| cube | E3 | 9 | 0.518 | 0.090 | 0.995 |
| cube | E4 | 6 | 0.537 | 0.095 | 0.995 |
| cube | E5 | 3 | 0.527 | 0.090 | 0.995 |
| cube | E6 | 1 | 0.975 | 0.975 | 0.975 |
| pusht | E1 | 5 | 0.890 | 0.855 | 0.905 |
| pusht | E2 | 2 | 0.497 | 0.090 | 0.905 |
| pusht | E3 | 15 | 0.625 | 0.075 | 0.940 |
| pusht | E4 | 9 | 0.620 | 0.085 | 0.915 |
| pusht | E5 | 9 | 0.635 | 0.070 | 0.930 |
| pusht | E6 | 1 | 0.930 | 0.930 | 0.930 |
| reacher | E1 | 3 | 0.778 | 0.685 | 0.840 |
| reacher | E2 | 2 | 0.392 | 0.090 | 0.695 |
| reacher | E3 | 9 | 0.543 | 0.065 | 0.845 |
| reacher | E4 | 6 | 0.554 | 0.075 | 0.845 |
| reacher | E5 | 3 | 0.577 | 0.090 | 0.845 |
| reacher | E6 | 1 | 0.685 | 0.685 | 0.685 |
| tworoom | E5 | 3 | 0.782 | 0.440 | 0.955 |
| tworoom | E6 | 1 | 0.965 | 0.965 | 0.965 |

## Issues

- None.
