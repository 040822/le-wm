# Repository Agent Instructions

## GPU Usage

- Prefer GPU0–3. Before starting a new task, check that the selected GPU has enough remaining VRAM and retain a reasonable safety margin for runtime fluctuations and other workloads.
- GPU4–7 may be used only with the user's explicit consent, and only for low-VRAM evaluation tasks. This is especially appropriate for evaluations with a large total workload when using the extra GPUs improves concurrency. GPU4–7 must not be used for high-VRAM training tasks.
- Between 10:00 and 23:00 Beijing time, avoid using GPU4–7 whenever reasonably possible.
- Every GPU command must explicitly restrict visibility with `CUDA_VISIBLE_DEVICES` to the selected device or devices. Before launching GPU work, verify that the selected IDs comply with the rules above and, when using GPU4–7, that the task is a user-approved low-VRAM evaluation.
- The sandbox may be unable to detect or access GPUs. If GPU inspection or a GPU task fails for that reason, request privilege escalation and rerun it with the same explicit `CUDA_VISIBLE_DEVICES` restriction.

## Goal-mode Progress Checks

- When an active goal is in progress, a batch of training tasks has already been launched, and the agent is monitoring those training tasks, perform one post-launch check to confirm startup, then wait at least 300 seconds (`sleep(300)`) between routine progress checks.
- If two consecutive routine checks show no state change, increase the interval to 600 seconds (`sleep(600)`). For very long-running, stable jobs, the interval may be increased further when timely detection is not important.
- Check sooner only when an error is suspected, tasks are near their expected completion time, or the user requests an update. Consolidate the status of all training tasks and GPUs into one query whenever possible.
- This delay and backoff policy applies only to that goal-mode training-monitoring phase. Non-goal work and non-monitoring phases do not require `sleep`.
