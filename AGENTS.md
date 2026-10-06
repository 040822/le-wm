# Repository Agent Instructions

## GPU Usage

- Prefer GPU0–3; GPU4–7 are also allowed.
- Before launching each experiment, check that the selected GPU has enough remaining VRAM and retain a reasonable safety margin for runtime fluctuations and other workloads. If VRAM is sufficient, multiple experiments may run concurrently on the same GPU to fully utilize GPU resources.
- Every GPU command must explicitly restrict visibility with `CUDA_VISIBLE_DEVICES` to the selected device or devices.
- The sandbox may be unable to detect or access GPUs. If GPU inspection or a GPU task fails for that reason, request privilege escalation and rerun it with the same explicit `CUDA_VISIBLE_DEVICES` restriction.

## Goal-mode Progress Checks

- When an active goal is in progress, a batch of training tasks has already been launched, and the agent is monitoring those training tasks, perform one post-launch check to confirm startup, then wait at least 300 seconds (`sleep(300)`) between routine progress checks.
- If two consecutive routine checks show no state change, increase the interval to 600 seconds (`sleep(600)`). For very long-running, stable jobs, the interval may be increased further when timely detection is not important.
- Check sooner only when an error is suspected, tasks are near their expected completion time, or the user requests an update. Consolidate the status of all training tasks and GPUs into one query whenever possible.
- This delay and backoff policy applies only to that goal-mode training-monitoring phase. Non-goal work and non-monitoring phases do not require `sleep`.
