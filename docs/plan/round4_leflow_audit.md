# LeFlow artifact and gradient audit

This is a source/config audit for the four frozen LeFlow payloads used as Round 4 references. It is separate from the R4-ABDE training results.

## Artifact checks

`config/round4/leflow_artifacts.json` passed both `validate` and `validate-leflow`. Each task has a planner checkpoint, dependent LeWM checkpoint, planner config, and SHA256. The revised dev/final cohorts match `config/round4/cohort_artifacts.json`.

The planner configs all declare seed 3072, train split 0.9, batch size 128, horizon 5, action block 5, evaluation samples 64, and evaluation flow steps 16. All four set `episode_split.enabled=false`.

## Training gradient path

The relevant implementation is `train_latent_planner.py`:

1. `encode_latents` runs the dependent LeWM encoder under `torch.no_grad()` and explicitly detaches `z_path`; LeWM is frozen and is not an optimizer parameter.
2. Flow matching receives the detached latent path, so gradients go to `LatentPathFlow` only.
3. Inverse-dynamics MSE sends gradients to `InverseDynamics` only.
4. All four configs set `loss.consistency.weight=0.1` and `detach_inverse=false`. The consistency term uses detached `z_path` and `predicted_actions` from the inverse head; because LeWM is frozen, its update path is limited to the inverse-dynamics parameters.
5. Smoothness is configured with weight 0.0. The optimizer contains only flow and inverse-dynamics parameters.

Thus the payloads do not update the dependent LeWM encoder/projector. The consistency term is active for inverse dynamics, while the latent path itself is detached. This explains why the R4 D/E training (which keeps encoder gradients for D anchors and E paths) is not directly comparable to the historical LeFlow training.

## Evaluation contract

The runtime uses action-normalized IDM outputs, fixes `z_start` and `z_goal` at every Euler step, and scores reranking from the decoded actions' world-model rollout endpoint. The Round 4 LeFlow final traces are in `outputs/round4/leflow_revised_final/<task>/leflow/final/`.
