# Share a Mode-Selectable DiT Across Fast-LeWAM Stages

Fast-LeWAM uses one checkpoint and one set of DiT blocks with mode-specific token layouts and attention masks. Goal-conditioned Stage A uses [z0, zg, a0, ..., aH-1]: action tokens read both clean anchors and each other, while z0 and zg may read each other but cannot read noisy action tokens. The goal is an explicit token and is not duplicated into the AdaLN task condition. Stage B remains goal-free internally and applies causal attention to clean-action and latent-query tokens; the CEM objective compares its terminal prediction with a separately encoded goal. Stage C remains a goal-free causal ablation.

The public stage_a_goal_injection switch defaults to none so legacy checkpoints retain their original parameter structure and strict loading behavior. New Fast-LeWAM training config selects token and is intended for from-scratch training; token and legacy checkpoints are not cross-structure strict-resume compatible.
