# LeFlow 集成

本仓库现在包含官方 LeFlow planner 的模型、训练和评测适配层：

- `source/model/leflow/latent_planner.py`：`LatentPathFlow`、`InverseDynamics`、冻结 LeWM 验证 rollout，以及 `LearnedLatentPathSolver`。
- `source/policy/leflow.py`：将 solver 接到现有 `stable_worldmodel.WorldModelPolicy`。
- `config/eval/solver/latent_flow.yaml`：官方默认的 `64` 条 path、`16` 个 flow steps 和 `rollout_goal` reranking。
- `train_latent_planner.py` / `config/train/latent_planner.yaml`：官方 planner 训练入口和配置。

LeWM reference 使用本仓库已有的 checkpoint remapper；官方 payload 中的
`tworoom/lewm`、`pusht/lewm`、`reacher/lewm`、`cube/lewm` 会映射到本地的
`quentinll/lewm-*` 权重。

## 评测命令

官方 planner 权重应放在：

```text
data/checkpoints/leflow/{tworoom,pusht,reacher,cube}/latent_planner.pt
```

旧协议（50 episodes）：

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --config-name=pusht.yaml \
  policy=leflow/pusht/latent_planner.pt \
  plan_config.horizon=5 \
  plan_config.receding_horizon=5 \
  plan_config.action_block=5 \
  eval.num_eval=50
```

Round 3 新协议的 legacy cohort：

```bash
python -m scripts.round3_phase1 legacy-cohort pusht --num-eval 50

CUDA_VISIBLE_DEVICES=0 python -m scripts.round3_phase1 evaluate \
  pusht leflow stage_b \
  --protocol-variant legacy \
  --cohort outputs/round3/phase1/cohorts/pusht/legacy_50.json \
  --checkpoint data/checkpoints/leflow/pusht/latent_planner.pt \
  --device cuda
```

Round 3 revised final cohort（200 episodes）：

```bash
python -m scripts.round3_phase1 cohort pusht --dev 50 --final 200

CUDA_VISIBLE_DEVICES=0 python -m scripts.round3_phase1 evaluate \
  pusht leflow stage_b \
  --protocol-variant round3_revised \
  --cohort outputs/round3/phase1/cohorts/pusht/final.json \
  --checkpoint data/checkpoints/leflow/pusht/latent_planner.pt \
  --device cuda
```

四个任务只需把 `pusht` 替换为 `cube`、`reacher` 或 `tworoom`，并使用对应
planner 权重。

## 当前权重状态

LeFlow README 指向的 Hugging Face 地址
`hsiangwei0903/LeFlow` 当前返回 404；官方 GitHub issue #1 也记录了这个 broken
link。因此本次迁移已完成并通过 CPU/runtime 测试，但在权重恢复前不会伪造旧协议
或新协议的 success rate。拿到四个 `latent_planner.pt` 后，上面的命令即可直接运行。
