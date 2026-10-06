# 官方 Sub-JEPA 权重使用与验证

本仓库接入 [intcomp/sub-jepa](https://huggingface.co/intcomp/sub-jepa) 发布的 TwoRoom、PushT、Reacher、Cube 模型。Hugging Face revision 固定为 `03466983bacc465e56464f800148275841b27b95`，不是浮动的 main。下载清单和原始 SHA256 在 `config/baselines/subjepa_official.json`。

Sub-JEPA 的正则化只参与训练，推理时仍是 LeWM 形状的 JEPA 和 CEM；本次仅接入官方训练权重，没有新增 Sub-JEPA 训练流程，没有接入 DeWM，也没有修改 CoWM。

## 1. 资产位置

```text
data/checkpoints/subjepa_official/
├── raw_source/                 # 固定版本的 upstream jepa.py、module.py、MIT LICENSE
├── tworoom/
│   ├── raw/config.yml
│   ├── raw/tworoom_subjepa_object.ckpt
│   ├── subjepa.pt              # 本仓库可严格加载的参数格式
│   ├── model_config.json
│   └── manifest.json           # 原始/转换哈希、来源、配套训练配置、数值检查
├── pusht/                     # 同样布局
├── reacher/
└── cube/
```

大文件位于已被 git 忽略的 `data/`。代码、固定下载清单和说明可以版本管理；搬迁项目时需要携带资产目录，或者运行下述准备命令重新下载。资产约576 MB，不含临时转换依赖。

官方配套配置报告的训练 seed=3072、10 epochs、latent=192；PushT K=16，其余 K=32，均为冻结正交投影。epoch 是发布配置的声明，转换过程没有独立重放训练来验证它。

## 2. 评测

准备好的四任务权重可以直接用：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_subjepa_official.py --task tworoom --seed 42
CUDA_VISIBLE_DEVICES=0 python scripts/eval_subjepa_official.py --task pusht --seed 42
CUDA_VISIBLE_DEVICES=0 python scripts/eval_subjepa_official.py --task reacher --seed 42
CUDA_VISIBLE_DEVICES=0 python scripts/eval_subjepa_official.py --task cube --seed 42
```

启动前检查所选 GPU0–3 有足够显存。脚本要求显式选择一张 GPU0–3，并绑定 MuJoCo EGL 到同一卡。支持 `--device cpu`，但 MuJoCo 环境仍需要可用的渲染后端。

默认值直接复用本仓库各任务 YAML：50 episodes、goal offset/budget=25/50、horizon/receding horizon/action block=5/5/5、CEM 300 samples×30 iterations、top-k=30。数据默认在本仓库 `data/datasets`；可用 `--cache-dir` 指定另一个 SWM 数据根目录。

每次运行写入 `outputs/subjepa_official/<task>/seed_<seed>/<UTC时间戳>/`，包括逐 episode 结果、resolved config、权重 SHA256 和官方来源。可用 `--output-dir` 指定结果目录；已有目录会被拒绝，以保留旧结果。

降低预算的接入检查示例：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_subjepa_official.py \
  --task tworoom --seed 42 \
  --override eval.num_eval=1 \
  --override solver.num_samples=8 \
  --override solver.topk=2 \
  --override solver.n_steps=2 \
  --override output.save_video=false
```

这类 smoke 仅验证接口，不是正式 benchmark。完整六个评测 seeds 为 `42,100,2026,3407,1234,4444`；本次没有派发24个完整评测单元。

原有通用入口也能加载转换后的模型，并将结果身份写为 `subjepa_official`：

```bash
STABLEWM_HOME="$PWD/data" CUDA_VISIBLE_DEVICES=0 python eval.py \
  --config-name=tworoom \
  policy="$PWD/data/checkpoints/subjepa_official/tworoom/subjepa.pt"
```

通用入口沿用既有输出目录行为；多 seed/多次尝试优先使用上面的专用脚本，以保持结果互不覆盖。

## 3. 重新准备权重

官方文件是旧版 Transformers 的 object pickle；正常运行仅加载新的 `subjepa.pt`，不需要安装旧依赖。

重新转换时先把兼容依赖安装到独立目录，不更改当前 `.venv`：

```bash
uv pip install --no-deps --target /tmp/subjepa-legacy-site \
  transformers==4.57.1 huggingface-hub==0.36.0

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python scripts/prepare_subjepa_official.py \
  --legacy-site-packages /tmp/subjepa-legacy-site
```

工具校验官方 LFS SHA256、配置的 Git blob hash，以及 upstream LeWM `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` 源码哈希。仅在独立子进程中反序列化这个已校验的官方对象；子进程使用固定的原始 `jepa.py`、`module.py` 和旧 Transformers。随后在当前环境重建模型、迁移旧 ViT key、严格加载全部参数，并核对完整推理行为。

可以用 `--tasks tworoom` 等选择部分任务，`--output-root` 指定新资产目录。已有原始文件哈希不符时拒绝覆盖；已有转换权重的来源/结构/参数不符时也拒绝覆盖。

已有全部原始文件后，可在 CPU 上离线重新验证转换及参数：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python scripts/prepare_subjepa_official.py \
  --legacy-site-packages /tmp/subjepa-legacy-site --offline
```

推理保留官方 eager ViT attention、原始 predictor、BatchNorm projection heads、3帧有效预测历史和动作 frameskip。不会把已有 LeWM/CoWM 的 latent、normalizer 或动作历史直接代入官方模型。

## 4. 验证记录

2026-10-03，四任务均通过旧 Transformers 4.57.1 与当前5.12.1之间的 CPU 数值检查：图像 latent、动作 embedding、下一 latent、5步 rollout 和候选 cost。容差为 `atol=rtol=2e-5`，固定候选的 argmin 全部一致。

| 任务 | 编码最大绝对误差 | 下一latent最大绝对误差 | rollout最大绝对误差 | cost最大绝对误差 |
|---|---:|---:|---:|---:|
| TwoRoom | 2.74e-6 | 1.30e-6 | 7.63e-6 | 7.93e-4 |
| PushT | 2.59e-6 | 2.38e-6 | 2.31e-6 | 5.34e-5 |
| Reacher | 9.54e-7 | 7.15e-7 | 1.37e-6 | 6.10e-5 |
| Cube | 8.34e-7 | 7.15e-7 | 8.34e-7 | 3.05e-5 |

cost 是192维误差平方和，数值量级与 latent 不同；其检查使用上述绝对与相对联合容差，不声称所有 cost 的绝对误差均小于2e-5。全部 action embedding 的最大误差为0。

本地测试共38项：37项通过，1项因本机没有旧 LeWM checkpoint 而跳过。覆盖直接/cache加载、严格参数检查、方法身份、有效history、候选分片一致性、转换检查对错误预测的拒绝，以及原始资产哈希保护；同时运行已有 checkpoint、评测协议、LeFlow、Fast-LeWAM 和 value-JEPA 的相关回归测试。

四任务均已在 GPU0 通过真实环境 smoke，每任务 seed=42、1 episode、CEM 8 samples×2 iterations、top-k=2。其余 goal offset/budget=25/50、horizon/receding/action block=5/5/5 保持统一协议。验收检查是正常完成、模型身份正确和结果来源完整；单回合成功/失败不作为正式成功率结论。详细记录见 `outputs/subjepa_official_integration/smoke/<task>/attempt1/`，汇总见 [接入验证记录](../../outputs/subjepa_official_integration/integration_summary.json)。

CoWM 可借鉴设计的展开分析见 [局部表示、分量评分和正则化](../report/cvpr/cowm_borrowing_from_dewm_20261003.md)。这些设计尚未实施。
