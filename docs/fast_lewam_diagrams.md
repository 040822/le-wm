# Fast-LeWAM 模型结构、训练与推理流程图

本文档按当前 `source/model/fast_lewam/`、`source/policy/fast_lewam*.py` 和
`config/train/fast_lewam.yaml` 绘制。Action Block 是 `frameskip` 个环境动作的拼接；
Action Horizon（H）是一次处理的 Action Block 数量，不是原始环境步数。

## 1. 总体训练结构

```mermaid
flowchart TD
    PIX["pixels<br/>B x (H+1) x 3 x 224 x 224<br/>uint8"] --> NORM["GPU float + ImageNet normalization"]
    NORM --> ENC["共享 LeWM ViT Encoder + Projector"]
    ENC --> Z0["当前 z0<br/>B x D"]
    ENC --> ZTGT["目标 z1:H<br/>B x H x D"]
    ENC --> SIG["SIGReg"]

    ACT["clean actions a0<br/>B x H x A"] --> FLOW["x_t=(1-t)noise+t*a0"]
    NOISE["Gaussian noise"] --> FLOW
    TIME["timestep t"] --> FLOW
    Z0 --> CONDA["Stage A condition<br/>z0 + t + optional task"]
    TIME --> CONDA
    FLOW --> DITA["SharedDiT<br/>双向 action self-attention"]
    CONDA --> DITA
    DITA --> AHEAD["Action Head"]
    AHEAD --> VEL["velocity v_hat"]
    VEL --> CLEAN["a0_hat=x_t+(1-t)*v_hat"]
    FLOW --> CLEAN

    CLEAN --> DETACH{"detach_clean_action?"}
    DETACH -->|false| MIX["课程混合<br/>预测动作或真值动作"]
    DETACH -->|true: stop-gradient| MIX
    ACT --> MIX
    MIX --> BTOK["z0,a1,q1,...,aH,qH"]
    MIX --> BTIME["Stage B timestep<br/>预测样本: t<br/>真值样本: 1"]
    Z0 --> BTOK
    BTOK --> DITB["同一套 SharedDiT 权重<br/>下三角 causal mask"]
    Z0 --> CONDB["Stage B condition"]
    BTIME --> CONDB
    CONDB --> DITB
    DITB --> LHEAD["Latent Head<br/>读取 q1:H"]
    LHEAD --> ZHAT["z_hat1:H<br/>B x H x D"]

    VEL --> LACT["action_loss"]
    ZHAT --> LLAT["latent_prefix_loss"]
    ZTGT --> LLAT
    TIME --> NW["noise_weight<br/>clamp(t/threshold, max=1)"]
    NW --> LLAT
    SIG --> LSIG["sigreg_loss"]
    LACT --> TOTAL["action + lambda_latent*latent<br/>+ lambda_sigreg*sigreg"]
    LLAT --> TOTAL
    LSIG --> TOTAL
```

`FastLeWAM.predictor` 只有一套 `SharedDiT` 权重，Stage A 与 Stage B 以不同 token
布局和 mask 分别调用。latent loss 始终通过 Stage B 调用更新共享 DiT；仅当课程混合
选中预测动作且 `detach_clean_action=false` 时，它才沿 clean estimate 间接回传到
Action Head。真值动作样本没有这条间接路径，其 Stage B timestep 被设为 `1`。

## 2. Stage A：双向 Action DiT

```mermaid
flowchart LR
    Z0["z0 + timestep + task"] --> DIT["SharedDiT<br/>bidirectional attention"]
    X1["x_t,1 + pos1"] --> DIT
    X2["x_t,2 + pos2"] --> DIT
    XD["..."] --> DIT
    XH["x_t,H + posH"] --> DIT
    DIT --> V1["v_hat1"]
    DIT --> V2["v_hat2"]
    DIT --> VH["v_hatH"]
```

所有动作 token 互相可见，一次并行输出整段 velocity；该调用不创建 latent query，
也不执行 `latent_head`。

## 3. Stage B：并行 causal-prefix latent 预测

```mermaid
flowchart TD
    TOK["单次并行 token layout<br/>z0,a1,q1,a2,q2,...,aH,qH"] --> DIT["SharedDiT<br/>下三角 causal mask"]
    DIT --> Q1["q1 -> z_hat1"]
    DIT --> Q2["q2 -> z_hat2"]
    DIT --> QK["..."]
    DIT --> QH["qH -> z_hatH"]
```

| query | 可访问 token | 不可访问 |
| --- | --- | --- |
| `q1` | `z0, a1, q1` | `a2:qH` |
| `q2` | `z0, a1, q1, a2, q2` | `a3:qH` |
| `qk` | `z0, a1, q1, ..., ak, qk` | `a(k+1):qH` |

因此 `qk` 看不到未来动作，并且一次前向并行产生 `z_hat1:H`，没有 autoregressive
latent rollout。当前普通下三角 mask 也允许 `qk` 访问此前的 query token。

## 4. Stage C 对照模式

```mermaid
flowchart LR
    TOK["z0,a1,q1,...,aH,qH"] --> DIT["SharedDiT<br/>causal mask"]
    DIT --> APOS["动作位置"]
    DIT --> QPOS["query 位置"]
    APOS --> AH["Action Head<br/>velocity1:H"]
    QPOS --> LH["Latent Head<br/>z_hat1:H"]
```

Stage C 用一次 causal pass 同时输出动作和 latent，动作位置也受 causal mask 限制，
因此不等价于 Stage A+B。`mode="stage_c"` 推理时，`sample_joint()` 每个 Euler step
都会执行动作头和 latent 头（循环只使用 velocity），终点还会额外执行一次 `t=1`
前向读取 endpoint latent；当前环境 policy 最终只使用返回的 actions。

## 5. 默认 Stage A 直接推理

```mermaid
flowchart TD
    OBS["当前 pixels"] --> PRE["图像预处理<br/>默认 eval: CPU transform<br/>raw uint8: 模型所在设备归一化"]
    PRE --> ENC["Encoder + Projector"]
    ENC --> Z0["z0"]
    RNG["Gaussian action noise"] --> EULER["Euler integration<br/>默认 10 steps"]
    Z0 --> EULER
    EULER --> DITA["Stage A SharedDiT + Action Head"]
    DITA --> EULER
    EULER --> CHUNK["action chunk<br/>B x H x A"]
    CHUNK --> BUF["receding-horizon buffer"]
    BUF --> ENV["inverse transform -> env.step"]
```

默认 eval transform 在 CPU 完成 float、resize 和归一化；直接传 raw uint8 时，
`encode_pixels()` 才在模型设备归一化。标准 checkpoint 仍实例化并加载完整模型，
但默认 Stage A 执行路径不会构造 Stage B token 或执行 latent head。

## 6. Stage B solver 推理

```mermaid
flowchart TD
    INFO["current pixels + goal pixels"] --> ENC["共享 Encoder + Projector"]
    ENC --> Z0["z0"]
    ENC --> GOAL["goal latent"]
    CEM["CEM candidates<br/>B x S x H x A"] --> SB["Stage B causal prediction"]
    Z0 --> SB
    SB --> LAST["z_hatH"]
    LAST --> COST["MSE(z_hatH, goal)<br/>B x S"]
    GOAL --> COST
    COST --> CEM
    CEM --> BEST["低 cost action sequence"]
```

`StageBModelView` 只向 solver 暴露 `get_cost()`。该路径用于显式 `mode=stage_b`，
不属于默认 Stage A 直接动作推理开销。

## 7. 关键 shape 与默认值

| 名称 | shape / 默认值 | 含义 |
| --- | --- | --- |
| `z0` / `z1:H` | `B x 192` / `B x H x 192` | 当前与未来视觉 latent |
| `action_dim` | `frameskip * env_action_dim` | 一个 Action Block 的维度 |
| `action_horizon` | `5` | 一次处理的 Action Block 数量 |
| `action_velocity` | `B x H x action_dim` | Stage A/C flow velocity |
| `predicted_latents` | `B x H x 192` | Stage B/C latent 输出 |
| `latent_head_layers` / `dim` | `6` / `192` | SharedDiT 深度与 token 维度 |
| `detach_clean_action` | `false` | 允许预测动作样本的间接梯度 |
| `latent_loss_noise_threshold` | `0.2` | latent loss 满权重的 t 阈值 |
| `inference_steps` | `10` | Euler 默认积分步数 |

关键实现位置：`source/model/fast_lewam/jepa.py`、`modules.py`、
`source/policy/fast_lewam.py`、`fast_lewam_eval.py` 和
`config/train/fast_lewam.yaml`。
