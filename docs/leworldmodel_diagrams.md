# LeWorldModel 模型、训练和评测流程图

本文档根据当前仓库中的 `source/model/lewm`、`source/policy/lewm.py`、`train.py`、`eval.py` 和 `config/` 配置绘制。图中的 LeWM 指 `source.model.lewm.jepa.JEPA`，训练入口为 `source.policy.lewm.LeWMPolicy`。它在评测阶段不是直接输出 action，而是为候选动作序列输出 cost，再由 planner 选择 action。

## 1. LeWorldModel 模型结构

```mermaid
flowchart TD
    P["像素序列 pixels<br/>B x T x C x H x W"] --> R["展平时间维<br/>(B*T) x C x H x W"]
    R --> E["ViT encoder<br/>stable_pretraining.backbone.utils.vit_hf"]
    E --> CLS["取 CLS token"]
    CLS --> PROJ["projector MLP"]
    PROJ --> Z["状态嵌入 emb<br/>B x T x D"]

    A["动作序列 action<br/>B x T x action_dim"] --> AE["action_encoder Embedder<br/>Conv1d + MLP"]
    AE --> U["动作嵌入 act_emb<br/>B x T x D"]

    Z --> PRED["ARPredictor<br/>位置编码 + 条件 Transformer"]
    U --> PRED
    PRED --> PP["pred_proj MLP"]
    PP --> ZHAT["预测状态嵌入 pred_emb<br/>B x T x D"]

    Z --> SIG["SIGReg 正则<br/>约束嵌入近似高斯"]
    ZHAT --> L1["pred_loss<br/>MSE pred_emb vs target emb"]
    SIG --> L2["sigreg_loss"]
    L1 --> LOSS["训练总损失<br/>pred_loss + lambda * sigreg_loss"]
    L2 --> LOSS
```

关键代码位置：

- `source.model.lewm.jepa.JEPA.encode()`：像素编码为 `emb`，动作编码为 `act_emb`。
- `source.model.lewm.jepa.JEPA.predict()`：用 `ARPredictor` 预测下一步 embedding。
- `ARPredictor`：定义在 `source/model/lewm/modules.py`，是带位置编码的条件 Transformer，输入状态 embedding，条件是动作 embedding。
- `source.policy.lewm.LeWMPolicy`：训练入口，复用 `stable_pretraining.spt.Module` 并封装 loss/optimizer/world-policy 构造。
- 默认配置：`wm.history_size=3`，`wm.num_preds=1`，`wm.embed_dim=192`，`loss.sigreg.weight=0.09`。

## 2. 训练流程

```mermaid
flowchart TD
    CFG["Hydra 配置<br/>config/train/lewm.yaml + policy/lewm.yaml"] --> DATA["读取 HDF5 数据集"]
    DATA --> TF["构造 transforms<br/>图像预处理 + 非图像列 z-score"]
    TF --> ADIM["设置 action_encoder.input_dim<br/>frameskip * action_dim"]
    ADIM --> SPLIT["按 train_split 划分 train/val"]
    SPLIT --> DL["DataLoader"]

    CFG --> MODEL["实例化 LeWMPolicy<br/>内嵌 JEPA + SIGReg"]
    DL --> BATCH["batch<br/>pixels, action, 其他状态列"]
    BATCH --> ENC["model.encode(batch)<br/>得到 emb 和 act_emb"]

    ENC --> CTX["取上下文<br/>ctx_emb = emb[:, :history_size]<br/>ctx_act = act_emb[:, :history_size]"]
    ENC --> TGT["取监督目标<br/>target_emb = emb[:, num_preds:]"]
    CTX --> PR["model.predict(ctx_emb, ctx_act)"]
    PR --> PL["pred_loss<br/>MSE(pred_emb, target_emb)"]
    ENC --> SR["SIGReg(emb)"]
    PL --> SUM["loss = pred_loss + 0.09 * sigreg_loss"]
    SR --> SUM
    SUM --> OPT["AdamW 反向传播更新"]
    OPT --> CKPT["保存 lewm_policy.ckpt<br/>lewm_object.ckpt<br/>和 weights checkpoint"]
```

训练时 LeWM 学的是 latent dynamics：给定当前若干帧的视觉 embedding 和动作 embedding，预测后续状态 embedding。训练阶段没有 CEM，也不会从模型里直接解码 action。

## 3. 评测和测试流程

```mermaid
flowchart TD
    ECFG["Hydra eval 配置<br/>config/eval/*.yaml"] --> WORLD["创建 swm.World 环境"]
    ECFG --> DSET["读取评测 HDF5 数据集"]
    DSET --> STAT["拟合 StandardScaler<br/>action, proprio/state 等"]
    DSET --> SAMPLE["采样 num_eval 个有效起点"]
    SAMPLE --> GOAL["目标来自同一 episode<br/>start_step + goal_offset_steps"]

    ECFG --> CKPT["优先加载 policy checkpoint<br/>lewm_policy.ckpt<br/>fallback lewm_object.ckpt"]
    CKPT --> LEWM["LeWMPolicy.make_world_policy<br/>或裸 JEPA fallback"]
    LEWM --> SOLVER["实例化 solver<br/>默认 CEMSolver"]
    STAT --> POLICY["WorldModelPolicy"]
    SOLVER --> POLICY
    POLICY --> WORLD

    WORLD --> RESET["按数据集起点重置环境"]
    GOAL --> SETGOAL["通过 callables 设置目标状态"]
    RESET --> LOOP["rollout eval_budget 步"]
    SETGOAL --> LOOP
    LOOP --> ACT["policy.get_action(world.infos)"]
    ACT --> STEP["env.step(action)"]
    STEP --> LOOP
    LOOP --> METRIC["输出 metrics, video, 结果文件"]
```

评测时的目标不是模型自己产生的，而是 `eval.py` 从数据集中构造的。以 TwoRoom 为例，目标是起点之后 `goal_offset_steps=25` 的 `goal_proprio`，再通过配置里的 `_set_goal_state` 写入环境。

## 4. 从候选动作到真实 action

```mermaid
flowchart TD
    INFO["当前 world.infos<br/>pixels, goal, action history, 状态列"] --> POL["WorldModelPolicy"]
    POL --> CEM["CEMSolver"]
    CEM --> SAMP["采样候选动作序列<br/>num_samples=300"]
    SAMP --> COST["LeWM.get_cost<br/>给每条候选序列打分"]
    COST --> TOPK["选 cost 最低 topk=30"]
    TOPK --> UPD["更新 CEM 均值和方差"]
    UPD --> MORE{"重复 n_steps=30?"}
    MORE -->|是| SAMP
    MORE -->|否| BEST["最优动作序列"]
    BEST --> BUF["按 receding horizon/action_block<br/>拆成 action buffer"]
    BUF --> ENV["执行当前 step action"]
```

这里最重要的一点是：LeWM 输出的是每条候选动作序列的 cost，不是直接输出 action。action 的来源是 planner。默认 planner 是 CEM，它在动作空间里反复采样、评分、保留低 cost 样本并更新采样分布，最后把最优动作序列交给 `WorldModelPolicy` 执行。

## 5. LeWM cost 计算细节

```mermaid
flowchart TD
    INFO["info_dict<br/>当前观测 pixels + goal + 状态列"] --> GOALCOPY["构造 goal dict<br/>goal pixels 替换为 goal 图像"]
    GOALCOPY --> GE["encode(goal)<br/>得到 goal_emb"]

    INFO --> INIT["取当前观测 history<br/>pixels 和已有状态"]
    INIT --> IE["encode(initial info)<br/>得到初始 emb"]

    CAND["action_candidates<br/>B x S x T x action_dim"] --> SPLIT["拆分 act_0 和 act_future"]
    SPLIT --> AE["action_encoder(act)<br/>得到 act_emb"]
    IE --> ROLL["rollout 自回归预测"]
    AE --> ROLL
    ROLL --> PRED["predicted_emb<br/>每条候选动作的预测未来 embedding"]

    GE --> LAST["取 goal_emb 最后一步"]
    PRED --> PLAST["取 predicted_emb 最后一步"]
    LAST --> MSE["MSE(pred_final_emb, goal_emb)"]
    PLAST --> MSE
    MSE --> COST["cost<br/>B x S，每个候选动作序列一个分数"]
```

`get_cost()` 内部先编码目标，再对每条候选动作序列做 latent rollout，最后用最后一步预测 embedding 和目标 embedding 的 MSE 作为 cost。CEM 只关心这个 cost 的大小，cost 越低表示该候选动作序列越可能到达目标。

默认 eval 配置中：

- `horizon: 5`
- `action_block: 5`
- `eval_budget: 50`
- CEM: `num_samples: 300`，`topk: 30`，`n_steps: 30`

所以一次规划会覆盖 `horizon * action_block = 25` 个环境步，不能超过 `eval_budget`。

## 6. 一句话总结

训练阶段：`LeWMPolicy` 通过 JEPA 学习 `当前视觉 embedding + 动作 embedding -> 下一状态 embedding`。

评测阶段：planner 生成候选动作，LeWM 给候选动作打 cost，planner 选择低 cost 动作序列，环境执行这些动作并统计指标。
