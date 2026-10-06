# DeWM / Sub-JEPA 实现审查与接入方案

日期：2026-10-03。本文记录接入前的代码、论文、权重元数据审查及集成设计；当时没有接入模型、修改训练/评测代码、转换正式权重或启动 GPU 实验。后续已按用户要求完成官方 Sub-JEPA 四任务权重接入，当前使用和验证记录见 [官方权重指南](../../guide/subjepa_official.md)。DeWM 仍处于接入设计阶段。

## 结论

1. DeWM 目录中的 `subjepa.py` 与 Sub-JEPA 官方版本逐字节相同；但当前 DeWM 的训练入口没有使用该模块，因此当前默认训练不能称为 Sub-JEPA。
2. DeWM 当前代码实现了 CLS/patch 两类表示、状态到状态的速度预测、Euler 积分和递归规划；但论文的角色专属损失、独立正则化、最终终点监督没有按公式落实，部分配置没有效果。不能据此确认它是论文主方法的完整复现。
3. 对 CoWM 最值得借鉴的是固定空间池化后的紧凑局部表示、分量化预测损失与规划代价、独立的表示正则化。直接替换为 DeWM 的递归 flow 动力学会损失 CoWM 的一次调用并行预测优势，优先级低。
4. 推荐先接入官方 Sub-JEPA 权重和统一评测。DeWM 的模块位置及接口可以确定，但用于正式比较的实现版本、四任务权重及训练来源还未确定，应先完成版本辨认及论文一致性验收。

## 1. 审查对象与证据范围

| 对象 | 固定来源 |
|---|---|
| 用户指定的 DeWM 工作目录 | `/data/users/majiahua/Flow-JEPA`，HEAD `f89eb9154689e1227a51e96e94745eb0601bf2cc`；工作树有大量未提交修改，结论针对实际工作树 |
| Sub-JEPA 官方仓库 | [intcomp/Sub-JEPA](https://github.com/intcomp/Sub-JEPA)，审查 commit `ef945ed434ce529bc7c5f1995f2e1cf173954843` |
| 官方绑定的 LeWM 子模块 | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` |
| Sub-JEPA 论文 | [arXiv:2605.09241v1](https://arxiv.org/html/2605.09241v1) |
| DeWM 论文 | 本仓库 `docs/paper/AAAI2027_Flow_JEPA.pdf`，正文标题为 DeWM: Role-Specialized Latent Dynamics for Reconstruction-Free World Models，匿名本地稿 |
| CoWM 方法 | 本仓库实际工作树：`paper/sec/3_method.tex`、`source/model/fast_lewam/`、`source/policy/fast_lewam.py`、`config/train/round4_ab.yaml` |

公开检索还存在 [arXiv:2608.29029](https://arxiv.org/abs/2608.29029) 的 Flow-JEPA，作者及方法均不同。不能仅凭目录名 Flow-JEPA 将它当成用户给出的 DeWM 论文，本次 DeWM 一致性判断依据上述本地稿。

关键内容 SHA256：

| 文件 | SHA256 |
|---|---|
| DeWM / 官方两份 `subjepa.py` | `38e1f9662e3adbae1aeb8b46fcec8e8c460f596ca1e02b3d8105c21b9092b3fb` |
| DeWM `train.py` | `e59e2d313ab80de7bfea28f62be9f3038348e9048e7bf4a27935d5961f1e053b` |
| DeWM `jepa.py` | `9997b7cadda780f45604c320077410f28e5c558d278fbbd554d9d146f358dbcb` |
| DeWM `module.py` | `93f0ba656b69c0c63454306fb7666a19e0d09dc7a1c8fe8aa0178340ea1af62b` |
| DeWM `config/train/lewm.yaml` | `8deacb056b8de6723cfe786f8a11843758148d6173ba2d795af87a5fcabc4050` |
| DeWM 本地 PDF | `1d5fa32f7649778cc937758899ba6abee9b56615c3d1e6b97d3e0950142509af` |

审查包含代码阅读、官方权重清单及配套配置查询、现有 DeWM pickle 的静态类名检查，以及 CPU 定向数值检查。没有验证论文成功率，也没有完成四任务权重加载和闭环评测。当前源码不能证明历史 checkpoint 当时使用的训练目标。

## 2. DeWM 中的 Sub-JEPA 与官方的区别

### 2.1 正则器文件没有区别

两份 `subjepa.py` 完全相同，均包含：

- `MultiSubspaceSIGReg`：先把 `[B,T,D]` 投影为 `[K,T,B,d_s]`，再在每个时间位置、每个子空间内计算 batch 样本的 SIGReg。
- `orthogonal_frozen`：每个子空间独立采样高斯矩阵并 QR 正交化，冻结行正交的投影矩阵。不同子空间之间没有被要求互相正交，也不是将 latent 按坐标切成互不重叠的块。
- `random_frozen` 与 `random_trainable_soft`：额外实现的变体；论文主方法使用前述冻结正交投影。
- 软正交罚项：只用于可训练投影模式。

官方论文主设置为 `D=192`；TwoRoom、Reacher、Cube 使用 `K=32, d_s=6`，PushT 使用 `K=16, d_s=12`。来源：[官方论文实现细节](https://arxiv.org/html/2605.09241v1)、[固定版本代码](https://github.com/intcomp/Sub-JEPA/blob/ef945ed434ce529bc7c5f1995f2e1cf173954843/subjepa.py)。

### 2.2 实际训练路径已经不同

| 项目 | 官方 Sub-JEPA | 当前 DeWM 工作树 |
|---|---|---|
| 视觉表示 | LeWM 的 CLS + MLP，192 维 | CLS 全局 192 维 + 4×4 patch 池化后局部 128 维，共 320 维 |
| 动力学 | 原 LeWM 离散下一状态预测 | 时间条件速度场，经 Euler 积分生成下一状态 |
| 预测损失 | 下一 latent 的 MSE | 随机桥上的速度损失，加积分轨迹上的速度/状态损失 |
| 正则器 | `MultiSubspaceSIGReg` | 从 `module.py` 导入的普通 `SIGReg` |
| 正则输入 | `self.sigreg(emb)`，接口为 `[B,T,D]` | `self.sigreg(emb.transpose(0,1))`，普通 SIGReg 接口为 `[T,B,D]` |
| 投影参数优化 | optimizer 匹配 `model\|sigreg` | optimizer 只匹配 `model` |
| 当前默认训练 | YAML 默认 100 epoch；README 主结果命令覆盖为 10 epoch，batch 128 | YAML 默认 1 epoch，batch 400 |
| 评测 | LeWM CEM + latent 终点距离 | CEM + 可调全局/局部终点距离 |

证据：[官方训练入口](https://github.com/intcomp/Sub-JEPA/blob/ef945ed434ce529bc7c5f1995f2e1cf173954843/train.py)，DeWM `train.py:17–18, 53–65, 121–172, 175–196`。

不能只取消 DeWM 中 `MultiSubspaceSIGReg` 的注释就恢复官方 Sub-JEPA：还需恢复 LeWM 的表示及预测模型，并正确传入 `embed_dim`、`K` 和 `[B,T,D]` 输入。若将正则器集成进本仓库，不能继续沿用普通 SIGReg 的转置调用。

DeWM 的 `output_model_name: subjepa` 和 README 仍沿用旧身份，因此名称不能证明所训练的方法。

## 3. DeWM 与本地论文方法的一致性

### 3.1 已落实的部分

- 一个共享 ViT 输出 CLS 与 patch tokens。
- CLS 经全局 MLP；patch 恢复二维网格，固定平均池化为 4×4，并用逐位置共享的 MLP 压缩，再展平拼接。对应论文的 task-state / spatial-detail 表示。
- 在当前 latent 与下一 latent 之间构造线性桥，目标速度为两者之差。
- 使用时间条件速度预测器及可微 Euler 积分，没有将像素重建加入世界模型训练目标。
- 多步规划递归调用下一状态预测，按预测终点与目标的全局/局部分量距离评分。`cost_local_weight` 的评分作用已实现。

证据：DeWM `module.py:342–433, 546–610`，`jepa.py:146–156, 218–288`。

### 3.2 主要偏离

| 严重程度 | 发现 | 对方法结论的影响 |
|---|---|---|
| 高 | `split_weighted_mse` 退化为整向量 MSE，不使用传入的局部权重；全局/局部指标恒为零 | 论文 Eq.14 的 role-specialized dynamics 没有落实，也无法从日志观察两类监督 |
| 高 | 普通 SIGReg 作用于拼接 latent；`BlockSIGReg` 初始化和调用均被注释 | 论文 Eq.16 的独立正则化未启用；配置中的两个正则权重不参与计算 |
| 高 | 每个 Euler 更新后都将中间状态与最终目标做 MSE 并累加 | 论文 Eq.13 只约束积分完成后的终点；当前目标额外鼓励提前到达目标，改变速度场的学习问题 |
| 高 | 另在每个模型积分状态上累加 `u_t` 与同一个 `z_{t+1}-z_t` 的速度 MSE | 论文 Eq.9 的随机桥速度监督之外又增加一组路径监督；积分步数变化会改变损失尺度 |
| 中 | 默认训练初始化一个混合全向量的 `ARPredictor`；`DualStreamPredictor` 被注释 | 不能据当前入口声称预测阶段保持两部分独立；论文文字中的独立性与统一场公式本身也需明确 |
| 中 | `predict(..., num_steps=2)` 固定默认步数，rollout 未传 `predictor.flow_steps` | 当前训练用配置的 4 步，rollout 默认 2 步；仅修改 flow_steps 不能构成推理步数消融 |
| 中 | 当前默认 cost 权重为 1，论文默认 balanced 为 0.25 | 当前对应 event-enhanced 规划偏好，不能直接作为论文 balanced 版本 |
| 中 | 当前 192+128 维、1 epoch 配置与文中优选 64+64 维和现有 100 epoch 权重并不相同 | 当前默认配置不是可直接复现论文表格的完整实验配方；文稿也没有提供完整训练超参数说明 |

证据：DeWM `jepa.py:174–210`、`train.py:53–65, 136–161, 186–196`、`config/train/lewm.yaml:16–26, 48–98`。论文依据为本地 PDF 的 Eq.9–16、规划段落和 Table 5。

### 3.3 目标函数的具体差异

用 `G`、`L` 表示两类 latent，论文定义：

\[
\mathcal L_{dyn}=\mathcal L_{pro}^{G}+\alpha\mathcal L_{out}^{G}
+\beta\mathcal L_{pro}^{L}+\mathcal L_{out}^{L},\quad \alpha,\beta<1.
\]

它要求全局强调过程、局部强调结果。当前代码则为：

\[
\mathcal L_{fm}=\operatorname{MSE}(V(z_\tau),\Delta z)
+\sum_{k=0}^{K-1}\operatorname{MSE}(V(\hat z^{(k)}),\Delta z),
\]

\[
\mathcal L_{pred}=\sum_{k=1}^{K}\operatorname{MSE}(\hat z^{(k)},z_{t+1}),
\quad \mathcal L=\mathcal L_{fm}+0.05\mathcal L_{pred}+0.09R([G;L]).
\]

当前整向量 MSE 隐含按维度加权：320 维中全局占 0.6、局部占 0.4；这与显式角色权重不同。

只恢复被注释的分量 MSE 仍不足以对齐论文。按现有配置恢复后，终点部分会是 `0.05*(L_out^G + 0.25*L_out^L)`，局部终点系数仅 0.0125，依然不符合 Eq.14 中局部终点的主导作用。后续忠实实现应显式写出四项系数，而不能把现有两个 local_weight 当成论文 α、β。

另外，代码保留原始 `src` 作为独立条件输入，预测器实际为 `V(x,src,a,τ)`；论文简写 `V(x,a,τ)`。它可以作为条件化的实现扩展，但应该明确披露。当前积分从 `eps=0.001` 开始、长度为 `1-eps`，而论文写从 0 到 1；这是较小的数值差异，冻结配置时同样应记录。

### 3.4 可运行检查

在 CPU 上使用固定速度 0.5 的玩具预测器，当前状态为 0、目标状态为 1、两步积分、`eps=0.001`：

| 检查 | 实际结果 |
|---|---|
| 修改 fm/pred 的 local_weight | 损失完全不变 |
| 全局/局部损失日志 | 四项均为 0 |
| 当前 fm_loss | 0.75；仅随机桥监督应为 0.25 |
| 当前 pred_loss | 0.813375；仅最终终点 MSE 为 0.250500 |
| predictor.flow_steps=9，调用默认 predict | 速度场调用次数仍为 2 |
| 官方正则器 `[5,4,192]`，K=32 | 投影为 `[32,4,5,6]`，投影冻结；行正交最大误差约 4.77e-7，latent 梯度有限 |

这些结果验证了调用和损失语义，不代表闭环性能结论。

### 3.5 历史权重的来源问题

静态读取 `data/checkpoints` 的六份 object checkpoint 的 pickle 类名，不反序列化模型：

- `tworoom-v2-192` 与 `tworoom-v2-320-entirereg` 包含 `CrossAttentionPooling` 和单 `ARPredictor`。
- `tworoom-fixed-64x64-joint`、`pusht-fixed-64x64-joint`、`pusht-v2-isolated`、`ogbscene-v2-isolated` 包含 `FixedSpatialPooling` 与 `DualStreamPredictor`。
- 当前训练入口为 `FixedSpatialPooling` 与单 `ARPredictor`，因此历史权重与当前入口并非同一结构。
- 部分目录没有配套训练配置；PushT 的 64+64 joint 权重为 epoch 1，其他示例有 epoch 100。此检查范围内没有 Cube/Reacher checkpoint。

此外，`DualStreamPredictor` 中局部分支接收经 `detach` 的全局特征：detach 切断梯度，却没有去掉前向依赖。启用它也不等于严格的双向独立预测。

object checkpoint 保存参数和对象属性，但 Python 方法从加载时的类定义取得；当前 `get_loss` 不能还原历史训练版本，文件名也不能证明 regularizer/监督方式。正式使用前需取得对应代码快照、完整 resolved config、训练日志和数据身份。

## 4. 对 CoWM 的借鉴判断

详细设计、公式、实验对照和验收指标见 [展开分析](cowm_borrowing_from_dewm_20261003.md)。2026-10-03 后续工作只接入官方 Sub-JEPA 权重；本文中的 DeWM 接入仍为设计。

### 4.1 两种 flow 解决的问题不同

| 方法分支 | 流的起点/终点 | 条件 | 推理方式 |
|---|---|---|---|
| DeWM dynamics | 当前 latent → 下一 latent | 当前状态、动作、flow 时间 | 每个物理 action block 内 K 次积分，再递归 H 个 block |
| CoWM A | 高斯动作噪声 → 动作 chunk | 当前/目标 latent、flow 时间 | S 次积分生成整个动作 chunk |
| CoWM B | 无生成流，直接回归 future latents | 当前 latent、动作前缀、训练条件 s | 一次 transformer 调用预测 H 个 future latents |

DeWM 不是从高斯噪声采样未来的随机世界模型。CoWM 的 full A+B+D+E 探索还包含额外的 latent-path flow，但当前 R4-AB 方法不使用该 D/E 目标；不能把它与 DeWM 的单步状态桥混为同一种训练任务。

CoWM 的 A/B 共享参数、生成动作输入 B、B 梯度通过动作回到 A，是与 DeWM 表示分解不同的设计轴。DeWM 不解决生成动作所对应的反事实未来缺少监督的问题。

### 4.2 推荐顺序

**优先级 1：保留 B 的并行结构，引入紧凑局部空间表示。**

本仓库 `FastLeWAM.encode_pixels` 当前仅使用 CLS。可在新的实验模型中从同一 ViT 提取 CLS 和固定 4×4 池化 patch，将全局/局部表示拼成紧凑向量。A/B 仍使用现有共享 DiT 和 causal-prefix mask，不引入全量 patch tokens，也不增加 horizon 上的串行调用。

为区分空间信息与模型容量，先做总 latent 仍为 192 的对照，例如 128 全局 + 16×4 局部；这个划分只是待验证设计，不是 DeWM 论文配方。论文的 64+64 可单列一个 128 维参考。新表示需要重新训练，不能复用旧 CoWM checkpoint 并宣称方法相同。

实施位置应集中在新的表示/模型实现中：现有 `encode_pixels` 向 projector 只传 CLS，而 DeWM projector 要 CLS+patch 且返回 tuple，不能仅替换 Hydra `_target_`。A/B 的调用接口可继续保持当前紧凑 latent 形状。

**优先级 2：分量化 B 的预测损失与规划代价。**

将未来误差拆成全局与局部两项，保留现有 generated-action 输入和梯度路径。规划时比较 `J_G + η J_L`，在开发集上冻结 η，再评测闭环选择和 PO。需要记录两类距离的维数、归约及实际尺度；η 不能跨不同表示空间直接视为同一物理偏好。

验收关注候选排名、实际执行后的目标改善及决策延迟，不能仅看训练 MSE。调整 inference-time η 是已训练分量表示上的评分选择；在当前全 CLS 表示中任意切维度不能取得同等语义。

**优先级 3：比较独立 SIGReg 和 Sub-JEPA 正则。**

只有建立显式全局/局部表示后，独立 SIGReg 才对应 DeWM 的功能分量。Sub-JEPA 的随机投影约束与这种语义分块不同，应单独做对照，先不要同时改表示、动力学和正则器。两路正则之和、加权平均和全向量正则的尺度不同，需要明确系数。

**低优先级：积分终点监督或 B 的 latent flow 化。**

可以探索训练期对完整生成终点的约束，或另建 B 的 latent velocity 变体，但都改变目标及训练成本。CoWM A 已有 clean-action estimate，不能把它称为 DeWM 完整积分终点监督。若另做积分终点监督，应约束最后的生成结果，而不是把每个中间状态都拉到终点。

把 B 直接换成 DeWM 动力学，H=5、K=2 时每批候选至少需要 10 次速度场查询，当前 CoWM B 则一次预测全部未来。实际延迟还取决于结构与历史长度，不能仅据调用次数预测绝对耗时，但这个串行依赖直接影响 CoWM 的设计目标。

## 5. 本仓库接入方案

### 5.1 共用评测接口，分开各方法实现

继续使用已有模型 `get_cost(info_dict, action_candidates) -> [B,S]` 接口、`WorldModelPolicy`、CEM、公共数据预处理、checkpoint 和评测样本流程。

各方法保留自己的 encoder、projector、动作 normalizer 和有效 history。公共协议可以固定同一物理动作序列、horizon 和起点，不能把 CoWM latent 直接传给另一个模型；若以后比较同一个 A 配不同评分器，要重新编码像素并通过物理动作坐标转换。

不把不同方法都塞进 `source/model/lewm` 的布尔开关，也不让评测入口依赖 `/data/users/majiahua/Flow-JEPA` 的 Python import 路径。

### 5.2 Sub-JEPA：先官方权重评测，再按需训练

已通过官方 Hugging Face 清单及配套配置确认四任务发布资产：

| 任务 | 官方资产目录 | 配置报告的 epoch / seed / D / K |
|---|---|---|
| TwoRoom | [subjepa/tworoom](https://huggingface.co/intcomp/sub-jepa/tree/main/subjepa/tworoom) | 10 / 3072 / 192 / 32 |
| PushT | [subjepa/pusht](https://huggingface.co/intcomp/sub-jepa/tree/main/subjepa/pusht) | 10 / 3072 / 192 / 16 |
| Reacher | [subjepa/reacher](https://huggingface.co/intcomp/sub-jepa/tree/main/subjepa/reacher) | 10 / 3072 / 192 / 32 |
| Cube | [subjepa/cube](https://huggingface.co/intcomp/sub-jepa/tree/main/subjepa/cube) | 10 / 3072 / 192 / 32 |

每个目录都有 `config.yml` 和 `*_subjepa_object.ckpt`。这是配置报告的训练来源，不是对训练过程的独立验证。发布配置仍使用旧 `wm.sigreg_subspaces` / `sigreg_init_mode` 布局，与当前官方 YAML 不同，转换时需要显式映射。

未来拟新增/扩展的文件：

| 文件/位置 | 责任 |
|---|---|
| `source/model/subjepa/regularizers.py` | 固定版本 `MultiSubspaceSIGReg`，明确 `[B,T,D]` 输入契约；保留作者和许可来源 |
| `source/policy/subjepa.py` | 仅在需要训练时新增；复用 LeWM 模型及生命周期，使用子空间正则；软投影变体才加入正交项和 sigreg 参数优化 |
| `config/train/subjepa.yaml`、`config/train/policy/subjepa.yaml` | 明确 D=192、按任务冻结 K、orthogonal_frozen；不沿用默认 PushT K=32 |
| `source/common/checkpoint.py` 的方法加载流程及独立转换工具 | 保存原始官方 object/config/hash，离线转成明确模型配置与 state_dict，严格加载 |
| 正式评测矩阵 | 方法身份 `subjepa_official`，复用公共评测；推理无需计算 SIGReg |

若只做官方权重评测，可先不增加训练 policy。Sub-JEPA 推理模型仍是 LeWM，因此完成 checkpoint 转换后可复用 LeWM World Policy，不需要新 planner。

转换必须在隔离的进程/对应源代码环境中加载旧 object，避免覆盖本仓库根目录 `jepa.py`、`module.py` 的兼容导出。随后验证迁移前后 latent、预测和 cost；已有 ViT state_dict remap 可以复用，但不能把任意非 LeWM object 一律套进该映射。

训练资源估计：官方向量化正则在 K=32、T=4、B=128、M=1024、knots=17 时，仅一个 FP32 频率展开张量约 1.06 GiB；梯度及其他中间结果还会增加显存。这个数是静态估算，没有测 GPU 峰值。可以研究按 K 分片的等价实现；按 B 分片后对 SIGReg 平均会改变统计目标，不能当作等价优化。使用官方权重可以先避开训练资源问题。

### 5.3 DeWM：先辨认版本，再接独立模型

未来拟新增：`source/model/dewm/{jepa,modules}.py`、`source/policy/dewm.py`、`config/train/dewm.yaml`、`config/train/policy/dewm.yaml`。评测继续使用相同 `get_cost` 和 CEM。

需要区分两种身份：

- `dewm_source_snapshot`：严格保留用户提供的某一源码/权重版本，标为本地实现参考；保留其损失、cost 和步数差异，不借论文成绩证明它。
- `dewm_paper_aligned`：四项角色损失、分量正则、最终终点监督及配置明确按论文对齐；这是本仓库重新实现版本，需取得或重新训练匹配权重，不能把旧结果转记到新实现名下。

正式 baseline 优先使用有完整来源、能证明对应论文主方法的版本。当前审查发现不足以确定应直接使用哪一份历史权重，也未找到本地 Cube/Reacher 成套权重；因此不能保证无需训练即可完成四任务接入。

论文对齐实现应明确以下契约：

1. 共享 ViT 的 CLS / patch 表示，以及 global/local 维数、池化和预测分支的依赖关系；对“独立预测”的解释需与论文/作者实现一致。
2. 随机桥速度监督与最终积分终点监督分开计算；四项系数独立配置和记录。
3. global/local 正则分别调用，明确是相加还是平均以及对应 λ；这不等于 Sub-JEPA 的随机多子空间正则。
4. 推理步数显式传入；训练步数、推理步数、积分器、eps 分开归档。论文默认 K=2 和 η=0.25，不能直接套用 CoWM A 的 S。
5. 明确支持的 history_size 和 num_preds；当前训练切片和递归 rollout 围绕 num_preds=1 设计，不能只改该数得到任意多步模型。
6. 集成代码不硬编码 `CUDA_VISIBLE_DEVICES=0` 或 `STABLEWM_HOME=./data`，而复用本仓库资源限制及资产路径。

### 5.4 统一评测及与现有主表计划的关系

沿用四任务公共起点/目标、six eval seeds `42,100,2026,3407,1234,4444`、每 seed 50 episodes、goal_offset/budget=25/50、horizon/receding_horizon/action_block=5/5/5。DeWM 本地四任务 YAML 确实采用 25/50 和 5/5/5；CEM 为 300×30、top-k=30、var_scale=1。

公共样本由本仓库评测模块产生或读入冻结清单；不复制 DeWM 的硬编码日志目录及完整视频流程。其上游 `len(valid_indices)-1` 抽样方式本仓库也显式保留，应继续以共同清单消除差异。

独立计时用同卡 batch=1，包括像素预处理、编码、候选生成/搜索、评分及输出，显式强制发生真实重规划；不把带环境/视频的 wall-clock 或 50 环境批量耗时除以 50 称为单决策延迟。对于各方法不同的 action normalization 和裁剪语义，参考已有 [主表计划复查](cvpr_table1_plan_review_20261003.md) 的物理动作对齐要求。

[当前主表计划](../../plan/cvpr_table1_plan.md) 明确包含后续 Sub-JEPA，并排除 DeWM。本次用户要求确认两种方法的接入方案，故设计 DeWM 的可选接入，但未据此重写正式 baseline 名单。每新增一个四任务方法增加 24 个评测单元、1,200 episodes；是否加入主表、是否包含原生/统一动作处理对照，需要在最终矩阵中重新计数。

### 5.5 后续实施顺序与验收

1. 固定来源：保存代码 commit、工作树 diff/hash、原始模型及训练/模型配置、数据身份与 normalizer；先辨认 DeWM 权重的方法版本。
2. 完成 Sub-JEPA 单任务的隔离转换与 CPU 数值一致性，再扩展四任务官方权重。
3. 将两方法接入公共评测；对 DeWM 先选择 source snapshot 或 paper-aligned 身份，避免混合四任务不同版本。
4. 检查编码/预测形状、frameskip 动作维度、有限的 `[B,S]` cost、候选/环境分片一致性、history 和真实推理调用次数；旧 object 转換需保持参数严格匹配。
5. 若训练，检查 Sub-JEPA 正则器输入轴、冻结投影及梯度；检查 DeWM 四项损失/权重效果、积分终点和分量正则。可复用本次玩具数值例作为语义验收。
6. 在开发样本上做单任务闭环 smoke，再冻结四任务正式评测和计时；最终报告区分实现一致性、checkpoint 来源和复现成绩。

本次已经完成源码/论文对照、官方资产清单确认与上述 CPU 语义检查。权重转换数值一致性、四任务闭环结果和 GPU 时间/显存尚未执行，属于后续实施的验收项。
