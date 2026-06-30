# LeWM

训练：

```bash
python train.py data=pusht
```

评测：

```bash
python eval.py --config-name=tworoom.yaml policy=checkpoints/tworoom/lewm
```

## 代码结构

本仓库现在使用类似 `DP_real` 的 `source/` 布局：

- `source/model/lewm/`：LeWM 的 JEPA latent dynamics 模型和相关 `nn.Module` 组件。
- `source/policy/lewm.py`：`LeWMPolicy` 训练入口，复用 `stable_pretraining.spt.Module` 的 Lightning 生命周期，并提供 `make_world_policy` 构造环境 eval 所需的 `stable_worldmodel` policy。
- `source/common/`：训练/评测共享的数据加载、日志、checkpoint、eval 兼容工具。

LeWM 评测阶段不会直接从模型解码 action。JEPA 负责给候选动作序列输出 latent cost，planner 选择低 cost 动作，再由 `WorldModelPolicy` 与环境交互。

根目录的 `jepa.py`、`module.py`、`utils.py` 只保留为旧 import/checkpoint 的兼容 re-export，新代码应从 `source.*` 导入。
