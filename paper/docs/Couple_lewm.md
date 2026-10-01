CoWM: coupled latent world model

## introduction

科学问题: 
在有限决策预算下，联合学习动作生成与动作条件动力学，能否提高模型对候选动作的评估（rerank）和修正（Guidance，PO/GF）能力，从而减少对大规模搜索的依赖？
1. 耦合模型，联合学习。
2. 并行评估action、使用cost梯度修正action。
3. 有限预算 / 推理速度 / fast

耦合： 
1. 参数耦合，使用同一个shared DiT 实现两个任务。
2. 梯度耦合，梯度回传。A=》B，B的latent loss回传到A。
3. 动作输入分布耦合。（特指B在训练的时候混入A的action，从而让B预先在A的输出action分布上进行优化，提升B对于A的action的rerank效果）

评估： 
1. 候选动作排序（P3 ， A输出action + B rerank ，速度快效果好）
2. 动作梯度修正方向（PO/GF，在reacher任务有提升）
3. 物理特性probe（复用lewm的实验，但重点关注cube任务中lewm做不好的几个物理量）

fast/硬件效率 ：
平均单次推理时间为 20ms 
闭环rollout后边测一下。

## 实验
主表：和baseline进行对比，baseline在DeWM的基础上增加LeFlow，任务预先选择lewm四任务，补充任务经过验证后再加上。

表2: 证明耦合的优势。 
1. 当前模型、分开DiT、action DiT + lewm
2. 梯度耦合与否。
3. 输入action分布。
表3: Probe实验
1. 候选动作排序，说明我们的模型的排序正确率是否更高（突出cube）
2. 动作修正方向，说明PO/GF的梯度是否能够有效修正action
3. 物理特性probe，说明我们的模型的latent相比lewm是否能够有效读出物理量。（突出cube）

表4: 对比不同推理。（P0~P3、PO/GF）
表5: 对比推理速度。单步推理时间、单轮rollout时间。
表6（maybe）：真机实验。

## 附录
放一下R4-AB和R4-ABCD的对比