# LEDP

# Idea



出发点：
1\. Fast\-WAM虽然证明无需推理时预测未来，训练阶段预测未来即可提升Action 预测的性能，但是 **推理时丢弃Video Model**实际上也使得Fast\-WAM不能进行推理时planner，且使其Video Model难以**Online Train、Test Time Train、持续学习。**同时推理时Video Model对action head的信息传递仅限于f0的KV cache，这样的信息传递**瓶颈**极大，对预测Action的增益有限。

2\. LeWM实现了在latent空间的预测未来，并证明latent可以用于对CEM solver的planner。但是LeWM的串行预测限制了其预测的速度，从而**限制其推理和online训练的效率 **，且LeWM过于注重obs对应的latent，action仅作为简单的调制condition用于引导future latent的预测，从而**忽视action chunk的时序性、因果性和高维结构信息**；LeWM训练时利用action非常好的专家数据训练而成，而推理时使用action随机初始化的CEM solver，具有天然的训练推理分布不匹配的问题，这进一步限制了LeWM planner的性能。



思路：
1\. 将LeWM设计为能够**并行预测**的Fast LeWM结构，在提升推理效率的同时增强其对action chunk的**时序性和因果性**信息的利用

2\. 给LeWM增加action head，扩展为LeWAM，从而增强LeWM对于action chunk信息的利用能力，从而避免LeWM陷入只关注latent的变化而忽视action对latent影响。

3\. 将Fast LeWAM构建为能够**online训练**的算法，从而**缓解训练推理分布不一致**的问题，使得Fast LeWAM具备online训练和持续学习的可能性，进一步提升LeWM的实用价值



**研究问题：**

1\. Action Head是否能**反向增强**World Model的能力，从而进一步提升World Model对Action Head的增益？我们真的要**丢弃World Model**吗？

2\. 能否**同时**实现**实时策略、显式planning和在线模型改进**？



算法设计

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OWQ4ODdhODdlNmIxMzAxZDgxYTIzNzU4NWQ3YzU2MDZfYWZmYTk3NjNmNWJlODU0NGUwNjNiZGVlNDZjYTA1ZDBfSUQ6NzY2MjY2NDU2Mzc3NTcwNDI3MV8xNzg0NzExOTk0OjE3ODQ3OTgzOTRfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NzJlYWM2NDk5NDI4ZDFlNzkyN2E1YTg4ZDJjZTVlY2ZfMWUzMDE3YWNiNjUyOWIyNDE1NDkxOWEyOGQ3ZGY0Y2RfSUQ6NzY2MjY3Njk5MDU3OTUyNjk0M18xNzg0NzExOTk0OjE3ODQ3OTgzOTRfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MzdmMjYwZmUwZGZmMmQxMzA1N2EwMDg3YWY3ZjRkNmVfMzBiNmU2YjY3MGNkY2JjMjk2MjA4MDE2YzFjZDEyOGJfSUQ6NzY2MjY3Njk5MjIyNzYyNTk0NF8xNzg0NzExOTk0OjE3ODQ3OTgzOTRfVjM)



在同一个DiT模型中同时实现预测future latent 和 action

Stage A （action pred train）：根据z0预测a1:aH的action chunk，提升模型对action的结构理解能力。

Stage B（future pred train）：根据ground truth action或者自身推理的action预测future latent，此时action不是简单的调制条件信号，而是通过因果注意力机制的方式来引导pred future的生成，提升模型对于充分利用action的因果性、时序性信息pred future的能力。

Online Train ：先预测action，然后利用action预测pred latent，根据env step获得的truth latent对LeWAM进行online train，使其性能不断提升，具备持续学习的可能性。



贡献：

1. 我们提出 Fast\-LeWAM，将 LeWM 扩展为一种能够并行预测 future latent 的 World Action Model，在避免自回归 rollout 开销的同时，保留因果推理结构，从而提升推理效率。

2. 我们系统探索了一个新的问题：Action Head 的训练是否能够反向增强 Future Prediction Head。我们发现，显式动作预测能够促使 World Model 更充分地理解 action chunk 的结构性、时序性与因果性，从而生成更符合动作约束的未来 latent 表征。

3. 我们进一步探索了 Fast\-LeWAM 在 online training 和持续学习中的应用潜力，使模型不仅能够进行离线世界建模与策略学习，还具备在线更新 world model 与 action model、实现持续改进的可能。





讨论：

1\. 为什么Video World Model难以发展出类似的结构（**为什么要用LeWM**）：因为Video Model的**训练成本过高**，为了节约探索成本，Video World Model通常会使用预训练模型，且在train action head的时候仅微调甚至冻结Video Model，而不破坏Video Model本身的结构和性能。 同时也因为Video Model的训练成本过高，online train和持续学习的效率都被进一步限制。 同时Video Model通常**不使用完整的action traj而仅使用prompt**进行pred future，这限制其对action的进一步利用。

2\. 可不可以**scale up**：**如果在LeWM上验证成功，那么Video Model可以采用同样的方式来提升性能并实现持续学习。**

3\. 在同一个模型中同时实现action pred和future pred可行吗，会不会导致两边都学不好： 参考**Actor\-Critic**，LeWAM实际上结构和Actor\-Critic非常类似，action head出动作，future pred head进行想象，间接起到评判的作用，因此有理由相信这样的结构具有一定的可行性。








# 预实验


| Method | Two Room | Reacher | Push-T | Cube |
|---|---:|---:|---:|---:|
| DINO-WM (paper) | 97% | 79% | 74% | 86% |
| LeWM (paper) | 87% | 86% | 96% | 74% |
| Ours (stage A) | 96% | 76% | 96% | 100% |
| Ours (stage B) | 100% | 84% | 84% | 70% |

| Method | Two Room | Reacher | Push-T | Cube |
|---|---:|---:|---:|---:|
| 5 epoch Stage A | 96% | 46% | 86% | 100% |
| 5 epoch Stage A shuffled goal | 40% | 14% | 8% | 40% |
| 5 epoch Stage B | 96% | 66% | 84% | 66% |
| 5 epoch Stage C | 52% | 4% | 10% | 48% |
| 10 epoch Stage A | 96% | 76% | 96% | 100% |
| 10 epoch Stage A shuffled goal | 38% | 10% | 12% | 40% |
| 10 epoch Stage B | 100% | 84% | 84% | 70% |
| 10 epoch Stage C | 60% | 6% | 6% | 40% |





### 复现 lewm





train参数未对齐项：

|任务/参数|论文|官方上游代码|当前本地配置|
|---|---|---|---|
|Train epoch |10|100|100|
|SIGReg lambda|0\.1|0\.09|0\.09|
|Two Room History<br>其他任务一致，为3|1|3|3|

eval参数未对齐项：

|任务/参数|论文|官方上游代码|当前本地配置|
|---|---|---|---|
|PushT CEM iterations|30|30|30|
|Cube CEM iterations|10|30|30|
|Reacher CEM iterations|10|30|30|
|TwoRoom CEM iterations|10|30|30|
|TwoRoom eval budget|150|50|50|
|TwoRoom goal offset|100|25|25|

官方参数未对齐项：

|项目|论文附录 D|作者 TwoRoom 权重|
|---|---|---|
|History length|1|3|



# 废案

1\. 在实现并行推理的同时保持 $z_t + a_t \to z_{t+1}$的因果性结构。

2\. **探索一个问题：Action head train是否能进一步提升Future pred train的性能，如果可以的话，是否能够通过online train和持续学习的方式，使得整个WAM的action head 和 future head的性能不断联合提升。 **Stage A =》 Stage B：通过显示预测action，增强World Model对于action的结构性、因果性、时序性（一整个horizon）的理解，**生成更能符合因果性和action结构性的latent信息 **；~~也可以利用生成action的hidden state/KV cache来增强stage B的效果和速度。~~
~~3\. 解决DP不能预测action对未来的影响的问题。Stage B =》 Stage A：梯度回传，通过利用所预测的action进行预测未来，来学习action对未来的影响，从而更好的预测action。~~

4\.探索lewm在online train和持续学习上的可能性。

## 组合world model 解决multi\-agent

Multi\-Agent
一阶段：GauDP的latent版本，不使用昂贵的3GDS重建，而是使用global latent与local latent相结合
二阶段：正式的分布式通信部署，主打降低数据传输量的低延迟，同时需要做异步推理、端侧部署和加速推理。



## 解耦LEDP

现有 World Action Model（WAM）通常依赖大规模视频生成或视频\-动作联合建模来支持机器人控制，导致测试时未来想象开销较高。本文提出 **LeWM\-Action**，一种基于 JEPA 隐空间动力学的轻量级 WAM。该方法以 LeWorldModel 作为世界主干，从像素和动作中学习紧凑的预测性隐空间，并使用扩散式动作解码器生成动作。我们指出，仅将世界模型隐向量作为动作条件并不能充分利用其预测性动力学：它只能描述当前状态，却无法显式刻画候选动作的目标价值与动力学可行性。为此，LeWM\-Action 设计了三种功能性解耦的世界到动作控制接口：状态通道将当前与目标隐状态注入动作解码器，价值能量通道利用隐空间 rollout 代价引导动作去噪，动力学先验通道通过逆动力学生成粗动作草图以初始化扩散采样。三种接口分别控制动作生成中的条件、方向和起点，使 JEPA world model 从被动表征模块转变为主动动作控制器。我们将在仿真和真实机械臂任务中系统评估三种通道的作用，验证紧凑 JEPA 世界模型能否在不生成未来视频的情况下，为机器人动作生成提供轻量、高效且可解释的控制信号。





![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MWIyYzE0MDdhMGQ3ODFiMWYyNzkzNTcxZWU2MWIwOWZfM2Q5ZmUwYTJhNTQ5OWVmZmU0MWI4YTRhNjI0YTliMDlfSUQ6NzY1NzU1OTY1Mzk1NzY2ODA0N18xNzg0NzExOTk0OjE3ODQ3OTgzOTRfVjM)



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MjFhNDY5MjZjYjY2YzkxODgyZjllNTZhOWU3Y2ZjYWRfNzcyYjljMTEzOTdlMjAwNjIyNmY1YzEyMTMyMDM2MDVfSUQ6NzY1NzU1OTY1MjE5NjE0MjAzMl8xNzg0NzExOTk0OjE3ODQ3OTgzOTRfVjM)





## 实验

50 episodes 数据



1\. 基于LeWM的obs encoder可以学习到更加有效的表征方式。

2\.数据量太少，难以训练能够学习到具有planner能力的LeWorldModel

3\. LeWorldModel的planner的样本效率有限。

4\. Goal的规划方式弱于Progress





all结论：会加大收敛难度，主要还是看局部相机是否对任务有帮助，帮助不能抵消收敛的损失就完了。

2a\_pass\_shoe: 相机1没能提供有效的局部信息。

2a\_place\_food: 局部还行，但是data中锅盖没飞，eval中锅盖飞了

2a\_lift\_barrier:局部可以

2a\_stack\_cube:局部还行。我不太理解为啥对DP2提升那么大。



### 探索性

突然发现为啥DP2效果不好而ACT效果可以了：因为resnet18没用pretrained模型，会导致训练速度比较慢。

