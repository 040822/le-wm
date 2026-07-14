1. check model设计，尤其是attention mask的合理性（stage A mask z0=》a0~aH-1 ，stage A是否要添加zg， stage B的大causal mask真的能建模zt+at=>zt+1的转移过程吗？——
2. check train设计，目前是简单的 stage A => stage B, 能不能更复杂一些，比如 stage B=>stage A或者stage A => stage B => stage A，或者训练的时候加入stage C。
3. 如何证明stage A 对 stage B的增益？
4. 当前benchmark 实际上是不能证明stage A的有效性的，因为目前的目标只是goal，而没有一个明确的task goal，因此目前没有goal latent的stage A是理论不能完成的。
5. 可视化训练数据，看一下训练数据到底是在训什么，尤其是goal相关数据。