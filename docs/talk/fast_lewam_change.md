1. 尝试将action维度展平，不要用5帧action在维度轴上堆叠为1帧action的方式，
2. 尝试修改position embedding，统一stage a和b所用的position embedding，并增加type embedding。尝试将可学习embedding改为sino，对应物理相对时间步，比如z0对应t=0,a0对应t=0,z1对应t=1
3. （收益不一定大）尝试把action的t加上0.5,比如z0对应t=0,a0对应t=0.5,z1对应t=1
4. 对于noise embedding：stage B不管输入是clean action还是stage A预测的action都统一用clean action对应的noise embedding或者不用noise embedding。因为stage B的主要任务是严格根据所给的action预测future latent，因此不管是clean action还是noise action都是一样的。

5. next-stage： online训练

考虑时间和计算资源限制，所有实验均遵循 1 seed 多任务原则，任务优先级顺序为（cube→push T→reacher），可以不做reacher和成功率已经达到100%的tworoom。