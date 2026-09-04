# Monte Carlo Control

Monte Carlo Control（蒙特卡洛控制）通过一条完整 episode 的实际回报来估计
动作价值函数，不需要已知环境模型，也不使用下一状态的估计值进行 bootstrap。

## 核心更新

一条 episode 结束后，从时间步 $t$ 开始的折扣回报为：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

对于第一次出现的状态动作对 $(S_t,A_t)$，使用所有已观察回报的平均值更新：

$$
Q(S_t,A_t) \leftarrow
\frac{1}{N(S_t,A_t)}
\sum_{i=1}^{N(S_t,A_t)}G_t^{(i)}
$$

行为策略采用 $\varepsilon$-greedy：

- 以概率 $1-\varepsilon$ 选择当前 Q 值最大的动作。
- 以概率 $\varepsilon$ 随机探索。

## 在实验中观察什么

1. `Episodes` 增加时，最近 100 轮的成功率通常会更加稳定。
2. `Epsilon` 太小可能无法探索到终点，太大则会长期执行随机动作。
3. `Slippery` 开启后，动作结果存在随机性，学习难度明显增加。
4. Monte Carlo 必须等 episode 结束才能更新；这和 TD 方法的逐步更新不同。

## 与现有算法的关系

- 与 Q-Learning 相同：都是 model-free control。
- 与 Q-Learning 不同：Monte Carlo 使用完整回报，Q-Learning 使用一步 TD target。
- 优点：目标直接来自实际回报。
- 局限：长 episode 的方差较大，而且必须等待 episode 结束。
