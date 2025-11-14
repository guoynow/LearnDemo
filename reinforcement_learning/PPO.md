## 强化学习的目标

在强化学习中，目标是训练一个神经网络 $Policy$ $\pi$ ，在所有状态 $s$ 下，给出相应的 $Action$ ，得到的 $Return$ 的期望值最大。即：
$$
E(R(\tau))_{\tau \sim P_{\theta}(\tau)} = \sum_{\tau} R(\tau) P_{\theta}(\tau)
$$

其中：
1. $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$：表示在策略 $P_{\theta}(\tau)$ 下轨迹 $\tau$ 的回报 $R(\tau)$ 的期望值。
2. $R(\tau)$：轨迹 $\tau$ 的回报，即从起始状态到终止状态获得的所有奖励的总和。
3. $\tau$：表示一条轨迹，即智能体在环境中的状态和动作序列。
4. $P_{\theta}(\tau)$：在参数 $\theta$ 下生成轨迹 $\tau$ 的概率。
5. $\theta$：策略的参数，控制着策略 $P_{\theta}$ 的行为。

所以，我们的目标是找到一个策略 $\pi$，使得 $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$ 最大。那怎么找到这个策略呢？我们使用梯度上升的办法，即不断地更新策略参数 $\theta$，使得 $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$ 不断增大。

首先，我们来计算梯度：

$$
\begin{align*}
\nabla E(R(\tau))_{\tau \sim P_{\theta}(\tau)} &= \nabla \sum_{\tau} R(\tau) P_{\theta}(\tau) \\
&= \sum_{\tau} R(\tau) \nabla P_{\theta}(\tau) \\
&= \sum_{\tau} R(\tau) \nabla P_{\theta}(\tau) \frac{P_{\theta}(\tau)}{P_{\theta}(\tau)} \\
&= \sum_{\tau} P_{\theta}(\tau) R(\tau) \frac{\nabla P_{\theta}(\tau)}{P_{\theta}(\tau)} \\
&= \sum_{\tau} P_{\theta}(\tau) R(\tau) \nabla \log P_{\theta}(\tau) \\
&\approx \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log P_{\theta}(\tau^n)
\end{align*}
$$

接下来，我们来看一下 $Trajectory$ 的概率 $P_{\theta}(\tau)$ 是怎么计算的：

1. $P_{\theta}(\tau ^n) = \rho _0(s_0) \prod_{t=0}^{T-1} P(s_{t+1} \mid s_t, a_t) P_{\theta}(a_t \mid s_t)=\prod_{t=1}^{T_n} P_{\theta}(a_n^t \mid s_n^t)$。
2. 上述公式的解释： 在大语言模型中，状态 $s_t$ 是已经生成的 $token$ 序列，动作 $a_t$ 是生成下一个 $token$，下一个状态 $ s_{t+1} $ 是已生成的 $token$ 序列加上新生成的 $token$。因此 $s_{t+1}=f(s_t,a_t)$ 是**确定的状态转移**，即 随机的状态转移 $P(s_{t+1}|s_t, a_t) = 1$；而大语言模型的初始状态是确定性的 $prompt$，故 $\rho _0(s_0)=1$，其中 $\rho _0$ 是初始状态的分布。

进一步计算梯度：
$$
\begin{align*}
\frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log P_{\theta}(\tau^n) &= \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log \prod_{t=1}^{T_n} P_{\theta}(a_n^t \mid s_n^t) \\
&= \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \sum_{t=1}^{T_n} \nabla \log P_{\theta}(a_n^t \mid s_n^t) \\
&= \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \nabla \log P_{\theta}(a_n^t \mid s_n^t) \\
\end{align*}
$$

1. 省略梯度符号后的形式：$\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \log P_{\theta}(a_n^t \mid s_n^t)$，将其作为**优化目标**，其中 $\theta$ 是 $Policy$ 神经网络的参数。

那我们应该如何训练一个 $Policy$ 网络呢？我们可以定义 $loss$ 函数为：

$$
loss = - \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \log P_{\theta}(a_n^t \mid s_n^t)
$$

在目标函数前加上负号，就可以转化为一个最小化问题,从而使用梯度下降的方法来求解这个问题。

上述公式的直观意义：如果当前的 $Trajectory$ 的回报 $R(\tau)$ 较大，那么我们就会增大这个 $Trajectory$ 下所有 $action$ 的概率，反之亦然。
这样，我们就可以不断地调整策略，使得回报最大化。

但这明显是存在**改进空间**的：

1. 是否减少或增大 $P_{\theta}(a_t|s_t)$，应该看做了这个动作之后，到结束游戏的累积 $Reward$，而不是整个 $Trajectory$ 累积的 $Reward$。
2. 一个 $action$ 可能只会影响接下来的几步 $Reward$，且影响逐步衰减，后面的 $Reward$ 更多的是被当时的 $action$ 影响。

针对这个问题，修改一下公式，令 $R(\tau^n) \rightarrow R_t^n = \sum_{t' = t}^{T_n} \gamma^{t' - t} r_{t'}^n$，得到新的优化目标函数：

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R^n_t \log P_{\theta}(a_n^t \mid s_n^t)
$$

还有一种情况会影响我们算法的稳定性，那就是在好的局势下和坏的局势下。比如在好的局势下，不论你做什么动作，你都会得到正的回报，这样算法就会增加所有动作的概率，这样会让训练很慢，也会不稳定。最好是能够让相对好的动作的概率增加，相对坏的动作的概率减小。

为了解决这个问题，我们可以对所有动作的 $Reward$ 都减去一个 $baseline$，这样就可以让相对好的动作的 $Reward$ 增加，相对坏的动作的 $Reward$ 减小，也能反映这个动作相对其他动作的价值。

所以优化目标函数变为：

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} (R_t^n - B(s_n^t)) \log P_{\theta}(a_n^t \mid s_n^t)
$$

其中，$B(s_n^t)$ 也需要用神经网络来拟合，这就是 $Actor-Critic$ 网络。$Actor$ 网络负责输出动作的概率，$Critic$ 网络负责评估 $Actor$ 网络输出动作的好坏。

接下来解释几个常见的强化学习概念：

- $Action-Value~Function$：$R_t^n$ 每次都是随机采样，方差很大。我们可以用 $Q_{\theta}(s, a)$ 来代替，$Q_{\theta}(s, a)$ 表示在状态 $s$ 下采取动作 $a$ 的价值，即从状态 $s$ 开始，采取动作 $a$ 后，按照某个策略 $\pi$ 执行，最终获得的回报的期望值。$Q(s, a)$ 可以用来评估在状态 $s$ 下采取动作 $a$ 的好坏，从而指导智能体的决策。

- $State-Value~Function$：$V_{\theta}(s)$ 表示在状态 $s$ 下的价值，即从状态 $s$ 开始，按照某个策略 $\pi$ 执行，最终获得的回报的期望值。$V(s)$ 可以用来评估状态 $s$ 的好坏，从而指导智能体的决策。

- $Advantage~Function$：$A_{\theta}(s, a) = Q_{\theta}(s, a) - V_{\theta}(s)$，表示在状态 $s$ 下采取动作 $a$ 相对于采取期望动作的优势。优势函数可以用来评估在状态 $s$ 下采取动作 $a$ 的优劣，从而指导智能体的决策，即优势函数。

其中：$R_t^n - B(s_n^t)$ 就是上述优势函数，表示在状态 $s_n^t$ 下采取动作 $a_n^t$ 相对于采取期望动作的优势。那我们的优化目标函数就变成了**最大化优势函数的期望**：

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta}(s_n^t, a_n^t) \log P_{\theta}(a_n^t \mid s_n^t)
$$

那如何计算优势函数呢？我们重新来看一下优势函数的定义：

$$
A_{\theta}(s, a) = Q_{\theta}(s, a) - V_{\theta}(s)
$$

$Q_{\theta}(s, a)$ 表示在状态 $s$ 下采取动作 $a$ 的价值，$V_{\theta}(s)$ 表示在状态 $s$ 下的价值。我们来看一下下面这个公式：

$$
Q_\theta(s_t, a) = r_t + \gamma \cdot V_\theta(s_{t+1})
$$


我们把上述公式代入到优势函数的定义中：

$$
\begin{align*}
A_{\theta}(s_t, a) &= Q_{\theta}(s_t, a) - V_{\theta}(s_t) \\ 
&= r_t + \gamma \cdot V_\theta(s_{t+1}) - V_\theta(s_t)
\end{align*}
$$

我们可以看到，现在优势函数中只剩下了状态价值函数 $V_\theta(s_t)$ 和下一个状态的价值函数 $ V_\theta(s_{t+1}) $，这样就由原来需要训练两个神经网络变成了只需要训练一个状态价值网络，这样就减少了训练的复杂度。

在上面的函数中，我们是对 $Reward$ 进行一步采样，下面我们对状态价值函数也进行 $action$ 和 $Reward$ 的一步采样。

$$
V_\theta(s_{t+1}) \approx r_{t+1} + \gamma \cdot V_\theta(s_{t+2})
$$

接下来，我们就可以对优势函数进行多步采样，也可以全部采样：

$$
A_\theta^1(s_t, a) = r_t + \gamma \cdot V_\theta(s_{t+1}) - V_\theta(s_t)
$$

$$
A_\theta^2(s_t, a) = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot V_\theta(s_{t+2}) - V_\theta(s_t)
$$

$$
A_\theta^3(s_t, a) = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \gamma^3 \cdot V_\theta(s_{t+3}) - V_\theta(s_t)
$$

$$
\vdots
$$

$$
A_\theta^T(s_t, a) = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \gamma^3 \cdot r_{t+3} + \cdots + \gamma^T \cdot r_T - V_\theta(s_t)
$$

我们知道，采样的步数越多，会导致方差越大，但偏差会越小。为了让式子更加简洁，定义：

$$
\delta_t^V = r_t + \gamma \cdot V_\theta(s_{t+1}) - V_\theta(s_t)
$$

$$
\delta_{t+1}^V = r_{t+1} + \gamma \cdot V_\theta(s_{t+2}) - V_\theta(s_{t+1})
$$

其中：

1. $\delta_t^V$：是时间步 $t$ 的优势函数，表示当前时刻$t$的即时奖励 $r_t$ 加上下一个状态的折扣价值 $\gamma \cdot V_\theta(s_{t+1})$ 减去当前状态的估计价值 $V_\theta(s_t)$。
2. $\delta_{t+1}^V$：是时间步 $ t+1 $ 的优势函数，类似地表示在时刻 $ t+1 $ 获得的即时奖励 $ r_{t+1}$ 加上状态 $ s_{t+2} $ 的折扣价值 $ \gamma \cdot V_\theta(s_{t+2}) $ 减去状态 $ s_{t+1} $ 的价值估计 $ V_\theta(s_{t+1}) $。

则对优势函数进行多步采样可简写为：
$$
A_\theta^1(s_t, a) = \delta_t^V \\

A_\theta^2(s_t, a) = \delta_t^V + \gamma \delta_{t+1}^V \\

A_\theta^3(s_t, a) = \delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V
$$


那我们究竟要采样几步呢？介绍一下**广义优势估计GAE**（Generalized Advantage Estimation），小孩子才做选择，我（GAE）全都要。

$$
A_\theta^{\text{GAE}}(s_t, a) = (1 - \lambda) (A_\theta^1(s_t, a) + \lambda A_\theta^2(s_t, a) + \lambda^2 A_\theta^3(s_t, a) + \cdots)
$$

将上面定义好的$\delta_t^V$和$\delta_{t+1}^V$代入到GAE优势函数中：

$$
A_{\theta}^{GAE}(s_t, a) = (1 - \lambda)\left( A_{\theta}^{1} + \lambda A_{\theta}^{2} + \lambda^2 A_{\theta}^{3} + \dots \right) \\
A_{\theta}^{GAE}(s_t, a) = (1 - \lambda)\left( \delta_t^V + \lambda (\delta_t^V + \gamma \delta_{t+1}^V) + \lambda^2 (\delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V) + \dots \right) \\
A_{\theta}^{GAE}(s_t, a) = (1 - \lambda)\left( \delta_t^V(1 + \lambda + \lambda^2 + \dots) + \gamma \delta_{t+1}^V(\lambda + \lambda^2 + \dots) + \dots \right) \\
A_{\theta}^{GAE}(s_t, a) = (1 - \lambda)\left( \delta_t^V \frac{1}{1 - \lambda} + \gamma \delta_{t+1}^V \frac{\lambda}{1 - \lambda} + \gamma^2 \delta_{t+2}^V \frac{\lambda^2}{1 - \lambda} \dots \right) \\
A_{\theta}^{GAE}(s_t, a) = ( \delta_t^V + \gamma \lambda \delta_{t+1}^V  + \gamma^2 \lambda^2 \delta_{t+2}^V + \dots ) \\
$$

最终我们可以得到：

$$
A_\theta^{\text{GAE}}(s_t, a) = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V
$$

那我们的策略梯度的优化目标函数就变成了**最大化广义优势函数的期望**:
$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_\theta^{\text{GAE}}(s_t, a) \log P_{\theta}(a_n^t \mid s_n^t)
$$

## Proximal Policy Optimization (PPO) 邻近策略优化

PPO 是 OpenAI 提出的一种基于策略梯度的强化学习算法，它通过对策略梯度的优化，来提高策略的稳定性和收敛速度。PPO 算法的核心思想是在更新策略时，通过引入一个重要性采样比例，来限制策略更新的幅度，从而保证策略的稳定性。

PPO 算法的目标函数为：

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{\nabla P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}
$$

其中：
1. $\frac{1}{N} \sum_{n=1}^{N}$：对 $N$ 条轨迹（采样的样本）取平均值。这里的 $N$ 表示采样轨迹的总数，通过对多个轨迹求平均来估计梯度，以获得更稳定的更新。

2. $\sum_{t=1}^{T_n}$：对每条轨迹 $n$ 中的 $T_n$ 个时间步求和，表示对单条轨迹中的所有时间步的累积。

3. $A_{\theta'}^{GAE}(s_n^t, a_n^t)$：广义优势估计（Generalized Advantage Estimation, GAE），由参数 $\theta'$ 估计，用于计算在状态 $s_n^t$ 下采取动作 $a_n^t$ 的优势。

4. $\frac{\nabla P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}$：表示策略的梯度，其中分母 $P_{\theta'}(a_n^t | s_n^t)$ 是旧策略（或目标策略），分子 $\nabla P_\theta(a_n^t | s_n^t)$ 是新策略的梯度。这个比值反映了新旧策略在同一状态-动作对上的相对概率密度，利用这一比值来更新策略参数 $\theta$。

整个公式的作用是通过优势估计来计算策略梯度，以优化策略参数，使得策略倾向于选择优势更高的动作，从而提升策略性能。GAE 可以有效降低方差，使得策略优化过程更加稳定和高效。

还是将loss函数取负号，转化为最小化问题，我们可以得到：

$$
loss = - \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{\nabla P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}
$$

具体来说，PPO 算法主要包括两个关键的技术：Adaptive KL Penalty Coefficient 和 Clipped Surrogate Objective。

PPO-惩罚（PPO-Penalty）用拉格朗日乘数法直接将 KL 散度的限制放进了目标函数中，这就变成了一个无约束的优化问题，在迭代的过程中不断更新 KL 散度前的系数。即：

$$
Loss_{kl} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)} + \beta KL(P_\theta, P_{\theta'})
$$

其中：

- $\beta KL(P_\theta, P_{\theta'})$：这是KL散度项，用于限制新旧策略之间的距离，其中 $KL(P_\theta, P_{\theta'})$ 表示策略$P_\theta$和旧策略$P_{\theta'}$之间的KL散度。超参数$\beta$控制KL散度项的权重，从而调节新旧策略之间的差异，防止策略更新过大导致不稳定。

整个PPO-KL损失函数的目的是通过限制新旧策略的差异（使用KL散度项）来优化策略，使其更稳定地朝着优势更高的方向进行更新。PPO的这种约束策略更新的方法相比于其他策略优化方法更为稳定且有效。

PPO截断（PPO-Clipped）是 PPO 的另一种变体，它通过对比新旧策略的比值，来限制策略更新的幅度，从而保证策略的稳定性。具体来说，PPO-Clipped 的目标函数为：

$$
Loss_{clip} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \min \left( A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}, \, \text{clip} \left( \frac{P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}, 1 - \epsilon, 1 + \epsilon \right) A_{\theta'}^{GAE}(s_n^t, a_n^t) \right)
$$

- $\text{clip} \left( \frac{P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}, 1 - \epsilon, 1 + \epsilon \right)$：裁剪函数，将概率比裁剪到 $[1 - \epsilon, 1 + \epsilon]$ 区间，防止策略的更新步长过大。这里$\epsilon$ 是一个超参数，控制裁剪的范围。

- $\min(\cdot, \cdot)$：在未裁剪的概率比项和裁剪后的项之间取最小值。这一操作的目的在于限制策略更新幅度，以防止策略偏离旧策略过远，从而导致不稳定的学习过程。

整个PPO-clip损失函数的作用是通过裁剪操作约束策略的变化幅度，使策略更新不会过于激进。这种方式相比于传统策略梯度方法更为稳定，并且在优化过程中能够有效平衡探索和利用。PPO2 的这种裁剪机制是其成功的关键，广泛用于实际的强化学习应用中。

好了，如果你坚持看到了这里，那想必你已经差不多掌握了强化学习的基本思想和PPO算法的基本思想。接下来你可以将PPO应用到大模型的训练中啦！

**参考文献**

1. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. [动手学强化学习](https://hrl.boyuai.com/chapter/1/%E5%88%9D%E6%8E%A2%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/)
3. [零基础学习强化学习算法：ppo](https://www.bilibili.com/video/BV1iz421h7gb/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=c102de6ffc75a54d6576f9fdc931e08a)