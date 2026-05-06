---
title: '从 PPO/DPO 到 GRPO 及其变体：LLM 后训练方法梳理'
description: '本文从统一的 RLHF 目标出发，梳理 PPO、DPO 与 GRPO 在 LLM 后训练中的技术路径、目标函数与工程取舍'
date: 2025-06-15 00:00:00+0000
math: true
categories:
    - '科研'
tags:
    - 'LLM'
    - '强化学习'
---


## 引言：后训练真正优化的是什么？

大语言模型的训练通常可以粗略分成两个阶段：预训练和后训练。预训练让模型从大规模语料中学习语言、知识与基本推理模式；后训练则试图回答另一个问题：**在已经会生成文本的基础上，如何让模型更符合人类偏好，或者更擅长完成某类任务？**

在今天的 LLM 训练语境里，后训练不只是简单的监督微调。SFT 可以让模型模仿高质量答案，但它本质上仍然是在做行为克隆：给定输入，学习参考答案的分布。如果我们关心的是“哪个回答更好”“哪个推理更可靠”“哪个答案更符合人类偏好”，那么仅仅模仿一个固定答案就不够了。这也是 RLHF、DPO、GRPO 等方法出现的背景。

这篇文章想梳理的不是“强化学习是什么”，而是下面这条技术主线：

> PPO 是如何把偏好信号转化成在线策略优化的；DPO 为什么可以绕过显式奖励模型和 RL 过程；GRPO 又为什么在推理模型训练中重新回到在线采样，但去掉了 PPO 中昂贵的价值模型。

因此，本文会尽量减少类比和科普式解释，直接从目标函数、训练数据和工程取舍出发，讨论 PPO、DPO 和 GRPO 三者之间的关系。

## 从 RLHF 的统一目标开始

无论是 PPO、DPO 还是 GRPO，它们都可以看作是在回答同一个问题：给定一个输入 $x$，我们希望训练后的策略模型 $\pi_\theta(y|x)$ 生成更高质量的输出 $y$，同时不要偏离原始参考模型 $\pi_{\text{ref}}(y|x)$ 太远。

一个常见的 KL 正则化 RLHF 目标可以写成：

$$
\max_\pi \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot|x)} \left[ r(x,y) \right] - \beta \, \mathbb{D}_{\mathrm{KL}}\left( \pi(y|x) \, \|\, \pi_{\text{ref}}(y|x) \right)
$$

其中：

- $\pi_\theta$ 是当前要训练的策略模型；
- $\pi_{\text{ref}}$ 通常是 SFT 模型或训练前的旧策略，用来约束模型不要偏离太远；
- $r(x,y)$ 是奖励函数，可以来自奖励模型，也可以来自规则打分；
- $\beta$ 控制奖励最大化和 KL 约束之间的权衡。

这个目标函数非常关键。PPO、DPO 和 GRPO 的差异，不在于它们想优化的方向完全不同，而在于它们对这个目标采取了不同的实现路径：

| 方法 | 是否需要奖励模型 | 是否需要在线采样 | 是否需要价值模型 / critic | 核心思想 |
|---|---:|---:|---:|---|
| PPO | 通常需要 | 需要 | 需要 | 用 actor-critic 估计优势并进行稳定的策略更新 |
| DPO | 不需要显式奖励模型 | 不需要 | 不需要 | 把偏好优化改写成二分类形式，直接优化 chosen/rejected 对 |
| GRPO | 可用奖励模型或规则奖励 | 需要 | 不需要 | 用同一 prompt 下的一组回答估计相对优势，替代 critic |

下面依次展开。

## 奖励模型：把偏好数据变成标量奖励

经典 RLHF 的第一步通常是训练奖励模型。训练数据不是单个标准答案，而是偏好对：

$$
(x, y_w, y_l),
$$

其中 $y_w$ 表示被人类或某种标注机制认为更好的回答，$y_l$ 表示较差的回答。奖励模型 $r_\phi(x,y)$ 的目标是给更好的回答更高分。

常用的 Bradley-Terry 形式可以写成：

$$
P(y_w \succ y_l | x) = \sigma\left(r_\phi(x,y_w) - r_\phi(x,y_l)\right),
$$

对应的损失函数为：

$$
\mathcal{L}_{\text{RM}}(\phi)
= - \mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}}
\left[\log \sigma\left(r_\phi(x,y_w) - r_\phi(x,y_l)\right)\right].
$$

训练完成后，奖励模型会被冻结，作为后续策略优化的打分器。PPO 和 GRPO 都可以使用这个奖励模型；而 DPO 的一个关键变化是：它不再显式训练这个奖励模型，而是把奖励模型隐含进策略模型本身。

## PPO：显式奖励模型 + 价值模型 + 在线策略优化

PPO 是早期 LLM RLHF 中非常重要的一条路线。它的基本结构是 actor-critic：

- **actor / policy model**：当前训练的语言模型 $\pi_\theta$；
- **reference model**：参考模型 $\pi_{\text{ref}}$，用于 KL 约束；
- **reward model**：冻结的奖励模型 $r_\phi$，用于给完整回答打分；
- **value model / critic**：价值模型 $V_\psi$，用于估计当前状态的期望回报，从而计算 advantage。

在 LLM 场景中，一个 prompt $x$ 对应一条生成轨迹：

$$
y = (a_1, a_2, \dots, a_T),
$$

其中每个 token $a_t$ 都可以看成一步 action，状态 $s_t$ 则是 prompt 加上此前已经生成的 token。

### PPO 的训练流程

PPO 在 LLM 后训练中的流程大致如下：

1. 从 prompt 数据集中采样一批输入 $x$；
2. 用当前策略 $\pi_\theta$ 生成回答 $y$；
3. 用奖励模型 $r_\phi(x,y)$ 给完整回答打分；
4. 额外计算当前策略相对参考策略的 KL 惩罚；
5. 用价值模型 $V_\psi(s_t)$ 估计每个 token 位置的 advantage；
6. 用 PPO clipped objective 更新策略模型；
7. 同时更新价值模型，使它更好地预测未来回报。

这里最重要的量是 advantage。PPO 并不是简单地说“这条回答 reward 高，所以整条回答都应该被强化”，而是希望知道：在生成过程的每一个 token 位置，当前模型选择的这个 token 相对于它在该状态下的平均表现，到底是更好还是更差。

形式上，advantage 定义为：

$$
A_t = Q(s_t,a_t) - V(s_t).
$$

这里的 $s_t$ 表示当前生成状态，也就是 prompt 加上已经生成出来的前缀；$a_t$ 表示当前位置生成的 token。$Q(s_t,a_t)$ 表示在状态 $s_t$ 下选择 token $a_t$ 之后，未来整条生成所能得到的期望回报；$V(s_t)$ 表示在状态 $s_t$ 下，按照当前策略继续生成所能得到的平均期望回报。

因此，$A_t$ 衡量的不是“这个 token 的绝对好坏”，而是：

> 在同一个上下文状态下，选择这个 token 是否比当前策略的平均选择更好。

如果 $A_t > 0$，说明这个 token 选择比当前策略的平均水平更好，训练时应该提高它的概率；如果 $A_t < 0$，说明这个 token 选择低于平均水平，训练时应该降低它的概率。

问题在于，真实的 $Q(s_t,a_t)$ 很难直接获得。对于 LLM 来说，奖励模型通常只会在完整回答生成结束后给一个 sequence-level reward，例如“这整段回答得分为 0.8”。但 PPO 的策略更新发生在 token 级别：模型需要知道第 1 个 token、第 2 个 token、一直到第 $T$ 个 token 分别应该被增强还是削弱。于是就需要一个方法，把最终的 sequence-level reward 转化为每个 token 位置上的训练信号。

这就是价值模型 $V_\psi(s_t)$ 的作用。它负责预测：给定当前已经生成到一半的状态 $s_t$，如果模型继续生成下去，最终大概能得到多少回报。换句话说，value model 是一个对“当前前缀未来质量”的估计器。

有了价值模型之后，就可以定义 temporal-difference error，也就是 TD error：

$$
\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t).
$$

这个式子的含义是：

> 生成 token $a_t$ 之后，模型对未来回报的估计有没有变好。

其中，$V_\psi(s_t)$ 是生成当前 token 之前，对最终回报的预测；$V_\psi(s_{t+1})$ 是生成当前 token 之后，对最终回报的预测；$r_t$ 是当前位置获得的即时奖励；$\gamma$ 是折扣因子。

在普通强化学习中，每一步 action 都可能获得即时奖励；但在 LLM 的 RLHF 里，主要奖励通常来自完整回答结束之后的奖励模型打分。因此很多中间 token 的 $r_t$ 可能接近 0，而最后一个位置才拿到主要 reward。当然，实际实现中还常常会把 KL penalty 作为 token-level reward 加进去，用来约束当前模型不要偏离 reference model 太远。

如果某个 token 生成之后，价值模型对未来最终回报的预测上升了，也就是 $V_\psi(s_{t+1}) > V_\psi(s_t)$，那么 $\delta_t$ 会偏正，说明这个 token 可能把生成带向了更好的方向；反之，如果生成这个 token 后，未来回报预测下降，$\delta_t$ 就会偏负。

不过，只看一步 TD error 仍然不够稳定。一个 token 的好坏不一定会在下一步立刻体现出来，它可能要经过后面很多 token 才能反映到最终答案质量上。例如在数学推理中，前面某一步推理是否正确，可能要到最终答案才看得出来。

因此，PPO 通常使用 GAE（Generalized Advantage Estimation）来把多步 TD error 累积起来：

$$
A_t^{\text{GAE}}
= \sum_{k=0}^{T-t-1} (\gamma \lambda)^k \delta_{t+k}.
$$

这个式子可以理解成：当前位置 $t$ 的 advantage，不只看当前一步的 TD error $\delta_t$，还会继续看后续若干步的 TD error：

$$
\delta_t,\quad \delta_{t+1},\quad \delta_{t+2},\quad \dots
$$

越靠后的 TD error，对当前位置的影响越小，由 $(\gamma\lambda)^k$ 控制衰减。这里的 $\lambda$ 是一个偏差和方差之间的折中参数：

- 当 $\lambda = 0$ 时，GAE 只看一步 TD error，方差较低，但偏差较大；
- 当 $\lambda = 1$ 时，GAE 更接近使用完整轨迹回报，偏差较低，但方差较大；
- 实际训练中通常取二者之间的值，让 advantage 估计既不过于短视，也不过于依赖高方差的完整回报。

所以，GAE 的作用可以概括为：

> 用价值模型提供 baseline，用多步 TD error 把完整回答的奖励信号向前传播，最终为每个 token 得到一个相对稳定的 advantage 估计。

这也是 PPO 在 LLM 后训练中需要 critic / value model 的核心原因。奖励模型只告诉我们“整条回答好不好”，而 PPO 需要知道“生成这条回答时，每个 token 的选择应该被强化还是削弱”。价值模型和 GAE 正是用来完成这个 credit assignment 的。

### PPO 的 clipped objective

PPO 的策略更新不是直接最大化 reward，而是使用一个被裁剪的概率比率来限制更新幅度。定义：

$$
\rho_t(\theta) =
\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}.
$$

PPO 的 clipped surrogate objective 为：

$$
\mathcal{L}_{\text{clip}}(\theta) = \mathbb{E}_t \left[\min \left(\rho_t(\theta) A_t,\operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right].
$$

这个裁剪项的作用是限制策略更新幅度，避免模型因为某一批 reward 信号过强而发生过大的分布漂移。

在 RLHF 中，最终的 PPO 训练目标通常还会加入 KL 惩罚和 value loss：

$$
\mathcal{L}_{\text{PPO}} = \mathcal{L}_{\text{clip}} - \beta \, \mathbb{D}_{\mathrm{KL}}(\pi_\theta \, \| \, \pi_{\text{ref}}) - c_v \mathcal{L}_{\text{value}} + c_e \, \mathcal{H}(\pi_\theta)
$$
其中 value loss 通常是：

$$
\mathcal{L}_{\text{value}}
= \mathbb{E}_t \left[\left(V_\psi(s_t) - R_t\right)^2\right].
$$

### PPO 的问题：稳定但昂贵

PPO 的优点是通用、稳定，并且可以直接优化在线采样得到的 reward。只要我们能定义奖励函数，PPO 就可以尝试把策略往高 reward 区域推。

但它的工程成本也很明显：

第一，PPO 需要维护多个模型。除了当前策略模型，还需要 reference model、reward model 和 value model。对于大语言模型来说，value model 往往也是一个体量接近策略模型的网络，这会显著增加显存和训练复杂度。

第二，PPO 是在线 RL。训练时要不断采样、打分、估计 advantage、更新策略，再继续采样。这比普通监督微调更复杂，也更容易受到 reward hacking、KL 控制、采样温度和 advantage 估计质量的影响。

第三，PPO 的 token-level credit assignment 并不自然。奖励模型通常只对完整回答给一个标量分数，但策略更新发生在 token 级别，因此需要 critic 把最终奖励转化成每个 token 的优势估计。

这些问题直接引出了两条不同方向的简化路线：DPO 和 GRPO。

## DPO：把 RLHF 改写成直接偏好分类

DPO（Direct Preference Optimization）的出发点非常不同。它并不是在 PPO 的基础上继续改进 advantage 估计，而是重新审视 KL 正则化 RLHF 目标本身。

回到前面的目标：

$$
\max_\pi \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot|x)} \left[ r(x,y) \right] - \beta \, \mathbb{D}_{\mathrm{KL}}\left( \pi(y|x) \, \|\, \pi_{\text{ref}}(y|x) \right)
$$

DPO 的关键观察是：在这个目标下，最优策略和奖励函数之间存在一个 closed-form 关系。忽略只与 $x$ 有关的归一化项，可以得到：

$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + C(x).
$$

这意味着：如果我们把策略模型本身看成隐式奖励模型，那么就不一定需要先训练一个显式的 reward model，再用 PPO 去优化它。我们可以直接用偏好数据训练 $\pi_\theta$，让它提高 chosen response 相对 reference model 的概率，同时降低 rejected response 的相对概率。

### DPO 的目标函数

给定偏好数据 $(x, y_w, y_l)$，DPO 的损失函数为：

$$
\mathcal{L}_{\text{DPO}}(\theta) 
= -\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}} 
\left[ 
  \log \sigma \Bigl( 
    \beta \Bigl( 
      \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} 
      - 
      \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} 
    \Bigr) 
  \Bigr) 
\right].
$$

这个式子可以拆开理解。DPO 比较的不是 $\pi_\theta(y_w|x)$ 和 $\pi_\theta(y_l|x)$ 的绝对概率，而是它们相对于参考模型 $\pi_{\text{ref}}$ 的概率提升：

$$
\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}.
$$

如果 chosen response 相对于 reference model 的提升幅度大于 rejected response，那么损失会降低；反之，损失会上升。

这就是 DPO 和普通 SFT 的关键区别。SFT 只会提高目标答案的似然；DPO 同时使用 chosen 和 rejected，并且通过 reference model 保留了一个隐式 KL 约束。

### DPO 为什么能省掉 reward model 和 PPO？

从工程角度看，DPO 的吸引力非常明显：

- 不需要训练显式奖励模型；
- 不需要在线 rollout；
- 不需要 value model；
- 训练形式接近普通监督学习，只需要偏好对数据；
- 实现上主要依赖 chosen/rejected 的 log probability。

因此，DPO 可以看作是把“奖励建模 + RL 优化”折叠成了一个 pairwise classification loss。它仍然在优化偏好，但不再显式执行强化学习过程。

不过，DPO 的简化也带来了边界。

首先，DPO 是离线方法。它依赖已有的偏好数据，如果数据覆盖不足，模型很难通过在线探索发现新的高质量行为。PPO 和 GRPO 可以在训练过程中不断采样当前模型的输出，再用奖励函数评价这些输出；DPO 通常不能做到这一点。

其次，DPO 对偏好数据质量非常敏感。因为它直接把 chosen/rejected 对作为训练信号，如果偏好对噪声很大，或者 chosen 与 rejected 的差异并不稳定，模型会直接学习到这些偏差。

第三，DPO 更适合对齐类任务和已有偏好数据充分的场景。对于数学、代码、复杂推理这类可以通过规则验证的任务，在线采样 + 规则奖励往往更有优势，因为模型可以生成大量候选解，再通过正确性信号筛选和强化。

这也是为什么在推理模型训练中，GRPO 重新变得重要。

## GRPO：保留在线 RL，去掉价值模型

GRPO（Group Relative Policy Optimization）可以看作是 PPO 的另一种简化方向。它没有像 DPO 那样完全放弃在线 RL，而是保留了“采样—打分—更新”的过程；但它去掉了 PPO 中最重的 critic / value model。

它的核心问题是：如果 PPO 中的 value model 主要是为了提供 baseline、降低 advantage 估计的方差，那么是否可以不用额外训练一个 critic，而是直接从同一个 prompt 的一组回答中构造 baseline？

GRPO 的答案是：可以。

### 组内相对优势

对于每个 prompt $x$，GRPO 不只采样一个回答，而是从当前策略中采样一组回答：

$$
\{y_1, y_2, \dots, y_G\} \sim \pi_{\theta_{\text{old}}}(\cdot|x).
$$

然后对每个回答计算奖励：

$$
\{r_1, r_2, \dots, r_G\}.
$$

奖励可以来自奖励模型，也可以来自规则函数。例如数学任务中可以根据最终答案是否正确给 reward；代码任务中可以根据测试用例是否通过给 reward。

接下来，GRPO 用组内归一化的方式估计 advantage：

$$
A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}.
$$

这个式子就是 GRPO 的核心。它不再问“这个 token 相对于 critic 预测值好多少”，而是问：

> 对于同一个 prompt，当前这条回答在同组候选答案中相对好多少？

因此，GRPO 的 baseline 是 prompt-specific 的组内平均奖励，而不是一个额外训练出来的 value function。

### GRPO 的目标函数

GRPO 仍然沿用了 PPO 的 clipped objective。对于第 $i$ 个回答中的第 $t$ 个 token，定义概率比率：

$$
\rho_{i,t}(\theta) 
= \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}.
$$

GRPO 的策略目标可以写成：

$$
\begin{aligned}
\mathcal{L}_{\text{GRPO}}(\theta) 
&= \mathbb{E}_{i,t} \Bigl[ 
  \min\Bigl( 
    \rho_{i,t}(\theta) A_i,\ 
    \operatorname{clip}\bigl(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon\bigr) A_i 
  \Bigr) 
\Bigr] \\
&\quad - \beta \, \mathbb{D}_{\mathrm{KL}}\bigl(\pi_\theta \,\|\, \pi_{\text{ref}}\bigr).
\end{aligned}
$$

这里的 $A_i$ 是 sequence-level advantage，但会被应用到该回答的 token-level log probability 上。也就是说，如果某个回答在组内 reward 更高，那么这条回答中所有 token 的生成概率整体上都会被提升；如果 reward 更低，则对应 token 概率会被压低。

### GRPO 相比 PPO 简化在哪里？

GRPO 最大的变化是去掉了 value model。

在 PPO 中，我们需要训练 $V_\psi(s_t)$ 来估计每个状态的期望回报；在 GRPO 中，同一 prompt 下的多个回答已经提供了一个局部比较集合，因此可以用组内均值作为 baseline，用组内标准差做归一化。

这带来几个直接好处：

第一，显存和计算成本更低。去掉 critic 以后，不需要再维护一个额外的价值网络，也不需要训练 value loss。

第二，训练信号更适合可验证任务。对于数学、代码这类任务，reward 往往可以由规则直接给出。此时最自然的问题不是“这个 token 的价值是多少”，而是“同一个题目下，哪些完整回答是正确的”。GRPO 的组内比较正好贴合这一点。

第三，GRPO 仍然保留在线探索能力。与 DPO 不同，GRPO 可以不断从当前策略采样新回答，然后用规则或奖励模型打分。这对于推理能力的自我提升非常重要。

当然，GRPO 也不是没有代价。它需要对每个 prompt 采样多个回答，因此 rollout 成本会增加；如果组大小太小，组内 baseline 的估计会不稳定；如果 reward 过于稀疏，比如一组回答全错或全对，advantage 信号也会变弱。

## PPO、DPO、GRPO 的关系

现在可以把三者放在同一条线索上重新看。

### PPO：最完整，但最重

PPO 是最接近经典 RLHF 目标的实现方式：先训练奖励模型，再通过在线 RL 最大化奖励，同时用 KL 约束防止策略偏离参考模型。它的优势是通用、直接、可以在线探索；缺点是系统复杂，需要 reward model、reference model、value model 和稳定的 RL 训练管线。

### DPO：最简洁，但更依赖离线偏好数据

DPO 直接把 KL 正则化 RLHF 目标改写成偏好对上的分类损失。它不需要 reward model，也不需要 PPO rollout，因此实现简单、训练稳定，非常适合已有高质量偏好数据的对齐场景。

但 DPO 不主动探索。它主要学习数据集中已经体现出来的偏好关系。因此，对于需要大量试错、采样和可验证反馈的复杂推理任务，DPO 往往不是最终答案，而更像是一个高效的偏好微调工具。

### GRPO：面向推理训练的轻量在线 RL

GRPO 保留了 PPO 的在线 RL 框架，但用组内相对奖励替代 critic。它尤其适合 reward 可自动验证的任务，例如数学、代码和结构化推理。相比 PPO，它更轻量；相比 DPO，它保留了在线采样和探索能力。

可以用一句话概括：

> PPO 是“显式奖励模型 + critic + 在线 RL”；DPO 是“无显式奖励模型的离线偏好优化”；GRPO 是“无 critic 的在线相对奖励优化”。

## 为什么推理模型更偏向 GRPO？

如果目标是一般偏好对齐，例如让回答更有帮助、更安全、更符合人类口味，那么 DPO 是一个非常自然的选择，因为这类偏好往往可以通过 chosen/rejected 数据表达。

但推理任务有一个不同点：它们经常存在可验证答案。

数学题可以检查最终答案；代码题可以跑测试；某些工具调用任务可以检查执行结果。这类任务不一定需要人类主观偏好，而更需要模型在大规模采样中发现“哪些解法真的能通向正确答案”。

在这种场景下，在线 RL 的价值会重新凸显。模型可以对同一道题生成多个候选推理轨迹，奖励函数只需要判断最终结果或格式是否正确。GRPO 通过组内比较，把这些候选轨迹转化成相对优势信号，进而强化高 reward 的生成模式。

这也是为什么 GRPO 在推理模型训练中很有吸引力：它没有 PPO 那么重，又比 DPO 更适合在线探索。

## 小结

PPO、DPO 和 GRPO 并不是三种彼此割裂的方法，而是围绕同一个偏好优化目标形成的三种工程路径。

PPO 的思路最传统：显式训练奖励模型，再用 actor-critic 做在线策略优化。它的表达能力强，但系统复杂。

DPO 的思路最优雅：利用 KL 正则化 RLHF 目标的解析形式，把奖励建模和 RL 优化折叠成一个偏好分类损失。它简单稳定，但依赖离线偏好数据，缺少在线探索。

GRPO 的思路最实用：保留在线采样和奖励优化，但去掉 PPO 中昂贵的价值模型，用同一 prompt 下多个回答的组内相对表现来估计优势。它尤其适合数学、代码等可验证推理任务。

因此，如果我们从后训练方法的演化角度看，这条线索并不是简单的“PPO 被 DPO 替代，DPO 又被 GRPO 替代”。更准确地说，它们分别服务于不同约束下的偏好优化：

- 当我们需要通用的在线奖励优化时，PPO 仍然是基础方案；
- 当我们有高质量偏好对并希望简单稳定训练时，DPO 很合适；
- 当任务可验证、需要大量在线采样提升推理能力时，GRPO 更有优势。

这也是 LLM 后训练方法演化中最值得注意的一点：算法并不是越复杂越好，而是要和奖励来源、数据形态、任务类型以及工程成本匹配。