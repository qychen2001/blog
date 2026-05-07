---
title: 'DPO 的核心秘密：从 RLHF 目标直接推导出“闭式最优解”'
description: 'DPO 里最优雅、最重要的一个数学洞见：KL 正则化 RLHF 目标的闭式关系。这个推导正是 DPO 能“跳过 Reward Model + PPO”两步，直接用人类偏好数据端到端训练策略模型的根本原因。'
date: 2025-06-18 00:00:00+0000
math: true
categories:
    - '科研'
tags:
    - 'LLM'
    - '强化学习'
---

## 先回顾：传统 RLHF 在优化什么？

RLHF（Reinforcement Learning from Human Feedback）的核心目标是让大模型生成的回答 **更符合人类偏好**，同时 **不要偏离原始 SFT 模型太远**（避免幻觉、胡说八道）。

数学上，这个目标通常写成：

$$
J(\pi) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot|x)} \Big[ r(x,y) \Big] - \beta \, \mathrm{KL}\big(\pi(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x)\big)
$$

- 第一项：期望奖励最大化（$r(x,y)$ 是 Reward Model 给出的分数）。
- 第二项：KL 散度惩罚，强迫新策略 $\pi$ 不能离参考模型（通常是 SFT 后的模型）$\pi_{\text{ref}}$ 太远。
- $\beta$ 是超参数，控制惩罚强度（$\beta$ 越大，模型越“保守”）。

传统做法是用 PPO（或其它 RL 算法）去最大化这个 $J(\pi)$。但 PPO 训练不稳定、采样开销大，还需要单独训练一个 Reward Model。**DPO 的天才之处在于：它发现这个目标其实有解析解！**

## 直观理解：最优策略长什么样？

想象一下：对于一个固定的问题 $x$，我们希望生成的回答 $y$ 既得分高（$r(x,y)$ 大），又不能离 $\pi_{\text{ref}}$ 太远。

最优策略应该怎么做？  
- 对奖励高的回答，**指数级地提高**它的概率；  
- 对奖励低的回答，**指数级地压低**它的概率；  
- 同时用 $\pi_{\text{ref}}$ 做“锚点”保持稳定性。

这听起来像 softmax，但奖励被指数放大了！  
数学上，我们猜测最优策略的形式是：

$$
\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \cdot \exp\left( \frac{r(x,y)}{\beta} \right)
$$

归一化之后就得到**闭式关系**（closed-form relation）：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y|x) \, \exp\left( \frac{r(x,y)}{\beta} \right)
$$

其中 $Z(x)$ 是归一化常数（partition function）：

$$
Z(x) = \sum_{y} \pi_{\text{ref}}(y|x) \exp\left( \frac{r(x,y)}{\beta} \right)
$$

（如果是连续 token 分布，就换成积分。）

**一句话直觉**：最优策略 = 参考模型 × “奖励指数放大器”。$\beta$ 控制放大器的强度。

## 严谨推导：为什么这个形式是最优的？

现在我们来证明上面的猜想是**唯一**的最优解。固定一个 $x$，目标简化为最大化内层期望：

$$
\max_{\pi(\cdot|x)} \ \mathbb{E}_{y \sim \pi} \left[ r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

这是一个带约束的优化问题（$\sum_y \pi(y|x) = 1$）。我们构造 Lagrangian：

$$
L(\pi, \lambda) = \mathbb{E}_{y\sim\pi} \Big[ r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \Big] + \lambda \left(1 - \sum_y \pi(y|x)\right)
$$

对 $\pi(y|x)$ 求变分导数（functional derivative）并令其为零：

$$
\frac{\partial L}{\partial \pi(y|x)} = r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \beta - \lambda = 0
$$

解得：

$$
\log \pi(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{r(x,y)}{\beta} - 1 - \frac{\lambda}{\beta}
$$

两边取指数并归一化（$\lambda$ 由归一化条件决定），正好得到我们之前的闭式关系：

$$
\boxed{\pi^*(y|x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y|x) \, \exp\left( \frac{r(x,y)}{\beta} \right)}
$$

这个推导其实是指数族分布的经典结果：在 KL 正则化下，**最优分布一定是参考分布乘以指数加权的似然**。DPO 只是把它应用到了语言模型上。

## 从最优策略反推奖励函数

现在我们把闭式关系“反过来用”。对两边取对数：

$$
\log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} = \frac{r(x,y)}{\beta} - \log Z(x)
$$

两边乘 $\beta$ 得到：

$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

关键观察来了：**$\beta \log Z(x)$ 只依赖于输入 $x$，与具体回答 $y$ 无关**。

在人类偏好数据里，我们总是成对比较两个回答（chosen $y_w$ 和 rejected $y_l$）。此时 $Z(x)$ 在两个回答上**完全抵消**：

$$
r(x,y_w) - r(x,y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

这意味着：**我们可以直接把策略模型 $\pi_\theta$ 本身当成一个隐式的 Reward Model**！

定义隐式奖励：

$$
r_\theta(x,y) \triangleq \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

## 自然导出 DPO 损失函数

有了这个等价关系，DPO 就水到渠成了。我们把隐式奖励代入 Bradley-Terry 偏好模型（人类更喜欢 $y_w$ 胜过 $y_l$ 的概率）：

$$
P(y_w \succ y_l | x) = \sigma \big( r(x,y_w) - r(x,y_l) \big)
$$

代入隐式奖励，得到 DPO 的最终损失（负对数似然）：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

一句话总结：**DPO 直接让策略提高 chosen 相对 reference 的概率，同时压低 rejected 相对 reference 的概率**，整个过程只用一个模型、一次 offline 训练，干净、稳定、高效。