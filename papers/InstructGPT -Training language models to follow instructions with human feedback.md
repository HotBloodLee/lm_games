# InstructGPT -Training language models to follow instructions with human feedback

# 一、价值

它标志着大语言模型（LLM）从单纯预测下一个词的“通用模型”向能够**理解并遵循人类指令的“助手型模型”**的关键转变。其核心思想和方法论后来直接催生了ChatGPT

**成本效益高**：RLHF对齐的成本远低于从头预训练一个大模型，但效果提升显著。

**可泛化性**：模型能将“遵循指令”的能力泛化到未监督的领域。

**低对齐税**：通过技术手段可以最小化对齐带来的性能损失，这有利于对齐技术的实际采用。

# 二、核心问题与动机

- **问题**：单纯扩大语言模型规模（如GPT-3）并不能让它们更好地理解和遵循用户的**真实意图**。这些模型经常产生不真实（捏造事实）、有毒（偏见、有害）或对用户无用的输出，即模型与用户“**未对齐**”。
- **目标**：开发一种方法，让语言模型在广泛的任务上**与人类意图对齐**，使其变得**有帮助、诚实、无害**。

# 核心方法：三阶段训练

这篇论文最重要的，就是这条三段式流程。

## 总流程

**Prompt 收集**→**人工写示范答案**→**SFT 监督微调**→**模型生成多个候选答案**→**人工排序偏好**→**训练奖励模型 RM**→**PPO 强化学习优化**→**得到 InstructGPT**

## 第一阶段：SFT（监督微调）

### 做什么

让人工标注员看到 prompt，然后直接写出一个“理想回答”。

得到的数据是：

- 输入：prompt
- 输出：人工写的高质量 answer

然后用这些数据对 GPT-3 做一次普通监督学习微调。

### 这一步的作用

先把模型从“通用语言模型”拉到“会按指令回答”的起点上。

### 为什么不能直接上强化学习

因为如果一开始就让模型靠 RL 自己探索，训练会很不稳定，成本也很高。SFT 相当于先给模型一个“像样的初始策略”。

你可以把它理解成：

> 先教会模型“像个助手一样回答”。
> 

### 数学上怎么训练

标准的 teacher forcing + 最大似然。

如果人工示范答案是y∗=(y1∗,...,yT∗)，那么损失函数就是：

$$
L_{S F T} \left(\right. \phi \left.\right) = - \mathbb{E}_{\left(\right. x , y^{*} \left.\right) sim D_{S F T}} \sum_{t = 1}^{T} log ⁡ \pi_{\phi} \left(\right. y_{t}^{*} \mid x , y_{< t}^{*} \left.\right)
$$

这就是经典的交叉熵 / NLL。

### 为什么说它是“稳定起点”

从 RL 角度看，SFT 的作用是：

### 第一，做 imitation learning / behavior cloning

SFT 让模型先逼近一个“专家策略”：

$$
\pi_{S F T} \approx \pi_{e x p e r t}
$$

虽然这个 expert 不是完美的，但至少让模型进入一个“回答像样”的区域。

### 第二，降低 RL 的探索难度

如果直接从一个普通预训练 LM 上做 RL，会有几个问题：

- 动作空间巨大（整个词表）
- 序列长，回报延迟
- 生成很容易跑到离谱区域
- 奖励模型也会被喂到大量离谱样本

SFT 后：

- 模型已经会基本 obey instruction
- RL 不需要“从零学会说人话”
- 只需要在一个合理分布附近“微调行为”

### 第三，SFT 后面还要当 reference model

后面 KL 惩罚要用到一个参考策略：

πref=πSFT

所以 SFT 不只是初始化，还相当于“行为锚点”。

### 1.5 直觉例子

比如 prompt 是：

> “用三句话解释 TCP 和 UDP 的区别。”
> 

预训练 GPT-3 可能会：

- 说很多，但不一定三句话
- 解释方向跑偏
- 输出格式不稳

SFT 后，模型至少学会：

- 这是一个 instruction
- 要遵守“三句话”
- 要给出类似“TCP 可靠、UDP 更快”的结构化回答

所以 SFT 的意义不是“最优”，而是：

> **先把模型拉到可控、可用、可继续优化的分布上。**
> 

## 第二阶段：奖励模型 RM（Reward Model）

### 做什么

对于同一个 prompt，不只生成一个答案，而是生成多个候选答案。

然后让标注员去比较：

- 哪个更好
- 哪个更差

不是简单打分，而是**排序/偏好比较**。

### 训练目标

把“人类更喜欢哪个回答”学成一个奖励函数：

- 输入：prompt + answer
- 输出：一个 reward 分数

它本质上是在学：

> **这个回答是否更符合人类偏好？**
> 

### 为什么这一步关键

因为“有帮助”“更真实”“更像人想要的回答”这些东西，很难手写规则。但人可以比较：**A 比 B 好**。

所以论文把“主观偏好”转成了可训练信号。

### 为什么需要 RM

SFT 有一个硬伤：

- 它只会模仿人工示范
- 但真实世界里，很多 prompt 没有唯一标准答案
- 而且“更好”通常是相对的，不是绝对的

比如这两个回答：

A. 正确但啰嗦B. 简洁、准确、结构清楚

你很难写一个“标准标签”说 B 就是 ground truth。但人类很容易说：**B 比 A 好。**

所以 RM 要学的是：

> 对同一个 prompt，哪个回答更受人类偏好。
> 

### RM 的数据怎么来

数据流程是：

1. 拿一个 prompt `x`
2. 让模型生成多个候选回答 y1,y2,...,yK
3. 让标注员排序，或者至少选出更好的那个
4. 把排序转成 pairwise preference 样本

论文相关资料里常见的数量级是：

- **约 33k 条 comparison / ranking 数据**
- 每个 prompt 通常生成 **4-9 个候选答案**

如果一个 prompt 有排序：

$y^{\left(\right. 1 \left.\right)} \succ y^{\left(\right. 2 \left.\right)} \succ y^{\left(\right. 3 \left.\right)} \succ y^{\left(\right. 4 \left.\right)}$

就可以拆成多个 pair：

- $y^{\left(\right. 1 \left.\right)} \succ y^{\left(\right. 2 \left.\right)}$
- $y^{\left(\right. 1 \left.\right)} \succ y^{\left(\right. 3 \left.\right)}$
- $y^{\left(\right. 1 \left.\right)} \succ y^{\left(\right. 4 \left.\right)}$
- $y^{\left(\right. 2 \left.\right)} \succ y^{\left(\right. 3 \left.\right)}$
- ...

### RM 的模型结构

RM 通常沿用 GPT 骨干，但把输出头换掉。

### 骨干

- 还是一个 decoder-only Transformer
- 输入是 prompt + completion 的拼接序列

### 输出

不像语言模型输出词表概率，RM 输出的是一个**标量 reward**：

$r_{\theta} \left(\right. x , y \left.\right)$

实现上可以写成：

$r_{\theta} \left(\right. x , y \left.\right) = w^{\top} h_{l a s t} + b$

其中：

- $h_{l a s t}$：最后一个 token 的隐藏状态
- $w , b$：一个线性 head

注意这点很重要：

> RM 不是逐 token 打分再求和，而是对整段回答给一个总分。
> 

论文/公开解读里经常强调：**奖励通常取最后一个 token 的表示做 sequence-level score。**

### RM 的数学原理：Bradley-Terry / pairwise preference learning

假设对同一个 prompt，标注员认为$y_{w}$比$y_{l}$更好（winner vs loser）。

那么模型假设：

$P \left(\right. y_{w} \succ y_{l} \mid x \left.\right) = \sigma \left(\right. r_{\theta} \left(\right. x , y_{w} \left.\right) - r_{\theta} \left(\right. x , y_{l} \left.\right) \left.\right)$

其中：

- $\sigma \left(\right. z \left.\right) = \frac{1}{1 + e^{- z}}$

这就是 Bradley-Terry 风格的偏好建模。

### 为什么这样合理

如果：

$r_{\theta} \left(\right. x , y_{w} \left.\right) \gg r_{\theta} \left(\right. x , y_{l} \left.\right)$

那么：

$P \left(\right. y_{w} \succ y_{l} \mid x \left.\right) \approx 1$

也就是模型认为 winner 被偏好的概率很高。

### 对应损失

最大化似然，等价于最小化：

$L_{R M} \left(\right. \theta \left.\right) = - \mathbb{E}_{\left(\right. x , y_{w} , y_{l} \left.\right) sim D_{R M}} log ⁡ \sigma \left(\right. r_{\theta} \left(\right. x , y_{w} \left.\right) - r_{\theta} \left(\right. x , y_{l} \left.\right) \left.\right)$

这就是论文里 RM 的核心数学形式。

### RM 为什么容易过拟合

因为 comparison 数据远少于预训练语料，而且偏好标签噪声很大：

- 标注员本身未必完全一致
- 排序样本规模有限
- 高分样本空间非常窄

所以公开资料里都强调：

- RM 很容易 overfit
- 往往训练得比较克制
- 会做 reward normalization / early stop

原因很简单：

> 一旦 RM 学偏，PPO 会把这个偏差放大。
> 

## 第三阶段：PPO 强化学习

### 做什么

现在有了：

- 一个初始模型（SFT）
- 一个会打分的奖励模型（RM）

就可以让模型自己生成回答，然后：

- 奖励模型给分
- 用 PPO 更新模型参数
- 让模型越来越倾向生成高分答案

### 为什么用 PPO

因为这里本质上是在优化一个“不可直接监督”的目标：不是固定标准答案，而是“人类更偏好的输出”。

要让模型真的更倾向输出高分回答，就得优化策略：

$\pi_{\phi} \left(\right. y \mid x \left.\right)$

这就是 PPO 阶段做的事。

### 把语言生成写成 RL

对于 token 生成过程：

- 状态：$s_{t} = \left(\right. x , y_{< t} \left.\right)$
- 动作：$a_{t} = y_{t}$
- 策略：$\pi_{\phi} \left(\right. a_{t} \mid s_{t} \left.\right)$

生成完一个完整回答y后，拿到 sequence-level reward：

$r_{\theta} \left(\right. x , y \left.\right)$

但是直接用这个终局奖励会很难训，所以论文实现里通常还会加入**per-token KL shaping**，后面我单独讲。

### PPO 之前的真正目标是什么

从论文角度，PPO 想优化的核心目标可以写成：

$J_{R L} \left(\right. \phi \left.\right) = \mathbb{E}_{x sim D , y sim \pi_{\phi}} \left[\right. r_{\theta} \left(\right. x , y \left.\right) - \beta log ⁡ \frac{\pi_{\phi} \left(\right. y \mid x \left.\right)}{\pi_{S F T} \left(\right. y \mid x \left.\right)} \left]\right.$

你可以把它理解成：

> **人类偏好奖励 - 偏离 SFT 的代价**
> 

这一步已经把：

- “更受偏好”
- “别跑太远”

同时写进目标函数了。

### 为什么不用普通 policy gradient，而用 PPO

如果你直接做 REINFORCE / vanilla policy gradient：

$\nabla J \left(\right. \phi \left.\right) = \mathbb{Eå} \left[\right. \nabla log ⁡ \pi_{\phi} \left(\right. y \mid x \left.\right) R \left]\right.$

问题很大：

- 方差高
- 更新不稳定
- 容易一步走太远
- 序列任务里特别容易崩

PPO 是 TRPO 的一个更实用近似，核心思想是：

> **每次更新要有，但不能太大。**
> 

### PPO 的核心数学形式

PPO 的 clipped objective：

$L_{P P O}^{c l i p} = \mathbb{E}_{t} \left[\right. min ⁡ \left(\right. \rho_{t} \left(\hat{A}\right)_{t} , \text{clip} \left(\right. \rho_{t} , 1 - \epsilon , 1 + \epsilon \left.\right) \left(\hat{A}\right)_{t} \left.\right) \left]\right.$

其中：

$\rho_{t} = \frac{\pi_{\phi} \left(\right. a_{t} \mid s_{t} \left.\right)}{\pi_{\phi_{o l d}} \left(\right. a_{t} \mid s_{t} \left.\right)}$

- $\pi_{\phi_{o l d}}$：旧策略
- $\left(\hat{A}\right)_{t}$：advantage estimate

### 直觉解释

如果新策略相对旧策略变化太大，ratio$\rho_{t}$会偏离 1 太多。PPO 就把这种更新“截住”，不让它过猛。

所以 PPO 的数学本质是：

> **一种带“保守更新”机制的策略梯度方法。**
> 

### 在语言模型里 advantage / value 是怎么来的

通常还要训练一个 value function：

$V_{\psi} \left(\right. s_{t} \left.\right) \approx \mathbb{E} \left[\right. R_{t} \mid s_{t} \left]\right.$

然后：

$\left(\hat{A}\right)_{t} = \left(\hat{R}\right)_{t} - V_{\psi} \left(\right. s_{t} \left.\right)$

这样可以减小方差。

### 在 LM 场景里

- `state` 是 prompt + 已生成前缀
- `return` 是该前缀继续生成后最终拿到的总 shaped reward
- `value head` 预测“从当前 token 状态出发，未来大概还能拿多少 reward”

所以 PPO 阶段实际不只是一个 policy model，通常还会有：

- **policy head**
- **value head**

有的实现共享 Transformer 骨干，有的分开。

# 论文里两个很重要的工程细节

## 1）KL 惩罚：别让模型训练跑飞

RL 很容易让模型为了刷高奖励而“变形”。

所以论文里加了**KL penalty**，约束新策略不要偏离 SFT 模型太远。

直觉上就是：

> 可以变好，但别变得不像正常语言模型了。
> 

## KL 惩罚写在哪里

论文核心目标里有这一项：

$- \beta log ⁡ \frac{\pi_{\phi} \left(\right. y \mid x \left.\right)}{\pi_{S F T} \left(\right. y \mid x \left.\right)}$

它等价于对新策略和参考策略之间的 KL 做惩罚。

更标准地说，你也可以把目标写成：

${max ⁡} \mathbb{E}_{y sim \pi_{\phi}} \left[\right. r_{\theta} \left(\right. x , y \left.\right) \left]\right. - \beta D_{K L} \left(\right. \pi_{\phi} \left(\right. \cdot \mid x \left.\right) \parallel \pi_{S F T} \left(\right. \cdot \mid x \left.\right) \left.\right)$

### 它的数学来源：拉格朗日松弛

其实 KL 惩罚不是拍脑袋加的，而是可以从一个**约束优化问题**推出来：

### 原问题

${max ⁡} \mathbb{E} \left[\right. r_{\theta} \left(\right. x , y \left.\right) \left]\right. \text{s}.\text{t}. \mathbb{E} \left[\right. D_{K L} \left(\right. \pi_{\phi} \parallel \pi_{S F T} \left.\right) \left]\right. \leq \delta$

意思是：

> 想提高 reward，但新策略不能离 SFT 太远。
> 

把这个约束问题写成拉格朗日形式，就得到：

${max ⁡} \mathbb{E} \left[\right. r_{\theta} \left(\right. x , y \left.\right) \left]\right. - \beta \mathbb{E} \left[\right. D_{K L} \left(\right. \pi_{\phi} \parallel \pi_{S F T} \left.\right) \left]\right.$

这就是 KL penalty 的理论来源。

### 序列级和 token 级怎么对应

因为：

$log ⁡ \pi_{\phi} \left(\right. y \mid x \left.\right) = \sum_{t = 1}^{T} log ⁡ \pi_{\phi} \left(\right. y_{t} \mid x , y_{< t} \left.\right)$

所以：

$log ⁡ \frac{\pi_{\phi} \left(\right. y \mid x \left.\right)}{\pi_{S F T} \left(\right. y \mid x \left.\right)} = \sum_{t = 1}^{T} log ⁡ \frac{\pi_{\phi} \left(\right. y_{t} \mid x , y_{< t} \left.\right)}{\pi_{S F T} \left(\right. y_{t} \mid x , y_{< t} \left.\right)}$

这意味着 KL 惩罚可以拆成 per-token 形式。

于是实现时常见做法是：

- 每个 token 都加一个 KL penalty
- 最后一个 token 再加上 RM 的终局 reward

所以 shaped reward 常写成：

$r_{t} = 
\begin{cases} 
-\beta \log \frac{\pi_{\phi} ( y_{t} \mid s_{t} )}{\pi_{SFT} ( y_{t} \mid s_{t} )}, & t < T \\ 
r_{\theta} ( x, y ) -\beta \log \frac{\pi_{\phi} ( y_{T} \mid s_{T} )}{\pi_{SFT} ( y_{T} \mid s_{T} )}, & t = T 
\end{cases}$

这就是为什么很多实现里说：

- 中间 token 是 non-score reward（主要来自 KL）
- 结尾 token 才叠加 RM score

### KL 惩罚 和 PPO clipping 不是一回事

这是技术面试里很加分的一点。

### PPO clipping 约束的是：

$\pi_{\phi} \text{vs } \pi_{\phi_{o l d}}$

也就是：

> **本次优化步长不要太大**
> 

它解决的是**优化稳定性**。

### KL penalty 约束的是：

$\pi_{\phi} \text{vs } \pi_{S F T}$

也就是：

> **最终行为不要偏离参考模型太远**
> 

它解决的是**行为锚定**和**reward hacking**。

一句话：

- **PPO clipping**：局部更新别太猛
- **KL penalty**：全局行为别跑偏

## 2）PPO-ptx：对齐的时候别把基本功练没了

论文发现，单纯 PPO 可能会让模型在一些通用 NLP 任务上退步。

这就是常说的**alignment tax（对齐税）**。

所以他们又混入了一部分预训练目标，形成**PPO-ptx**。

作用是：

- 一边让模型更符合人类偏好
- 一边尽量保住原本语言建模能力

这是非常关键的工程折中。

### 为什么会出现

因为 PPO 优化的目标太窄了：

$max ⁡ \mathbb{E} \left[\right. r_{\theta} \left(\right. x , y \left.\right) - \beta K L \left]\right.$

这个目标只关心：

- 奖励模型高不高
- 离 SFT 远不远

但它**不直接关心原始预训练分布**上的建模能力。

所以如果你一直沿着 reward 方向优化，模型可能会：

- 忘掉一些语言建模能力
- 风格变窄
- 分布变得更“服务于 reward model”

这本质上是一个**灾难性遗忘 + 目标偏置**问题。

### PPO-ptx 是怎么做的

论文采用**PPO-ptx**，ptx = pretraining mix。

做法就是在 PPO 更新时，混入一部分原始 LM 目标。

可以写成：

$J_{P P O - p t x} \left(\right. \phi \left.\right) = J_{P P O} \left(\right. \phi \left.\right) + \gamma \mathbb{E}_{x sim D_{p t x}} \left[\right. {\sum} log ⁡ \pi_{\phi} \left(\right. x_{t} \mid x_{< t} \left.\right) \left]\right.$

如果写成 loss，更常见是：

$L_{t o t a l} = L_{P P O} + \lambda L_{L M}$

其中：

$L_{L M} = - \mathbb{E}_{x sim D_{p t x}} {\sum} log ⁡ \pi_{\phi} \left(\right. x_{t} \mid x_{< t} \left.\right)$

### 数学上它为什么有效

它其实是一个**多目标优化 / 正则化**问题。

纯 PPO 在优化：

偏好对齐目标

PPO-ptx 在优化：

偏好对齐目标+原始语言建模目标

所以它相当于把参数更新方向拆成两部分：

- 一部分往“更符合人类偏好”走
- 一部分往“别忘了自己本来会建模自然语言”拉

从几何角度看，就是用 LM 梯度给 PPO 梯度加了一个反向约束，降低过拟合 reward 的风险。

# Bradley-Terry 模型

Bradley-Terry 模型，本质上是一个**“成对比较(pairwise comparison) 的概率模型”**。

它回答的问题是：

> 给两个候选项 A 和 B，它们各自有一个“隐藏分数/效用/奖励”，那么**A 被选中胜过 B 的概率**是多少？
> 

## 1. 最核心定义

假设两个候选回答分别是$y_{1}$和$y_{2}$，每个回答都有一个分数：

$r \left(\right. x , y_{1} \left.\right) , r \left(\right. x , y_{2} \left.\right)$

这里的r(x,y)可以理解成：

- 质量分数
- 人类偏好强度
- reward
- latent utility（潜在效用）

那 Bradley-Terry 模型定义：

$P \left(\right. y_{1} \succ y_{2} \mid x \left.\right) = \frac{exp ⁡ \left(\right. r \left(\right. x , y_{1} \left.\right) \left.\right)}{exp ⁡ \left(\right. r \left(\right. x , y_{1} \left.\right) \left.\right) + exp ⁡ \left(\right. r \left(\right. x , y_{2} \left.\right) \left.\right)}$

意思是：

- $y_{1}$分数越高，被选中的概率越大
- 但不是硬比较，而是变成概率形式

## 2. 为什么它常写成 sigmoid 形式

把上式变形一下：

$P \left(\right. y_{1} \succ y_{2} \mid x \left.\right) = \sigma \left(\right. r \left(\right. x , y_{1} \left.\right) - r \left(\right. x , y_{2} \left.\right) \left.\right)$

其中

$\sigma \left(\right. z \left.\right) = \frac{1}{1 + e^{- z}}$

这就很直观了：

- 如果 $r \left(\right. x , y_{1} \left.\right) \gg r \left(\right. x , y_{2} \left.\right)$，概率接近 1
- 如果两者差不多，概率接近 0.5
- 如果 $r \left(\right. x , y_{1} \left.\right) \ll r \left(\right. x , y_{2} \left.\right)$，概率接近 0

所以它本质上就是：

> **“胜负概率 = 两个分数差的 sigmoid”**
> 

## 3. 它本质上是个什么模型

从机器学习角度，你可以把 Bradley-Terry 看成：

- 一个 **pairwise ranking model**
- 一个 **logistic preference model**
- 一个 **二分类概率模型**

因为它最后就是在学：

chosen 是否胜过 rejected

而判别依据是两者的分数差。

## 8. 它的优点

Bradley-Terry 在偏好学习里很常见，主要因为它有几个优点：

### 1）很自然

人类更容易做相对比较，而不是打绝对分数。“哪个更好” 比 “给它打 8.3 分” 容易得多。

### 2）公式简单

只需要分数差，不需要复杂建模。

### 3）和 logistic loss 完全兼容

容易训练，梯度稳定，实现简单。

## 9. 它的局限

也别把它看得太神。

### 1）只看分数差

它假设偏好只由一个标量分数决定，现实里人类偏好可能是多维的：

- 准确性
- 风格
- 安全性
- 长度
- 礼貌性

但 BT 把这些全压成一个数。

### 2）偏好噪声可能很大

人类标注不一定稳定，甚至不同人标准不同。

### 3）成对独立假设比较理想化

真实偏好有时受上下文、候选集大小、展示顺序影响，但 BT 没显式建这些。

# 待解决问题

- 有监督微调里的最终的SFT模型选择基于验证集上的RM得分是什么意思，模型在验证损失上1个epoch后就会过拟合是什么表现（loss很低的意思吗）

> **用“更像人类偏好”这个指标选 SFT 模型，而不是只看 token-level loss。**
> 
> 
> ## 为什么不用 validation loss 直接选？
> 
> 因为 SFT 的 validation loss 衡量的是：
> 
> > 模型生成的 token，和验证集那条“参考答案”逐 token 有多接近
> > 
> 
> 但开放式问题里，一个 prompt 往往有很多合理答案。
> 
> 比如 prompt 是：
> 
> > “解释一下 TCP 和 UDP 的区别”
> > 
> 
> 验证集参考答案可能是 A。但模型生成了 B：
> 
> - 内容也对
> - 更简洁
> - 更像人喜欢的回答
> 
> 这种情况下：
> 
> - **validation loss 可能不低**
> - 但 **RM score 可能更高**
> - 甚至人类也更喜欢 B
> 
> 所以在 instruction-following 任务里：
> 
> > **验证集 NLL 不一定等于真正的回答质量。**
> > 
> 
> ## “1 个 epoch 后过拟合”具体是什么表现？
> 
> 不是单纯“loss 很低”这么简单。**过拟合的标准不是 training loss 低，而是 train / val 开始分叉。**
> 
> 典型表现是：
> 
> - **训练集 loss 持续下降**
> - **验证集 loss 不再下降，甚至开始上升**
> 
> 也就是：
> 
> > 模型越来越会“复现训练集示范答案的具体写法”，但对没见过的验证样本，逐 token 复现能力反而变差。
> > 
> 
> ## 为什么论文里会出现这种现象？
> 
> 因为 SFT 数据量不大，示范答案风格又比较集中。
> 
> 所以模型容易学成：
> 
> - 越来越贴训练数据的措辞
> - 越来越像“在背参考答案风格”
> 
> 这在 token-level loss 上就是过拟合。
> 
> ## 那为什么过拟合后 RM score 可能还在涨？
> 
> 因为 RM 衡量的是：
> 
> > “这个回答整体上更像人类偏好的回答吗？”
> > 
> 
> 而 validation loss 衡量的是：
> 
> > “这串 token 有没有贴参考答案”
> > 
> 
> 这两个目标不是同一个东西。
> 
> 所以论文的意思不是：
> 
> > 过拟合没事
> > 
> 
> 而是：
> 
> > **instruction tuning 里，NLL 不是唯一靠谱的选模标准**
> > 
- 训练奖励模型里的移除SFT模型最后的反嵌入层，替换为一个投影层以输出标量值是什么意思，损失函数怎么理解

> **原**“换成投影层输出标量”是什么意思？
> 
> 
> 就是把最后输出层改成：
> 
> $r_{\theta} \left(\right. x , y \left.\right) = w^{\top} h_{\text{final}} + b$
> 
> 其中：
> 
> - $h_{\text{final}}$：读完整个 prompt + response 后的最终隐藏状态
> - $w , b$：一个线性层参数
> - 输出是 **1 个标量**
> 
> 也就是：
> 
> - 原来输出 vocab 大小的向量
> - 现在输出 1 个 reward 分数
> 
> ## 奖励模型的损失函数怎么理解？
> 
> 如果对于同一个 promptx，人类觉得：
> 
> - $y_{w}$比 $y_{l}$更好
> 
> 那就希望模型满足：
> 
> $r_{\theta} \left(\right. x , y_{w} \left.\right) > r_{\theta} \left(\right. x , y_{l} \left.\right)$
> 
> 于是论文用的就是 pairwise preference loss：
> 
> $L_{R M} = - log ⁡ \sigma \left(\right. r_{\theta} \left(\right. x , y_{w} \left.\right) - r_{\theta} \left(\right. x , y_{l} \left.\right) \left.\right)$
> 
> 更完整写法是对整个数据集取期望。
> 
> ## 这个损失的直觉是什么？
> 
> ### 当 winner 分数确实更高时
> 
> 如果：
> 
> $r \left(\right. x , y_{w} \left.\right) - r \left(\right. x , y_{l} \left.\right)$
> 
> 很大，那么 sigmoid 接近 1，loss 很小。
> 
> ### 当 loser 反而分数更高时
> 
> 如果差值是负的，sigmoid 很小，loss 很大。
> 
> 所以它在逼模型学会：
> 
> > **对同一个 prompt，排序方向别搞反。**
> > 
> 
> ## 这不是分类“好/坏”，而是排序学习
> 
> 这是重点：
> 
> RM 不是在学“这个回答绝对值是 8 分还是 9 分”，而是在学：
> 
> > **A 和 B 比，哪个更好**
> > 
> 
> 所以它更像**ranking model**，不是标准分类器。
> 
- 训练奖励模型里的防过拟合技巧是什么意思

> RM 很容易记住训练数据里的偏好关系，但泛化到新 prompt / 新回答会变差，所以要刻意抑制它过拟合。
> 
> 
> ## 为什么 RM 比 SFT 更容易过拟合？
> 
> 因为 RM 的数据更稀疏、更贵、更主观：
> 
> - 偏好数据量比预训练数据少太多
> - 标注噪声更大
> - 样本之间高度相关
> - 模型容量又很强
> 
> 所以 RM 特别容易学成：
> 
> - 训练集排序很准
> - 新样本一塌糊涂
> 
> ## 论文里最关键的防过拟合技巧
> 
> ### 技巧 1：只训练 1 个 epoch
> 
> 这是最直接的 early stopping。
> 
> 意思是：
> 
> - 不让 RM 在小规模偏好数据上反复刷太久
> - 训练一点就停
> 
> 这是非常典型的“强制别背训练集”。
> 
> ### 技巧 2：同一个 prompt 下的 pair 不要当成大量独立样本乱放大
> 
> 如果一个 prompt 生成了 K 个答案，排序后会拆成很多 pair：
> 
> $\left(\right. \frac{K}{2} \left.\right)$
> 
> 这些 pair 高度相关，因为它们共享：
> 
> - 同一个 prompt
> - 同一批回答
> 
> 如果全当独立样本去训，就会导致：
> 
> - 某些回答被重复强化很多次
> - 梯度偏得很厉害
> - 更容易过拟合
> 
> 所以要对同 prompt 下的 pair 做更谨慎的聚合处理。
> 
> ### 用验证集看泛化，而不是只看训练 loss
> 
> 这属于常规操作，但在 RM 上尤其重要。因为 RM 训练集 loss 持续变好，不代表真的更适合拿去 PPO。
> 
> ## 你可以把 RM 过拟合理解成什么？
> 
> 比如它学会了某些表面偏好：
> 
> - 长答案更像“认真”
> - 有 numbered list 就高分
> - 语气更像标注员就高分
> 
> 那它可能会把这些“伪特征”当成真正的人类偏好。
> 
> 一旦后面 PPO 开始最大化 RM 分数，模型就会疯狂利用这些漏洞。
> 
> 这就是为什么：
> 
> > **RM 的过拟合，比普通分类器过拟合更危险。**
> > 
> 
> 因为它会被下游 RL 放大。
> 
> **论文/复现资料里的关键做法是：**
> 
> > **不把这些 pair 全部打散成彼此独立的训练样本；而是把同一个 prompt 下产生的所有 comparisons，作为一个整体来计算 loss。**
> > 
> 
> 也就是：
> 
> - 不是 `一个 pair = 一个训练样本`
> - 而是 `一个 prompt 下的一组 completion = 一个训练单元`
> 
> ## 做法核心：按 prompt 分组训练
> 
> 对于一个 promptx：
> 
> 1. 先拿到这一组 completion：
>     
>     $y_{1} , y_{2} , . . . , y_{K}$
>     
> 2. RM 分别给每个 completion 打分：
>     
>     $r_{\theta} \left(\right. x , y_{1} \left.\right) , r_{\theta} \left(\right. x , y_{2} \left.\right) , . . . , r_{\theta} \left(\right. x , y_{K} \left.\right)$
>     
> 3. 再在**同一个 prompt 内部**，根据人工排序构造所有 pairwise loss
> 4. 把这些 loss **先在 prompt 内聚合/平均**
> 5. 最后把这个 prompt 的聚合 loss 当成一个 batch element 参与训练
> 
> ## 数学上怎么写
> 
> 假设同一个 prompt 下，构造出的偏好对集合为：
> 
> $P ( x ) = { ( i , j ) \mid y_{i} \succ y_{j} }$
> 
> 那么这个 prompt 的 reward model loss 可以写成：
> 
> $L_{x} ( \theta ) = - \frac{1}{\mid \mathcal{P} ( x ) \mid} ( x ){\sum} log ⁡ \sigma ( r_{\theta} ( x , y_{i} ) - r_{\theta} ( x , y_{j} ) )$
> 
> 然后整个 batch 的 loss 再对多个 prompt 做平均：
> 
> $L ( \theta ) = \frac{1}{B} {\sum} L_{x} ( \theta )$
> 
> ## 按 prompt 聚合方式
> 
> 先对 prompt 内部做平均：
> 
> $L_{x} = \text{prompt }内平均$
> 
> 再跨 prompt 平均：
> 
> $L = \text{prompt }间平均$
> 
> 这样每个 prompt 的影响更均衡，不会因为某个 prompt 导出了更多 pair，就把它的训练权重人为放大。
> 
> ## 1）降低相关样本的“重复计数”
> 
> 同一个 prompt 下的 comparisons 很相关。按 prompt 聚合后，相当于告诉优化器：
> 
> > 这些 pair 是同一个标注任务的不同视角，不是完全新的独立样本。
> > 
> 
> 这样就不会把一组相关样本的信号无限放大。
> 
> ## 2）让模型更关注“排序结构”，而不是记住某几个 pair
> 
> 如果拆成独立 pair，模型可能很快学会：
> 
> - “这个句子总是赢”
> - “那个回答总是输”
> 
> 但按 prompt 聚合后，它更像是在学：
> 
> > 对这组 completion 的整体排序关系该怎么建模
> > 
> 
> 这会更接近 RM 真正想学的东西。
> 
- 训练奖励模型里的• 奖励归一化怎么理解

> **让 reward 的数值尺度稳定，方便 PPO 训练。**
> 
> 
> ## 为什么 reward 需要归一化？
> 
> 奖励模型输出的分数本身没有天然单位。
> 
> 比如一个 RM 可能输出范围是：
> 
> - 0.3 到 0.8
> 
> 另一个 checkpoint 可能输出：
> 
> - 12 到 17
> 
> 虽然排序关系可能差不多，但对 PPO 来说，这种尺度差异很要命。
> 
> 因为 PPO 里：
> 
> - advantage 大小会受 reward 影响
> - policy update 激进程度会变
> - KL 系数的相对强弱也会变
> 
> ## 归一化怎么做？
> 
> 论文里本质是做一个线性变换：
> 
> $r \left(\right. x , y \left.\right) = g \cdot r \left(\right. x , y \left.\right) + b$
> 
> 让 reward 在某个参考分布上满足：
> 
> $\mathbb{E} \left[\right. r \left]\right. = 0 , V a r \left(\right. r \left.\right) = 1$
> 
> 也就是：
> 
> - 均值 0
> - 方差 1
> 
> 等价写法就是标准化：
> 
> $r \left(\right. x , y \left.\right) = \frac{r \left(\right. x , y \left.\right) - \mu}{\sigma}$
> 
> ## 这个参考分布是什么？
> 
> 通常是：
> 
> - 从 prompt 数据分布里采样 prompt
> - 用一个固定参考策略生成 response
> - 统计这些 response 上 reward 的均值和方差
> 
> 然后按这个分布去归一化。
>