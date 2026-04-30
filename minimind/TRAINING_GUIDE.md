# MiniMind 训练完整指南

本文档整理了 MiniMind 项目中 **SFT（有监督微调）**、**PPO（近端策略优化）**、**DPO（直接偏好优化）** 三个阶段的完整训练指令（含所有可选参数）、数据准备工作，以及训练结束后使用 **lm-evaluation-harness** 进行客观评测的具体步骤。

---

## 目录

- [一、数据准备](#一数据准备)
  - [1.1 数据下载](#11-数据下载)
  - [1.2 各数据集说明](#12-各数据集说明)
  - [1.3 数据格式](#13-数据格式)
- [二、SFT（有监督微调）训练](#二sft有监督微调训练)
  - [2.1 前置条件](#21-前置条件)
  - [2.2 训练指令](#22-训练指令)
  - [2.3 完整参数列表](#23-完整参数列表)
  - [2.4 训练输出](#24-训练输出)
- [三、DPO（直接偏好优化）训练](#三dpo直接偏好优化训练)
  - [3.1 前置条件](#31-前置条件)
  - [3.2 训练指令](#32-训练指令)
  - [3.3 完整参数列表](#33-完整参数列表)
  - [3.4 训练输出](#34-训练输出)
- [四、PPO（近端策略优化）训练](#四ppo近端策略优化训练)
  - [4.1 前置条件](#41-前置条件)
  - [4.2 训练指令](#42-训练指令)
  - [4.3 完整参数列表](#43-完整参数列表)
  - [4.4 训练输出](#44-训练输出)
- [五、使用 lm-evaluation-harness 评测](#五使用-lm-evaluation-harness-评测)
  - [5.1 安装评测框架](#51-安装评测框架)
  - [5.2 模型格式转换](#52-模型格式转换)
  - [5.3 运行评测](#53-运行评测)
  - [5.4 参考基准分数](#54-参考基准分数)
- [六、通用说明](#六通用说明)

---

## 一、数据准备

### 1.1 数据下载

MiniMind 训练数据集下载地址：
- [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
- [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

> 无需全部 clone，可单独下载所需的文件

将下载的数据集文件放到 `./dataset/` 目录下：

```bash
./dataset/
├── dpo.jsonl (53MB)                  # DPO 偏好训练数据
├── pretrain_t2t_mini.jsonl (1.2GB)   # 轻量预训练数据 ✨
├── pretrain_t2t.jsonl (10GB)         # 主线预训练数据
├── rlaif.jsonl (24MB)                # PPO/GRPO/CISPO 训练数据 ✨
├── sft_t2t_mini.jsonl (1.6GB)        # 轻量 SFT 数据 ✨
└── sft_t2t.jsonl (14GB)              # 主线 SFT 数据
```

### 1.2 各数据集说明

| 数据文件 | 用途 | 推荐 max_seq_len | 说明 |
|---|---|---|---|
| `pretrain_t2t_mini.jsonl` | 轻量预训练 | ≈768 | 适合快速复现 |
| `pretrain_t2t.jsonl` | 主线预训练 | ≈380 | 完整复现 minimind-3 |
| `sft_t2t_mini.jsonl` | 轻量 SFT | ≈768 | 已混入 Tool Call 样本 |
| `sft_t2t.jsonl` | 主线 SFT | - | 完整复现，含 Tool Call |
| `dpo.jsonl` | DPO 偏好训练 | 1024 | 抽样自 DPO-En-Zh-20k |
| `rlaif.jsonl` | PPO/GRPO/CISPO | 768 | RLAIF 训练数据 |

**推荐训练方案：**
- 快速复现 Zero 模型：`pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`
- 完整复现 minimind-3：`pretrain_t2t` + `sft_t2t` + `rlaif/agent_rl`

### 1.3 数据格式

#### SFT 数据格式

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个有用的助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}
  ]
}
```

#### DPO 数据格式

```json
{
  "chosen": [
    {"content": "用户问题", "role": "user"},
    {"content": "好的回答", "role": "assistant"}
  ],
  "rejected": [
    {"content": "用户问题", "role": "user"},
    {"content": "差的回答", "role": "assistant"}
  ]
}
```

#### RLAIF（PPO/GRPO）数据格式

```json
{
  "conversations": [
    {"role": "user", "content": "请解释一下什么是光合作用？"},
    {"role": "assistant", "content": "无"}
  ]
}
```

> 注：RLAIF 数据中 assistant 内容不重要（填"无"即可），因为训练时完全由策略模型实时采样生成回答。

---

## 二、SFT（有监督微调）训练

### 2.1 前置条件

1. 已完成预训练，`./out/` 目录下存在 `pretrain_*.pth` 权重文件
2. 已下载 SFT 数据集（`sft_t2t_mini.jsonl` 或 `sft_t2t.jsonl`）到 `./dataset/` 目录

### 2.2 训练指令

```bash
# 进入 trainer 目录
cd trainer

# 单卡训练
python train_full_sft.py

# 多卡训练（DDP，N 为 GPU 数量）
torchrun --nproc_per_node N train_full_sft.py

# 带参数的完整示例
python train_full_sft.py \
  --save_dir "../out" \
  --save_weight "full_sft" \
  --epochs 2 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --device "cuda:0" \
  --dtype "bfloat16" \
  --num_workers 8 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 1000 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --use_moe 0 \
  --data_path "../dataset/sft_t2t_mini.jsonl" \
  --from_weight "pretrain" \
  --from_resume 0 \
  --use_wandb \
  --wandb_project "MiniMind-Full-SFT" \
  --use_compile 0
```

### 2.3 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--save_dir` | str | `../out` | 模型保存目录 |
| `--save_weight` | str | `full_sft` | 保存权重的前缀名，最终文件名为 `{save_weight}_{hidden_size}.pth` |
| `--epochs` | int | `2` | 训练轮数 |
| `--batch_size` | int | `16` | 每个 GPU 的 batch size |
| `--learning_rate` | float | `1e-5` | 初始学习率（使用余弦退火调度） |
| `--device` | str | `cuda:0` | 训练设备（多卡时自动分配） |
| `--dtype` | str | `bfloat16` | 混合精度类型，可选 `bfloat16`/`float16` |
| `--num_workers` | int | `8` | DataLoader 数据加载线程数 |
| `--accumulation_steps` | int | `1` | 梯度累积步数（等效增大 batch size） |
| `--grad_clip` | float | `1.0` | 梯度裁剪阈值（防止梯度爆炸） |
| `--log_interval` | int | `100` | 日志打印间隔（每多少 step 打印一次） |
| `--save_interval` | int | `1000` | 模型保存间隔（每多少 step 保存一次） |
| `--hidden_size` | int | `768` | 隐藏层维度（需与预训练模型一致） |
| `--num_hidden_layers` | int | `8` | Transformer 层数（需与预训练模型一致） |
| `--max_seq_len` | int | `768` | 训练的最大序列长度（token 数，中文 1token≈1.5~1.7 字符） |
| `--use_moe` | int | `0` | 是否使用 MoE 架构（0=否，1=是） |
| `--data_path` | str | `../dataset/sft_t2t_mini.jsonl` | 训练数据路径 |
| `--from_weight` | str | `pretrain` | 基于哪个权重训练（对应 `out/{from_weight}_{hidden_size}.pth`），设为 `none` 则从头训练 |
| `--from_resume` | int | `0` | 是否自动检测并续训（0=否，1=是） |
| `--use_wandb` | flag | - | 是否使用 wandb/swanlab 可视化（添加此参数即开启） |
| `--wandb_project` | str | `MiniMind-Full-SFT` | wandb 项目名 |
| `--use_compile` | int | `0` | 是否使用 `torch.compile` 加速（0=否，1=是） |

### 2.4 训练输出

- 权重文件：`out/full_sft_{hidden_size}.pth`（如 `out/full_sft_768.pth`）
- MoE 模型：`out/full_sft_{hidden_size}_moe.pth`
- 检查点（续训用）：`checkpoints/full_sft_{hidden_size}_resume.pth`

**验证训练结果：**

```bash
python eval_llm.py --weight full_sft
```

---

## 三、DPO（直接偏好优化）训练

### 3.1 前置条件

1. 已完成 SFT 训练，`./out/` 目录下存在 `full_sft_*.pth` 权重文件
2. 已下载 DPO 数据集（`dpo.jsonl`）到 `./dataset/` 目录

> DPO 是 off-policy 算法，使用静态偏好数据对比 chosen vs rejected，不需要 Reward Model。

### 3.2 训练指令

```bash
# 进入 trainer 目录
cd trainer

# 单卡训练
python train_dpo.py

# 多卡训练（DDP）
torchrun --nproc_per_node N train_dpo.py

# 带参数的完整示例
python train_dpo.py \
  --save_dir "../out" \
  --save_weight "dpo" \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 4e-8 \
  --device "cuda:0" \
  --dtype "bfloat16" \
  --num_workers 8 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 100 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 1024 \
  --use_moe 0 \
  --data_path "../dataset/dpo.jsonl" \
  --from_weight "full_sft" \
  --from_resume 0 \
  --beta 0.15 \
  --use_wandb \
  --wandb_project "MiniMind-DPO" \
  --use_compile 0
```

### 3.3 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--save_dir` | str | `../out` | 模型保存目录 |
| `--save_weight` | str | `dpo` | 保存权重的前缀名 |
| `--epochs` | int | `1` | 训练轮数 |
| `--batch_size` | int | `4` | 每个 GPU 的 batch size（DPO 内部会 concat chosen 和 rejected，实际前向 2×batch_size） |
| `--learning_rate` | float | `4e-8` | 初始学习率（**建议 ≤5e-8**，过大会导致灾难性遗忘） |
| `--device` | str | `cuda:0` | 训练设备 |
| `--dtype` | str | `bfloat16` | 混合精度类型 |
| `--num_workers` | int | `8` | 数据加载线程数 |
| `--accumulation_steps` | int | `1` | 梯度累积步数 |
| `--grad_clip` | float | `1.0` | 梯度裁剪阈值 |
| `--log_interval` | int | `100` | 日志打印间隔 |
| `--save_interval` | int | `100` | 模型保存间隔 |
| `--hidden_size` | int | `768` | 隐藏层维度 |
| `--num_hidden_layers` | int | `8` | Transformer 层数 |
| `--max_seq_len` | int | `1024` | 最大序列长度（DPO 需要对比完整回答，建议稍大） |
| `--use_moe` | int | `0` | 是否使用 MoE 架构（0=否，1=是） |
| `--data_path` | str | `../dataset/dpo.jsonl` | DPO 训练数据路径 |
| `--from_weight` | str | `full_sft` | 基于哪个权重训练（通常为 SFT 后的权重） |
| `--from_resume` | int | `0` | 是否自动检测并续训（0=否，1=是） |
| `--beta` | float | `0.15` | DPO 中的 β 参数，控制偏离参考模型的程度（越大越保守） |
| `--use_wandb` | flag | - | 是否使用 wandb/swanlab 可视化 |
| `--wandb_project` | str | `MiniMind-DPO` | wandb 项目名 |
| `--use_compile` | int | `0` | 是否使用 `torch.compile` 加速 |

### 3.4 训练输出

- 权重文件：`out/dpo_{hidden_size}.pth`（如 `out/dpo_768.pth`）
- 检查点：`checkpoints/dpo_{hidden_size}_resume.pth`

**DPO 训练要点：**
- DPO 学习率建议极低（≤5e-8），过高会导致灾难性遗忘
- DPO 需要同时加载策略模型和参考模型（ref_model，冻结），显存约为单模型的 2 倍
- DPO 是离线训练，不需要 Reward Model

---

## 四、PPO（近端策略优化）训练

### 4.1 前置条件

1. 已完成 SFT 训练，`./out/` 目录下存在 `full_sft_*.pth` 权重文件
2. 已下载 RLAIF 数据集（`rlaif.jsonl`）到 `./dataset/` 目录
3. **已下载 Reward Model**：[InternLM2-1.8B-Reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward)

**Reward Model 放置位置（minimind 项目同级目录）：**

```
root/
├── minimind/                    # MiniMind 项目
│   ├── model/
│   ├── trainer/
│   └── ...
└── internlm2-1_8b-reward/       # 奖励模型
    ├── config.json
    ├── model.safetensors
    └── ...
```

### 4.2 训练指令

```bash
# 进入 trainer 目录
cd trainer

# 单卡训练
python train_ppo.py

# 多卡训练（DDP）
torchrun --nproc_per_node N train_ppo.py

# 带参数的完整示例
python train_ppo.py \
  --save_dir "../out" \
  --save_weight "ppo_actor" \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 3e-7 \
  --critic_learning_rate 5e-7 \
  --device "cuda:0" \
  --dtype "bfloat16" \
  --num_workers 8 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 1 \
  --save_interval 10 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 768 \
  --max_gen_len 1024 \
  --data_path "../dataset/rlaif.jsonl" \
  --clip_epsilon 0.2 \
  --vf_coef 0.5 \
  --kl_coef 0.02 \
  --gamma 1.0 \
  --lam 0.95 \
  --cliprange_value 0.2 \
  --ppo_update_iters 2 \
  --early_stop_kl 0.25 \
  --mini_batch_size 2 \
  --from_weight "full_sft" \
  --reward_model_path "../../internlm2-1_8b-reward" \
  --from_resume 0 \
  --use_wandb \
  --wandb_project "MiniMind-PPO" \
  --use_compile 0 \
  --debug_mode \
  --debug_interval 20 \
  --thinking_ratio 0.9 \
  --rollout_engine "torch" \
  --sglang_base_url "http://localhost:8998" \
  --sglang_model_path "../model" \
  --sglang_shared_path "./sglang_ckpt_ppo"
```

### 4.3 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| **基础训练参数** | | | |
| `--save_dir` | str | `../out` | 模型保存目录 |
| `--save_weight` | str | `ppo_actor` | 保存权重的前缀名 |
| `--epochs` | int | `1` | 训练轮数 |
| `--batch_size` | int | `2` | 每个 GPU 的 batch size（PPO 显存开销大，建议小 batch） |
| `--learning_rate` | float | `3e-7` | Actor（策略模型）学习率 |
| `--critic_learning_rate` | float | `5e-7` | Critic（价值模型）学习率 |
| `--device` | str | `cuda:0` | 训练设备 |
| `--dtype` | str | `bfloat16` | 混合精度类型 |
| `--num_workers` | int | `8` | 数据加载线程数 |
| `--accumulation_steps` | int | `1` | 梯度累积步数 |
| `--grad_clip` | float | `1.0` | 梯度裁剪阈值 |
| `--log_interval` | int | `1` | 日志打印间隔 |
| `--save_interval` | int | `10` | 模型保存间隔 |
| **模型结构参数** | | | |
| `--hidden_size` | int | `768` | 隐藏层维度 |
| `--num_hidden_layers` | int | `8` | Transformer 层数 |
| `--use_moe` | int | `0` | 是否使用 MoE 架构（0=否，1=是） |
| `--max_seq_len` | int | `768` | Prompt 最大长度 |
| `--max_gen_len` | int | `1024` | 模型生成（rollout）的最大 token 长度 |
| **PPO 算法超参数** | | | |
| `--clip_epsilon` | float | `0.2` | PPO 策略裁剪参数 ε（控制新旧策略比率的裁剪范围） |
| `--vf_coef` | float | `0.5` | Value function 损失系数（Critic loss 权重） |
| `--kl_coef` | float | `0.02` | KL 散度惩罚系数（约束策略不要偏离参考模型太远） |
| `--gamma` | float | `1.0` | GAE 折扣因子（1.0 表示不折扣未来奖励） |
| `--lam` | float | `0.95` | GAE λ 参数（平衡偏差与方差，越高方差越大但偏差越小） |
| `--cliprange_value` | float | `0.2` | Value function 裁剪范围（防止价值估计跳跃过大） |
| `--ppo_update_iters` | int | `2` | 同一批 rollout 数据重复更新次数 |
| `--early_stop_kl` | float | `0.25` | PPO early stop 的 KL 阈值（超过后停止当前 batch 更新） |
| `--mini_batch_size` | int | `2` | PPO 每次参数更新使用的 mini-batch 大小 |
| **数据与模型加载** | | | |
| `--data_path` | str | `../dataset/rlaif.jsonl` | RLAIF 训练数据路径 |
| `--from_weight` | str | `full_sft` | 基于哪个权重训练（Actor 和 Ref 模型均从此加载） |
| `--reward_model_path` | str | `../../internlm2-1_8b-reward` | Reward Model 路径 |
| `--from_resume` | int | `0` | 是否自动检测并续训（0=否，1=是） |
| **可视化与调试** | | | |
| `--use_wandb` | flag | - | 是否使用 wandb/swanlab |
| `--wandb_project` | str | `MiniMind-PPO` | wandb 项目名 |
| `--use_compile` | int | `0` | 是否使用 `torch.compile` 加速 |
| `--debug_mode` | flag | - | 是否打印训练调试采样（展示 prompt、response、reward） |
| `--debug_interval` | int | `20` | debug 模式下每隔多少 step 打印一次采样 |
| **推理与生成** | | | |
| `--thinking_ratio` | float | `0.9` | 按概率开启 thinking（0.0~1.0），控制多少比例的训练样本启用 `<think>` 标签 |
| `--rollout_engine` | str | `torch` | rollout 引擎类型，可选 `torch`（本地推理）/ `sglang`（SGLang 服务） |
| `--sglang_base_url` | str | `http://localhost:8998` | SGLang 服务器 URL（仅 rollout_engine=sglang 时生效） |
| `--sglang_model_path` | str | `../model` | SGLang tokenizer 路径 |
| `--sglang_shared_path` | str | `./sglang_ckpt_ppo` | SGLang 共享存储路径（用于策略权重同步） |

### 4.4 训练输出

- Actor 权重文件：`out/ppo_actor_{hidden_size}.pth`（如 `out/ppo_actor_768.pth`）
- 检查点（含 Critic）：`checkpoints/ppo_actor_{hidden_size}_resume.pth`

**PPO 训练要点：**
- PPO 需要同时维护 4 个模型：Actor、Critic、Ref（冻结）、Reward Model，显存需求大
- `batch_size` 建议从小值（2~4）开始，根据显存情况逐步增大
- 监控 reward 的方差，若持续接近 0 说明奖励信号稀疏，需调整数据或奖励机制
- PPO 收敛较慢（Critic 需要逐步准确估计价值函数），patience 要高于 DPO

---

## 五、使用 lm-evaluation-harness 评测

### 5.1 安装评测框架

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness && pip install -e .
```

### 5.2 模型格式转换

lm-evaluation-harness 需要 transformers 格式的模型。MiniMind 训练产出的是 `.pth` 文件，需要先转换：

```bash
cd scripts
```

**编辑 `convert_model.py` 中的路径**（根据你要评测的权重）：

```python
# 评测 SFT 模型
torch_path = "../out/full_sft_768.pth"
transformers_path = '../minimind-3'

# 评测 DPO 模型
torch_path = "../out/dpo_768.pth"
transformers_path = '../minimind-3-dpo'

# 评测 PPO 模型
torch_path = "../out/ppo_actor_768.pth"
transformers_path = '../minimind-3-ppo'
```

**运行转换：**

```bash
python convert_model.py
```

转换完成后，目录结构如下：

```
minimind/
├── minimind-3/              # 转换后的 transformers 格式模型
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
```

**转换说明：**
- 默认使用 `convert_torch2transformers()` 转为 **Qwen3 兼容格式**（推荐，生态兼容性好，lm-eval 原生识别）
- 也可用 `convert_torch2transformers_minimind()` 转为 MiniMind 自定义格式（需要 `--trust_remote_code`）
- MoE 模型也支持转换（自动重组为 `Qwen3MoeForCausalLM` 格式）

### 5.3 运行评测

```bash
# 基本评测命令（使用国内 HF 镜像）
HF_ENDPOINT=https://hf-mirror.com lm_eval \
  --model hf \
  --model_args pretrained="/path/to/minimind-3",dtype=auto \
  --tasks "ceval-valid,cmmlu,arc_easy,piqa,openbookqa,hellaswag,social_iqa" \
  --batch_size 16 \
  --device cpu \
  --trust_remote_code \
  --apply_chat_template
```

**参数说明：**

| 参数 | 说明 |
|---|---|
| `--model hf` | 使用 HuggingFace 模型加载器 |
| `--model_args pretrained="..."` | 模型路径（绝对路径或相对路径） |
| `--model_args dtype=auto` | 自动推断精度 |
| `--tasks` | 评测数据集，多个用逗号分隔 |
| `--batch_size 16` | 评测 batch size，根据显存调整 |
| `--device cpu` | 设备（可改为 `cuda:0`） |
| `--trust_remote_code` | 信任自定义代码（MiniMind 格式需要，Qwen3 格式可省略） |
| `--apply_chat_template` | **SFT/RL 后的指令模型必须加此参数**；纯 pretrain 基座不加 |

**支持的评测数据集：**

| 数据集 | 语言 | 领域 | 说明 |
|---|---|---|---|
| `ceval-valid` | 中文 | 综合学科 | 中文多学科选择题 |
| `cmmlu` | 中文 | 综合学科 | 中文大规模多任务理解 |
| `arc_easy` | 英文 | 科学常识 | AI2 科学推理（简单级） |
| `piqa` | 英文 | 物理直觉 | 日常物理常识问答 |
| `openbookqa` | 英文 | 科学常识 | 开放知识科学问答 |
| `hellaswag` | 英文 | 常识推理 | 句子补全/情境推理 |
| `social_iqa` | 英文 | 社交常识 | 社交情境理解与推理 |

**查看所有可用数据集：**

```bash
lm_eval ls tasks
```

**完整评测示例（评测 SFT 模型 + GPU 加速）：**

```bash
HF_ENDPOINT=https://hf-mirror.com lm_eval \
  --model hf \
  --model_args pretrained="./minimind-3",dtype=auto \
  --tasks "ceval-valid,cmmlu,arc_easy,piqa,openbookqa,hellaswag,social_iqa" \
  --batch_size 32 \
  --device cuda:0 \
  --trust_remote_code \
  --apply_chat_template
```

**评测 pretrain 基座模型（不加 chat_template）：**

```bash
HF_ENDPOINT=https://hf-mirror.com lm_eval \
  --model hf \
  --model_args pretrained="./minimind-3-pretrain",dtype=auto \
  --tasks "ceval-valid,cmmlu,arc_easy,piqa,openbookqa,hellaswag,social_iqa" \
  --batch_size 32 \
  --device cuda:0 \
  --trust_remote_code
```

### 5.4 参考基准分数

| 模型 | 参数量 | zh (ceval / cmmlu) | en (arc / piqa / obqa / hellaswag / siqa) |
|---|---|---|---|
| minimind-3 | 64M | 24.89 / 25.38 | 28.49 / 50.65 / 23.60 / 28.28 / 34.19 |
| minimind-3-moe | 198M | 25.48 / 24.32 | 27.74 / 50.71 / 26.20 / 27.43 / 34.03 |
| minimind-3-exam | 64M | 30.98 / 26.12 | 35.61 / 56.26 / 24.20 / 28.40 / 34.19 |

> **注意：** 评测原理是比较各候选项的条件概率 `p(x|y)` 取最大者，而非让模型自由生成。MiniMind 训练数据偏中文且规模较小，英文表现通常在随机基线附近。

---

## 六、通用说明

### 断点续训

所有训练脚本均支持断点续训，添加 `--from_resume 1` 即可：

```bash
python train_full_sft.py --from_resume 1
python train_dpo.py --from_resume 1
python train_ppo.py --from_resume 1
```

- 检查点自动保存在 `./checkpoints/` 目录
- 支持跨不同 GPU 数量恢复
- 支持 wandb 训练记录连续性

### 多卡训练

```bash
# N 为 GPU 数量
torchrun --nproc_per_node N train_full_sft.py
torchrun --nproc_per_node N train_dpo.py
torchrun --nproc_per_node N train_ppo.py
```

### WandB/SwanLab 可视化

```bash
# 任何训练脚本添加 --use_wandb 即可
python train_full_sft.py --use_wandb --wandb_project "My-SFT"
```

> 当前默认使用 [SwanLab](https://swanlab.cn/) 作为可视化工具（国内访问友好，API 与 WandB 兼容）。

### 训练阶段依赖关系

```
Pretrain → SFT → DPO（可选，偏好对齐）
                 → PPO（可选，RLAIF 强化学习）
```

- **SFT** 必须基于预训练权重（`--from_weight pretrain`）
- **DPO/PPO** 必须基于 SFT 权重（`--from_weight full_sft`）
- DPO 和 PPO 是两条独立的 RL 路径，通常选择其一即可

### 推荐训练成本参考（单卡 3090）

| 阶段 | minimind-3 (64M) | minimind-3-moe (198M) |
|---|---|---|
| pretrain_t2t_mini | ≈1.21h / ≈1.57￥ | ≈1.69h / ≈2.20￥ |
| sft_t2t_mini | ≈1.10h / ≈1.43￥ | ≈1.54h / ≈2.00￥ |
| RLAIF (PPO/GRPO) | ≈1.1h / ≈1.43￥ | ≈1.54h / ≈2.00￥ |
