# Chinese Name Generator

基于深度学习的中文人名生成器，使用 PyTorch 实现，支持 Transformer 和 GPT 两种模型架构。

## 功能特性

- **双模型架构**：独立实现 Transformer 和 GPT 两种语言模型
- **多设备支持**：自动检测并支持 CUDA（NVIDIA GPU）、MPS（Mac Apple Silicon）、CPU
- **三种生成模式**：
  - 给定姓氏生成名字（如：输入"张"，输出"张伟"）
  - 完全随机生成完整人名
  - 给定部分字补全人名
- **多种采样策略**：贪婪解码、温度采样、Top-K、Top-P（nucleus sampling）

## 环境要求

- Python 3.9+
- PyTorch 2.0+

## 安装

```bash
# 克隆项目
cd chinese_name_generator

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据

将人名语料放入 `data/names.txt`，每行一个人名：

```
张伟
王芳
李娜
...
```

### 2. 训练模型

**训练 Transformer 模型：**

```bash
python scripts/train_transformer.py --epochs 100 --device auto
```

**训练 GPT 模型：**

```bash
python scripts/train_gpt.py --epochs 100 --device auto
```

**设备选择参数：**
- `--device auto`：自动选择最佳设备（CUDA > MPS > CPU）
- `--device cuda`：强制使用 NVIDIA GPU
- `--device mps`：强制使用 Mac MPS
- `--device cpu`：强制使用 CPU

### 3. 生成人名

```bash
# 随机生成人名
python scripts/generate.py --model gpt --mode random --num 10

# 给定姓氏生成
python scripts/generate.py --model transformer --mode surname --prefix "张"

# 补全人名
python scripts/generate.py --model gpt --mode complete --prefix "李小"
```

## 项目结构

```
chinese_name_generator/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖列表
├── data/
│   ├── vocabulary.py            # 词表管理
│   ├── dataset.py               # 数据集类
│   └── names.txt                # 训练语料
├── models/
│   ├── transformer.py           # Transformer 模型
│   └── gpt.py                   # GPT 模型
├── trainers/
│   └── trainer.py               # 训练器
├── utils/
│   ├── config.py                # 配置管理
│   └── device.py                # 设备管理
├── scripts/
│   ├── train_transformer.py     # Transformer 训练脚本
│   ├── train_gpt.py             # GPT 训练脚本
│   └── generate.py              # 推理脚本
└── checkpoints/                 # 模型保存目录
```

## 模型说明

### Transformer 模型

基于标准 Transformer Encoder 架构，使用自注意力机制捕捉字符间的依赖关系。

### GPT 模型

基于 GPT（Decoder-only）架构，使用因果注意力掩码实现自回归生成。

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--embed_dim` | 128 | 嵌入维度 |
| `--num_heads` | 4 | 注意力头数 |
| `--num_layers` | 4 | 层数 |
| `--dropout` | 0.1 | Dropout 率 |
| `--device` | auto | 训练设备 |

## License

MIT License
