# 分词器使用指南

本项目支持两种分词方式：**字符级分词 (Char)** 和 **BPE 分词 (Byte-Pair Encoding)**。

## 快速对比

| 特性 | 字符级分词 (Char) | BPE 分词 |
|-----|-----------------|---------|
| **词表大小** | ~3000 字符 | 300-5000（可配置） |
| **序列长度** | 较长 | 较短（压缩 20-50%） |
| **训练速度** | 快 | 稍慢 |
| **适用场景** | 中文姓名（2-4字） | 大数据集、长文本 |
| **优势** | 简单、零准备 | 学习常见组合、更高效 |

---

## 一、字符级分词（默认）

### 特点
- ✅ **开箱即用**：无需预训练分词器
- ✅ **简单直接**：每个汉字对应一个 token
- ✅ **适合短文本**：中文姓名通常 2-4 个字符

### 使用方法

#### GPT 模型
```bash
# 默认使用字符级分词
python scripts/train_gpt.py --epochs 100 --device auto

# 显式指定
python scripts/train_gpt.py --tokenizer char --epochs 100
```

#### Transformer 模型
```bash
# 默认使用字符级分词
python scripts/train_transformer.py --epochs 100 --device auto

# 显式指定
python scripts/train_transformer.py --tokenizer char --epochs 100
```

---

## 二、BPE 分词

### 特点
- ✅ **更短序列**：平均减少 20-50% 的 token 数量
- ✅ **学习组合**：自动发现常见字符组合（如"小明"、"建国"）
- ✅ **可配置**：灵活调整词表大小

### 使用方法

#### 1. 首次使用（自动训练分词器）

如果 BPE 分词器不存在，脚本会自动从数据集训练一个新的分词器。

```bash
# GPT 模型 - 使用 BPE 分词，词表大小 500
python scripts/train_gpt.py \
    --tokenizer bpe \
    --bpe_vocab_size 500 \
    --epochs 100

# Transformer 模型 - 使用 BPE 分词，词表大小 500
python scripts/train_transformer.py \
    --tokenizer bpe \
    --bpe_vocab_size 500 \
    --epochs 100
```

**自动过程**：
1. 检测到 `checkpoints/bpe_tokenizer.json` 不存在
2. 从 `data/names.txt` 训练 BPE 分词器
3. 保存分词器到 `checkpoints/bpe_tokenizer.json`
4. 开始模型训练

#### 2. 使用已有分词器

如果已经训练过 BPE 分词器，直接指定路径：

```bash
# GPT 模型 - 使用已有的 BPE 分词器
python scripts/train_gpt.py \
    --tokenizer bpe \
    --bpe_tokenizer_path checkpoints/bpe_tokenizer.json \
    --epochs 100

# Transformer 模型 - 使用已有的 BPE 分词器
python scripts/train_transformer.py \
    --tokenizer bpe \
    --bpe_tokenizer_path checkpoints/bpe_tokenizer.json \
    --epochs 100
```

#### 3. 手动训练 BPE 分词器

也可以先单独训练分词器，再用于模型训练：

```bash
# 训练 BPE 分词器
python scripts/build_bpe_tokenizer.py \
    --vocab-size 500 \
    --output checkpoints/bpe_tokenizer.json

# 然后使用它训练模型
python scripts/train_gpt.py \
    --tokenizer bpe \
    --bpe_tokenizer_path checkpoints/bpe_tokenizer.json \
    --epochs 100
```

---

## 三、参数说明

### 字符级分词参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--tokenizer` | str | `char` | 分词器类型 |

### BPE 分词参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--tokenizer` | str | `char` | 设置为 `bpe` 启用 BPE 分词 |
| `--bpe_tokenizer_path` | str | `checkpoints/bpe_tokenizer.json` | BPE 分词器文件路径 |
| `--bpe_vocab_size` | int | `500` | BPE 词表大小（训练新分词器时） |

---

## 四、完整示例

### 示例 1：快速开始（字符级）
```bash
# 使用默认字符级分词训练 GPT
python scripts/train_gpt.py --epochs 50 --device auto
```

### 示例 2：使用 BPE（自动训练分词器）
```bash
# 第一次运行：自动训练 BPE 分词器并训练模型
python scripts/train_gpt.py \
    --tokenizer bpe \
    --bpe_vocab_size 500 \
    --epochs 100 \
    --device cuda \
    --batch_size 128

# 第二次运行：直接使用已有的 BPE 分词器
python scripts/train_gpt.py \
    --tokenizer bpe \
    --epochs 50 \
    --device cuda
```

### 示例 3：对比两种分词方式
```bash
# 训练字符级模型
python scripts/train_gpt.py \
    --tokenizer char \
    --checkpoint_dir checkpoints/gpt_char \
    --epochs 100

# 训练 BPE 模型
python scripts/train_gpt.py \
    --tokenizer bpe \
    --bpe_vocab_size 500 \
    --checkpoint_dir checkpoints/gpt_bpe \
    --epochs 100
```

### 示例 4：Transformer 模型使用 BPE
```bash
# Transformer + BPE
python scripts/train_transformer.py \
    --tokenizer bpe \
    --bpe_vocab_size 500 \
    --epochs 100 \
    --device mps \
    --batch_size 64
```

---

## 五、词表大小建议

根据数据集大小选择合适的 BPE 词表大小：

| 数据集大小 | 推荐词表大小 | 说明 |
|-----------|------------|------|
| < 10k | 300-500 | 小数据集，避免过拟合 |
| 10k-100k | 500-1000 | 中等数据集 |
| 100k-1M | 1000-3000 | 大数据集，学习更多组合 |
| > 1M | 3000-5000 | 超大数据集 |

**当前数据集（114 万姓名）**：推荐使用 **1000-2000** 的词表大小。

---

## 六、查看分词效果

### 测试 BPE 分词器
```bash
python scripts/test_bpe_tokenizer.py
```

### 对比两种分词方式
```bash
python scripts/compare_tokenizers.py
```

输出示例：
```
=== 字符级分词 vs BPE 分词对比 ===

示例: 张伟
  字符级: ['[BOS]', '张', '伟', '[EOS]'] (4 tokens)
  BPE:    ['[BOS]', '张伟', '[EOS]'] (3 tokens)
  压缩率: 25.00%

示例: 王小明
  字符级: ['[BOS]', '王', '小', '明', '[EOS]'] (5 tokens)
  BPE:    ['[BOS]', '王小明', '[EOS]'] (3 tokens)
  压缩率: 40.00%
```

---

## 七、常见问题

### Q1: 什么时候应该使用 BPE 分词？
**A**: 推荐在以下场景使用 BPE：
- 数据集较大（>10k）
- 希望减少序列长度以提高训练速度
- 希望模型学习常见字符组合

### Q2: BPE 分词器训练需要多久？
**A**: 通常很快：
- 10k 姓名：~5 秒
- 100k 姓名：~30 秒
- 114 万姓名：~2-3 分钟

### Q3: 可以在训练中途切换分词器吗？
**A**: 不建议。不同分词器的词表完全不同，模型参数不兼容。应该从头开始训练。

### Q4: BPE 分词器保存在哪里？
**A**: 默认保存在 `checkpoints/bpe_tokenizer.json`，可以通过 `--bpe_tokenizer_path` 修改。

### Q5: 如何选择词表大小？
**A**: 参考上面的"词表大小建议"表格。经验法则：
- 词表太小（<300）：压缩效果差
- 词表太大（>5000）：接近字符级，失去 BPE 优势

---

## 八、技术细节

### 模型参数对比

**字符级分词**（词表 ~3000）：
```python
model = GPTModel(
    vocab_size=3000,
    pad_idx=0,
    tokenizer_type='char'
)
# 参数量: ~1.2M
```

**BPE 分词**（词表 500）：
```python
model = GPTModel(
    vocab_size=500,
    pad_idx=0,
    tokenizer_type='bpe'
)
# 参数量: ~1.0M
```

### 数据加载

两种分词器使用相同的数据加载接口：

```python
from data.vocabulary import Vocabulary
from data.bpe_tokenizer import BPETokenizer
from data.dataset import create_data_loaders

# 字符级
vocab = Vocabulary()
vocab.build_from_texts(names)
train_loader, val_loader = create_data_loaders(names, vocab, batch_size=64)

# BPE
tokenizer = BPETokenizer.load('checkpoints/bpe_tokenizer.json')
train_loader, val_loader = create_data_loaders(names, tokenizer, batch_size=64)
```

---

## 九、总结

| 场景 | 推荐分词方式 | 命令 |
|-----|------------|------|
| **快速开始** | 字符级 | `python scripts/train_gpt.py` |
| **生产环境** | BPE | `python scripts/train_gpt.py --tokenizer bpe` |
| **小数据集（<10k）** | 字符级 | `python scripts/train_gpt.py --tokenizer char` |
| **大数据集（>100k）** | BPE | `python scripts/train_gpt.py --tokenizer bpe --bpe_vocab_size 1000` |

现在你可以根据需求灵活选择分词方式了！🎯
