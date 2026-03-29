# 🚀 分词器快速入门

## 一行命令开始训练

### 字符级分词（默认）
```bash
python scripts/train_gpt.py --epochs 50
```

### BPE 分词（自动训练分词器）
```bash
python scripts/train_gpt.py --tokenizer bpe --epochs 50
```

---

## 核心参数

| 参数 | 值 | 说明 |
|-----|---|------|
| `--tokenizer` | `char` (默认) | 字符级分词 |
|  | `bpe` | BPE 分词 |
| `--bpe_vocab_size` | `500` (默认) | BPE 词表大小 |
| `--bpe_tokenizer_path` | `checkpoints/bpe_tokenizer.json` | BPE 分词器路径 |

---

## 常用命令

### GPT 模型

```bash
# 字符级（默认）
python scripts/train_gpt.py --epochs 100 --device auto

# BPE（词表 500）
python scripts/train_gpt.py --tokenizer bpe --bpe_vocab_size 500 --epochs 100

# BPE（使用已有分词器）
python scripts/train_gpt.py --tokenizer bpe --bpe_tokenizer_path checkpoints/bpe_tokenizer.json
```

### Transformer 模型

```bash
# 字符级（默认）
python scripts/train_transformer.py --epochs 100 --device auto

# BPE（词表 500）
python scripts/train_transformer.py --tokenizer bpe --bpe_vocab_size 500 --epochs 100

# BPE（使用已有分词器）
python scripts/train_transformer.py --tokenizer bpe --bpe_tokenizer_path checkpoints/bpe_tokenizer.json
```

---

## 选择指南

| 场景 | 推荐 | 词表大小 |
|-----|-----|---------|
| 快速原型 | `char` | N/A |
| 小数据集 (<10k) | `char` | N/A |
| 大数据集 (>100k) | `bpe` | 1000-2000 |
| 超大数据集 (>1M) | `bpe` | 3000-5000 |

---

## 完整工作流

```bash
# 1. 训练 BPE 分词器（可选，首次使用会自动训练）
python scripts/build_bpe_tokenizer.py --vocab-size 500

# 2. 训练模型
python scripts/train_gpt.py \
    --tokenizer bpe \
    --epochs 100 \
    --device cuda \
    --batch_size 128

# 3. 测试集成
python scripts/test_tokenizer_integration.py
```

---

## 查看帮助

```bash
# GPT 模型
python scripts/train_gpt.py --help

# Transformer 模型
python scripts/train_transformer.py --help
```

---

## 测试验证

```bash
# 运行所有集成测试
python scripts/test_tokenizer_integration.py

# 对比两种分词方式
python scripts/compare_tokenizers.py
```

---

## 更多信息

- **详细文档**: `docs/TOKENIZER_GUIDE.md`
- **完整总结**: `TOKENIZER_INTEGRATION_SUMMARY.md`
- **BPE 教程**: `docs/BPE_TOKENIZER.md`

现在开始训练你的模型吧！🎯
