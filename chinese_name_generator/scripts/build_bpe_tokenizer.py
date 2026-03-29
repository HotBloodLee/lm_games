"""
构建 BPE 分词器

从 names.txt 数据集训练 BPE 分词器
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.bpe_tokenizer import BPETokenizer
import argparse


def main():
    parser = argparse.ArgumentParser(description='从 names.txt 构建 BPE 分词器')
    parser.add_argument(
        '--data-file',
        type=str,
        default='data/names.txt',
        help='训练数据文件路径'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=500,
        help='词表大小（包含特殊 token）'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=2,
        help='最小词频阈值'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='checkpoints/bpe_tokenizer.json',
        help='输出分词器文件路径'
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    data_file = project_root / args.data_file
    output_file = project_root / args.output
    
    print("=" * 60)
    print("BPE 分词器构建工具")
    print("=" * 60)
    print(f"数据文件: {data_file}")
    print(f"词表大小: {args.vocab_size}")
    print(f"最小词频: {args.min_frequency}")
    print(f"输出路径: {output_file}")
    print("=" * 60)
    
    # 检查数据文件
    if not data_file.exists():
        print(f"错误: 数据文件不存在: {data_file}")
        return
    
    # 读取数据统计
    with open(data_file, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    
    print(f"\n✓ 成功读取 {len(names):,} 个姓名")
    
    # 统计字符数
    all_chars = set()
    for name in names:
        all_chars.update(name)
    print(f"✓ 数据集包含 {len(all_chars)} 个不同字符")
    
    # 统计长度分布
    lengths = [len(name) for name in names]
    print(f"✓ 姓名长度范围: {min(lengths)} - {max(lengths)}")
    print(f"✓ 平均长度: {sum(lengths) / len(lengths):.2f}")
    
    # 创建并训练分词器
    print(f"\n开始训练 BPE 分词器...")
    tokenizer = BPETokenizer()
    tokenizer.train_from_file(
        data_file,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True
    )
    
    # 保存分词器
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_file)
    print(f"\n✓ 分词器已保存到: {output_file}")
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("分词器统计信息:")
    print("=" * 60)
    stats = tokenizer.get_stats()
    for key, value in stats.items():
        print(f"{key:20s}: {value}")
    
    # 测试编码解码
    print("\n" + "=" * 60)
    print("测试样例:")
    print("=" * 60)
    
    # 随机选择一些姓名进行测试
    import random
    random.seed(42)
    test_samples = random.sample(names, min(10, len(names)))
    
    for name in test_samples:
        encoded = tokenizer.encode(name, add_special_tokens=True)
        tokens = tokenizer.tokenize(name)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        
        # 检查是否正确解码
        status = "✓" if decoded == name else "✗"
        print(f"{status} 原文: {name:10s} | Tokens: {len(tokens):2d} | 编码: {encoded}")
    
    # 显示一些高频 token
    print("\n" + "=" * 60)
    print("词表预览 (前 30 个 token):")
    print("=" * 60)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])[:30]
    
    for token, idx in sorted_vocab:
        # 转义特殊字符以便显示
        display_token = token.replace('\n', '\\n').replace('\t', '\\t')
        print(f"  [{idx:3d}] {display_token}")
    
    print("\n" + "=" * 60)
    print("✓ BPE 分词器构建完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
