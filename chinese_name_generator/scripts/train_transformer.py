#!/usr/bin/env python
"""
Transformer 模型训练脚本

使用方法:
    # 字符级分词（默认）
    python scripts/train_transformer.py --epochs 100 --device auto
    python scripts/train_transformer.py --epochs 50 --device cuda --batch_size 128
    
    # BPE 分词
    python scripts/train_transformer.py --tokenizer bpe --bpe_vocab_size 500 --epochs 100
    python scripts/train_transformer.py --tokenizer bpe --bpe_tokenizer_path checkpoints/bpe_tokenizer.json
    
    # 自定义数据
    python scripts/train_transformer.py --device mps --data_path data/custom_names.txt
"""

import argparse
import sys
import random
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.vocabulary import Vocabulary
from data.bpe_tokenizer import BPETokenizer
from data.dataset import load_names_from_file, create_data_loaders
from models.transformer import TransformerModel
from trainers.trainer import Trainer
from utils.config import ModelConfig, TrainConfig
from utils.device import get_device, print_device_info, should_pin_memory


def set_seed(seed: int):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="训练 Transformer 人名生成模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据参数
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/names.txt",
        help="人名数据文件路径"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="char",
        choices=["char", "bpe"],
        help="分词器类型: 'char' 为字符级分词，'bpe' 为 BPE 分词"
    )
    parser.add_argument(
        "--bpe_tokenizer_path",
        type=str,
        default="checkpoints/bpe_tokenizer.json",
        help="BPE 分词器路径（仅当 --tokenizer=bpe 时使用）"
    )
    parser.add_argument(
        "--bpe_vocab_size",
        type=int,
        default=500,
        help="BPE 词表大小（如果分词器不存在则训练新的）"
    )
    
    # 模型参数
    parser.add_argument("--d_model", type=int, default=128, help="模型维度")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=4, help="层数")
    parser.add_argument("--d_ff", type=int, default=512, help="前馈网络隐藏层维度")
    parser.add_argument("--max_len", type=int, default=20, help="最大序列长度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比例")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=100, help="学习率预热步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值")
    
    # 设备参数
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="训练设备"
    )
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载进程数")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度训练")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/transformer",
        help="检查点保存目录"
    )
    parser.add_argument("--save_freq", type=int, default=10, help="保存频率")
    parser.add_argument("--val_freq", type=int, default=1, help="验证频率")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 打印设备信息
    print_device_info()
    
    # 解析数据路径
    data_path = project_root / args.data_path
    if not data_path.exists():
        print(f"错误: 数据文件不存在: {data_path}")
        sys.exit(1)
    
    # 加载数据
    print(f"\n加载数据: {data_path}")
    names = load_names_from_file(str(data_path))
    print(f"人名数量: {len(names)}")
    
    # 构建或加载分词器
    print(f"\n分词器类型: {args.tokenizer}")
    
    if args.tokenizer == "char":
        # 字符级分词
        tokenizer = Vocabulary()
        tokenizer.build_from_texts(names)
        print(f"词表大小: {len(tokenizer)}")
        tokenizer_type = "char"
    elif args.tokenizer == "bpe":
        # BPE 分词
        bpe_path = project_root / args.bpe_tokenizer_path
        
        if bpe_path.exists():
            print(f"加载 BPE 分词器: {bpe_path}")
            tokenizer = BPETokenizer.load(str(bpe_path))
        else:
            print(f"BPE 分词器不存在，训练新的分词器...")
            tokenizer = BPETokenizer()
            tokenizer.train_from_file(
                str(data_path),
                vocab_size=args.bpe_vocab_size,
                show_progress=True
            )
            # 保存分词器
            bpe_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(bpe_path))
            print(f"BPE 分词器已保存到: {bpe_path}")
        
        print(f"BPE 词表大小: {tokenizer.get_vocab_size()}")
        tokenizer_type = "bpe"
    else:
        raise ValueError(f"不支持的分词器类型: {args.tokenizer}")
    
    # 统一接口：获取词表大小和 pad_idx
    vocab_size = len(tokenizer) if args.tokenizer == "char" else tokenizer.get_vocab_size()
    pad_idx = tokenizer.pad_idx
    
    # 创建模型配置
    model_config = ModelConfig(
        model_type="transformer",
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
        pad_idx=pad_idx,
    )
    
    # 创建训练配置
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        train_ratio=args.train_ratio,
        patience=args.patience,
        val_freq=args.val_freq,
        save_freq=args.save_freq,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        checkpoint_dir=str(project_root / args.checkpoint_dir),
        use_amp=args.use_amp,
    )
    
    # 获取设备
    device = get_device(train_config.device)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        names=names,
        tokenizer=tokenizer,
        batch_size=train_config.batch_size,
        train_ratio=train_config.train_ratio,
        max_length=model_config.max_len,
        shuffle=True,
        pin_memory=should_pin_memory(device),
        num_workers=train_config.num_workers,
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # 创建模型
    model = TransformerModel(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        d_ff=model_config.d_ff,
        max_len=model_config.max_len,
        dropout=model_config.dropout,
        pad_idx=model_config.pad_idx,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_config=train_config,
        model_config=model_config,
        vocab=tokenizer,  # 统一使用 tokenizer（兼容 Vocabulary 和 BPETokenizer）
    )
    
    # 开始训练
    history = trainer.train(train_loader, val_loader)
    
    print("\n训练完成！")
    print(f"检查点保存在: {train_config.checkpoint_dir}")
    print(f"最佳模型: {train_config.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
