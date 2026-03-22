#!/usr/bin/env python
"""
人名生成推理脚本

支持三种生成模式：
1. random: 完全随机生成人名
2. surname: 给定姓氏生成名字
3. complete: 给定部分字补全人名

使用方法:
    # 随机生成
    python scripts/generate.py --model gpt --mode random --num 10
    
    # 给定姓氏生成
    python scripts/generate.py --model transformer --mode surname --prefix "张" --num 5
    
    # 补全人名
    python scripts/generate.py --model gpt --mode complete --prefix "李小" --num 5
    
    # 使用不同采样策略
    python scripts/generate.py --model gpt --mode random --temperature 0.8 --top_k 50
    python scripts/generate.py --model gpt --mode random --top_p 0.9
"""

import argparse
import sys
import torch
from pathlib import Path
from typing import List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.vocabulary import Vocabulary
from models.transformer import TransformerModel
from models.gpt import GPTModel
from utils.device import get_device, get_device_info, safe_multinomial


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="中文人名生成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        default="gpt",
        choices=["transformer", "gpt"],
        help="模型类型"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="检查点路径（默认使用 best.pt）"
    )
    
    # 生成模式
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "surname", "complete"],
        help="生成模式: random=随机生成, surname=姓氏生成, complete=补全"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="前缀（用于 surname 和 complete 模式）"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="生成数量"
    )
    
    # 采样参数
    parser.add_argument(
        "--max_length",
        type=int,
        default=10,
        help="最大生成长度"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="温度参数（越高越随机）"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-K 采样"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-P (nucleus) 采样"
    )
    
    # 设备参数
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="推理设备"
    )
    
    return parser.parse_args()


def load_model(
    model_type: str,
    checkpoint_path: str,
    device: torch.device
):
    """
    加载模型
    
    Args:
        model_type: 模型类型
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        (model, vocab) 元组
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # 创建模型
    if model_type == "transformer":
        model = TransformerModel(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            d_ff=model_config['d_ff'],
            max_len=model_config['max_len'],
            dropout=0.0,  # 推理时不使用 dropout
            pad_idx=model_config['pad_idx'],
        )
    else:
        model = GPTModel(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            d_ff=model_config['d_ff'],
            max_len=model_config['max_len'],
            dropout=0.0,
            pad_idx=model_config['pad_idx'],
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 加载词表
    vocab_path = Path(checkpoint_path).parent / "vocab.json"
    vocab = Vocabulary.load(str(vocab_path))
    
    return model, vocab


def generate_names(
    model,
    vocab: Vocabulary,
    device: torch.device,
    mode: str = "random",
    prefix: str = "",
    num_samples: int = 10,
    max_length: int = 10,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
) -> List[str]:
    """
    生成人名
    
    Args:
        model: 模型实例
        vocab: 词表
        device: 设备
        mode: 生成模式
        prefix: 前缀
        num_samples: 生成数量
        max_length: 最大长度
        temperature: 温度
        top_k: Top-K 采样
        top_p: Top-P 采样
    
    Returns:
        生成的人名列表
    """
    generated_names = []
    
    for _ in range(num_samples):
        # 构建起始序列
        if mode == "random":
            # 随机生成：只有 BOS
            start_tokens = torch.tensor([[vocab.bos_idx]], device=device)
        elif mode == "surname":
            # 姓氏生成：BOS + 姓氏
            if not prefix:
                prefix = "张"  # 默认姓氏
            encoded = vocab.encode(prefix, add_bos=True, add_eos=False)
            start_tokens = torch.tensor([encoded], device=device)
        else:  # complete
            # 补全模式：BOS + 已知字符
            if not prefix:
                prefix = "李"
            encoded = vocab.encode(prefix, add_bos=True, add_eos=False)
            start_tokens = torch.tensor([encoded], device=device)
        
        # 生成
        with torch.no_grad():
            generated = model.generate(
                start_tokens=start_tokens,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_idx=vocab.eos_idx,
            )
        
        # 解码
        tokens = generated[0].tolist()
        name = vocab.decode(tokens, remove_special=True)
        generated_names.append(name)
    
    return generated_names


def main():
    """主函数"""
    args = parse_args()
    
    # 获取设备
    device = get_device(args.device)
    device_info = get_device_info(device)
    print(f"使用设备: {device_info['device']} ({device_info['name']})")
    
    # 确定检查点路径
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = project_root / f"checkpoints/{args.model}/best.pt"
    
    if not checkpoint_path.exists():
        print(f"错误: 检查点不存在: {checkpoint_path}")
        print("请先训练模型，或指定正确的检查点路径")
        sys.exit(1)
    
    print(f"加载模型: {checkpoint_path}")
    
    # 加载模型
    model, vocab = load_model(args.model, str(checkpoint_path), device)
    
    print(f"词表大小: {len(vocab)}")
    print(f"模型参数量: {model.get_num_params():,}")
    
    # 生成人名
    print(f"\n生成模式: {args.mode}")
    if args.prefix:
        print(f"前缀: {args.prefix}")
    print(f"生成数量: {args.num}")
    print(f"采样参数: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("-" * 30)
    
    names = generate_names(
        model=model,
        vocab=vocab,
        device=device,
        mode=args.mode,
        prefix=args.prefix,
        num_samples=args.num,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    
    # 打印结果
    print("生成的人名:")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    
    # 去重统计
    unique_names = set(names)
    print(f"\n唯一人名数: {len(unique_names)}/{len(names)}")


if __name__ == "__main__":
    main()
