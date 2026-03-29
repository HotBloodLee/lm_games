"""
数据集类

实现 PyTorch Dataset 接口，用于加载和预处理人名数据
支持字符级分词（Vocabulary）和 BPE 分词（BPETokenizer）
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Union
from pathlib import Path
from .vocabulary import Vocabulary

try:
    from .bpe_tokenizer import BPETokenizer
    BPE_AVAILABLE = True
except ImportError:
    BPETokenizer = None
    BPE_AVAILABLE = False


class NameDataset(Dataset):
    """
    人名数据集类
    
    将人名转换为模型可用的张量格式
    支持字符级分词（Vocabulary）和 BPE 分词（BPETokenizer）
    """
    
    def __init__(
        self,
        names: List[str],
        tokenizer: Union[Vocabulary, 'BPETokenizer'],
        max_length: Optional[int] = None
    ):
        """
        初始化数据集
        
        Args:
            names: 人名列表
            tokenizer: 分词器实例（Vocabulary 或 BPETokenizer）
            max_length: 最大序列长度（包含 BOS 和 EOS），None 表示使用数据中的最大长度
        """
        self.names = names
        self.tokenizer = tokenizer
        
        # 检测分词器类型
        self.is_bpe = BPE_AVAILABLE and isinstance(tokenizer, BPETokenizer)
        
        # 计算最大长度（+2 是为了 BOS 和 EOS）
        if max_length is None:
            self.max_length = max(len(name) for name in names) + 2
        else:
            self.max_length = max_length
        
        # 预处理：编码所有人名
        if self.is_bpe:
            # BPE 分词器
            self.encoded_names = [
                tokenizer.encode(name, add_special_tokens=True) 
                for name in names
            ]
        else:
            # 字符级分词器
            self.encoded_names = [tokenizer.encode(name) for name in names]
    
    def __len__(self) -> int:
        """数据集大小"""
        return len(self.names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            (input_ids, target_ids) 元组
            - input_ids: 输入序列 [BOS, token1, token2, ..., PAD, ...]
            - target_ids: 目标序列 [token1, token2, ..., EOS, PAD, ...]
        """
        encoded = self.encoded_names[idx]
        
        # 输入：BOS + token 序列（不含 EOS）
        input_ids = encoded[:-1]  # 移除 EOS
        
        # 目标：token 序列 + EOS（不含 BOS）
        target_ids = encoded[1:]  # 移除 BOS
        
        # 填充到最大长度
        input_ids = self._pad_sequence(input_ids, self.max_length - 1)
        target_ids = self._pad_sequence(target_ids, self.max_length - 1)
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long)
        )
    
    def _pad_sequence(self, seq: List[int], max_len: int) -> List[int]:
        """
        填充序列到指定长度
        
        Args:
            seq: 原始序列
            max_len: 目标长度
        
        Returns:
            填充后的序列
        """
        if len(seq) >= max_len:
            return seq[:max_len]
        
        # 获取 PAD 索引
        if self.is_bpe:
            pad_idx = self.tokenizer.pad_idx
        else:
            pad_idx = self.tokenizer.pad_idx
        
        return seq + [pad_idx] * (max_len - len(seq))
    
    def get_raw_name(self, idx: int) -> str:
        """获取原始人名"""
        return self.names[idx]


def load_names_from_file(file_path: str) -> List[str]:
    """
    从文件加载人名列表
    
    Args:
        file_path: 文件路径（每行一个人名）
    
    Returns:
        人名列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def create_data_loaders(
    names: List[str],
    tokenizer: Union[Vocabulary, 'BPETokenizer'],
    batch_size: int = 64,
    train_ratio: float = 0.9,
    max_length: Optional[int] = None,
    shuffle: bool = True,
    pin_memory: bool = False,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        names: 人名列表
        tokenizer: 分词器实例（Vocabulary 或 BPETokenizer）
        batch_size: 批次大小
        train_ratio: 训练集比例
        max_length: 最大序列长度
        shuffle: 是否打乱数据
        pin_memory: 是否使用 pin_memory（CUDA 优化）
        num_workers: 数据加载的工作进程数
    
    Returns:
        (train_loader, val_loader) 元组
    """
    # 划分数据集
    split_idx = int(len(names) * train_ratio)
    train_names = names[:split_idx]
    val_names = names[split_idx:]
    
    # 创建数据集
    train_dataset = NameDataset(train_names, tokenizer, max_length)
    val_dataset = NameDataset(val_names, tokenizer, max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


class NameDatasetForGeneration(Dataset):
    """
    用于生成任务的数据集
    
    提供前缀（姓氏或部分字符）作为输入
    支持字符级分词和 BPE 分词
    """
    
    def __init__(
        self,
        names: List[str],
        tokenizer: Union[Vocabulary, 'BPETokenizer'],
        prefix_length: int = 1
    ):
        """
        初始化数据集
        
        Args:
            names: 人名列表
            tokenizer: 分词器实例（Vocabulary 或 BPETokenizer）
            prefix_length: 前缀长度（用于条件生成）
        """
        self.names = names
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.is_bpe = BPE_AVAILABLE and isinstance(tokenizer, BPETokenizer)
    
    def __len__(self) -> int:
        return len(self.names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        获取样本
        
        Returns:
            (prefix_ids, full_name) 元组
        """
        name = self.names[idx]
        prefix = name[:self.prefix_length]
        
        if self.is_bpe:
            # BPE 分词器：只添加 BOS，不添加 EOS
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            prefix_ids = [self.tokenizer.bos_idx] + prefix_ids
        else:
            # 字符级分词器
            prefix_ids = self.tokenizer.encode(prefix, add_bos=True, add_eos=False)
        
        return torch.tensor(prefix_ids, dtype=torch.long), name


if __name__ == "__main__":
    # 测试代码
    from pathlib import Path
    
    # 测试数据
    test_names = ["张伟", "王芳", "李娜", "刘洋", "陈静", "杨敏", "黄强", "周杰"]
    
    # 创建词表
    vocab = Vocabulary()
    vocab.build_from_texts(test_names)
    
    # 创建数据集
    dataset = NameDataset(test_names, vocab)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"最大长度: {dataset.max_length}")
    
    # 测试单个样本
    input_ids, target_ids = dataset[0]
    print(f"\n样本 0:")
    print(f"  原始人名: {dataset.get_raw_name(0)}")
    print(f"  输入 IDs: {input_ids.tolist()}")
    print(f"  目标 IDs: {target_ids.tolist()}")
    print(f"  输入解码: {vocab.decode(input_ids.tolist())}")
    print(f"  目标解码: {vocab.decode(target_ids.tolist())}")
    
    # 测试数据加载器
    train_loader, val_loader = create_data_loaders(
        test_names, vocab, batch_size=2, train_ratio=0.75
    )
    
    print(f"\n训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
