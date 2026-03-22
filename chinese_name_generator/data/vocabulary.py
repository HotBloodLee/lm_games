"""
词表管理类

实现字符到索引的双向映射，包含特殊 token：
- PAD: 填充符
- BOS: 序列开始符
- EOS: 序列结束符  
- UNK: 未知字符
"""

from typing import List, Dict, Optional
import json
from pathlib import Path


class Vocabulary:
    """
    字符级词表类
    
    用于管理字符与索引之间的映射关系
    """
    
    # 特殊 token
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self):
        """初始化词表"""
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """初始化特殊 token"""
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for token in special_tokens:
            self._add_char(token)
    
    def _add_char(self, char: str) -> int:
        """
        添加字符到词表
        
        Args:
            char: 要添加的字符
        
        Returns:
            字符对应的索引
        """
        if char not in self.char2idx:
            idx = len(self.char2idx)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
        return self.char2idx[char]
    
    def build_from_texts(self, texts: List[str]):
        """
        从文本列表构建词表
        
        Args:
            texts: 文本列表（如人名列表）
        """
        for text in texts:
            for char in text:
                self._add_char(char)
    
    def build_from_file(self, file_path: str):
        """
        从文件构建词表（每行一个文本）
        
        Args:
            file_path: 文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        self.build_from_texts(texts)
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        将文本编码为索引序列
        
        Args:
            text: 输入文本
            add_bos: 是否添加序列开始符
            add_eos: 是否添加序列结束符
        
        Returns:
            索引序列
        """
        indices = []
        
        if add_bos:
            indices.append(self.bos_idx)
        
        for char in text:
            if char in self.char2idx:
                indices.append(self.char2idx[char])
            else:
                indices.append(self.unk_idx)
        
        if add_eos:
            indices.append(self.eos_idx)
        
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        将索引序列解码为文本
        
        Args:
            indices: 索引序列
            remove_special: 是否移除特殊 token
        
        Returns:
            解码后的文本
        """
        chars = []
        special_indices = {self.pad_idx, self.bos_idx, self.eos_idx, self.unk_idx}
        
        for idx in indices:
            if remove_special and idx in special_indices:
                continue
            if idx in self.idx2char:
                chars.append(self.idx2char[idx])
        
        return ''.join(chars)
    
    @property
    def pad_idx(self) -> int:
        """PAD token 的索引"""
        return self.char2idx[self.PAD_TOKEN]
    
    @property
    def bos_idx(self) -> int:
        """BOS token 的索引"""
        return self.char2idx[self.BOS_TOKEN]
    
    @property
    def eos_idx(self) -> int:
        """EOS token 的索引"""
        return self.char2idx[self.EOS_TOKEN]
    
    @property
    def unk_idx(self) -> int:
        """UNK token 的索引"""
        return self.char2idx[self.UNK_TOKEN]
    
    def __len__(self) -> int:
        """词表大小"""
        return len(self.char2idx)
    
    def __contains__(self, char: str) -> bool:
        """检查字符是否在词表中"""
        return char in self.char2idx
    
    def save(self, file_path: str):
        """
        保存词表到文件
        
        Args:
            file_path: 保存路径
        """
        data = {
            'char2idx': self.char2idx,
            'idx2char': {str(k): v for k, v in self.idx2char.items()}
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> 'Vocabulary':
        """
        从文件加载词表
        
        Args:
            file_path: 词表文件路径
        
        Returns:
            Vocabulary 实例
        """
        vocab = cls.__new__(cls)
        vocab.char2idx = {}
        vocab.idx2char = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab.char2idx = data['char2idx']
        vocab.idx2char = {int(k): v for k, v in data['idx2char'].items()}
        
        return vocab
    
    def get_stats(self) -> dict:
        """
        获取词表统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'vocab_size': len(self),
            'special_tokens': 4,
            'regular_chars': len(self) - 4,
        }


if __name__ == "__main__":
    # 测试代码
    vocab = Vocabulary()
    
    # 测试从文本构建
    names = ["张伟", "王芳", "李娜", "刘洋", "陈静"]
    vocab.build_from_texts(names)
    
    print(f"词表大小: {len(vocab)}")
    print(f"词表统计: {vocab.get_stats()}")
    
    # 测试编码解码
    text = "张伟"
    encoded = vocab.encode(text)
    decoded = vocab.decode(encoded)
    print(f"原文: {text}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")
