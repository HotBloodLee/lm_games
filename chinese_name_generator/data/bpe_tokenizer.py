"""
BPE 分词器

基于 Byte-Pair Encoding (BPE) 算法的分词器实现
使用 tokenizers 库训练和构建分词器
"""

from typing import List, Optional, Union
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


class BPETokenizer:
    """
    BPE 分词器类
    
    用于训练和使用 BPE 分词器，支持字符级和子词级分词
    """
    
    # 特殊 token
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        """
        初始化 BPE 分词器
        
        Args:
            tokenizer: 已有的 tokenizer 实例（可选）
        """
        if tokenizer is None:
            self.tokenizer = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
        else:
            self.tokenizer = tokenizer
        
        # 特殊 token 列表
        self.special_tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN
        ]
    
    def train_from_file(
        self,
        file_path: Union[str, Path],
        vocab_size: int = 500,
        min_frequency: int = 2,
        show_progress: bool = True
    ):
        """
        从文件训练 BPE 分词器
        
        Args:
            file_path: 训练数据文件路径（每行一个文本）
            vocab_size: 词表大小（包含特殊 token）
            min_frequency: 最小词频阈值
            show_progress: 是否显示训练进度
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 创建 trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
            show_progress=show_progress,
            initial_alphabet=[],  # 自动从数据中学习
        )
        
        # 不使用预处理器，直接处理字符
        # 中文字符不需要特殊的预分词
        
        # 训练
        self.tokenizer.train([str(file_path)], trainer)
        
        # 配置后处理（添加 BOS 和 EOS）
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.BOS_TOKEN} $A {self.EOS_TOKEN}",
            special_tokens=[
                (self.BOS_TOKEN, self.bos_idx),
                (self.EOS_TOKEN, self.eos_idx),
            ],
        )
        
        # 启用填充
        self.tokenizer.enable_padding(
            pad_id=self.pad_idx,
            pad_token=self.PAD_TOKEN
        )
    
    def train_from_texts(
        self,
        texts: List[str],
        vocab_size: int = 500,
        min_frequency: int = 2,
        show_progress: bool = True
    ):
        """
        从文本列表训练 BPE 分词器
        
        Args:
            texts: 文本列表
            vocab_size: 词表大小
            min_frequency: 最小词频阈值
            show_progress: 是否显示训练进度
        """
        # 创建 trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
            show_progress=show_progress,
            initial_alphabet=[],
        )
        
        # 不使用预处理器
        
        # 训练
        self.tokenizer.train_from_iterator(texts, trainer)
        
        # 配置后处理
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.BOS_TOKEN} $A {self.EOS_TOKEN}",
            special_tokens=[
                (self.BOS_TOKEN, self.bos_idx),
                (self.EOS_TOKEN, self.eos_idx),
            ],
        )
        
        # 启用填充
        self.tokenizer.enable_padding(
            pad_id=self.pad_idx,
            pad_token=self.PAD_TOKEN
        )
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        编码文本为 token IDs
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊 token (BOS/EOS)
        
        Returns:
            token ID 列表
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True
    ) -> List[List[int]]:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            add_special_tokens: 是否添加特殊 token
        
        Returns:
            token ID 列表的列表
        """
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        解码 token IDs 为文本
        
        Args:
            token_ids: token ID 列表
            skip_special_tokens: 是否跳过特殊 token
        
        Returns:
            解码后的文本
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_batch(
        self,
        token_ids_batch: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        批量解码
        
        Args:
            token_ids_batch: token ID 列表的列表
            skip_special_tokens: 是否跳过特殊 token
        
        Returns:
            解码后的文本列表
        """
        return [
            self.decode(token_ids, skip_special_tokens)
            for token_ids in token_ids_batch
        ]
    
    def get_vocab(self) -> dict:
        """
        获取词表
        
        Returns:
            词表字典 {token: id}
        """
        return self.tokenizer.get_vocab()
    
    def get_vocab_size(self) -> int:
        """
        获取词表大小
        
        Returns:
            词表大小
        """
        return self.tokenizer.get_vocab_size()
    
    @property
    def pad_idx(self) -> int:
        """PAD token 的索引"""
        return self.tokenizer.token_to_id(self.PAD_TOKEN)
    
    @property
    def bos_idx(self) -> int:
        """BOS token 的索引"""
        return self.tokenizer.token_to_id(self.BOS_TOKEN)
    
    @property
    def eos_idx(self) -> int:
        """EOS token 的索引"""
        return self.tokenizer.token_to_id(self.EOS_TOKEN)
    
    @property
    def unk_idx(self) -> int:
        """UNK token 的索引"""
        return self.tokenizer.token_to_id(self.UNK_TOKEN)
    
    def save(self, file_path: Union[str, Path], pretty: bool = True):
        """
        保存分词器到文件
        
        Args:
            file_path: 保存路径
            pretty: 是否格式化 JSON
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(file_path), pretty=pretty)
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'BPETokenizer':
        """
        从文件加载分词器
        
        Args:
            file_path: 分词器文件路径
        
        Returns:
            BPETokenizer 实例
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        tokenizer = Tokenizer.from_file(str(file_path))
        return cls(tokenizer)
    
    def get_stats(self) -> dict:
        """
        获取分词器统计信息
        
        Returns:
            统计信息字典
        """
        vocab = self.get_vocab()
        return {
            'vocab_size': self.get_vocab_size(),
            'special_tokens': len(self.special_tokens),
            'regular_tokens': self.get_vocab_size() - len(self.special_tokens),
            'pad_idx': self.pad_idx,
            'bos_idx': self.bos_idx,
            'eos_idx': self.eos_idx,
            'unk_idx': self.unk_idx,
        }
    
    def tokenize(self, text: str) -> List[str]:
        """
        将文本分词为 token 字符串列表（用于调试）
        
        Args:
            text: 输入文本
        
        Returns:
            token 字符串列表
        """
        encoding = self.tokenizer.encode(text)
        return encoding.tokens


if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("BPE 分词器测试")
    print("=" * 50)
    
    # 创建示例数据
    sample_names = [
        "张伟", "王芳", "李娜", "刘洋", "陈静",
        "杨帆", "赵敏", "孙悦", "周杰", "吴涛",
        "阿安", "阿彬", "阿冰", "阿超", "阿晨"
    ]
    
    # 训练分词器
    print("\n1. 训练 BPE 分词器...")
    tokenizer = BPETokenizer()
    tokenizer.train_from_texts(sample_names, vocab_size=200, min_frequency=1)
    
    # 统计信息
    print("\n2. 分词器统计信息:")
    stats = tokenizer.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 测试编码解码
    print("\n3. 测试编码解码:")
    test_name = "张伟"
    encoded = tokenizer.encode(test_name)
    tokens = tokenizer.tokenize(test_name)
    decoded = tokenizer.decode(encoded)
    
    print(f"   原文: {test_name}")
    print(f"   Tokens: {tokens}")
    print(f"   编码: {encoded}")
    print(f"   解码: {decoded}")
    
    # 测试批量编码
    print("\n4. 测试批量编码:")
    test_names = ["李娜", "刘洋", "陈静"]
    encoded_batch = tokenizer.encode_batch(test_names)
    decoded_batch = tokenizer.decode_batch(encoded_batch)
    
    for original, encoded, decoded in zip(test_names, encoded_batch, decoded_batch):
        print(f"   原文: {original} -> 编码: {encoded} -> 解码: {decoded}")
    
    print("\n" + "=" * 50)
