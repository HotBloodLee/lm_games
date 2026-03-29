"""
Transformer 模型

基于标准 Transformer Encoder 架构实现的语言模型
支持字符级分词和 BPE 分词
用于中文人名生成任务
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    位置编码
    
    使用正弦和余弦函数生成位置编码，为模型提供序列位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout 比例
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # 注册为 buffer（不参与训练，但会保存到模型）
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout 比例
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, d_model]
            key: 键张量 [batch_size, seq_len, d_model]
            value: 值张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        
        Returns:
            注意力输出 [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(context)


class FeedForward(nn.Module):
    """
    前馈神经网络
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        初始化前馈网络
        
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout 比例
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder 层
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        初始化 Encoder 层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
        
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 自注意力 + 残差连接 + LayerNorm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerModel(nn.Module):
    """
    Transformer 语言模型
    
    使用 Transformer Encoder 架构进行字符级语言建模
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 20,
        dropout: float = 0.1,
        pad_idx: int = 0,
        tokenizer_type: str = 'char'
    ):
        """
        初始化 Transformer 模型
        
        Args:
            vocab_size: 词表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: Encoder 层数
            d_ff: 前馈网络隐藏层维度
            max_len: 最大序列长度
            dropout: Dropout 比例
            pad_idx: PAD token 的索引
            tokenizer_type: 分词器类型，'char' 为字符级，'bpe' 为 BPE 分词
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.tokenizer_type = tokenizer_type
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder 层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        创建填充掩码
        
        Args:
            x: 输入序列 [batch_size, seq_len]
        
        Returns:
            掩码张量 [batch_size, 1, 1, seq_len]
        """
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len]
            mask: 可选的注意力掩码
        
        Returns:
            logits [batch_size, seq_len, vocab_size]
        """
        # 创建填充掩码
        if mask is None:
            mask = self._create_padding_mask(x)
        
        # 嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Encoder 层
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # 输出
        logits = self.output_layer(x)
        
        return logits
    
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_length: int = 10,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_idx: int = 2
    ) -> torch.Tensor:
        """
        自回归生成
        
        Args:
            start_tokens: 起始 token [batch_size, start_len]
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-K 采样的 K 值
            top_p: Top-P (nucleus) 采样的阈值
            eos_idx: EOS token 索引
        
        Returns:
            生成的序列 [batch_size, seq_len]
        """
        self.eval()
        device = start_tokens.device
        batch_size = start_tokens.size(0)
        
        # 初始化生成序列
        generated = start_tokens.clone()
        
        # 记录每个序列是否已结束
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # 前向传播
                logits = self.forward(generated)
                
                # 取最后一个位置的 logits
                next_logits = logits[:, -1, :] / temperature
                
                # Top-K 采样
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                
                # Top-P (nucleus) 采样
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = float('-inf')
                
                # 采样
                probs = F.softmax(next_logits, dim=-1)
                
                # 兼容 MPS 的采样
                from utils.device import safe_multinomial
                next_token = safe_multinomial(probs, num_samples=1)
                
                # 更新已结束的序列
                next_token[finished] = eos_idx
                
                # 拼接新 token
                generated = torch.cat([generated, next_token], dim=1)
                
                # 检查是否所有序列都已结束
                finished = finished | (next_token.squeeze(-1) == eos_idx)
                if finished.all():
                    break
        
        return generated
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试代码
    vocab_size = 1000
    batch_size = 4
    seq_len = 10
    
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        max_len=20,
        dropout=0.1
    )
    
    print(f"模型参数数量: {model.get_num_params():,}")
    
    # 测试前向传播
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    
    # 测试生成
    start_tokens = torch.tensor([[1, 5, 6]])  # BOS + 两个字符
    generated = model.generate(start_tokens, max_length=5, temperature=1.0)
    print(f"生成序列: {generated}")
