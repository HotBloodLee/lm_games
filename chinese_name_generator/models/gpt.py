"""
GPT 模型

基于 GPT (Decoder-only) 架构实现的字符级语言模型
使用因果注意力掩码实现自回归生成
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """
    因果自注意力
    
    使用因果掩码确保只能看到当前位置及之前的 token
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_len: int = 100,
        dropout: float = 0.1
    ):
        """
        初始化因果自注意力
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            max_len: 最大序列长度
            dropout: Dropout 比例
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # QKV 合并投影
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # 因果掩码
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('mask', mask.view(1, 1, max_len, max_len))
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            padding_mask: 填充掩码 [batch_size, seq_len]
        
        Returns:
            注意力输出 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # QKV 投影
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 分头
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用因果掩码
        causal_mask = self.mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # 应用填充掩码
        if padding_mask is not None:
            # padding_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask == 0, float('-inf'))
        
        # Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 计算输出
        context = torch.matmul(attn_weights, v)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.resid_dropout(self.out_proj(context))


class GPTBlock(nn.Module):
    """
    GPT Decoder 块
    
    包含因果自注意力和前馈网络
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_len: int = 100,
        dropout: float = 0.1
    ):
        """
        初始化 GPT 块
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            max_len: 最大序列长度
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, max_len, dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播（Pre-LN 架构）
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            padding_mask: 填充掩码
        
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 自注意力 + 残差
        x = x + self.attn(self.ln1(x), padding_mask)
        
        # 前馈网络 + 残差
        x = x + self.mlp(self.ln2(x))
        
        return x


class GPTModel(nn.Module):
    """
    GPT 语言模型
    
    Decoder-only 架构，使用因果注意力实现自回归生成
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
        pad_idx: int = 0
    ):
        """
        初始化 GPT 模型
        
        Args:
            vocab_size: 词表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: Decoder 层数
            d_ff: 前馈网络隐藏层维度
            max_len: 最大序列长度
            dropout: Dropout 比例
            pad_idx: PAD token 的索引
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_len = max_len
        
        # Token 嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # 位置嵌入（可学习）
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # GPT 块
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, max_len, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终 LayerNorm
        self.ln_f = nn.LayerNorm(d_model)
        
        # 输出层（与嵌入层共享权重）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.token_embedding.weight
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len]
            padding_mask: 填充掩码 [batch_size, seq_len]
        
        Returns:
            logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.size()
        device = x.device
        
        # 创建填充掩码
        if padding_mask is None:
            padding_mask = (x != self.pad_idx)
        
        # Token 嵌入
        tok_emb = self.token_embedding(x)
        
        # 位置嵌入
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        pos_emb = self.position_embedding(positions)
        
        # 合并嵌入
        x = self.dropout(tok_emb + pos_emb)
        
        # GPT 块
        for block in self.blocks:
            x = block(x, padding_mask)
        
        # 最终 LayerNorm
        x = self.ln_f(x)
        
        # 输出
        logits = self.lm_head(x)
        
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
            temperature: 温度参数（越高越随机）
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
                # 截断到最大长度
                if generated.size(1) > self.max_len:
                    context = generated[:, -self.max_len:]
                else:
                    context = generated
                
                # 前向传播
                logits = self.forward(context)
                
                # 取最后一个位置的 logits
                next_logits = logits[:, -1, :] / temperature
                
                # Top-K 采样
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')
                
                # Top-P (nucleus) 采样
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过 top_p 的 token
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
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        获取模型参数数量
        
        Args:
            non_embedding: 是否排除嵌入层参数
        
        Returns:
            参数数量
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()
        return n_params


class GPTModelWithKVCache(GPTModel):
    """
    带 KV Cache 的 GPT 模型
    
    在生成时缓存 Key 和 Value，避免重复计算
    """
    
    def generate_with_cache(
        self,
        start_tokens: torch.Tensor,
        max_length: int = 10,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_idx: int = 2
    ) -> torch.Tensor:
        """
        使用 KV Cache 的生成方法
        
        注意：当前实现简化处理，未完全实现 KV Cache
        在小模型场景下，普通生成已经足够高效
        """
        # 对于小模型，直接使用基类方法
        return super().generate(
            start_tokens, max_length, temperature, top_k, top_p, eos_idx
        )


if __name__ == "__main__":
    # 测试代码
    vocab_size = 1000
    batch_size = 4
    seq_len = 10
    
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        max_len=20,
        dropout=0.1
    )
    
    print(f"模型参数数量: {model.get_num_params():,}")
    print(f"非嵌入参数数量: {model.get_num_params(non_embedding=True):,}")
    
    # 测试前向传播
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    
    # 测试生成
    start_tokens = torch.tensor([[1, 5, 6]])  # BOS + 两个字符
    generated = model.generate(start_tokens, max_length=5, temperature=1.0)
    print(f"生成序列: {generated}")
