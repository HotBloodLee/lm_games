"""
配置管理

定义模型和训练的超参数配置类
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """
    模型配置
    
    定义模型架构相关的超参数
    """
    # 模型类型
    model_type: Literal["transformer", "gpt"] = "gpt"
    
    # 词表大小（由数据决定，训练时自动设置）
    vocab_size: int = 1000
    
    # 模型维度
    d_model: int = 128
    
    # 注意力头数
    num_heads: int = 4
    
    # 层数
    num_layers: int = 4
    
    # 前馈网络隐藏层维度
    d_ff: int = 512
    
    # 最大序列长度
    max_len: int = 20
    
    # Dropout 比例
    dropout: float = 0.1
    
    # PAD token 索引
    pad_idx: int = 0
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "max_len": self.max_len,
            "dropout": self.dropout,
            "pad_idx": self.pad_idx,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, file_path: str):
        """保存配置到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> "ModelConfig":
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class TrainConfig:
    """
    训练配置
    
    定义训练相关的超参数
    """
    # 训练轮数
    epochs: int = 100
    
    # 批次大小
    batch_size: int = 64
    
    # 学习率
    learning_rate: float = 1e-3
    
    # 权重衰减
    weight_decay: float = 0.01
    
    # 学习率预热步数
    warmup_steps: int = 100
    
    # 梯度裁剪
    grad_clip: float = 1.0
    
    # 训练集比例
    train_ratio: float = 0.9
    
    # 早停耐心值
    patience: int = 10
    
    # 验证频率（每多少个 epoch 验证一次）
    val_freq: int = 1
    
    # 保存频率（每多少个 epoch 保存一次）
    save_freq: int = 10
    
    # 日志频率（每多少个 batch 打印一次）
    log_freq: int = 10
    
    # 设备类型
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    
    # 数据加载的工作进程数
    num_workers: int = 0
    
    # 随机种子
    seed: int = 42
    
    # 检查点目录
    checkpoint_dir: str = "checkpoints"
    
    # 是否使用混合精度训练
    use_amp: bool = False
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "grad_clip": self.grad_clip,
            "train_ratio": self.train_ratio,
            "patience": self.patience,
            "val_freq": self.val_freq,
            "save_freq": self.save_freq,
            "log_freq": self.log_freq,
            "device": self.device,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "checkpoint_dir": self.checkpoint_dir,
            "use_amp": self.use_amp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, file_path: str):
        """保存配置到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> "TrainConfig":
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class GenerateConfig:
    """
    生成配置
    
    定义推理相关的参数
    """
    # 最大生成长度
    max_length: int = 10
    
    # 温度参数
    temperature: float = 1.0
    
    # Top-K 采样
    top_k: Optional[int] = None
    
    # Top-P (nucleus) 采样
    top_p: Optional[float] = None
    
    # 生成数量
    num_samples: int = 10
    
    # 设备类型
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_samples": self.num_samples,
            "device": self.device,
        }


# 预定义配置

def get_small_config() -> ModelConfig:
    """小型模型配置（适合快速实验）"""
    return ModelConfig(
        d_model=64,
        num_heads=2,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
    )


def get_medium_config() -> ModelConfig:
    """中型模型配置（默认配置）"""
    return ModelConfig(
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
    )


def get_large_config() -> ModelConfig:
    """大型模型配置（更强的表达能力）"""
    return ModelConfig(
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        dropout=0.1,
    )


if __name__ == "__main__":
    # 测试配置
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    print("模型配置:")
    print(json.dumps(model_config.to_dict(), indent=2))
    
    print("\n训练配置:")
    print(json.dumps(train_config.to_dict(), indent=2))
    
    # 测试保存和加载
    model_config.save("/tmp/model_config.json")
    loaded_config = ModelConfig.load("/tmp/model_config.json")
    print(f"\n配置加载测试: {loaded_config.d_model == model_config.d_model}")
