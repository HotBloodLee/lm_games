"""
训练器

统一的模型训练器，支持多设备训练、验证、早停和模型保存
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm

from utils.config import ModelConfig, TrainConfig
from utils.device import get_device, get_device_info, should_pin_memory, get_autocast_device_type
from data.vocabulary import Vocabulary


class Trainer:
    """
    统一训练器
    
    支持 Transformer 和 GPT 模型的训练
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        model_config: ModelConfig,
        vocab: Vocabulary
    ):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            train_config: 训练配置
            model_config: 模型配置
            vocab: 词表实例
        """
        self.model = model
        self.train_config = train_config
        self.model_config = model_config
        self.vocab = vocab
        
        # 设置设备
        self.device = get_device(train_config.device)
        self.model = self.model.to(self.device)
        
        # 打印设备信息
        device_info = get_device_info(self.device)
        print(f"训练设备: {device_info['device']} ({device_info['name']})")
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练
        self.use_amp = train_config.use_amp and self.device.type in ["cuda", "mps"]
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == "cuda" else None
        self.autocast_dtype = get_autocast_device_type(self.device)
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 创建检查点目录
        self.checkpoint_dir = Path(train_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
    
    def _create_scheduler(self) -> LambdaLR:
        """
        创建学习率调度器
        
        使用线性预热 + 余弦退火
        """
        warmup_steps = self.train_config.warmup_steps
        
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        训练一个 epoch
        
        Args:
            train_loader: 训练数据加载器
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            # 移动数据到设备
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.use_amp and self.autocast_dtype:
                with torch.autocast(device_type=self.autocast_dtype):
                    logits = self.model(input_ids)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1)
                    )
                
                # 混合精度反向传播
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_config.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_config.grad_clip
                    )
                    self.optimizer.step()
            else:
                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.grad_clip
                )
                self.optimizer.step()
            
            self.scheduler.step()
            self.global_step += 1
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits = self.model(input_ids)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
        
        Returns:
            训练历史
        """
        print(f"\n开始训练 (共 {self.train_config.epochs} 个 epoch)")
        print(f"模型参数数量: {self.model.get_num_params():,}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.train_config.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            # 验证
            val_loss = None
            if val_loader and (epoch + 1) % self.train_config.val_freq == 0:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                # 早停检查
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint("best.pt")
                else:
                    self.patience_counter += 1
                
                print(f"Epoch {epoch + 1}/{self.train_config.epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"Best: {self.best_val_loss:.4f}")
                
                if self.patience_counter >= self.train_config.patience:
                    print(f"\n早停触发！验证损失已 {self.patience_counter} 个 epoch 未改善")
                    break
            else:
                print(f"Epoch {epoch + 1}/{self.train_config.epochs} - "
                      f"Train Loss: {train_loss:.4f}")
            
            # 定期保存
            if (epoch + 1) % self.train_config.save_freq == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")
        
        # 保存最终模型
        self.save_checkpoint("final.pt")
        
        elapsed_time = time.time() - start_time
        print("-" * 50)
        print(f"训练完成！耗时: {elapsed_time / 60:.2f} 分钟")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """
        保存检查点
        
        Args:
            filename: 检查点文件名
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_config': self.model_config.to_dict(),
            'train_config': self.train_config.to_dict(),
            'history': self.history,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # 同时保存词表
        vocab_path = self.checkpoint_dir / "vocab.json"
        self.vocab.save(str(vocab_path))
    
    def load_checkpoint(self, filename: str, load_optimizer: bool = True):
        """
        加载检查点
        
        Args:
            filename: 检查点文件名
            load_optimizer: 是否加载优化器状态
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        # 使用 map_location 确保跨设备加载
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint.get('history', self.history)
        
        print(f"加载检查点: {filename}")
        print(f"  - Epoch: {self.current_epoch}")
        print(f"  - Best Val Loss: {self.best_val_loss:.4f}")


def load_model_for_inference(
    checkpoint_path: str,
    device_type: str = "auto"
) -> Tuple[nn.Module, Vocabulary, ModelConfig]:
    """
    加载模型用于推理
    
    Args:
        checkpoint_path: 检查点路径
        device_type: 设备类型
    
    Returns:
        (model, vocab, config) 元组
    """
    from models import TransformerModel, GPTModel
    
    device = get_device(device_type)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载配置
    model_config = ModelConfig.from_dict(checkpoint['model_config'])
    
    # 创建模型
    if model_config.model_type == "transformer":
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
    else:
        model = GPTModel(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            d_ff=model_config.d_ff,
            max_len=model_config.max_len,
            dropout=model_config.dropout,
            pad_idx=model_config.pad_idx,
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 加载词表
    checkpoint_dir = Path(checkpoint_path).parent
    vocab_path = checkpoint_dir / "vocab.json"
    vocab = Vocabulary.load(str(vocab_path))
    
    return model, vocab, model_config
