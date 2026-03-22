"""
设备管理工具

支持 CUDA、MPS、CPU 三种计算设备的自动检测和手动指定
优先级：CUDA > MPS > CPU
"""

import torch
from typing import Optional, Literal

DeviceType = Literal["auto", "cuda", "mps", "cpu"]


def get_device(device_type: DeviceType = "auto") -> torch.device:
    """
    获取计算设备
    
    Args:
        device_type: 设备类型
            - "auto": 自动选择最佳设备（CUDA > MPS > CPU）
            - "cuda": 强制使用 NVIDIA GPU
            - "mps": 强制使用 Mac MPS
            - "cpu": 强制使用 CPU
    
    Returns:
        torch.device 对象
    
    Raises:
        RuntimeError: 当指定的设备不可用时
    """
    if device_type == "auto":
        return _auto_detect_device()
    elif device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请检查 NVIDIA 驱动和 PyTorch CUDA 版本")
        return torch.device("cuda")
    elif device_type == "mps":
        if not _is_mps_available():
            raise RuntimeError("MPS 不可用，请确保使用 Mac Apple Silicon 并安装支持 MPS 的 PyTorch")
        return torch.device("mps")
    elif device_type == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"未知的设备类型: {device_type}")


def _auto_detect_device() -> torch.device:
    """
    自动检测并返回最佳可用设备
    
    优先级：CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif _is_mps_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _is_mps_available() -> bool:
    """
    检查 MPS (Metal Performance Shaders) 是否可用
    
    MPS 是 Apple Silicon Mac 上的 GPU 加速后端
    """
    if not hasattr(torch.backends, 'mps'):
        return False
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """
    获取设备详细信息
    
    Args:
        device: 要查询的设备，如果为 None 则查询当前自动检测的设备
    
    Returns:
        包含设备信息的字典
    """
    if device is None:
        device = _auto_detect_device()
    
    info = {
        "device": str(device),
        "type": device.type,
    }
    
    if device.type == "cuda":
        info.update({
            "name": torch.cuda.get_device_name(device),
            "memory_total": f"{torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB",
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        })
    elif device.type == "mps":
        info.update({
            "name": "Apple Silicon GPU (MPS)",
            "note": "MPS 内存与系统统一内存共享",
        })
    else:
        info.update({
            "name": "CPU",
            "threads": torch.get_num_threads(),
        })
    
    return info


def to_device(tensor_or_model, device: torch.device):
    """
    将张量或模型移动到指定设备
    
    Args:
        tensor_or_model: PyTorch 张量或模型
        device: 目标设备
    
    Returns:
        移动后的张量或模型
    """
    return tensor_or_model.to(device)


def get_autocast_device_type(device: torch.device) -> Optional[str]:
    """
    获取用于 torch.autocast 的设备类型字符串
    
    Args:
        device: 计算设备
    
    Returns:
        设备类型字符串，CPU 返回 None（不支持自动混合精度）
    """
    if device.type == "cuda":
        return "cuda"
    elif device.type == "mps":
        # MPS 在 PyTorch 2.0+ 支持有限的自动混合精度
        return "mps" if hasattr(torch, 'autocast') else None
    else:
        return None


def should_pin_memory(device: torch.device) -> bool:
    """
    判断 DataLoader 是否应该使用 pin_memory
    
    pin_memory 只在 CUDA 设备上有意义
    
    Args:
        device: 计算设备
    
    Returns:
        是否应该启用 pin_memory
    """
    return device.type == "cuda"


def safe_multinomial(probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    """
    安全的多项式采样，兼容 MPS 设备
    
    某些 PyTorch 版本的 MPS 不支持 multinomial，需要回退到 CPU
    
    Args:
        probs: 概率分布张量
        num_samples: 采样数量
    
    Returns:
        采样结果张量
    """
    original_device = probs.device
    
    if original_device.type == "mps":
        # MPS 可能不支持 multinomial，回退到 CPU
        try:
            return torch.multinomial(probs, num_samples)
        except RuntimeError:
            cpu_probs = probs.cpu()
            result = torch.multinomial(cpu_probs, num_samples)
            return result.to(original_device)
    else:
        return torch.multinomial(probs, num_samples)


def print_device_info():
    """
    打印当前可用设备的详细信息
    """
    print("=" * 50)
    print("设备检测信息")
    print("=" * 50)
    
    # CUDA 信息
    print(f"\nCUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA 版本: {torch.version.cuda}")
        print(f"  - GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  - GPU {i}: {props.name}")
            print(f"    - 显存: {props.total_memory / 1024**3:.2f} GB")
    
    # MPS 信息
    print(f"\nMPS 可用: {_is_mps_available()}")
    if _is_mps_available():
        print("  - Apple Silicon GPU (Metal Performance Shaders)")
    
    # CPU 信息
    print(f"\nCPU 线程数: {torch.get_num_threads()}")
    
    # 自动选择结果
    auto_device = _auto_detect_device()
    print(f"\n自动选择设备: {auto_device}")
    print("=" * 50)


if __name__ == "__main__":
    print_device_info()
