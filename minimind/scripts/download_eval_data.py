"""
评测数据集离线下载脚本

将 lm-evaluation-harness 所需的评测数据集下载到本地指定目录，
之后评测时通过 HF_DATASETS_CACHE 环境变量指定该目录即可离线运行。

用法:
    # 下载到默认目录 ./eval_data/
    python scripts/download_eval_data.py

    # 下载到指定目录
    python scripts/download_eval_data.py --save_dir /data/eval_datasets

    # 使用镜像加速（国内推荐）
    python scripts/download_eval_data.py --mirror

    # 指定要下载的数据集（逗号分隔）
    python scripts/download_eval_data.py --tasks ceval,cmmlu,arc_easy
"""

import os
import argparse
from pathlib import Path


# 评测数据集映射表：task_name -> (HuggingFace repo, subset/config)
EVAL_DATASETS = {
    "ceval": {
        "repo": "ceval/ceval-exam",
        "subset": None,
        "description": "中文多学科选择题（ceval-valid）",
    },
    "cmmlu": {
        "repo": "haonan-li/cmmlu",
        "subset": None,
        "description": "中文大规模多任务语言理解",
    },
    "arc_easy": {
        "repo": "allenai/ai2_arc",
        "subset": "ARC-Easy",
        "description": "AI2 推理挑战（简单）",
    },
    "piqa": {
        "repo": "ybisk/piqa",
        "subset": None,
        "description": "物理直觉问答",
    },
    "openbookqa": {
        "repo": "allenai/openbookqa",
        "subset": "main",
        "description": "开卷科学问答",
    },
    "hellaswag": {
        "repo": "Rowan/hellaswag",
        "subset": None,
        "description": "常识自然语言推理",
    },
    "social_iqa": {
        "repo": "allenai/social_i_qa",
        "subset": None,
        "description": "社交情境推理",
    },
}


def download_datasets(save_dir: str, tasks: list, use_mirror: bool = False):
    """下载指定的评测数据集到本地目录"""

    # 设置镜像
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("✅ 已启用 HuggingFace 镜像: https://hf-mirror.com")

    # 设置缓存目录为指定的保存目录
    save_path = Path(save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(save_path)

    print(f"📁 数据集保存目录: {save_path}")
    print(f"📋 待下载数据集: {', '.join(tasks)}")
    print("-" * 60)

    # 延迟导入，避免在帮助信息时就需要 datasets 库
    from datasets import load_dataset

    success = []
    failed = []

    for task_name in tasks:
        if task_name not in EVAL_DATASETS:
            print(f"⚠️  未知数据集: {task_name}，跳过")
            print(f"   可选: {', '.join(EVAL_DATASETS.keys())}")
            failed.append(task_name)
            continue

        info = EVAL_DATASETS[task_name]
        repo = info["repo"]
        subset = info["subset"]
        desc = info["description"]

        print(f"\n🔽 正在下载: {task_name}")
        print(f"   来源: {repo} (subset={subset})")
        print(f"   说明: {desc}")

        try:
            ds = load_dataset(
                repo,
                subset,
                trust_remote_code=True,
                cache_dir=str(save_path),
            )
            # 打印数据集基本信息
            total_samples = sum(len(split) for split in ds.values())
            splits = list(ds.keys())
            print(f"   ✅ 下载完成! splits={splits}, 总样本数={total_samples}")
            success.append(task_name)
        except Exception as e:
            print(f"   ❌ 下载失败: {e}")
            failed.append(task_name)

    # 打印总结
    print("\n" + "=" * 60)
    print("📊 下载总结:")
    print(f"   成功: {len(success)}/{len(tasks)} - {', '.join(success) if success else '无'}")
    if failed:
        print(f"   失败: {len(failed)}/{len(tasks)} - {', '.join(failed)}")
    print(f"\n📁 数据已保存到: {save_path}")
    print(f"\n💡 后续评测时使用以下命令（离线模式）:")
    print(f"   export HF_DATASETS_CACHE={save_path}")
    print(f"   export HF_DATASETS_OFFLINE=1")
    print(f"   export HF_HUB_OFFLINE=1")
    print(f"   lm_eval --model hf --model_args pretrained=\"./minimind-3\",dtype=auto \\")
    print(f"     --tasks \"ceval-valid,cmmlu,arc_easy,piqa,openbookqa,hellaswag,social_iqa\" \\")
    print(f"     --batch_size auto --device cuda:0 --trust_remote_code")


def main():
    parser = argparse.ArgumentParser(
        description="下载 lm-evaluation-harness 评测数据集到本地",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载全部数据集到默认目录
  python scripts/download_eval_data.py --mirror

  # 下载到指定目录
  python scripts/download_eval_data.py --save_dir /data/eval_datasets --mirror

  # 只下载中文数据集
  python scripts/download_eval_data.py --tasks ceval,cmmlu --mirror

  # 只下载英文数据集
  python scripts/download_eval_data.py --tasks arc_easy,piqa,openbookqa,hellaswag,social_iqa --mirror
        """,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./eval_data",
        help="数据集保存目录（默认: ./eval_data）",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="要下载的数据集，逗号分隔（默认: 全部）。"
        f"可选: {', '.join(EVAL_DATASETS.keys())}",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用 HuggingFace 镜像加速下载（国内推荐）",
    )

    args = parser.parse_args()

    # 解析 tasks
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = list(EVAL_DATASETS.keys())

    download_datasets(
        save_dir=args.save_dir,
        tasks=tasks,
        use_mirror=args.mirror,
    )


if __name__ == "__main__":
    main()
