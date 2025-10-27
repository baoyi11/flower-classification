#!/usr/bin/env python3
"""修复版本的花卉分类训练脚本 - 修复训练器初始化"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def safe_imports():
    """安全导入所需的模块"""
    try:
        # 尝试从app包导入
        from app.src.data.data_loader import get_data_loaders
        from app.src.model.cnn_model import create_model
        from app.src.model.trainer import SimpleTrainer
        from app.src.utils.config import config

        return get_data_loaders, create_model, SimpleTrainer, config
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("尝试替代导入方式...")

        # 尝试直接导入
        try:
            # 添加可能的路径
            sys.path.append(str(project_root / "app" / "src"))

            from data.data_loader import get_data_loaders
            from model.cnn_model import create_model
            from model.trainer import SimpleTrainer
            from utils.config import config

            return get_data_loaders, create_model, SimpleTrainer, config
        except ImportError:
            print("❌ 所有导入方式都失败")
            return None, None, None, None


def setup_environment():
    """设置环境"""
    print("设置训练环境...")

    # 创建必要目录
    directories = ["models", "logs", "data/v1", "data/v2"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ 环境设置完成")


def train_model(data_version="v1", model_type="simple"):
    """训练模型的主函数"""
    print(f"开始训练 {model_type} 模型，使用数据集: {data_version}")

    # 安全导入
    get_data_loaders, create_model, SimpleTrainer, config = safe_imports()

    if None in [get_data_loaders, create_model, SimpleTrainer, config]:
        print("❌ 模块导入失败，无法继续训练")
        return False

    try:
        # 设置数据路径
        if data_version == "v1":
            data_dir = config.get("data.v1_path", "data/v1")
        else:
            data_dir = config.get("data.v2_path", "data/v2")

        print(f"使用数据目录: {data_dir}")

        # 检查数据目录是否存在
        if not Path(data_dir).exists():
            print(f"❌ 数据目录不存在: {data_dir}")
            print("请先运行数据准备脚本: python mL/data_pipeline.py")
            return False

        # 获取配置参数
        batch_size = config.get("data.batch_size", 32)
        image_size = config.get("data.image_size", [128, 128])

        print(f"数据配置 - 批量大小: {batch_size}, 图像尺寸: {image_size}")

        # 获取数据加载器
        train_loader, test_loader, class_to_idx = get_data_loaders(
            data_dir=data_dir, batch_size=batch_size
        )

        # 从 class_to_idx 中获取类别信息
        class_names = list(class_to_idx.keys())
        num_classes = len(class_names)

        print(f"✅ 数据加载成功 - 类别数: {num_classes}, 类别: {class_names}")

        # 创建模型
        model = create_model(model_type, num_classes=num_classes)
        print(f"创建模型: {model_type}, 类别数: {num_classes}")

        # 训练模型
        epochs = config.get("model.epochs", 5)
        learning_rate = config.get("model.learning_rate", 0.001)

        print(f"训练配置 - 轮次: {epochs}, 学习率: {learning_rate}")

        # 🔧 修复：移除 num_classes 参数
        trainer = SimpleTrainer(
            model,
            model_name=f"{model_type}_{data_version}",
            # 移除了 num_classes=num_classes 参数
        )

        trainer.train(train_loader, test_loader, epochs=epochs)

        print("✅ 训练完成!")
        return True

    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="修复版本的花卉分类训练脚本")
    parser.add_argument(
        "--data-version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="使用的数据集版本",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="simple",
        choices=["simple", "cnn"],
        help="模型类型",
    )
    parser.add_argument("--fix-config", action="store_true", help="自动修复配置问题")

    args = parser.parse_args()

    print("=" * 50)
    print("花卉分类模型训练 (修复版本)")
    print("=" * 50)

    # 如果需要修复配置
    if args.fix_config:
        print("自动修复配置...")
        os.system("python scripts/fix_config.py")

    # 设置环境
    setup_environment()

    # 训练模型
    success = train_model(args.data_version, args.model_type)

    if success:
        print("\n🎉 训练成功完成!")
    else:
        print("\n❌ 训练失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
