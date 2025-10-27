#!/usr/bin/env python3
"""简化的模型训练脚本"""

import argparse
import sys
import json
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from app.src.data.data_loader import get_data_loaders
    from app.src.model.cnn_model import create_model
    from app.src.model.trainer import SimpleTrainer
    from app.src.utils.config import config
except ImportError as e:
    print(f"导入错误: {e}")
    print("使用备用配置...")
    
    # 备用配置类
    class Config:
        def __init__(self):
            self.data = {
                "v1_path": "data/v1",
                "v2_path": "data/v2",
                "batch_size": 32,
                "image_size": [128, 128]
            }
            self.model = {
                "epochs": 5,
                "learning_rate": 0.001,
                "num_classes": 5
            }
            self.mlflow = {
                "tracking_uri": "./mlruns"
            }
        
        def get(self, key, default=None):
            keys = key.split(".")
            value = self.__dict__
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
    
    config = Config()


def train_model(data_version: str = "v1", model_type: str = "simple"):
    """训练模型"""
    print(f"开始训练 {model_type} 模型，使用数据集: {data_version}")

    # 确定数据目录
    if data_version == "v1":
        data_dir = config.get("data.v1_path", "data/v1")
    else:
        data_dir = config.get("data.v2_path", "data/v2")

    # 获取数据加载器 - 现在返回3个值
    train_loader, test_loader, class_to_idx = get_data_loaders(
        data_dir=data_dir, batch_size=config.get("data.batch_size", 32)
    )

    # 创建目录
    os.makedirs("ml/registry", exist_ok=True)
    # 保存类别映射
    class_mapping_path = "ml/registry/class_to_idx.json"
    with open(class_mapping_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"✅ 保存类别映射到: {class_mapping_path}")

    # 更新模型类别数量
    num_classes = len(class_to_idx)
    print(f"✅ 检测到 {num_classes} 个类别")

    # 创建模型
    model = create_model(model_type, num_classes=num_classes)
    print(f"创建模型: {model_type}, 类别数: {num_classes}")

    # 训练模型
    trainer = SimpleTrainer(model, model_name=f"{model_type}_{data_version}")
    trainer.train(train_loader, test_loader,
                  epochs=config.get("model.epochs", 5))

    print("训练完成!")


def main():
    parser = argparse.ArgumentParser(description="训练花卉分类模型")
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
        choices=["simple"],
        help="模型类型",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("花卉分类模型训练")
    print("=" * 50)

    train_model(args.data_version, args.model_type)


if __name__ == "__main__":
    main()
