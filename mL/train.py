#!/usr/bin/env python3
"""简化的模型训练脚本"""

import argparse
import sys
sys.path.append('../app')

from src.data.data_loader import get_data_loaders
from src.model.cnn_model import create_model
from src.model.trainer import SimpleTrainer
from src.utils.config import config

def train_model(data_version: str = "v1", model_type: str = "simple"):
    """训练模型"""
    print(f"开始训练 {model_type} 模型，使用数据集: {data_version}")
    
    # 设置数据路径
    if data_version == "v1":
        data_dir = config.get('data.v1_path', 'data/v1')
    else:
        data_dir = config.get('data.v2_path', 'data/v2')
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=config.get('data.batch_size', 32)
    )
    
    # 创建模型
    model = create_model(model_type)
    print(f"创建模型: {model_type}")
    
    # 训练模型
    trainer = SimpleTrainer(model, model_name=f"{model_type}_{data_version}")
    trainer.train(train_loader, test_loader, epochs=config.get('model.epochs', 5))
    
    print("训练完成!")

def main():
    parser = argparse.ArgumentParser(description='训练花卉分类模型')
    parser.add_argument('--data-version', type=str, default='v1',
                       choices=['v1', 'v2'],
                       help='使用的数据集版本')
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple'],
                       help='模型类型')
    
    args = parser.parse_args()
    
    print("="*50)
    print("花卉分类模型训练")
    print("="*50)
    
    train_model(args.data_version, args.model_type)

if __name__ == "__main__":
    main()