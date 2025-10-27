#!/usr/bin/env python3
"""支持Dagshub MLflow追踪的训练脚本"""

import os
import argparse
import mlflow
import torch
from app.src.data.data_loader import get_data_loaders
from app.src.model.cnn_model import create_model
from app.src.model.trainer import SimpleTrainer
from app.src.utils.config import config

def setup_mlflow():
    """设置MLflow远程追踪"""
    tracking_uri = config.get('mlflow.tracking_uri')
    username = config.get('mlflow.username')
    password = config.get('mlflow.password')
    
    if tracking_uri and username and password:
        # 设置MLflow追踪URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # 对于Dagshub，需要设置认证
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password
        
        print(f"✅ MLflow追踪URI: {tracking_uri}")
        return True
    else:
        print("⚠️  使用本地MLflow追踪")
        mlflow.set_tracking_uri(config.get('mlflow.tracking_uri', 'mL/mlruns'))
        return True

def train_with_remote_tracking(data_version: str = "v1", model_type: str = "simple"):
    """使用远程追踪训练模型"""
    print(f"开始训练 {model_type} 模型，使用数据集: {data_version}")
    
    # 设置MLflow
    if not setup_mlflow():
        print("❌ MLflow设置失败，使用本地模式")
        mlflow.set_tracking_uri('mL/mlruns')
    
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
    
    # 使用MLflow自动记录
    mlflow.autolog()
    
    # 训练模型
    run_name = f"{model_type}_{data_version}_{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
    
    with mlflow.start_run(run_name=run_name):
        # 记录参数
        mlflow.log_params({
            'data_version': data_version,
            'model_type': model_type,
            'epochs': config.get('model.epochs', 5),
            'learning_rate': config.get('model.learning_rate', 0.001),
            'batch_size': config.get('data.batch_size', 32)
        })
        
        trainer = SimpleTrainer(model, model_name=run_name)
        trainer.train(train_loader, test_loader, epochs=config.get('model.epochs', 5))
    
    print("训练完成!")

def main():
    parser = argparse.ArgumentParser(description='使用Dagshub MLflow训练花卉分类模型')
    parser.add_argument('--data-version', type=str, default='v1',
                       choices=['v1', 'v2'],
                       help='使用的数据集版本')
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple'],
                       help='模型类型')
    
    args = parser.parse_args()
    
    print("="*50)
    print("花卉分类模型训练 (Dagshub MLflow)")
    print("="*50)
    
    train_with_remote_tracking(args.data_version, args.model_type)

if __name__ == "__main__":
    main()