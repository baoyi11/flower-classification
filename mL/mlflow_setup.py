#!/usr/bin/env python3
"""设置MLflow与Dagshub集成"""

import os
import subprocess
import requests
import json

def setup_mlflow_tracking():
    """设置MLflow追踪到Dagshub"""
    print("设置MLflow追踪...")
    
    username = os.getenv('DAGSHUB_USERNAME')
    token = os.getenv('DAGSHUB_TOKEN')
    
    if not username or not token:
        print("❌ 请设置环境变量 DAGSHUB_USERNAME 和 DAGSHUB_TOKEN")
        return False
    
    try:
        # MLflow追踪URI
        tracking_uri = f"https://dagshub.com/{username}/flower-classification-mlops.mlflow"
        
        # 更新配置文件
        config_content = f"""
app:
  host: "0.0.0.0"
  port: 8000

data:
  image_size: [128, 128]
  batch_size: 32

model:
  num_classes: 5
  learning_rate: 0.001
  epochs: 5

mlflow:
  tracking_uri: "{tracking_uri}"
  username: "{username}"
  password: "{token}"
"""
        
        with open('mL/configs/config.yaml', 'w') as f:
            f.write(config_content)
        
        print("✅ MLflow追踪URI已配置")
        
        # 设置环境变量（用于训练脚本）
        os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token
        
        return True
        
    except Exception as e:
        print(f"❌ MLflow设置失败: {e}")
        return False

def test_mlflow_connection():
    """测试MLflow连接"""
    print("测试MLflow连接...")
    
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    username = os.getenv('MLFLOW_TRACKING_USERNAME')
    password = os.getenv('MLFLOW_TRACKING_PASSWORD')
    
    if not all([tracking_uri, username, password]):
        print("❌ MLflow环境变量未设置")
        return False
    
    try:
        # 简单的连接测试
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        
        # 创建测试运行
        with mlflow.start_run(run_name="connection_test") as run:
            mlflow.log_param("test_param", "connection_test")
            mlflow.log_metric("test_metric", 1.0)
        
        print("✅ MLflow连接测试成功")
        return True
        
    except Exception as e:
        print(f"❌ MLflow连接测试失败: {e}")
        return False

def main():
    """主函数"""
    print("="*50)
    print("MLflow与Dagshub集成设置")
    print("="*50)
    
    steps = [
        ("MLflow追踪设置", setup_mlflow_tracking),
        ("连接测试", test_mlflow_connection),
    ]
    
    all_passed = True
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        try:
            if not step_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {step_name} 出错: {e}")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 MLflow集成设置完成!")
        username = os.getenv('DAGSHUB_USERNAME')
        print(f"📈 查看实验: https://dagshub.com/{username}/flower-classification-mlops/experiments")
    else:
        print("⚠️  设置部分完成，请检查问题")
    print("="*50)

if __name__ == "__main__":
    main()