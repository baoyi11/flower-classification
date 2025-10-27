#!/usr/bin/env python3
"""完整的数据管理脚本"""

import os
import sys
import argparse
from pathlib import Path

def setup_dagshub():
    """设置Dagshub环境"""
    print("设置Dagshub环境...")
    os.system('python scripts/dvc_setup.py')

def create_new_dataset_version():
    """创建新的数据集版本"""
    print("创建新的数据集版本...")
    os.system('python mL/data_pipeline.py')

def upload_data():
    """上传数据到Dagshub"""
    print("上传数据到Dagshub...")
    os.system('python scripts/upload_data.py')

def setup_mlflow():
    """设置MLflow"""
    print("设置MLflow远程追踪...")
    os.system('python mL/mlflow_setup.py')

def train_with_tracking():
    """使用远程追踪训练模型"""
    print("使用Dagshub MLflow训练模型...")
    os.system('python mL/train_dagshub.py --data-version v2')

def full_workflow():
    """完整工作流程"""
    print("执行完整MLOps工作流程...")
    
    steps = [
        ("创建数据集版本", create_new_dataset_version),
        ("设置Dagshub环境", setup_dagshub),
        ("上传数据", upload_data),
        ("设置MLflow", setup_mlflow),
        ("训练模型", train_with_tracking),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*40}")
        print(f"步骤: {step_name}")
        print(f"{'='*40}")
        try:
            step_func()
        except Exception as e:
            print(f"❌ {step_name} 失败: {e}")
            return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据管理脚本')
    parser.add_argument('--setup', action='store_true', help='设置Dagshub环境')
    parser.add_argument('--create-data', action='store_true', help='创建新数据集版本')
    parser.add_argument('--upload', action='store_true', help='上传数据到Dagshub')
    parser.add_argument('--mlflow', action='store_true', help='设置MLflow')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--full', action='store_true', help='执行完整工作流程')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_dagshub()
    elif args.create_data:
        create_new_dataset_version()
    elif args.upload:
        upload_data()
    elif args.mlflow:
        setup_mlflow()
    elif args.train:
        train_with_tracking()
    elif args.full:
        full_workflow()
    else:
        print("请指定操作，使用 --help 查看选项")

if __name__ == "__main__":
    main()