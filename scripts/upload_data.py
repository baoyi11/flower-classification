#!/usr/bin/env python3
"""自动化数据上传到Dagshub的脚本"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def check_prerequisites():
    """检查前提条件"""
    print("检查前提条件...")
    
    # 检查环境变量
    if not os.getenv('DAGSHUB_USERNAME') or not os.getenv('DAGSHUB_TOKEN'):
        print("❌ 请设置环境变量 DAGSHUB_USERNAME 和 DAGSHUB_TOKEN")
        return False
    
    # 检查数据目录是否存在
    if not Path('data/v1').exists() or not Path('data/v2').exists():
        print("❌ 数据目录不存在，请先运行数据管道")
        print("运行: python mL/data_pipeline.py")
        return False
    
    print("✅ 前提条件检查通过")
    return True

def create_data_version(version_name=None):
    """创建新的数据版本"""
    if version_name is None:
        version_name = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    print(f"创建数据版本: {version_name}")
    
    try:
        # 创建新版本的数据目录
        new_data_dir = Path('data') / version_name
        new_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 这里可以添加数据处理的逻辑
        # 目前我们复制v2的数据作为示例
        if Path('data/v2').exists():
            import shutil
            for item in Path('data/v2').iterdir():
                if item.is_dir():
                    shutil.copytree(item, new_data_dir / item.name)
                else:
                    shutil.copy2(item, new_data_dir / item.name)
        
        print(f"✅ 数据版本 {version_name} 创建完成")
        return str(new_data_dir)
        
    except Exception as e:
        print(f"❌ 创建数据版本失败: {e}")
        return None

def track_new_version(data_dir):
    """跟踪新数据版本"""
    print(f"使用DVC跟踪新版本: {data_dir}")
    
    try:
        # 添加数据到DVC
        subprocess.run(['dvc', 'add', data_dir], check=True)
        
        # 添加.dvc文件到Git
        dvc_file = f"{data_dir}.dvc"
        subprocess.run(['git', 'add', dvc_file], check=True)
        
        print(f"✅ 数据版本跟踪完成: {data_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 跟踪数据版本失败: {e}")
        return False

def push_data_and_code(commit_message):
    """推送数据和代码"""
    print("推送数据和代码到Dagshub...")
    
    try:
        # 推送数据到DVC远程
        print("推送数据到DVC远程存储...")
        subprocess.run(['dvc', 'push'], check=True)
        
        # 提交代码
        print("提交代码更改...")
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        
        # 推送到Dagshub
        print("推送到Dagshub...")
        subprocess.run(['git', 'push', 'dagshub', 'main'], check=True)
        
        print("✅ 数据和代码推送完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 推送失败: {e}")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='上传数据到Dagshub')
    parser.add_argument('--version', type=str, help='数据版本名称')
    parser.add_argument('--message', type=str, default='Update dataset', 
                       help='提交消息')
    
    args = parser.parse_args()
    
    print("="*50)
    print("数据上传到Dagshub")
    print("="*50)
    
    # 检查前提条件
    if not check_prerequisites():
        sys.exit(1)
    
    # 创建新数据版本（可选）
    new_data_dir = None
    if args.version:
        new_data_dir = create_data_version(args.version)
        if not new_data_dir:
            sys.exit(1)
    
    # 跟踪数据
    if new_data_dir:
        if not track_new_version(new_data_dir):
            sys.exit(1)
    else:
        # 跟踪现有数据
        data_dirs = ['data/v1', 'data/v2']
        for data_dir in data_dirs:
            if Path(data_dir).exists():
                if not track_new_version(data_dir):
                    sys.exit(1)
    
    # 推送数据
    commit_msg = args.message
    if new_data_dir:
        commit_msg = f"Add dataset version {Path(new_data_dir).name}"
    
    if not push_data_and_code(commit_msg):
        sys.exit(1)
    
    print("\n" + "="*50)
    print("🎉 数据上传完成!")
    username = os.getenv('DAGSHUB_USERNAME')
    print(f"📊 查看项目: https://dagshub.com/{username}/flower-classification-mlops")
    print("="*50)

if __name__ == "__main__":
    main()