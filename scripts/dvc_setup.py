#!/usr/bin/env python3
"""DVC和Dagshub设置脚本"""

import os
import subprocess
import sys
from pathlib import Path

def setup_dvc_remote():
    """设置DVC远程存储"""
    print("设置DVC远程存储到Dagshub...")
    
    # 获取环境变量
    username = os.getenv('DAGSHUB_USERNAME')
    token = os.getenv('DAGSHUB_TOKEN')
    
    if not username or not token:
        print("❌ 请设置环境变量 DAGSHUB_USERNAME 和 DAGSHUB_TOKEN")
        print("export DAGSHUB_USERNAME='your-username'")
        print("export DAGSHUB_TOKEN='your-token'")
        return False
    
    try:
        # 初始化DVC（如果尚未初始化）
        if not Path('.dvc').exists():
            subprocess.run(['dvc', 'init'], check=True, capture_output=True)
            print("✅ DVC初始化完成")
        
        # 设置远程存储
        repo_url = f"https://dagshub.com/{username}/flower-classification-mlops.dvc"
        
        # 检查是否已设置远程
        result = subprocess.run(['dvc', 'remote', 'list'], capture_output=True, text=True)
        if 'origin' not in result.stdout:
            subprocess.run(['dvc', 'remote', 'add', 'origin', repo_url], check=True)
            print("✅ DVC远程存储添加完成")
        
        # 配置认证
        subprocess.run(['dvc', 'remote', 'modify', 'origin', '--local', 'auth', 'basic'], check=True)
        subprocess.run(['dvc', 'remote', 'modify', 'origin', '--local', 'user', username], check=True)
        subprocess.run(['dvc', 'remote', 'modify', 'origin', '--local', 'password', token], check=True)
        
        print("✅ DVC远程存储认证配置完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ DVC设置失败: {e}")
        return False

def track_data_with_dvc():
    """使用DVC跟踪数据"""
    print("使用DVC跟踪数据目录...")
    
    data_dirs = ['data/v1', 'data/v2']
    
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            try:
                # 使用DVC跟踪目录
                subprocess.run(['dvc', 'add', data_dir], check=True)
                print(f"✅ 跟踪目录: {data_dir}")
                
                # 添加对应的.dvc文件到Git
                dvc_file = f"{data_dir}.dvc"
                subprocess.run(['git', 'add', dvc_file], check=True)
                print(f"✅ 添加 {dvc_file} 到Git")
                
            except subprocess.CalledProcessError as e:
                print(f"❌ 跟踪 {data_dir} 失败: {e}")
                return False
        else:
            print(f"⚠️  目录不存在: {data_dir}")
    
    return True

def push_to_dagshub():
    """推送数据和代码到Dagshub"""
    print("推送数据和代码到Dagshub...")
    
    try:
        # 推送数据到DVC远程存储
        print("推送数据到DVC远程存储...")
        subprocess.run(['dvc', 'push'], check=True)
        print("✅ 数据推送完成")
        
        # 提交代码更改
        print("提交代码更改...")
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Add dataset versions and DVC tracking'], check=True)
        print("✅ 代码提交完成")
        
        # 推送到Dagshub
        print("推送到Dagshub远程仓库...")
        subprocess.run(['git', 'push', 'dagshub', 'main'], check=True)
        print("✅ 代码推送完成")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 推送失败: {e}")
        return False

def main():
    """主函数"""
    print("="*50)
    print("DVC和Dagshub设置")
    print("="*50)
    
    steps = [
        ("DVC远程存储设置", setup_dvc_remote),
        ("数据跟踪", track_data_with_dvc),
        ("推送数据到Dagshub", push_to_dagshub),
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
        print("🎉 DVC和Dagshub设置完成!")
        print(f"📊 查看项目: https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/flower-classification-mlops")
    else:
        print("⚠️  设置部分完成，请检查问题")
    print("="*50)

if __name__ == "__main__":
    main()
