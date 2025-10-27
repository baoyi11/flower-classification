# Dagshub 设置指南

## 1. 初始设置

### 1.1 创建Dagshub账户和项目
1. 访问 [dagshub.com](https://dagshub.com) 注册账户
2. 创建新项目 `flower-classification-mlops`
3. 复制项目URL

### 1.2 环境变量设置
```bash
# 设置环境变量（添加到 ~/.bashrc 或 ~/.zshrc）
export DAGSHUB_USERNAME="your-username"
export DAGSHUB_TOKEN="your-token"

# 立即生效
source ~/.bashrc