# 简化版花卉分类 MLOps 项目

一个简化的机器学习驱动的花卉分类应用，包含完整的MLOps流程。

## 项目特点

- 🌸 花卉图像分类
- 🚀 FastAPI Web服务  
- 📊 MLflow实验跟踪
- 📈 DVC数据版本控制
- 🐳 Docker容器化
- 🔄 CI/CD自动化

## 快速开始

### 1. 环境设置
```bash
git clone <repository>
cd flower-classification-mlops

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
flower-classification-mlops/
├── app/                    # 应用代码
│   ├── main.py            # FastAPI服务
│   ├── test_app.py        # 测试
│   └── src/               # 核心模块
├── mL/                    # 机器学习组件
│   ├── train.py           # 训练脚本
│   ├── data_pipeline.py   # 数据预处理
│   └── configs/           # 配置
├── data/                  # 数据目录
├── .github/workflows/     # CI/CD配置
├── requirements.txt       # 依赖
├── Dockerfile            # 容器配置
└── README.md             # 文档