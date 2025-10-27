#!/usr/bin/env python3
"""简化的花卉分类API"""

import io
import json
import os
import sys
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

# 导入项目模块
try:
    from app.src.model.cnn_model import create_model
    from app.src.utils.config import config
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 全局变量
model = None
class_names = {}


def load_model():
    """加载模型"""
    global model, class_names

    try:
        # 加载类别映射
        class_mapping_path = "ml/registry/class_to_idx.json"
        if os.path.exists(class_mapping_path):
            with open(class_mapping_path, "r") as f:
                class_to_idx = json.load(f)
            # 反转映射：索引 -> 类别名称
            class_names = {idx: name for name, idx in class_to_idx.items()}
            print(f"✅ 加载类别映射: {len(class_names)} 个类别")
        else:
            # 默认类别
            class_names = {i: f"class_{i}" for i in range(5)}
            print("⚠️  使用默认类别映射")

        # 加载模型
        model_path = "ml/registry/best_model.pth"
        if not os.path.exists(model_path):
            print("⚠️  模型文件不存在，使用随机初始化模型")
            model = create_model(num_classes=len(class_names))
        else:
            model = create_model(num_classes=len(class_names))
            model.load_state_dict(
                torch.load(model_path, map_location="cpu")
            )
            print("✅ 模型加载成功")

        model.eval()
        return True

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理器"""
    # 启动时加载模型
    print("启动花卉分类API服务...")
    if not load_model():
        raise RuntimeError("模型加载失败，服务无法启动")
    yield
    # 关闭时清理资源
    print("关闭花卉分类API服务...")


app = FastAPI(
    title="花卉分类API",
    description="简化的花卉图像分类服务",
    version="1.0.0",
    lifespan=lifespan,
)


def preprocess_image(image_bytes):
    """预处理图像"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 图像变换
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize(config.get("data.image_size", [128, 128])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
            ]
        )

        return transform(image).unsqueeze(0)  # 添加batch维度

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"图像处理失败: {str(e)}"
        )


@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "花卉分类API服务",
        "status": "运行中",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """预测图像类别"""
    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图像文件")

    try:
        # 读取图像
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        # 预测
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # 获取结果
        class_id = predicted.item()
        class_name = class_names.get(class_id, f"未知类别_{class_id}")
        confidence_score = confidence.item()

        return {
            "predicted_class": class_name,
            "confidence": round(confidence_score, 4),
            "all_classes": class_names,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"预测失败: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.get("app.host", "0.0.0.0"),
        port=config.get("app.port", 8000)
    )
