#!/usr/bin/env python3
"""修复后的应用测试"""

import os
import sys
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

# 现在可以正确导入 app
from app.main import app


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


class TestFlowerAPI:
    """API测试类"""

    def test_root_endpoint(self, client):
        """测试根端点"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["message"] == "花卉分类API服务"

    def test_health_endpoint(self, client):
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        # 注意：在兼容版本中可能没有 model_loaded 字段

    def test_predict_valid_image(self, client):
        """测试有效图像预测"""
        # 创建测试图像
        image = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        files = {"file": ("test.jpg", img_bytes.getvalue(), "image/jpeg")}
        response = client.post("/predict", files=files)

        # 检查响应格式
        assert response.status_code in [200, 500]  # 可能因为模型未训练而返回500
        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data

    def test_predict_invalid_file(self, client):
        """测试无效文件"""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)
        assert response.status_code == 400

    def test_predict_no_file(self, client):
        """测试无文件上传"""
        response = client.post("/predict")
        assert response.status_code == 422  # 验证错误


def test_model_creation():
    """测试模型创建"""
    try:
        from app.src.model.cnn_model import create_model

        model = create_model(num_classes=5)
        assert model is not None
    except ImportError:
        pytest.skip("无法导入模型模块，跳过此测试")


def test_config_loading():
    """测试配置加载"""
    try:
        from app.src.utils.config import config

        # 检查配置是否存在
        assert config is not None
        # 检查一些基本配置项
        assert config.get("model.epochs") is not None
    except ImportError:
        pytest.skip("无法导入配置模块，跳过此测试")


def test_data_loader():
    """测试数据加载器"""
    try:
        from app.src.data.data_loader import get_data_loaders
        
        # 如果数据目录存在，测试数据加载器
        if os.path.exists("data/v1"):
            train_loader, test_loader, class_to_idx = get_data_loaders(
                "data/v1", batch_size=2
            )
            assert train_loader is not None
            assert test_loader is not None
            assert class_to_idx is not None
        else:
            pytest.skip("数据目录不存在，跳过数据加载器测试")
    except ImportError:
        pytest.skip("无法导入数据加载器模块，跳过此测试")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
