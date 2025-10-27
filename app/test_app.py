#!/usr/bin/env python3
"""简化的应用测试"""

import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

class TestFlowerAPI:
    """API测试类"""
    
    def test_root_endpoint(self):
        """测试根端点"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
    
    def test_health_endpoint(self):
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
    
    def test_predict_valid_image(self):
        """测试有效图像预测"""
        # 创建测试图像
        image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        response = client.post("/predict", files=files)
        
        # 检查响应格式
        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert "all_classes" in data
            assert 0 <= data["confidence"] <= 1
    
    def test_predict_invalid_file(self):
        """测试无效文件"""
        files = {'file': ('test.txt', b'not an image', 'text/plain')}
        response = client.post("/predict", files=files)
        assert response.status_code == 400
    
    def test_predict_empty_file(self):
        """测试空文件"""
        files = {'file': ('empty.jpg', b'', 'image/jpeg')}
        response = client.post("/predict", files=files)
        assert response.status_code == 400

def test_model_creation():
    """测试模型创建"""
    from src.model.cnn_model import create_model
    model = create_model()
    assert model is not None

def test_config_loading():
    """测试配置加载"""
    from src.utils.config import config
    assert config.get('app.host') == '0.0.0.0'
    assert config.get('model.epochs') == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])