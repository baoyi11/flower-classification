import torch
import torch.nn as nn
from ..utils.config import config

class SimpleFlowerCNN(nn.Module):
    """简化的花卉分类CNN"""
    
    def __init__(self, num_classes: int = None):
        super().__init__()
        
        if num_classes is None:
            num_classes = config.get('model.num_classes', 5)
        
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 自适应池化层，适应不同输入尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

def create_model(model_type: str = 'simple', num_classes: int = None):
    """创建模型工厂函数"""
    if num_classes is None:
        num_classes = config.get('model.num_classes', 5)
    
    if model_type == 'simple':
        return SimpleFlowerCNN(num_classes)
    else:
        raise ValueError(f"未知模型类型: {model_type}")