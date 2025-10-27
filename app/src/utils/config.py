import os
import yaml
from typing import Dict, Any

class Config:
    """简化的配置管理"""
    
    _default_config = {
        'app': {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False
        },
        'data': {
            'raw_path': 'data/raw',
            'v1_path': 'data/v1',
            'v2_path': 'data/v2',
            'image_size': [128, 128],
            'batch_size': 32
        },
        'model': {
            'num_classes': 5,
            'learning_rate': 0.001,
            'epochs': 5
        },
        'mlflow': {
            'tracking_uri': 'mL/mlruns'
        }
    }
    
    def __init__(self, config_path: str = "mL/configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._create_dirs()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return {**self._default_config, **yaml.safe_load(f)}
        return self._default_config
    
    def _create_dirs(self):
        """创建必要的目录"""
        dirs = [
            self.config['data']['raw_path'],
            self.config['data']['v1_path'], 
            self.config['data']['v2_path'],
            'mL/registry',
            'mL/mlruns'
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def get(self, key: str, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default

# 全局配置实例
config = Config()