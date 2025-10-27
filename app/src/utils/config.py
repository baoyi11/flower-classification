import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """配置管理类 - 修复版本"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # 尝试多个可能的配置文件路径
            possible_paths = [
                "mL/configs/config.yaml",
                "app/configs/config.yaml",
                "config.yaml",
                "../mL/configs/config.yaml",
            ]

            for path in possible_paths:
                if Path(path).exists():
                    config_path = path
                    break
            else:
                raise FileNotFoundError("找不到配置文件，请检查 config.yaml 是否存在")

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._create_dirs()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            if config is None:
                config = {}

            # 设置默认配置
            config = self._set_defaults(config)
            return config

        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            print("使用默认配置...")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "app": {"host": "0.0.0.0", "port": 8000},
            "data": {
                "raw_path": "data/raw",
                "processed_path": "data/processed",
                "v1_path": "data/v1",
                "v2_path": "data/v2",
                "image_size": [128, 128],
                "batch_size": 32,
                "num_workers": 4,
            },
            "model": {
                "num_classes": 5,
                "learning_rate": 0.001,
                "epochs": 5,
                "save_path": "models",
            },
            "logs": {"path": "logs"},
            "mlflow": {
                "tracking_uri": "mL/mlruns",
                "experiment_name": "flower-classification",
            },
        }

    def _set_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """设置默认值"""
        defaults = self._get_default_config()

        # 递归合并配置
        def merge_dicts(default, user):
            result = default.copy()
            for key, value in user.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result

        return merge_dicts(defaults, config)

    def _create_dirs(self):
        """创建必要的目录 - 修复版本"""
        try:
            # 使用安全的get方法访问配置
            dirs_to_create = []

            # 数据目录
            data_dirs = [
                self.get("data.raw_path", "data/raw"),
                self.get("data.processed_path", "data/processed"),
                self.get("data.v1_path", "data/v1"),
                self.get("data.v2_path", "data/v2"),
            ]
            dirs_to_create.extend(data_dirs)

            # 模型目录
            model_dir = self.get("model.save_path", "models")
            dirs_to_create.append(model_dir)

            # 日志目录
            log_dirs = [
                self.get("logs.path", "logs"),
                self.get("logs.tensorboard_dir", "logs/tensorboard"),
            ]
            dirs_to_create.extend(log_dirs)

            # MLflow目录
            mlflow_dir = "mL/mlruns"
            dirs_to_create.append(mlflow_dir)

            # 创建目录
            for dir_path in dirs_to_create:
                if dir_path:  # 确保路径不为空
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    print(f"✅ 创建目录: {dir_path}")

        except Exception as e:
            print(f"⚠️ 目录创建过程中出现警告: {e}")
            # 不抛出异常，继续执行

    def get(self, key: str, default: Any = None) -> Any:
        """安全地获取配置值"""
        try:
            keys = key.split(".")
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def save(self, path: Optional[str] = None):
        """保存配置到文件"""
        if path is None:
            path = self.config_path

        try:
            with open(path, "w", encoding="utf-8") as file:
                yaml.dump(
                    self.config, file, default_flow_style=False, allow_unicode=True
                )
            print(f"✅ 配置已保存到: {path}")
        except Exception as e:
            print(f"❌ 配置保存失败: {e}")

    def update(self, updates: Dict[str, Any]):
        """更新配置"""

        def update_dict(original, new):
            for key, value in new.items():
                if (
                    key in original
                    and isinstance(original[key], dict)
                    and isinstance(value, dict)
                ):
                    update_dict(original[key], value)
                else:
                    original[key] = value

        update_dict(self.config, updates)
        self._create_dirs()  # 更新后重新创建目录


# 创建全局配置实例
try:
    config = Config()
    print("✅ 配置加载成功")
except Exception as e:
    print(f"❌ 配置加载失败: {e}")
    # 使用默认配置创建实例
    config = Config.__new__(Config)
    config.config = config._get_default_config()
    config._create_dirs()
