#!/usr/bin/env python3
"""简化的数据预处理管道"""

import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance


def create_dataset_version_v1(input_dir: str, output_dir: str):
    """创建版本v1 - 基础数据集"""
    print("创建数据集版本 v1...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 复制原始数据
    if Path(input_dir).exists():
        for class_name in os.listdir(input_dir):
            class_dir = Path(input_dir) / class_name
            if not class_dir.is_dir():
                continue

            dest_dir = output_path / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_file in class_dir.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    shutil.copy2(img_file, dest_dir / img_file.name)

    # 创建数据集信息
    class_count = len([d for d in output_path.iterdir() if d.is_dir()])
    total_images = sum(
        len(list(d.iterdir())) for d in output_path.iterdir() if d.is_dir()
    )

    dataset_info = {
        "version": "v1",
        "num_classes": class_count,
        "total_images": total_images,
        "description": "基础数据集",
    }

    # 保存数据集信息
    with open(output_path / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"版本 v1 创建完成: {output_path}")
    print(f"类别: {class_count}, 图像: {total_images}")
    return output_path


def create_dataset_version_v2(input_dir: str, output_dir: str):
    """创建版本v2 - 增强数据集"""
    print("创建数据集版本 v2...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 数据增强函数
    def apply_augmentation(image):
        """应用随机增强"""
        augmentations = [
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            lambda img: img.rotate(random.randint(-15, 15)),
            lambda img: ImageEnhance.Brightness(img).enhance(
                random.uniform(0.8, 1.2)
            ),
        ]

        # 随机选择一种增强
        aug_func = random.choice(augmentations)
        return aug_func(image)

    # 处理每个类别
    for class_name in os.listdir(input_dir):
        class_dir = Path(input_dir) / class_name
        if not class_dir.is_dir():
            continue

        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        # 复制并增强图像
        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # 复制原始图像
            original_img = Image.open(img_path)
            original_img.save(output_class_dir / f"original_{img_path.name}")

            # 创建增强版本
            for i in range(2):  # 每个图像创建2个增强版本
                augmented_img = apply_augmentation(original_img.copy())
                augmented_img.save(
                    output_class_dir / f"augmented_{i}_{img_path.name}"
                )

    # 创建数据集信息
    class_count = len([d for d in output_path.iterdir() if d.is_dir()])
    total_images = sum(
        len(list(d.iterdir())) for d in output_path.iterdir() if d.is_dir()
    )

    dataset_info = {
        "version": "v2",
        "num_classes": class_count,
        "total_images": total_images,
        "description": "增强数据集",
        "augmentation": "flip, rotate, brightness",
    }

    with open(output_path / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"版本 v2 创建完成: {output_path}")
    print(f"类别: {class_count}, 图像: {total_images}")
    return output_path


def validate_dataset(data_dir: str):
    """验证数据集"""
    path = Path(data_dir)
    if not path.exists():
        return False, "数据集路径不存在"

    classes = [d for d in path.iterdir() if d.is_dir()]
    if len(classes) == 0:
        return False, "没有找到类别目录"

    total_images = 0
    for class_dir in classes:
        images = list(class_dir.glob("*.*"))
        total_images += len(images)
        if len(images) == 0:
            return False, f"类别 {class_dir.name} 中没有图像"

    return True, f"验证通过: {len(classes)} 个类别, {total_images} 张图像"


def main():
    """主函数"""
    print("开始创建数据集版本...")

    # 验证原始数据
    valid, message = validate_dataset("data/raw")
    if not valid:
        print(f"原始数据验证失败: {message}")
        return

    # 创建v1版本
    print("\n" + "=" * 40)
    v1_path = create_dataset_version_v1("data/raw", "data/v1")
    valid, message = validate_dataset(v1_path)
    print(f"v1验证: {message}")

    # 创建v2版本
    print("\n" + "=" * 40)
    v2_path = create_dataset_version_v2("data/raw", "data/v2")
    valid, message = validate_dataset(v2_path)
    print(f"v2验证: {message}")

    print("\n数据集版本创建完成!")
    print("使用 DVC 跟踪数据版本:")
    print("dvc add data/v1 data/v2")
    print("git add data/v1.dvc data/v2.dvc .gitignore")


if __name__ == "__main__":
    main()
