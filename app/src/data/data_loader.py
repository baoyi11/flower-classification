import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from ..utils.config import config


class FlowerDataset(Dataset):
    """简化的花卉数据集"""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()
        self.class_to_idx = self._create_class_mapping()

    def _load_samples(self):
        """加载数据样本"""
        samples = []

        if not os.path.exists(self.data_dir):
            print(f"⚠️  数据目录不存在: {self.data_dir}")
            return samples

        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_name))

        print(f"✅ 从 {self.data_dir} 加载了 {len(samples)} 个样本")
        return samples

    def _create_class_mapping(self):
        """创建类别到索引的映射"""
        classes = sorted(list(set([sample[1] for sample in self.samples])))
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        print(f"✅ 类别映射: {class_to_idx}")
        return class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            # 将字符串标签转换为整数索引
            label_idx = self.class_to_idx[label]

            return image, label_idx
        except Exception as e:
            print(f"❌ 加载图像失败 {img_path}: {e}")
            # 返回一个占位符图像和默认标签
            image = Image.new("RGB", (128, 128), color="gray")
            if self.transform:
                image = self.transform(image)
            return image, 0


def get_data_loaders(data_dir: str, batch_size: int = None,
                     train_split: float = 0.8):
    """获取数据加载器"""

    # 使用配置值或默认值
    if batch_size is None:
        batch_size = config.get("data.batch_size", 32)

    image_size = config.get("data.image_size", [128, 128])

    # 数据变换
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # 创建数据集
    dataset = FlowerDataset(data_dir, transform=transform)

    if len(dataset) == 0:
        raise ValueError(f"数据目录 {data_dir} 中没有找到图像")

    # 分割数据集
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    if train_size == 0 or test_size == 0:
        raise ValueError("数据集太小，无法分割")

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"✅ 创建数据加载器: 训练集 {train_size} 样本, "
          f"测试集 {test_size} 样本")
    print(f"✅ 类别数量: {len(dataset.class_to_idx)}")

    return train_loader, test_loader, dataset.class_to_idx
