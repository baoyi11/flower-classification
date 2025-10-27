import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..utils.config import config


class SimpleTrainer:
    """简化的模型训练器"""

    def __init__(self, model, model_name: str = "simple"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.model_name = model_name

        # 设置优化器和损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("model.learning_rate", 0.001)
        )

        # 设置MLflow
        mlflow.set_tracking_uri(config.get("mlflow.tracking_uri"))
        mlflow.set_experiment("flower-classification")

    def train_epoch(self, train_loader, epoch: int):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (images, labels) in enumerate(pbar):
            # 确保标签是张量
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)

            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{100.*correct/total:.2f}%"
                }
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                # 确保标签是张量
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)

                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def train(self, train_loader, test_loader, epochs: int = None):
        """完整训练流程"""
        if epochs is None:
            epochs = config.get("model.epochs", 5)

        best_acc = 0.0

        with mlflow.start_run(run_name=self.model_name):
            # 记录参数
            mlflow.log_params(
                {
                    "model": self.model_name,
                    "epochs": epochs,
                    "learning_rate": config.get("model.learning_rate", 0.001),
                    "batch_size": config.get("data.batch_size", 32),
                }
            )

            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(train_loader, epoch)
                test_acc = self.evaluate(test_loader)

                # 记录指标
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "test_accuracy": test_acc,
                    },
                    step=epoch,
                )

                print(f"Epoch {epoch+1}/{epochs}:")
                print(
                    f"  Train Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_acc:.2f}%"
                )
                print(f"  Test Acc: {test_acc:.2f}%")

                # 保存最佳模型
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.save_model("best_model.pth")

            # 记录最佳准确率
            mlflow.log_metric("best_test_accuracy", best_acc)
            print(f"最佳测试准确率: {best_acc:.2f}%")

    def save_model(self, filename: str):
        """保存模型"""
        model_path = f"ml/registry/{filename}"
        torch.save(self.model.state_dict(), model_path)
        print(f"模型保存到: {model_path}")

    def load_model(self, filename: str):
        """加载模型"""
        model_path = f"ml/registry/{filename}"
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        print(f"模型从 {model_path} 加载")
