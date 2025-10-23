import csv
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.init as init
from torch import nn, optim
from tqdm import tqdm

from dataset.mnist import get_dataloader
from model.resnet import get_resnet


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_backend() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _initialize_weights(model: nn.Module, mode: str) -> None:
    for module in model.modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        if mode == "normal":
            init.normal_(module.weight, mean=0.0, std=0.02)
        elif mode == "uniform":
            init.uniform_(module.weight, a=-0.1, b=0.1)
        elif mode == "xavier_normal":
            init.xavier_normal_(module.weight)
        elif mode == "xavier_uniform":
            init.xavier_uniform_(module.weight)
        elif mode == "kaiming_normal":
            init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        elif mode == "kaiming_uniform":
            init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

        if module.bias is not None:
            init.zeros_(module.bias)


@torch.inference_mode()
def _evaluate_accuracy(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def _capture_training_activations(activations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    snapshot: Dict[str, np.ndarray] = {}
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        data = activations.get(layer_name)
        if data is not None:
            snapshot[layer_name] = data.copy()
    return snapshot


def _save_activations(
    activations: Dict[str, np.ndarray],
    output_dir: str,
    epoch: int,
    mode: str,
) -> None:
    for layer_name, array in activations.items():
        file_path = os.path.join(output_dir, f"epoch_{epoch}_{layer_name}_activation_{mode}.npy")
        np.save(file_path, array)


def _save_accuracy_csv(
    output_path: str,
    accuracy_log: List[float],
    test_accuracy_log: List[float],
    generalization_gap: List[float],
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_accuracy", "test_accuracy", "generalization_gap"])
        for epoch, (train_acc, test_acc, gap) in enumerate(
            zip(accuracy_log, test_accuracy_log, generalization_gap)
        ):
            writer.writerow([epoch, train_acc, test_acc, gap])


def run_training_pipeline(
    mode: str,
    *,
    total_epoch: int = 20,
    seed: int = 1022,
    output_tag: str = "1022",
) -> Tuple[List[float], List[float], List[float], str]:
    """
    Train ResNet on MNIST, capturing per-layer activations and accuracy logs.

    Returns
    -------
    accuracy_log, test_accuracy_log, generalization_gap, output_dir
    """
    _set_seed(seed)
    _configure_backend()
    device = _get_device()
    print(f"使用中のデバイス: {device}")
    print(f"初期化モード: {mode}, エポック数: {total_epoch}")

    train_loader, test_loader = get_dataloader(
        root="data",
        batch_size=256,
        num_workers=4,
        seed=seed,
    )

    model = get_resnet(pretrained=False).to(device)
    _initialize_weights(model, mode=mode)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    activations: Dict[str, np.ndarray] = {}

    def make_hook(name: str):
        def hook(module, inputs, output):
            activations[name] = output.detach().cpu().numpy()

        return hook

    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        getattr(model, layer_name).register_forward_hook(make_hook(layer_name))

    output_dir = os.path.join("./output", f"{mode}_{output_tag}")
    os.makedirs(output_dir, exist_ok=True)

    accuracy_log: List[float] = []
    test_accuracy_log: List[float] = []
    generalization_gap: List[float] = []

    train_acc = _evaluate_accuracy(model, train_loader, device)
    train_activation_snapshot = _capture_training_activations(activations)
    test_acc = _evaluate_accuracy(model, test_loader, device)

    accuracy_log.append(train_acc)
    test_accuracy_log.append(test_acc)
    generalization_gap.append(train_acc - test_acc)
    _save_activations(train_activation_snapshot, output_dir, epoch=0, mode=mode)

    for epoch in range(1, total_epoch + 1):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epoch}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        scheduler.step()  # ← エポックごとにLRを更新
        train_activation_snapshot = _capture_training_activations(activations)

        train_acc = _evaluate_accuracy(model, train_loader, device)
        test_acc = _evaluate_accuracy(model, test_loader, device)

        accuracy_log.append(train_acc)
        test_accuracy_log.append(test_acc)
        generalization_gap.append(train_acc - test_acc)

        avg_loss = running_loss / len(train_loader)
        print(
            f"Epoch [{epoch}/{total_epoch}] "
            f"| Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
        )

        _save_activations(train_activation_snapshot, output_dir, epoch=epoch, mode=mode)

    accuracy_csv_path = os.path.join(output_dir, f"epoch_accuracies_{mode}.csv")
    _save_accuracy_csv(accuracy_csv_path, accuracy_log, test_accuracy_log, generalization_gap)
    print(f"✅ 精度ログを保存しました: {accuracy_csv_path}")

    return accuracy_log, test_accuracy_log, generalization_gap, output_dir
