import os
import torch
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
import numpy as np

from dataset.mnist import get_dataloader
from model.resnet import get_resnet  

def train(total_epoch: int = 20, save_path="output/"):
    # 保存フォルダを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 日時ベースのフォルダ名
    save_path = os.path.join(save_path, timestamp)
    os.makedirs(save_path, exist_ok=True)
    print(f"Results will be saved in: {save_path}")

    # デバイス設定
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # データローダーの取得
    dataloader = get_dataloader(root="data", batch_size=64)

    # モデルの作成とデバイスへの転送
    model = get_resnet(pretrained=False, num_classes=10)
    model = model.to(device)

    # オプティマイザーの定義
    optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=1e-3, total_steps=len(dataloader) * total_epoch
    )
    criterion = nn.CrossEntropyLoss()

    # 中間層出力を保存するためのフック関数を定義
    intermediate_outputs = {}

    def hook_fn(module, input, output):
        intermediate_outputs["layer4"] = output.detach().cpu().numpy()

    # フックを登録
    model.layer4.register_forward_hook(hook_fn)

    model.train()
    for epoch in range(total_epoch):
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # モデルからの出力
            out = model(images)

            # lossの算出
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            # 推測値
            preds = out.argmax(axis=1)

            # **🔹 バッチごとの正答率を保存**
            batch_accuracy = torch.sum(preds == labels).item() / len(labels)
            accuracy_save_path = os.path.join(save_path, f"epoch_{epoch + 1}_batch_{batch_idx + 1}_accuracy.txt")
            with open(accuracy_save_path, "w") as f:
                f.write(str(batch_accuracy))

            # **🔹 中間層の保存 (NumPy形式)**
            if "layer4" in intermediate_outputs:  
                intermediate_save_path = os.path.join(
                    save_path, f"epoch_{epoch + 1}_batch_{batch_idx + 1}_intermediate.npy"
                )
                np.save(intermediate_save_path, intermediate_outputs["layer4"])

        scheduler.step()

        # **🔹 エポックごとのモデルを保存**
        model_save_path = os.path.join(save_path, f"epoch_{epoch + 1}_model.pt")
        torch.save(model.state_dict(), model_save_path)

        print(f"✅ Epoch {epoch + 1} completed!")

if __name__ == "__main__":
    train()