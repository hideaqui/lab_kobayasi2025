import json
from test import evaluate_test_accuracy
from dataset.mnist import get_dataloader
from model.resnet import get_resnet
import torch

def save_test_accuracy():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = get_resnet(num_classes=10).to(device)
    _, test_dataloader = get_dataloader(root="data", batch_size=64)
    
    # モデルのロード（必要に応じて）
    # model.load_state_dict(torch.load("path_to_model.pth"))

    test_acc = evaluate_test_accuracy(model, test_dataloader, device)
    
    # 結果を保存
    with open("test_accuracy.json", "w") as f:
        json.dump({"accuracy": test_acc}, f, indent=4)
    
    print(f"Test accuracy saved in test_accuracy.json: {test_acc:.4f}")

if __name__ == "__main__":
    save_test_accuracy()