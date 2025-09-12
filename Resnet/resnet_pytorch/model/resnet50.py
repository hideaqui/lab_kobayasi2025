from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


def get_resnet50(pretrained: bool = True, num_classes: int = 10) -> nn.Module:
    """
    ResNet50モデルを取得し、指定された設定にカスタマイズします。
    
    Args:
        pretrained (bool): 事前学習済みモデルを使用するかどうか。
        num_classes (int): 出力層のクラス数。
    
    Returns:
        nn.Module: カスタマイズされたResNet50モデル。
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None

    try:
        model = resnet50(weights=weights)
    except Exception as e:
        raise RuntimeError(f"モデルのロードに失敗しました: {e}")

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
    )

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )

    return model


def disable_bn50(model):
    """ResNet50 の Batch Normalization を完全に無効化"""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()  # 推論モードに設定
            module.track_running_stats = False  # running_mean, running_var を更新しない
            module.affine = False  # γ, β を削除
            if module.weight is not None:
                module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
    return model


# ✅ モデルのテスト
if __name__ == "__main__":
    model = get_resnet50(pretrained=False)
    model = disable_bn50(model)  # BatchNorm を無効化
    print(model)  # モデルの構造を確認