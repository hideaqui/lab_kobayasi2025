# ResNet18モデルの定義（モノクロ画像対応）
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def get_resnet(pretrained: bool = True, num_classes: int = 10) -> nn.Module:
    """
    ResNet18モデルを取得し、指定された設定にカスタマイズします。
    
    Args:
        pretrained (bool): 事前学習済みモデルを使用するかどうか。
        num_classes (int): 出力層のクラス数。
    
    Returns:
        nn.Module: カスタマイズされたResNet18モデル。
    """
    # 事前学習済みの重みをロード（最新方式で対応）
    weights = ResNet18_Weights.DEFAULT if pretrained else None

    try:
        # ResNet18モデルのロード
        model = resnet18(weights=weights)
    except Exception as e:
        raise RuntimeError(f"モデルのロードに失敗しました: {e}")

    # 入力層を変更（モノクロ画像対応）
    model.conv1 = nn.Conv2d(
        in_channels=1,  # 入力チャンネル数（モノクロ画像）
        out_channels=64,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
    )

    # 出力層を変更（指定クラス数に対応）
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )

    return model


def disable_bn(model):
    """ResNet の Batch Normalization を無効化する"""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()  # BatchNorm を推論モードにする
            module.weight.requires_grad = False
            module.bias.requires_grad = False
    return model


# モデルを取得
model = get_resnet(pretrained=False)

# Batch Normalization を無効化
model = disable_bn(model)