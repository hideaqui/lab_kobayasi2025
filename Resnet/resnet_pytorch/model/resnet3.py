from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def get_resnet(pretrained: bool = True, num_classes: int = 10, disable_bn: bool = True) -> nn.Module:
    """
    ResNet18モデルを取得し、CIFAR-10 用にカスタマイズし、BN を無効化するオプションを追加。

    Args:
        pretrained (bool): 事前学習済みモデルを使用するかどうか。
        num_classes (int): 出力層のクラス数。
        disable_bn (bool): True の場合、Batch Normalization を完全に無効化する。

    Returns:
        nn.Module: カスタマイズされた ResNet18 モデル。
    """
    # 事前学習済みの重みをロード
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # **入力チャンネルを 3 に変更（CIFAR-10 は RGB 画像）**
    model.conv1 = nn.Conv2d(
        in_channels=3,  # **RGB画像用に変更**
        out_channels=64,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
    )

    # **出力層を変更（CIFAR-10 用に 10 クラスに対応）**
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # **Batch Normalization を完全に無効化**
    if disable_bn:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()  # 推論モードに固定
                module.weight.requires_grad = False  # 学習しない
                module.bias.requires_grad = False
                module.track_running_stats = False  # running mean & var を更新しない

    return model


# **Batch Normalization を無効化した CIFAR-10 用 ResNet**
model = get_resnet(pretrained=False, num_classes=10, disable_bn=True)