import torch
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor

net = resnet18()
feature_extractor = create_feature_extractor(net, {"avgpool": "feature"})

x = torch.rand((1, 3, 224, 224)).float()
feature_dict = feature_extractor(x)
print(feature_dict["feature"].shape)