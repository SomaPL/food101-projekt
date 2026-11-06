"""ResNet-18 w trzech wariantach: od zera, z wagami ImageNet, jako ekstraktor cech."""
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .config import NUM_CLASSES


def build_resnet18(pretrained: bool, num_classes: int = NUM_CLASSES) -> nn.Module:
    """pretrained=True -> start z wag ImageNet (transfer), False -> losowe wagi (od zera).

    W obu przypadkach wymieniamy ostatnią warstwę: zamiast 1000 klas ImageNet ma 20 naszych.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 512 -> 20
    return model


def build_feature_extractor() -> nn.Module:
    """Sieć z ImageNet bez ostatniej warstwy: zamienia zdjęcie w wektor 512 cech."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model.eval()
