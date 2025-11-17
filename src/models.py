import torch
import torch.nn as nn
from torchvision import models


# Backbones individuales

def _make_resnet18(num_classes=2, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _make_resnet50(num_classes=2, pretrained=True):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    m = models.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _make_densenet121(num_classes=2, pretrained=True):
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    m = models.densenet121(weights=weights)
    # DenseNet usa "classifier" como capa final
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m


def _make_efficientnet_b0(num_classes=2, pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    # EfficientNet de torchvision tiene classifier = Sequential(Dropout, Linear)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


# API genérica: make_backbone(arch=...)

_BACKBONE_FACTORY = {
    "resnet18": _make_resnet18,
    "resnet50": _make_resnet50,
    "densenet121": _make_densenet121,
    "efficientnet_b0": _make_efficientnet_b0,
}


def make_backbone(arch: str = "resnet18", num_classes: int = 2, pretrained: bool = True):
    """
    Construye un backbone según el nombre de arquitectura.

    arch puede ser:
      - "resnet18"
      - "resnet50"
      - "densenet121"
      - "efficientnet_b0"
    lo que nos permite probar distintos modelos

    Todos funcionan correctamente con imágenes 3x224x224.
    """
    key = arch.lower()
    if key not in _BACKBONE_FACTORY:
        raise ValueError(f"Arquitectura no soportada: {arch}. "
                         f"Opciones: {list(_BACKBONE_FACTORY.keys())}")
    return _BACKBONE_FACTORY[key](num_classes=num_classes, pretrained=pretrained)


# ============================================================
# Compatibilidad hacia atrás: make_resnet18
# ============================================================

def make_resnet18(num_classes=2, pretrained=True):
    """
    Función original mantenida por compatibilidad.
    Internamente llama al mismo constructor que make_backbone("resnet18").
    """
    return _make_resnet18(num_classes=num_classes, pretrained=pretrained)
