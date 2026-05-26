import torch
import torch.nn as nn
import torchvision.models as models

def build_model(pretrained=True, dropout_rate=0.5, num_classes=2):
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=None)
    
    # Замораживаем все слои
    for param in model.parameters():
        param.requires_grad = False
    
    # Размораживаем последние 2 слоя (layer4 + fc)
    # В ResNet50 children: 0: conv1, 1: bn1, 2: relu, 3: maxpool, 4: layer1, 5: layer2, 6: layer3, 7: layer4, 8: avgpool, 9: fc
    for i, layer in enumerate(model.children()):
        if i >= 7:          # начиная с layer4
            for param in layer.parameters():
                param.requires_grad = True
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(512, num_classes)
    )
    return model
