import torch.nn as nn
import torchvision.models as models


class CatsDogsModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.vgg19_bn(
            weights=models.VGG19_BN_Weights.DEFAULT
        )

        self.features = backbone.features

        for param in self.features.parameters():
            param.requires_grad = False

        for i, layer in enumerate(self.features.children()):
            if i >= 44:
                for param in layer.parameters():
                    param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 2)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x
