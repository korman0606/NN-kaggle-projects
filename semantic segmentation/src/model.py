import torch
import torch.nn as nn
import torchvision.models as models
import config

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            skip = skip[:, :, diff_h//2 : diff_h//2 + x.shape[2],
                                 diff_w//2 : diff_w//2 + x.shape[3]]
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)

class UNetWithResNet50(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.dec1 = DecoderBlock(in_channels=2048, skip_channels=1024, out_channels=1024)
        self.dec2 = DecoderBlock(in_channels=1024, skip_channels=512,  out_channels=512)
        self.dec3 = DecoderBlock(in_channels=512,  skip_channels=256,  out_channels=256)
        self.dec4 = DecoderBlock(in_channels=256,  skip_channels=64,   out_channels=64)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip0 = x
        x = self.maxpool(x)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        d1 = self.dec1(s4, s3)
        d2 = self.dec2(d1, s2)
        d3 = self.dec3(d2, s1)
        d4 = self.dec4(d3, skip0)
        out = self.final_up(d4)
        out = self.final_conv(out)
        return out

def init_model(device):
    model = UNetWithResNet50().to(device)
    # Заморозка всех слоёв, затем разморозка layer3, layer4 и декодера
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(layer in name for layer in ['layer3', 'layer4', 'dec', 'final_up', 'final_conv']):
            param.requires_grad = True
    return model

def get_optimizer(model):
    encoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and ('layer3' in n or 'layer4' in n)]
    decoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and not ('layer3' in n or 'layer4' in n)]
    import torch.optim as optim
    return optim.RMSprop([
        {'params': decoder_params, 'lr': config.LEARNING_RATE_DECODER},
        {'params': encoder_params, 'lr': config.LEARNING_RATE_ENCODER}
    ])
