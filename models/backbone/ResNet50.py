import torch
import torchvision.models as models
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self,):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.in_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4


    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        return x1, x2, x3, x4, x5

if __name__ == '__main__':
    A = torch.rand(1, 3, 224, 224)
    model = ResNet50()
    a1, a2, a3, a4, a5 = model(A)
    print(a1.shape)
    print(a2.shape)
    print(a3.shape)
    print(a4.shape)
    print(a5.shape)



