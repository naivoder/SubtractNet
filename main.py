import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout3 = nn.Dropout2d(p=0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.maxpool(x)
        return x

class SubtractNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SubtractNet, self).__init__()
        self.block1a = ConvBlock(in_channels, 32)
        self.block1b = ConvBlock(in_channels, 32)

        self.block2a = ConvBlock(32, 64)
        self.block2b = ConvBlock(32, 64)

        self.block4a = ConvBlock(64, 128)
        self.block4b = ConvBlock(64, 128)

        self.block5a = ConvBlock(128, 256)
        self.block5b = ConvBlock(128, 256)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.output = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x1 = self.block1a(x)
        x2 = self.block1b(x)
        x = torch.abs(x1 - x2)
        
        x1 = self.block2a(x)
        x2 = self.block2b(x)
        x = torch.abs(x1 - x2)
        
        x1 = self.block3a(x)
        x2 = self.block3b(x)
        x = torch.abs(x1 - x2)

        x1 = self.block4a(x)
        x2 = self.block4b(x)
        x = torch.abs(x1 - x2)

        x1 = self.block5a(x)
        x2 = self.block5b(x)
        x = torch.abs(x1 - x2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        return self.output(x)
    
if __name__=="__main__":
    model = SubtractNet(in_channels=3, num_classes=10)
    
    x1 = torch.randn(1, 3, 128, 128)
    x2 = torch.randn(1, 3, 128, 128)
    
    output = model(x1, x2)
    print("Output shape:", output.shape)