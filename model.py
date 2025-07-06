import torch
import torch.nn as nn

# Created Reusable Residual Block
class ResidualBlock(nn):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        # This process checks if the input and output are of same size , else it transforms the size of output
        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        out = self.conv2(x)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_added = out + shortcut
        out = torch.relu(out_added)
        
        return out
    
class AudioCNN(nn.Module):
    # created the constructer
    def __init__(self, num_classes=50):
        super().__init__()                                                                                           
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), 
            nn.MaxPool2d(3, stride=2, padding=1)
            )
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for i in range(3)])
        self.layer2 = nn.ModuleList([ResidualBlock(64 if i == 0 else 128, 128) for i in range(4)])
        self.layer3 = nn.ModuleList([ResidualBlock(128 if i == 0 else 256, 256) for i in range(6)])
        self.layer4 = nn.ModuleList([ResidualBlock(256 if i == 0 else 512, 512) for i in range(3)])
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)    
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1) # To reshape the tensor without changing its data
        x = self.dropout(x)
        x = self.fc(x)
        return x