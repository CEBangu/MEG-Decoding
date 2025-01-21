import torch.nn as nn
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #Conv blocks
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1) #(16, 247, 241)
        self.bn1 = nn.BatchNorm2d(16) #idea from Defossez (King) et al
        self.relu1 = nn.GELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #(16, 123, 120)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) #(32, 123, 120)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.GELU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #(32, 61, 60)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # (64, 61, 60)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.GELU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) #(64, 30, 30)

        #fc layers
        self.fc1 = nn.Linear(57600, 512)
        self.fc2 = nn.Linear(512, 18)
        

    def forward(self, x):
        # (1, 247, 241)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        #print("After conv1:", x.shape)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        #print("After conv2:", x.shape)
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        #print("After conv3:", x.shape)

        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x