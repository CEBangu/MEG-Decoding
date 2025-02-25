import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class AlexNetMPSFinalOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(weights=True)

        # Keep all layers as they are, including AdaptiveAvgPool2d
        self.features = self.model.features
        self.avgpool = self.model.avgpool  # This is AdaptiveAvgPool2d
        self.classifier = self.model.classifier
        self.classifier[-1] = nn.Linear(in_features=4096, out_features=3)

    def forward(self, x):
        x = self.features(x)  # Feature extraction
        x = self.avgpool(x.to("cpu")).to(x.device)  # Move to CPU, then back to MPS
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class AlexNetMPSSuddenDescend(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(weights=True)

        # Keep all layers as they are, including AdaptiveAvgPool2d
        self.features = self.model.features
        self.avgpool = self.model.avgpool  # This is AdaptiveAvgPool2d
        self.classifier = classifier_sudden

    def forward(self, x):
        x = self.features(x)  # Feature extraction
        x = self.avgpool(x.to("cpu")).to(x.device)  # Move to CPU, then back to MPS
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNetMPSLongDescend(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(weights=True)

        # Keep all layers as they are, including AdaptiveAvgPool2d
        self.features = self.model.features
        self.avgpool = self.model.avgpool  # This is AdaptiveAvgPool2d
        self.classifier = classifier_gradual_descend_long

    def forward(self, x):
        x = self.features(x)  # Feature extraction
        x = self.avgpool(x.to("cpu")).to(x.device)  # Move to CPU, then back to MPS
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class AlexNetMPSDescend(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(weights=True)

        # Keep all layers as they are, including AdaptiveAvgPool2d
        self.features = self.model.features
        self.avgpool = self.model.avgpool  # This is AdaptiveAvgPool2d
        self.classifier = classifier_descend

    def forward(self, x):
        x = self.features(x)  # Feature extraction
        x = self.avgpool(x.to("cpu")).to(x.device)  # Move to CPU, then back to MPS
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

###########
# Layer Adaptation
###########


# let's give it some options
classifier_sudden = nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=9216, out_features=4096), 
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=4096, out_features=3)
)

classifier_descend = nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=9216, out_features=4096), 
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=2048),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=2048, out_features=3)
)

classifier_gradual_descend_long = nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(9216, 4096), 
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=2048),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=2048, out_features=1024),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=1024, out_features=512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=512, out_features=3)
)