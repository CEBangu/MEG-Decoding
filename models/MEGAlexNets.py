import torch
import torch.nn as nn
from torchvision import models


#######################
# Layer freezing code #
#######################
class LayerFreezeMixin:
    """This class allows you to apply layer freezing to all of the alexnet combos
    without having to repeat this code over and over again"""
   
    def freeze_type(self, freeze_type=None):
        freeze_types = ["final", "full", "most", "none"]
        if freeze_type not in freeze_types:
            raise ValueError(f'freeze_type must be {freeze_types}')

        if freeze_type == "final":
            self._final_only()
        elif freeze_type == "full":
            self._features_only()
        elif freeze_type == "most":
            self._freeze_most()
        elif freeze_type == "none":
            self._no_freeze()

    def _final_only(self):
        """Freeze all layers except the last classifier layer."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier[:-1].parameters(): 
            param.requires_grad = False
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("last classifier layer unfrozen")
        print(f"Number of trainable parameters: {trainable_params}")

    def _features_only(self):
        """Freeze the entire feature extractor."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("feature layer frozen; classifier unfrozen")
        print(f"Number of trainable parameters: {trainable_params}")

    def _freeze_most(self):
        """Freeze all but the last 5 feature extraction layers."""
        # Unfreeze everything first
        for param in self.features.parameters():
            param.requires_grad = True
        # Freeze all but the last 5 modules
        feature_modules = list(self.features.children())
        for layer in feature_modules[:-5]:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("last 5 layers of feature extractor and all classifier layers are unfrozen")
        print(f"Number of trainable parameters: {trainable_params}")

    def _no_freeze(self):
        """No freezing, full network trainable."""
        for param in self.features.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("All parameters available")
        print(f"Number of trainable parameters: {trainable_params}")

    def reset_parameters(self):
        """Resets all layers to their original initialization safely."""
        for name, module in self.named_children():
            print('resetting ', name)
            module.reset_parameters()
#######################################################################################################################################

# Models

class AlexNetFinalOnly(nn.Module, LayerFreezeMixin):
    def __init__(self, num_classes=3):
        super().__init__()
        pretrained =  models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        # pretrained =  models.alexnet(weights=None)
        pretrained.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
        # Keep all layers as they are, including AdaptiveAvgPool2d
        self.features = pretrained.features
        self.avgpool = pretrained.avgpool  # This is AdaptiveAvgPool2d
        self.classifier = pretrained.classifier

        del pretrained #because otherwise torch gets discombobulated

    def forward(self, x):
        x = self.features(x)  # Feature extraction
        if x.device.type == "mps":
            x = self.avgpool(x.to("cpu")).to(x.device)
        else:
            x = self.avgpool(x)    
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class AlexNetSuddenDescend(nn.Module, LayerFreezeMixin):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        # self.model =  models.alexnet(weights=None)
        # Keep all layers as they are, including AdaptiveAvgPool2d
        self.features = self.model.features
        self.avgpool = self.model.avgpool  # This is AdaptiveAvgPool2d

        del self.model.classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=9216, out_features=4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)  # Feature extraction
        if x.device.type == "mps":
            x = self.avgpool(x.to("cpu")).to(x.device)
        else:
            x = self.avgpool(x)    
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNetLongDescend(nn.Module, LayerFreezeMixin):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

        # Keep all layers as they are, including AdaptiveAvgPool2d
        self.features = self.model.features
        self.avgpool = self.model.avgpool  # This is AdaptiveAvgPool2d

        del self.model.classifier
        self.classifier = nn.Sequential( #NB! these are global objects, so they can't be created outside of the class because then they will be held over the entire time! 
            nn.Dropout(p=0.5, inplace=False), # and KFolds will fail!!!
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
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)  # Feature extraction
        if x.device.type == "mps":
            x = self.avgpool(x.to("cpu")).to(x.device)
        else:
            x = self.avgpool(x)    
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetDescend(nn.Module, LayerFreezeMixin):
    def __init__(self, num_classes=3):
        super().__init__()
        base = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.features = base.features

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) # can mess around with this
    
        self.classifier = nn.Sequential( # can also mess around with this
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(9216, num_classes)
        )

        
    def forward(self, x):
        x = self.features(x)  # Feature extraction
        if x.device.type == "mps":
            x = self.avgpool(x.to("cpu")).to(x.device)
        else:
            x = self.avgpool(x)    
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=9216, out_features=4096), 
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=4096, out_features=2048),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=2048, out_features=num_classes)
        # self.classifier = nn.Sequential(
        # )
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=9216, out_features=512), 
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=512, out_features=num_classes),
        # )