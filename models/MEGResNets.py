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
        freeze_types = ["final", "feature", "most", "none"]
        if freeze_type not in freeze_types:
            raise ValueError(f'freeze_type must be {freeze_types}')

        if freeze_type == "final":
            self._final_only()
        elif freeze_type == "feature":
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

class ResNet101SmallHead(nn.Module, LayerFreezeMixin):
    def __init__(self, num_classes=3):
        super().__init__()
        base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*(list(base.children())[:-2]))  # All layers up to avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Replace base.avgpool

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes)# 2048 is resnet101's final feature size
        )

    def forward(self, x):
        x = self.features(x)
        if x.device.type == "mps":
            x = self.avgpool(x.to("cpu")).to(x.device)
        else:
            x = self.avgpool(x)
        x = self.classifier(x)
        return x

class ResNet101LongHead(nn.Module, LayerFreezeMixin):
    def __init__(self, num_classes=3):
        super().__init__()
        base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*(list(base.children())[:-2]))  # All layers up to avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Replace base.avgpool

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
              # 2048 is resnet101's final feature size
        )

    def forward(self, x):
        x = self.features(x)
        if x.device.type == "mps":
            x = self.avgpool(x.to("cpu")).to(x.device)
        else:
            x = self.avgpool(x)
        x = self.classifier(x)
        return x


