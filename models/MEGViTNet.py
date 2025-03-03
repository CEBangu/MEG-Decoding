from transformers import ViTForImageClassification

class MEGVisionTransformer(ViTForImageClassification):
    """Custom VIT wrapper that makes sure the positional encodings are interpolated
    This inherits from the base ViTForImageClassification class, doesn't need any arguments
    It adds the interpolate_pos_encoding to the forward method
    It also adds a method that allows you to freeze/unfreeze layers:
        You can either unfreeze the classifier only (around 2k trainable params) : 'classifier'
        Or you can unfreeze the classifier and the attention heads (around 28m trainable params) : 'attention'
        Or you can unfreeze the entire model (around 80m trainable params) 'all'
        """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # parent init

    def forward(self, pixel_values=None, labels=None, **kwargs):
        """Custom forward pass because we need positional interpolation"""
        return super().forward(
            pixel_values=pixel_values,
            labels=labels,
            interpolate_pos_encoding=True, # need to define custom forward method so that we can interpolate the encodings
            **kwargs
        )
    
    def freeze_type(self, freeze_type=None):
        if freeze_type not in ["classifier", "attention", "all"]:
            raise ValueError('freeze_type is either classifier, attention, or all')

        if freeze_type == "classifier":
            self.classifier_only()
        elif freeze_type == "attention":
            self.mha_training()
        elif freeze_type == 'all':
            self.train_all()

    ### layer freezing methods
    def classifier_only(self):
        """this keeps only the classifier open to training"""
        for name, param in self.named_parameters():
            # Unfreeze if the parameter belongs to the classifier or an attention layer.
            if "classifier" in name:
              param.requires_grad = True
              print(f"Unfreezing {name}")
            else:
                param.requires_grad = False
                print(f"Freezing {name}")

    def mha_training(self):
        """This keeps the classifier and attention heads open to training, following
        Touvron et al."""
        for name, param in self.named_parameters():
            # Unfreeze if the parameter belongs to the classifier or an attention layer.
            if "classifier" in name or "attention" in name:
                param.requires_grad = True
                print(f"Unfreezing {name}")
            else:
                param.requires_grad = False
                print(f"Freezing {name}")
    
    def train_all(self):
        """Full model training"""
        for name, param in self.named_parameters():
            param.requires_grad = True
            print(f"Unfreezing {name}")