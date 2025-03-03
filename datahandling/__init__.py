# datahandling/__init__.py
__all__ = ['BcomMEG', 'AlexNetDataHandler', 'ViTDataHandler']

class LazyLoader:
    def __init__(self, import_path, class_name):
        self.import_path = import_path
        self.class_name = class_name
        self._class = None
        
    def __call__(self, *args, **kwargs):
        if self._class is None:
            # Dynamically import when first called
            module = __import__(self.import_path, fromlist=[self.class_name])
            self._class = getattr(module, self.class_name)
        return self._class(*args, **kwargs)

# Create lazy loaders for each class
BcomMEG = LazyLoader("datahandling.BcomMEG", "BcomMEG")
AlexNetDataHandler = LazyLoader("datahandling.AlexNetDataHandler", "AlexNetDataHandler")
ViTDataHandler = LazyLoader("datahandling.ViTDataHandler", "ViTDataHandler")