__all__ = ['BcomMEG', 'AlexNetDataHandler', 'ViTDataHandler']

def __getattr__(name):
    if name == "BcomMEG":
        from .BcomMEG import BcomMEG
        return BcomMEG
    elif name == "AlexNetDataHandler":
        from .AlexNetDataHandler import AlexNetDataHandler
        return AlexNetDataHandler
    elif name == "ViTDataHandler":
        from .ViTDataHandler import ViTDataHandler
        return ViTDataHandler
    raise AttributeError(f"Module {__name__} has no attribute {name}")
