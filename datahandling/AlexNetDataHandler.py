from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

class AlexNetDataHandler(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :]),  # remove alpha channel
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = str(self.data.iloc[idx, 0])
        label = int(self.data.iloc[idx, 1])

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

