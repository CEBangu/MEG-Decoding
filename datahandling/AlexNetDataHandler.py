from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from PIL import Image

class AlexNetDataClass(Dataset):
    def __init__(self, csv_file, img_directory, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_directory = img_directory
        self.transform = transform


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_directory, str(self.data.iloc[idx, 0]))
        label = int(self.data.iloc[idx, 1])

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label