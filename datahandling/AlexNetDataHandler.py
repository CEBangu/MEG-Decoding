from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn.functional as F
import math
import pandas as pd
import os
from PIL import Image
import numpy as np
from matplotlib import cm

class AlexNetDataHandler(Dataset):
    def __init__(self, csv_file, sensor_indices, output_size=(224, 224),
                 coeff_augment_fn=None):
        

        self.data=pd.read_csv(csv_file)

        self.sensor_indices = sensor_indices
        self.output_size = output_size
        self.coeff_augment_fn = coeff_augment_fn

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = row['FileName'] #double check this
        label = row['Label'] # double check this

        coeffs = np.load(file_path)

        if "covert_producing" in file_path:
            vmin, vmax = None, None
        elif "covert_reading" in file_path:
            vmin, vmax = None, None
        elif "overt_producing" in file_path:
            vmin, vmax = None, None
        else:
            raise ValueError("incorrect file path")

        if self.coeff_augment_fn:
            coeffs = self.coeff_augment_fn(coeffs)
        
        image = self._generate_scalogram(coeffs, self.sensor_indices,
                                    vmin=vmin, vmax=vmax,
                                    output_size=self.output_size)
        
        return image, label
    
    def _zscore_to_rgb_tensor(self, z, vmin, vmax):
        """This function takes in the zscored coefficeints, and turns them into an RGB tensor via the turbo colormap"""
        z_clipped = np.clip((z - vmin )/(vmax - vmin), 0, 1)
        rgb = cm.turbo(z_clipped)[..., :3]
        rgb = np.transpose(rgb, (2, 0, 1))
        return torch.from_numpy(rgb).float()

    def _auto_layout_square(self, n):
        """This helper function computes the dimensions of the scalogram collage"""
        return math.ceil(math.sqrt(n)), math.ceil(math.sqrt(n))

    def _stitch_scalograms(self, rgb_list, layout):
        """This function stitched the scalograms together for the collage"""
        rows, cols = layout
        H, W = rgb_list[0].shape[1:] # get the right dimension
        total_tiles = rows * cols
        n_patches = len(rgb_list)
        n_missing = total_tiles - n_patches
        
        blank = torch.zeros((3, H, W), dtype=rgb_list[0].dtype) # make the padding arrays
        
        # it gets plotted from front to back so I need to the empty ones first
        full_list = rgb_list[:]
        if n_missing > 0:
            full_list += [blank] * n_missing
        
        stitched_rows = [ # stitches all of the rows together
            torch.cat(full_list[r * cols: (r + 1) * cols], dim=2)
            for r in range(rows)
        ] 
        stitched_rows = stitched_rows[::-1] # otherwise the 0s get plotted on top
        return torch.cat(stitched_rows, dim=1) # concatenate on the height

    def _generate_scalogram(self, coefficients, sensor_indices, vmin, vmax, output_size=(224, 224)):
        rgb_list = []
        for idx in sensor_indices:
            z = coefficients[idx] # get the coefficients
            rgb = self.zscore_to_rgb_tensor(z, vmin=vmin, vmax=vmax) #transform them into rgb_tensors
            rgb_list.append(rgb)

        layout=self._auto_layout_square(len(sensor_indices)) # get layout dimensions
        collage = self._stitch_scalograms(rgb_list, layout=layout) # stitch em
        collage_resized = F.interpolate(collage.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False).squeeze(0) #resize to AlexNet specs

        # return collage
        return collage_resized
    

# class AlexNetDataHandler(Dataset):
#     def __init__(self, csv_file):
#         self.data = pd.read_csv(csv_file)
#         self.transform = transforms.Compose([
#             transforms.Resize((224,224)),
#             # transforms.RandomCrop(224),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x[:3, :, :]),  # remove alpha channel
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             ])

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         img_path = str(self.data.iloc[idx, 0])
#         label = int(self.data.iloc[idx, 1])

#         image = Image.open(img_path)

#         if self.transform:
#             image = self.transform(image)

#         return image, label

