from torch.utils.data import Dataset
import random
import numpy as np
import json
import os

class TripletMEGDataset(Dataset):
    def __init__(self, BcomMEG_object, label_map, indices, normalize=False):
        self.data_dict = BcomMEG_object 
        self.label_map = self.get_label_map(label_map)
        self.labels = self.get_labels(self.data_dict, self.label_map)

        self.data = self.data_dict.data_to_tensor(normalize)

        #ONLY TRAINING SAMPLES (or test)
        self.indices = indices
        self.data = self.data[self.indices]
        self.labels = [self.labels[i] for i in self.indices]

       # Build an index mapping from labels to indices for efficient sampling
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def get_label_map(self, map_name)->dict:
        
        '''This method is used by the class to populate its label_map attribute. 
        If a dictionary is passed as the map_name, then this method returns that dictionary.
        If a string is passed, then this method searches the directory for the label_maps.json, and recovers the corresponding
        label map if it is found in the file'''

        if type(map_name) == dict:
            label_map = map_name
        else:
            dir = os.getcwd()
            with open(os.path.join(dir, 'label_maps.json'), 'r') as f:
                label_maps = json.load(f)
                if map_name not in label_maps:
                    legal_names = list(label_maps.keys())
                    raise ValueError(f"map_name '{map_name}' not found in label_maps.json. Legal names are: {legal_names}")
                label_map = label_maps.get(map_name)
        
        return label_map
    

    def get_labels(self, data_dict, label_map) -> list:
        '''This method is used by the class to populate its labels attribute.
        It returns a list which matches the data at each index to its label'''
        labels = []
        syllable_counts = data_dict.get_syllable_counts()
        for subject in syllable_counts:
            for syllable in syllable_counts[subject]:
                labels.extend([label_map[syllable]] * syllable_counts[subject][syllable])
        return labels
    



    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        anchor = self.data[idx]
        anchor_label = self.labels[idx]
        
        # Positive samples from the same class, excluding the current idx
        positive_indices = self.label_to_indices[anchor_label]
        positive_indices = [i for i in positive_indices if i != idx]
        random_positive_index = random.choice(positive_indices)
        positive = self.data[random_positive_index]

        # Negative samples from different classes
        negative_label = random.choice([label for label in self.label_to_indices if label != anchor_label])
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative = self.data[negative_index]

        return anchor, positive, negative