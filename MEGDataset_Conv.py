from torch.utils.data import Dataset
import numpy as np
import json
import os

class MEGDataset_Conv(Dataset):
    def __init__(self, BcomMEG_object, label_map):
        self.data_dict = BcomMEG_object 
        self.label_map = self.get_label_map(label_map)
        self.labels = self.get_labels(self.data_dict, self.label_map)

        self.data, _  = self.data_dict.data_to_tensor() #NB! this also returns indexes
                                                  # which we want to avoid (the '_')

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
    
    def __getitem__(self, index):
        sample = self.data[index]
        sample = np.expand_dims(sample, axis=0)
        label = self.labels[index]

        # if self.transform:
        #     sample = self.transform(sample)

        return sample, label