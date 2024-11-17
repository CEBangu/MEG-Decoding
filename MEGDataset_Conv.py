from torch.utils.data import Dataset 
import json
import os

class MEGDataset_Conv(Dataset):
    def __init__(self, BcomMEG_object, label_map):
        self.data_dict = BcomMEG_object 
        self.label_map = self.get_label_map(label_map)
        self.labels = self.get_labels(self.data_dict, self.label_map)

        self.data, _  = BcomMEG_object.data_to_tensor() #NB! this is already a torch tensor, and also returns indexes
                                                  # which we want to avoid (the '_')

    def get_label_map(self, map_name):
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
    

    def get_labels(self, data_dict, label_map):
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
        sample = sample.unsqueeze(0)
        label = self.labels[index]

        # if self.transform:
        #     sample = self.transform(sample)

        return sample, label