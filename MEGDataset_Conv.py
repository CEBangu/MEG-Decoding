from torch.utils.data import Dataset 

class MEGDataset_Conv(Dataset):
    def __init__(self, BcomMEG_object, label_map):
        # Ok, now the project of doing this correctly starts
        self.data_dict = BcomMEG_object 
        self.label_map = label_map #actually, I think this should be a property of this class, because it only really matters for the decoding. 

        self.labels = self.get_labels(self.data_dict, self.label_map)

        self.data, _  = BcomMEG_object.data_to_tensor() #NB! this is already a torch tensor, and also returns indexes
                                                  # which we want to avoid (the '_')


    def get_labels(self, data_dict, label_map):
        labels = []
        for subject in data_dict:
            labels.extend(label_map[syllable] for syllable in data_dict[subject])
        return labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        # if self.transform:
        #     sample = self.transform(sample)

        return sample, label