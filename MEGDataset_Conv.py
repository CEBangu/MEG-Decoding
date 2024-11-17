from torch.utils.data import Dataset 

class MEGDataset_Conv(Dataset):
    def __init__(self, BcomMEG_object):
        # Ok, now the project of doing this correctly starts
        self.data_dict = BcomMEG_object 
        self.label_map = { #TODO: thisis blatently inellegant - is there a better way of doing this? 
            #Covert Speech
            'a_112': 0,'e_114': 1,'i_116': 2,'la_122': 3,'le_124': 4,'li_126': 5,'ma_132': 6,'me_134': 7,'mi_136': 8, 
            'ra_142': 9,'re_144': 10,'ri_146': 11,'sa_152': 12,'se_154': 13,'si_156': 14,'ta_162': 15,'te_164': 16,'ti_166': 17,
            #Overt Speech
            'a_12': 20,'e_14': 21,'i_16': 22,'la_22': 23,'le_24': 24,'li_26': 25,'ma_32': 26,'me_34': 27,'mi_36': 28, 'ra_42': 29,
            're_44': 30,'ri_46': 31,'sa_52': 32,'se_54': 33,'si_56': 34,'ta_62': 35,'te_64': 36,'ti_66': 37,
        }

        self.labels = self.get_labels(self.data_dict, self.label_map)

        self.data, _  = BcomMEG_object.data_to_tensor() #NB! this is already a torch tensor, and also returns indexes
                                                  # which we want to avoid (the '_')


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
        label = self.labels[index]

        # if self.transform:
        #     sample = self.transform(sample)

        return sample, label