import os
import mne
import numpy as np
import pandas as pd
from typing import Tuple
from numpy.typing import NDArray
import torch
import copy


class BcomMEG():
    def __init__(self, dir, subjects, picks=None, avoid_reading=True):
        self.dir = dir
        self.subjects = subjects
        self.picks = picks
        self.avoid_reading = avoid_reading
        self.data = self.load_data()

    def load_data(self) -> dict:
        data_dict = {}
        for subject in self.subjects:
            if subject not in data_dict:
                data_dict[subject] = {}
            for root, dirs, files in os.walk(self.dir):
                for file in files:
                    if file.startswith(subject):
                        epo_name = file[10:-8]
                        if (self.avoid_reading == True) and (sum(c.isdigit() for c in epo_name) < 3):
                            continue
                        file = os.path.join(self.dir, file)
                        data_dict[subject][epo_name] = mne.read_epochs(file, preload=True).pick(picks=self.picks).get_data()
        return data_dict

    def get_epo_pca(self) -> Tuple[NDArray, NDArray]:
        all_epochs = []
        labels = []
        i = 1
        for subject in self.data:
            for syllable in self.data[subject]:
                for epoch in self.data[subject][syllable]:
                    all_epochs.append(epoch)
                    labels.append(i)
                i += 1
        return np.array(all_epochs), np.array(labels)

    def sensor_correlations(self, syllable_epochs: NDArray):
        i = 1
        correlations = pd.DataFrame({'Epoch': [], 'Max Correlation Value': [], 'Max Correlation Indices': []})
        for trial in syllable_epochs:
            correlation_matrix = np.corrcoef(trial, rowvar=True)
            max_corr_value = np.max(np.abs(correlation_matrix)[np.abs(correlation_matrix) < 0.99])
            max_corr_indices = np.where(np.abs(correlation_matrix) == max_corr_value)
            max_corr_indices = list(zip(*max_corr_indices))
            max_corr_indices = list(set(tuple(sorted(pair)) for pair in max_corr_indices))
            correlations = pd.concat([correlations, pd.DataFrame({'Epoch': i, 'Max Correlation Value': max_corr_value, 'Max Correlation Indices': max_corr_indices})], ignore_index=True)
            i += 1
        return correlations

    def get_syllable_counts(self):
        syllable_counts = {}
        for subject in self.data:
            syllable_counts[subject] = {}
            for syllable in self.data[subject]:
                syllable_counts[subject][syllable] = len(self.data[subject][syllable])
        return syllable_counts

    def get_max_length(self, syllable_count):
        max_length = {}
        for subjects in syllable_count:
            max_length[subjects] = max(syllable_count[subjects].values())
        return max_length

    def padding(self, rows=None, columns=None): #Depricated?
        if rows is None:
            rows = list(list(self.data.values())[0].values())[0][0].shape[0]
        if columns is None:
            columns = list(list(self.data.values())[0].values())[0][0].shape[1]
        for subject in self.data:
            max_length = self.get_max_length(self.get_syllable_counts())
            for syllable in self.data[subject]:
                if len(self.data[subject][syllable]) < max_length[subject]:
                    padding = np.zeros([max_length[subject] - len(self.data[subject][syllable]), rows, columns])
                    self.data[subject][syllable] = np.concatenate((self.data[subject][syllable], padding))
        return self.data

    def concat_padded(self, rows=None, columns=None):
        if rows is None:
            rows = list(list(self.data.values())[0].values())[0][0].shape[0]
        if columns is None:
            columns = list(list(self.data.values())[0].values())[0][0].shape[1]
        concatenated = np.zeros((0, rows, columns))
        for subject in self.data:
            for syllable in self.data[subject]:
                concatenated = np.concatenate((concatenated, self.data[subject][syllable]), axis=0)
        return concatenated

    def remove_padding(self, concatenated, rows=None, columns=None):
        if rows is None:
            rows = list(list(self.data.values())[0].values())[0][0].shape[0]
        if columns is None:
            columns = list(list(self.data.values())[0].values())[0][0].shape[1]
        padding_array = np.zeros([rows, columns])
        index_list = []
        i = -1
        for slice in concatenated:
            i += 1
            if np.array_equal(slice, padding_array):
                index_list.append(i)
        concatenated = np.delete(concatenated, index_list, axis=0)
        return concatenated

    def syllable_indexes(self):
        syllable_indexes = []
        i = 0
        counts = self.get_syllable_counts()
        for subject in counts:
            for syllable in counts[subject]:
                syllable_indexes.extend([i] * counts[subject][syllable])
                i += 1
        return syllable_indexes

    def data_to_tensor(self, rows=None, columns=None): #TODO: Just double check this works properly
        if rows is None:
            rows = list(list(self.data.values())[0].values())[0][0].shape[0]
        if columns is None:
            columns = list(list(self.data.values())[0].values())[0][0].shape[1]

        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        # data_dictionary_copy = copy.deepcopy(self.data)
        syllable_idxs = self.syllable_indexes()

        # padded = self.padding(rows, columns) #I guess I don't actually need the padding? Need to revist this later. 

        concatenated = self.concat_padded(rows, columns)
        unpadded = self.remove_padding(concatenated, rows, columns)
        tensor = torch.tensor(unpadded, dtype=torch.float32, device=device)
        return tensor, syllable_idxs

    def slicer(self, time_start, time_end):
        sliced_data = {}
        for subject in self.data:
            sliced_data[subject] = {}
            for syllable in self.data[subject]:
                sliced_data[subject][syllable] = self.data[subject][syllable][:, :, time_start:time_end]
        
        new_instance = copy.deepcopy(self)
        new_instance.data = sliced_data
        return new_instance
    
    def get_trial(self, trial_name:str):
        new_instance = copy.deepcopy(self)
        new_instance.data = self.data[trial_name]
        return new_instance
