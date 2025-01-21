import os
import mne
import numpy as np
import pandas as pd
from typing import Tuple
from numpy.typing import NDArray
import sklearn.preprocessing as sk
import torch
import copy
import warnings
import warnings


class BcomMEG():
    def __init__(self, subjects, dir=None, data=None, picks=None, avoid_reading=True):
        self.dir = dir
        self.subjects = subjects
        self.picks = picks
        self.avoid_reading = avoid_reading

        if data:
            self.data = data
        else:
            self.data = self.load_data()

    def load_data(self) -> dict:
        '''This method populates the class's data attribute if data was not already passed to it.
        It searches the directory for the subjects and fetches and loads the epochs'''
        if not self.dir:
            raise ValueError("Directory not specified. Please provide a valid directory path.")
        
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
                        data_dict[subject][epo_name] = mne.read_epochs(file, preload=True).pick(picks=self.picks)
        return data_dict

    def get_raw_data(self):
        '''This method unpacks the epoch data into its raw form in the data_dict'''
        for subject in self.data:
            for epoch in self.data[subject]:
                self.data[subject][epoch] = self.data[subject][epoch].get_data()

    #TODO: this currently does epoch by epoch transformation. I should add an option to do it syllable by syllable
    def get_spectrogram(self, frequencies:NDArray, cycle_divisor:int, baseline:tuple, mode='logratio', data_only=False):
        '''This method applies the spectrogram transformation to the data from
        its epo form.'''
        #Because I can't alter in place for some reason...
        transformed_data = {}
        if data_only:
            print("NB! Data_only is set to true. To get accurate plots, the extent will have to be set manually, and channels \n"
                  "will have to be specified as indexes, not names")
        for subject in self.data:
            if subject not in transformed_data:
                transformed_data[subject] = {}
            for syllable in self.data[subject]:
                if syllable not in transformed_data[subject]:
                    transformed_data[subject][syllable] = None
                powers = []
                for epoch in range(len(self.data[subject][syllable])):
                    power = mne.time_frequency.tfr_morlet(
                        self.data[subject][syllable][epoch],
                        freqs=frequencies,
                        n_cycles= frequencies/cycle_divisor,
                        use_fft=True,
                        return_itc=False,
                        decim=3,
                        n_jobs=10
                    ).apply_baseline(
                        baseline=baseline,
                        mode=mode
                    )
                    if data_only:
                        powers.append(power.data)
                        
                    else:
                        powers.append(power)
                
                transformed_data[subject][syllable] = powers


        return BcomMEG(subjects=self.subjects, data=transformed_data)
    
    #TODO: ez plotting function for the spectrograms

    # def plot_spectrogram(self, channels:list, syllables:list): 
        
    #     pass


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
        '''This method counts the number of epochs for each syllable in the object'''
        syllable_counts = {}
        for subject in self.data:
            syllable_counts[subject] = {}
            for syllable in self.data[subject]:
                syllable_counts[subject][syllable] = len(self.data[subject][syllable])
        return syllable_counts

    def get_max_length(self, syllable_count):
        '''This method gets the maximum number of epochs in the object'''
        max_length = {}
        for subjects in syllable_count:
            max_length[subjects] = max(syllable_count[subjects].values())
        return max_length

    def syllable_indexes(self):
        '''This method returns a list that corresponds to each syllables index if data_dict were an array'''
        syllable_indexes = []
        i = 0
        counts = self.get_syllable_counts()
        for subject in counts:
            for syllable in counts[subject]:
                syllable_indexes.extend([i] * counts[subject][syllable])
                i += 1
        return syllable_indexes

    def data_to_tensor(self, normalize=False): # it was not smart to return a torch tensor
        '''This method converts the object into one large tensor'''
        #Numpy Acceleration 
        total_epochs = 0
        for subject in self.data:
            total_epochs += sum(len(self.data[subject][syllable]) for syllable in self.data[subject])
        epoch_shape = ()
        for subject in self.data:
            sample_epoch = next(iter(self.data[subject].values()))[0]
            epoch_shape = sample_epoch.shape
            break
        
        tensor = np.empty((total_epochs, *epoch_shape))
        
        index = 0
        for subject in self.data:
            for syllable in self.data[subject]:
                for epoch in self.data[subject][syllable]:
                    tensor[index] = epoch
                    
                    #TODO: normalization
                    # if normalize:
                    #     tensor[index][epoch] = (tensor[index][epoch] - tensor[index][epoch].mean(axis=(0, 1), keepdims=True)) / tensor[index][epoch].std(axis=(0, 1), keepdims=True)

                    index += 1
        
        return tensor

    def slicer(self, time_start, time_end):
        '''This method returns a new instance of the object that has every epoch sliced time-wise per the values in the arguments'''
        sliced_data = {}
        for subject in self.data:
            sliced_data[subject] = {}
            for syllable in self.data[subject]:
                sliced_data[subject][syllable] = self.data[subject][syllable][:, :, time_start:time_end]
        
        return BcomMEG(subjects=self.subjects, data=sliced_data)
    
    def upscale(self, order_of_magnitude:int):
        '''This method modifies the data.data attribute in the object by multiplying each value by the specified order of magnitude'''
        for subject in self.data:
            for syllable in self.data[subject]:
                self.data[subject][syllable] *= 10 ** order_of_magnitude

    def get_trial(self, trial_names:list):
        '''This method returns a new instance of the object subsetted by the specified trial_names'''
        data = {trial: self.data[trial] for trial in trial_names}
        subjects = list(data.keys())
        return BcomMEG(subjects=subjects, data=data)
