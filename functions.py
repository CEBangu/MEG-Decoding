import os
import mne
import numpy as np
import pandas as pd
from typing import Tuple
from numpy.typing import NDArray
import torch
import copy


# class BcomMEG():
#     def __init__(self, dir, subjects, picks, avoid_reading=True):
#         self.dir = dir
#         self.subjects = subjects
#         self.picks = picks
#         self.avoid_reading = avoid_reading
        
#         #hardcode
#         self.possible_syllables = [

#         ]
        
#         #get all the data
#         self.data = load_data(self.dir, self.subjects, self.picks, self.avoid_reading)

#         self.label_to_in_map = {label: index for index, label in enumerate(possible_syllables)}
#         self.int_to_label_map = {index: label for index, label in enumerate(possible_syllables)}


        






#TODO: make this into a class

def load_data(dir, subjects, picks=None, avoid_reading=True) -> dict:
    '''This function takes in a directory, the desired subjects, the desired channels, and a boolean of whether or not to avoid overt trials
    i.e., those coded with 3 digits.
    It returns a dictionary with the data of the desired subjects and trials, indexed first by subject, and then by syllable.'''

    #TODO: add a way to subset by trial - at the moment this needs to be specified in the subjects list

    #initialize the dictionary
    data_dict = {}

    #loop through the subjects
    for subject in subjects:
        if subject not in data_dict:
            data_dict[subject] = {}
    
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.startswith(subject):
                    epo_name = file[10:-8]

                    if (avoid_reading == True) and (sum(c.isdigit() for c in epo_name) < 3): #segments where the subject is actually just reading rather than imagining have 2 digits
                        continue
            
                    file = os.path.join(dir, file)

                    data_dict[subject][epo_name] = mne.read_epochs(file, preload=True).pick(picks=picks).get_data()


    return data_dict


def get_epo_pca(data_dict) -> Tuple[NDArray, NDArray]:
    '''This function organizes the epochs to be used in PCA Analysis.
    It takes in the data dictionary, and retunrs all of the epochs and their corresponding labels.
    NB! This currently assumes there is only 1 subject in the dictionary'''

    #TODO: This assumes only 1 subject at the moment! This is bad because we probably want to see the points
    # over all subjects (or at least 2)
    # maybe also add a way to subset by syllable?

    all_epochs = [] # init list to store all epochs
    labels = [] # init list to store all labels
    i = 1 # init counter for labels

    for subject in data_dict: #by subject
        for syllable in (data_dict[subject]): #by syllable
            for epoch in data_dict[subject][syllable]: #by epoch
                all_epochs.append(epoch)
                labels.append(i)
            i += 1 #increment the label counter

    return np.array(all_epochs), np.array(labels)


def sensor_correlations(syllable_epochs: NDArray): 
    '''This function takes in the epochs for a single syllable and returns the correlation matrix between the sensors during
    each epoch'''
    
    i = 1
    correlations = pd.DataFrame({'Epoch': [], 'Max Correlation Value': [], 'Max Correlation Indices': []})
    for trial in syllable_epochs:
        correlation_matrix = np.corrcoef(trial, rowvar=True)
        max_corr_value = np.max(np.abs(correlation_matrix)[np.abs(correlation_matrix) < 0.99])
        max_corr_indices = np.where(np.abs(correlation_matrix) == max_corr_value)
    
        # Ensure indices are symmetrical
        max_corr_indices = list(zip(*max_corr_indices))
        max_corr_indices = list(set(tuple(sorted(pair)) for pair in max_corr_indices))
    
        correlations = pd.concat([correlations, pd.DataFrame({'Epoch': i, 'Max Correlation Value': max_corr_value, 'Max Correlation Indices': max_corr_indices})], ignore_index=True)

        i += 1

    return correlations

def get_syllable_counts(data_dict):
    '''This function takes in the data dictionary and returns a dictionary with the counts of each syllable for each subject'''
    syllable_counts = {}
    for subject in data_dict:
        syllable_counts[subject] = {}
        for syllable in data_dict[subject]:
            syllable_counts[subject][syllable] = len(data_dict[subject][syllable])

    return syllable_counts

def get_max_length(syllable_count):
    '''This function takes the output of get_syllable_counts and returns the maximum syllable length per subject'''
    max_length = {}
    for subjects in syllable_count:
        max_length[subjects] = max(syllable_count[subjects].values())

    return max_length


def padding(data_dictionary, rows=None, columns=None):
    '''This function pads the data in order to assemble it into a tensor in case there are different numbers of epochs per syllable'''
    if rows is None:
        rows = list(list(data_dictionary.values())[0].values())[0][0].shape[0]
    if columns is None:
        columns = list(list(data_dictionary.values())[0].values())[0][0].shape[1]

    for subject in data_dictionary:
        max_length = get_max_length(get_syllable_counts(data_dictionary))

        for syllable in data_dictionary[subject]:

            if len(data_dictionary[subject][syllable]) < max_length[subject]: # if the length is smaller than the max length

                padding = np.zeros([max_length[subject] - len(data_dictionary[subject][syllable]), rows, columns]) # create a padding array
                data_dictionary[subject][syllable] = np.concatenate((data_dictionary[subject][syllable], padding)) # concatenate the padding to the original array
    return data_dictionary

def concat_padded(padded_data_dict, rows=None, columns=None):
    '''This function takes in the padded data dictionary and returns a tensor'''
    if rows is None:
        rows = list(list(padded_data_dict.values())[0].values())[0][0].shape[0]
    if columns is None:
        columns = list(list(padded_data_dict.values())[0].values())[0][0].shape[1]

    concatenated = np.zeros((0, rows, columns)) # create an empty array to concatenate the data to
    for subject in padded_data_dict: # for each subject
        for syllable in padded_data_dict[subject]: # for each syllable
            concatenated = np.concatenate((concatenated, padded_data_dict[subject][syllable]), axis=0)
    
    return concatenated

def remove_padding(concatenated, rows=None, columns=None):
    '''This function takes in the concatenated padded data and removes the padding'''

    if rows is None:
        rows = list(list(concatenated.values())[0].values())[0][0].shape[0]
    if columns is None:
        columns = list(list(concatenated.values())[0].values())[0][0].shape[1]
    
    padding_array = np.zeros([rows, columns]) #create a padding array to compare to

    index_list = []
    i = -1 #if you start at 0 you will miss the first index 

    for slice in concatenated:
        i += 1
        if np.array_equal(slice, padding_array):
            index_list.append(i)

    concatenated = np.delete(concatenated, index_list, axis=0)

    return concatenated

def syllable_indexes(data_dictionary):
    '''When the data_dictionary gets turned into a tensor, which slices belong to which syllable get lost.
    This function solves this problem by returning an array with the indexes of the syllables'''

    syllable_indexes = []
    i = 0
    counts = get_syllable_counts(data_dictionary)
    for subject in counts:
        for syllable in counts[subject]:
            syllable_indexes.extend([i] * counts[subject][syllable])
            i += 1
    
    return syllable_indexes


def data_to_tensor(data_dictionary, rows=None, columns=None):
    '''This function combines all of the individual steps outlined above, and converts the data into a 3d tensor'''

    if rows is None:
        rows = list(list(data_dictionary.values())[0].values())[0][0].shape[0]
    if columns is None:
        columns = list(list(data_dictionary.values())[0].values())[0][0].shape[1] 

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Create a deep copy of the data_dictionary to avoid altering the original object
    data_dictionary_copy = copy.deepcopy(data_dictionary)

    syllable_idxs = syllable_indexes(data_dictionary_copy)
    padded = padding(data_dictionary_copy, rows, columns)
    concatenated = concat_padded(padded, rows, columns)
    unpadded = remove_padding(concatenated, rows, columns)
    tensor = torch.tensor(unpadded, dtype=torch.float32, device=device) #Is it smart to have this be a torch tensor already?

    return tensor, syllable_idxs

def slicer(data_dictionary, time_start, time_end):
    '''This function slices the data into the desired time interval'''
    sliced_data = {}

    for subject in data_dictionary:
        sliced_data[subject] = {}
        for syllable in data_dictionary[subject]:
            sliced_data[subject][syllable] = data_dictionary[subject][syllable][:, :, time_start:time_end]

    return sliced_data
# %% Cell 1 Testing Cell
# dir = '/Volumes/@neurospeech/PROJECTS/BCI/BCOM/DATA_ANALYZED/EVOKED/DATA/WITHOUT_BADS/COVERT'
# subjects = ['BCOM_18_2']
# picks=['MEG 130', 'MEG 139','MEG 133','MEG 117','MEG 140','MEG 127','MEG 128','MEG 109','MEG 135','MEG 132','MEG 137',
#  'MEG 131','MEG 129','MEG 118','MEG 134','MEG 136','MEG 141','MEG 116','MEG 114','MEG 115']



# #Let's put them all in a dictionary for easy access
# data_dict = data_load(dir, subjects, picks, avoid_overt=True)

# aepca, labels = get_epo_pca(data_dict)

# print(aepca.shape)