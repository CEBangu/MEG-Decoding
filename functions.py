import os
import mne
import numpy as np
import pandas as pd
from typing import Tuple
from numpy.typing import NDArray


def data_load(dir, subjects, picks, avoid_overt=True) -> dict:
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

                    if (avoid_overt == True) and (sum(c.isdigit() for c in epo_name) >= 3): #avoid the ones with 3 digits in them because those are the out-loud trials I believe - 
                                                                                            #will have to double check this but good to know if it works anyways
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


# %% Cell 1 Testing Cell
# dir = '/Volumes/@neurospeech/PROJECTS/BCI/BCOM/DATA_ANALYZED/EVOKED/DATA/WITHOUT_BADS/COVERT'
# subjects = ['BCOM_18_2']
# picks=['MEG 130', 'MEG 139','MEG 133','MEG 117','MEG 140','MEG 127','MEG 128','MEG 109','MEG 135','MEG 132','MEG 137',
#  'MEG 131','MEG 129','MEG 118','MEG 134','MEG 136','MEG 141','MEG 116','MEG 114','MEG 115']



# #Let's put them all in a dictionary for easy access
# data_dict = data_load(dir, subjects, picks, avoid_overt=True)

# aepca, labels = get_epo_pca(data_dict)

# print(aepca.shape)