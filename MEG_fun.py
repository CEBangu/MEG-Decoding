#README:
# This file is mostly just to play around with the data, to get a feel for it. The real decoding work will be done in a different file. 
# Also, this probably should be a jupyter notebook. However, it was originally done in the Zed editor with REPL since Zed does not have Jupyter support at this time (Oct 30, 2024),
# and I wanted to practice using Zed since it seems like a promising project. 

# %% Cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import os
from functions import data_load, get_epo_pca, sensor_correlations



# a note on the file names:
# basically, the file names are subject, number between 2 and 4 (inclusive; I suppose trial), syllable, syllable label, and epo_a.fif
# %% Cell 2
dir = '/Volumes/@neurospeech/PROJECTS/BCI/BCOM/DATA_ANALYZED/EVOKED/DATA/WITHOUT_BADS/COVERT'
dir2 = '/Users/ciprianbangu/Cogmaster/M2 Internship/BCI code'
subjects = ['BCOM_18_2']
picks=['MEG 130', 'MEG 139','MEG 133','MEG 117','MEG 140','MEG 127','MEG 128','MEG 109','MEG 135','MEG 132','MEG 137',
 'MEG 131','MEG 129','MEG 118','MEG 134','MEG 136','MEG 141','MEG 116','MEG 114','MEG 115']



#Let's put them all in a dictionary for easy access
data_dict = data_load(dir2, subjects, picks, avoid_overt=True)

epo_a = mne.read_epochs('BCOM_18_2_a_12-epo.fif', preload=True).pick(picks=picks).get_data() #so in this case, this is subect 1, trial 2, syllable a whose label is 12
epo_ti = mne.read_epochs('BCOM_18_2_ti_66-epo.fif', preload=True).pick(picks=picks).get_data() #so in this case, this is subect 1, trial 2, syllable ti whose label is 12

assert epo_a.shape == data_dict['BCOM_18_2']['a_12'].shape # just to make sure that the data is the same as the data loaded from the function


epo_a.shape # (17_epochs, 20_channels, 241_timespoints)
# %% Cell 3
# first epoch, in the first channel, over time.
first_epo_a = epo_a[0, 0, :]
first_epo_a.shape
plt.plot(first_epo_a)

# %% Cell 4
# all channels in the first epoch, over time
first_epo_all_a = epo_a[0, :, :]
first_epo_all_a.shape
plt.matshow(first_epo_all_a, aspect='auto')
plt.colorbar()

num_epochs = epo_a.shape[0]
fig, axes = plt.subplots(num_epochs, 1, figsize=(10, 2 * num_epochs))

for i in range(num_epochs):
    ax = axes[i] if num_epochs > 1 else axes
    ax.matshow(epo_a[i, :, :], aspect='auto')
    ax.set_title(f'Epoch {i+1}')

plt.tight_layout()
plt.show()

# %% Cell 5
# lets check out what the data looks like for the ti syllable, since it is rather different than a. At least it sounds different in my head...
epo_ti.shape # (6_epochs, 20_channels, 241_timespoints) NB! this is less than half of the 'a'

# %% Cell 6
# lets check out what the first epoch looks like
first_epo_all_ti = epo_ti[0, :, :]
plt.matshow(first_epo_all_ti, aspect='auto') # it looks like the dip is in a different location basically.
plt.colorbar()

# %% Cell 7
# Generate a figure of the matshow() for each of the epochs in the data_ti object
num_epochs = epo_ti.shape[0]
fig, axes = plt.subplots(num_epochs, 1, figsize=(10, 2 * num_epochs))

for i in range(num_epochs):
    ax = axes[i] if num_epochs > 1 else axes
    ax.matshow(epo_ti[i, :, :], aspect='auto')
    ax.set_title(f'Epoch {i+1}')

plt.tight_layout()
plt.show()
# hmm, ok now im not sure there is really a patter. There seems to be a pattern within epochs - i.e., the channels seem to sync in the dip.
# But there does not seem to be a visual consistency accross epochs.
###########################################################################################################################################################################################
# %% Cell 8
# PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

aepca, labels = get_epo_pca(data_dict)

# Standardize data
aepca = StandardScaler().fit_transform(aepca)

# Apply PCA
pca = PCA(n_components=3)
projected_data = pca.fit_transform(aepca)

# Plot the projected data
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot each class separately to add labels to the legend
unique_labels = np.unique(labels)
for label in unique_labels:
    indices = labels == label
    ax.scatter(projected_data[indices, 0], projected_data[indices, 1], projected_data[indices, 2],
               label=f'Class {label}', marker='o')

# Set labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('Projection of Data onto Principal Components')

# Show legend
ax.legend(title="Classes")
plt.tight_layout()
plt.show()

# Well, at least for this subject, plotting the first 3 PCs didn't  yield any clear separation between the classes.
# Could try doing it for all of the subjects? Or maybe some of the classes are seperable, but not others.
# I guess that just takes experimentation to suss out, but it could at least be guided by that phonetic stuff that was
# mentioned in one of the papers.


# %% Cell 10
#Let's see if there are any interesting correlational patterns in the data, between sensors for example.

a = data_dict['BCOM_18_2']['a_12'] # (17_epochs, 20_channels, 241_timespoints)

a_trial = a[3] # (20_channels, 241_timespoints), the first epoch of a

# plt.matshow(a_1, aspect='auto') #sensors are the rows, time is the columns

# plt.matshow(a_1[0:2], aspect='auto') # the first 2 sensors over time. 

# Compute the pairwise correlation between each row of the third epoch of a
correlation_matrix = np.corrcoef(a_trial, rowvar=True)

# Plot the correlation matrix
plt.matshow(correlation_matrix, aspect='auto')

plt.xlabel('Sensor Index')
plt.ylabel('Sensor Index')
plt.title('Pairwise Correlation between Sensors')
plt.colorbar()
plt.show()

# actually it might be nice to know which sensors are the most highly correlated accross in each epoch.

# Find the maximum correlation value less than 1
max_corr_value = np.max(np.abs(correlation_matrix)[np.abs(correlation_matrix) < 0.99]) # avoid the diagonal

# Find the indices of the maximum correlation value
max_corr_indices = np.where(np.abs(correlation_matrix) == max_corr_value)

print(f"Maximum correlation value less than 1: {max_corr_value}")
print(f"Indices of maximum correlation: {max_corr_indices}")

# %%Cell 11

sensor_correlations(a)

# i = 1
# correlations_a = pd.DataFrame({'Epoch': [], 'Max Correlation Value': [], 'Max Correlation Indices': []})
# for trial in a:
#     correlation_matrix = np.corrcoef(trial, rowvar=True)
#     max_corr_value = np.max(np.abs(correlation_matrix)[np.abs(correlation_matrix) < 0.99])
#     max_corr_indices = np.where(np.abs(correlation_matrix) == max_corr_value)
    
#     # Ensure indices are symmetrical
#     max_corr_indices = list(zip(*max_corr_indices))
#     max_corr_indices = list(set(tuple(sorted(pair)) for pair in max_corr_indices))
    
#     correlations_a = pd.concat([correlations_a, pd.DataFrame({'Epoch': i, 'Max Correlation Value': max_corr_value, 'Max Correlation Indices': max_corr_indices})], ignore_index=True)

#     i += 1

# correlations_a.sort_values(by='Max Correlation Value', ascending=False)
# %% Cell 12
# Compute the pairwise covariance between each row of a_1
covariance_matrix = np.cov(a_trial, rowvar=True)

# Plot the covariance matrix
plt.matshow(covariance_matrix, aspect='auto')

plt.xlabel('Sensor Index')
plt.ylabel('Sensor Index')
plt.title('Pairwise Covariance between Sensors')
plt.colorbar()
plt.show()

# %% Cell 13
## Try Vision Transformer - maybe pretrained one though
