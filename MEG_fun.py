# %% Cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import os
from functions import data_load, get_epo_pca

# a note on the file names:
# basically, the file names are subject, number between 2 and 4 (inclusive; I suppose trial), syllable, syllable label, and epo_a.fif
# %% Cell 2
dir = '/Volumes/@neurospeech/PROJECTS/BCI/BCOM/DATA_ANALYZED/EVOKED/DATA/WITHOUT_BADS/COVERT'
subjects = ['BCOM_18_2']
picks=['MEG 130', 'MEG 139','MEG 133','MEG 117','MEG 140','MEG 127','MEG 128','MEG 109','MEG 135','MEG 132','MEG 137',
 'MEG 131','MEG 129','MEG 118','MEG 134','MEG 136','MEG 141','MEG 116','MEG 114','MEG 115']



#Let's put them all in a dictionary for easy access
data_dict = data_load(dir, subjects, picks, avoid_overt=True)

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

# def get_epo_pca(data_dict):
#     all_epochs = []
#     labels = []
#     i = 0
#     for key in data_dict:
#         for epoch, data in enumerate(data_dict[key]):
#             for t in data:
#                 all_epochs.append(t)
#                 labels.append(i)
#         i += 1

#     return np.array(all_epochs), np.array(labels)


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




# %% Cell 11
## Try Vision Transformer - maybe pretrained one though
