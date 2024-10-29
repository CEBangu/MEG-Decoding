# %% Cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import os
# a note on the file names:
# basically, the file names are subject, number between 2 and 4 (inclusive; I suppose trial), syllable, syllable label, and epo_a.fif

picks=['MEG 130',
 'MEG 139',
 'MEG 133',
 'MEG 117',
 'MEG 140',
 'MEG 127',
 'MEG 128',
 'MEG 109',
 'MEG 135',
 'MEG 132',
 'MEG 137',
 'MEG 131',
 'MEG 129',
 'MEG 118',
 'MEG 134',
 'MEG 136',
 'MEG 141',
 'MEG 116',
 'MEG 114',
 'MEG 115']

epo_a = mne.read_epochs('BCOM_18_2_a_12-epo.fif', preload=True).pick(picks=picks).get_data() #so in this case, this is subect 1, trial 2, syllable a whose label is 12
epo_e = mne.read_epochs('BCOM_18_2_e_14-epo.fif', preload=True).pick(picks=picks).get_data() #so in this case, this is subect 1, trial 2, syllable e whose label is 14
epo_i = mne.read_epochs('BCOM_18_2_i_16-epo.fif', preload=True).pick(picks=picks).get_data() #so in this case, this is subect 1, trial 2, syllable i whose label is 16
epo_la = mne.read_epochs('BCOM_18_2_la_22-epo.fif', preload=True).pick(picks=picks).get_data() #so in this case, this is subect 1, trial 2, syllable la whose label is 22
epo_le = mne.read_epochs('BCOM_18_2_le_24-epo.fif', preload=True).pick(picks=picks).get_data() #so in this case, this is subect 1, trial 2, syllable le whose label is 24
epo_li = mne.read_epochs('BCOM_18_2_li_26-epo.fif', preload=True).pick(picks=picks).get_data() #so in this case, this is subect 1, trial 2, syllable li whose label is 26
epo_ma = mne.read_epochs('BCOM_18_2_ma_32-epo.fif', preload=True).pick(picks=picks).get_data() #so in this case, this is subect 1, trial 2, syllable ma whose label is 32
epo_me = mne.read_epochs('BCOM_18_2_me_34-epo.fif', preload=True).pick(picks=picks).get_data() #so in this case, this is subect 1, trial 2, syllable me whose label is 34
epo_mi = mne.read_epochs('BCOM_18_2_mi_36-epo.fif', preload=True).pick(picks=picks).get_data() #so in this case, this is subect 1, trial 2, syllable mi whose label is 36
epo_ra = mne.read_epochs('BCOM_18_2_ra_42-epo.fif', preload=True).pick(picks=picks).get_data()
epo_re = mne.read_epochs('BCOM_18_2_re_44-epo.fif', preload=True).pick(picks=picks).get_data()
epo_ri = mne.read_epochs('BCOM_18_2_ri_46-epo.fif', preload=True).pick(picks=picks).get_data()
epo_sa = mne.read_epochs('BCOM_18_2_sa_52-epo.fif', preload=True).pick(picks=picks).get_data()
epo_se = mne.read_epochs('BCOM_18_2_se_54-epo.fif', preload=True).pick(picks=picks).get_data()
epo_si = mne.read_epochs('BCOM_18_2_si_56-epo.fif', preload=True).pick(picks=picks).get_data()
epo_ta = mne.read_epochs('BCOM_18_2_ta_62-epo.fif', preload=True).pick(picks=picks).get_data()
epo_te = mne.read_epochs('BCOM_18_2_te_64-epo.fif', preload=True).pick(picks=picks).get_data()
epo_ti = mne.read_epochs('BCOM_18_2_ti_66-epo.fif', preload=True).pick(picks=picks).get_data() #so in this case, this is subect 1, trial 2, syllable ti whose label is 12


#Let's put them all in a dictionary for easy access

data_dict = {'a': epo_a, 'e': epo_e, 'i': epo_i,
    'la': epo_la, 'le': epo_le, 'li': epo_li,
    'ma': epo_ma, 'me': epo_me, 'mi': epo_mi,
    'ra': epo_ra, 're': epo_re, 'ri': epo_ri,
    'sa': epo_sa, 'se': epo_se, 'si': epo_si,
    'ta': epo_ta, 'te': epo_te, 'ti': epo_ti}



# %% Cell 2
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

def get_epo_pca(data_dict):
    all_epochs = []
    labels = []
    i = 0
    for key in data_dict:
        for epoch, data in enumerate(data_dict[key]):
            for t in data:
                all_epochs.append(t)
                labels.append(i)
        i += 1

    return np.array(all_epochs), np.array(labels)


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
