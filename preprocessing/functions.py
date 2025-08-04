import os
import glob
import numpy as np
from pymatreader import read_mat
from sklearn.neighbors import LocalOutlierFactor

# This code was written by PhD student Soufiane Jhilal, who co-supervised the project

class Epoching:
    """ Functions used in epoching script"""

    def __init__(self, mat_path, events):
        """
        Parameters:
        ----------
        - mat_path: string
            path of the folder where to find the original raw file before it was cleaned
            this path also contains the matlab files containing sentences event times

        - cleaned_raw: raw object
            the cleaned raw data (after preprocessing) to extract events from
        """
        self.mat_path = mat_path
        self.events = events

    def get_overt_trials(self):
        '''

        Function to get the interval of the overt trials (whole trial including the read in head)

        Return:
        ----------
        - overt_intervals: list of 3 (x,y) tuples
            each tuple marks the start time in samples (x) and end time in samples (y) of the overt trials

        '''

        sent_times = read_mat(os.path.join(self.mat_path, 'Syll1_sent1.mat'))['Trigger_Sent1_sel'].tolist()
        overt_times = read_mat(os.path.join(self.mat_path, 'Syll1_sent1_Overt.mat'))['Trigger_Sent1_Overt'].tolist()

        overt_idx = [sent_times.index(time) for time in overt_times]

        start_samples = [event[0] for event in self.events if int(event[2]) == int(4)]

        overt_intervals = []

        for idx in overt_idx:
            x = start_samples[idx]
            end_idx = idx + 1
            if end_idx >= len(sent_times):
                y = self.events[-1][0] + 1
            else:
                y = start_samples[end_idx]
            overt_intervals.append((x,y))

        return overt_intervals

    def is_overt(self, event):
        '''

        Function that checks if the event occurred during and overt trial (whole trial including the read in head)

        Parameters:
        ----------
        - event: mne event, array of shape (3,)

        Return:
        ----------
        - True or False
            True if event is overt, False otherwise

        '''

        overt_intervals = self.get_overt_trials()

        for x,y in overt_intervals:
            if x < event[0] < y:
                return True

        return False

    def get_bad_syll(self, raws_folder, sub, read_syll_label, read_triggers, syllables, i):
        '''

        Function that takes a syllable and returns the index (order of occurrence in the block) of the syllable in the bad attempts to be excluded
            bad attempts refer to instances where the syllable was not pronouced the way it should due to issues in the design of the stimilus
            (e.g. removing 'le' because it is not pronouced as 'lé')

        Parameters:
        ----------
        - raws_folder: string
            path of the folder where to find all the original uncleaned raw files of all the subjects

        - sub: string
            name of subject

        - read_syll_label: int
            integer label (trigger) of the targeted syllable in the read in head version (not produce in head)

        - read_triggers: list of integers
            list of integer labels (triggers) of all the syllables in the read in head version (not produce in head)

        - syllables: list of strings
            list of all the syllables

        - i: int
            block (integer 2, 3 or 4)

        Return:
        ----------
        - bad_idx: list of integers
            list containing the index of all bad syllables

        '''

        bad_idx = []

        syll = syllables[read_triggers.index(read_syll_label)]

        target_syllables = ['se','sa','si','le']

        if syll not in target_syllables:
            return []

        else:
            num = sub[-2:]
            if num[0] == '0':
                num = sub[-1]

            phrases = read_mat(glob.glob(os.path.join(raws_folder, sub, 'subject' + num + '_block' + str(i - 1) + '_*'))[0])['sentences']['phrases']

            sent_syllables = read_mat(glob.glob(os.path.join(raws_folder, sub, 'subject' + num + '_block' + str(i - 1) + '_*'))[0])['sentences']['syllables']

            labelsNum = read_mat(glob.glob(os.path.join(raws_folder, sub, 'subject' + num + '_block' + str(i - 1) + '_*'))[0])['sentences']['labelsNum']

            syll_count = -1
            for x in range(1,len(phrases)):
                for j in range(int(labelsNum[x].shape[0] / 2)):
                    if labelsNum[x][j] == read_syll_label:
                        syll_count += 1

                        if syll == 'le':
                            if sent_syllables[x].lower().split('-')[j] == syll:
                                bad_idx.append(syll_count)
                                # print(' ')
                                # print(syll)
                                # print(phrases[x])
                                # print(sent_syllables[x].split('-')[j-1:j+1])

                        elif syll == 'si':
                            if sent_syllables[x].lower().split('-')[j-1:j+1] == ["l'a", 'sie']:
                                bad_idx.append(syll_count)
                            elif j > 0 and sent_syllables[x].lower().split('-')[j-1] == 'sai':
                                bad_idx.append(syll_count)

                        elif syll == 'sa':
                            if sent_syllables[x].split('-')[j-1:j+1] == ['I', 'sa']:
                                bad_idx.append(syll_count)
                            elif j > 0 and sent_syllables[x].lower().split('-')[j-1] == 'ra':
                                bad_idx.append(syll_count)

                        elif syll == 'se':
                            if sent_syllables[x].lower().split('-')[j] == 'ce':
                                bad_idx.append(syll_count)
                            elif j < (labelsNum[x].shape[0]/2)-1 and sent_syllables[x].lower().split('-')[j+1][0] == 'm':
                                bad_idx.append(syll_count)
                            elif sent_syllables[x].lower().split('-')[j][:2] == 'sé':
                                bad_idx.append(syll_count)

        return bad_idx


class Preprocessing:
    """ Functions used in preprocessing script"""

    def __init__(self, raw):
        '''

        Function that detects flat channels

        Parameters:
        ----------
        - raw: raw object
            raw EEG/MEG file

        '''

        self.raw = raw
        self.Signal = self.raw._data[:247]
        self.SRate = self.raw.info['sfreq']
        self.ch_names = self.raw.info['ch_names']

    def get_flatlines(self, MaxFlatlineDuration=5, MaxAllowedJitter=20):
        '''

        Function that detects flat channels

        Parameters:
        ----------
        - MaxFlatlineDuration: int
            maximum tolerated flatline duration in seconds
            If a channel has a longer flatline than this, it will be considered abnormal. Default: 5

        - MaxAllowedJitter: int
            maximum tolerated jitter during flatlines as a multiple of epsilon. Default: 20

        Return:
        ----------
        - flat_chans: list of strings
            list containing the names of all flat channels

        '''

        flat_channels = np.array([False for i in range(len(self.Signal))])

        for c in range(len(self.Signal)):

            zero_intervals = np.reshape(np.where(
                np.diff(
                    [False] + list(abs(np.diff(self.Signal[c, :])) < (MaxAllowedJitter * np.finfo(float).eps)) + [False])),
                (2, -1)).T

            if (len(zero_intervals) > 0):
                if (np.max(zero_intervals[:, 1] - zero_intervals[:, 0]) > MaxFlatlineDuration * self.SRate):
                    flat_channels[c] = True
        flat_channels_idx = np.where(flat_channels)
        flat_chans = [self.ch_names[x] for x in flat_channels_idx[0]]

        return flat_chans

    def get_outliers(self):
        '''

        Function that detects bad channels using LOF

        Return:
        ----------
        - LOF_bads: list of strings
            list containing the names of all bad channels

        '''


        clf = LocalOutlierFactor()
        LOF = clf.fit_predict(self.Signal)

        LOF_bads = [self.ch_names[idx] for idx in range(len(LOF)) if LOF[idx] == -1]

        return LOF_bads


#
# for trigger in produce_triggers:
#     for sub in subjects:
#         for i in range(2, 5):
#             # start = time.time()
#             f_path = os.path.join(cleaned_path, sub, str(i), "cleaned_raw.fif")
#             cleaned_raw = mne.io.read_raw_fif(f_path, preload=True)
#
#             path = os.path.join(raws_folder, sub, "MEG", sub, "BCom")
#
#             for file in os.listdir(path):
#                 if os.path.isdir(os.path.join(path, file)):
#                     if sub == 'BCOM_08':
#                         mat_path = os.path.join(path, file, str(i + 1))
#                     else:
#                         mat_path = os.path.join(path, file, str(i))
#                     break
#
#             events = mne.find_events(cleaned_raw, shortest_event=1)
#
#             produce_trigger = trigger
#             read_trigger = trigger - 100
#
#             all_read_events = [event for event in events if int(event[2]) == int(read_trigger)]
#             all_produce_events = [event for event in events if int(event[2]) == int(produce_trigger)]
#
#             # covert_events = [event for event in events if int(event[2]) == int(produce_trigger) and not is_overt(event)]
#             # overt_events = [event for event in events if int(event[2]) == int(produce_trigger) and is_overt(event)]
#             # read_events = [event for event in events if int(event[2]) == int(read_trigger)]
#
#             bad_idx = get_bad_syll(raws_folder, sub, read_trigger, read_triggers, i)
#
#             cleaned_read_events = all_read_events.copy()
#             cleaned_produce_events = all_produce_events.copy()
#             for idx in bad_idx[::-1]:
#                 cleaned_read_events.pop(idx)
#                 cleaned_produce_events.pop(idx)
#
#             events_list = []
#
#             all_read_events_covert = np.array([event for event in all_read_events if not is_overt_v2(event)])
#             events_list.append(all_read_events_covert)
#             all_read_events_overt = np.array([event for event in all_read_events if is_overt_v2(event)])
#             events_list.append(all_read_events_overt)
#
#             all_produce_events_covert = np.array([event for event in all_produce_events if not is_overt_v2(event)])
#             events_list.append(all_produce_events_covert)
#             all_produce_events_overt = np.array([event for event in all_produce_events if is_overt_v2(event)])
#             events_list.append(all_produce_events_overt)
#
#             cleaned_read_events_covert = np.array([event for event in cleaned_read_events if not is_overt_v2(event)])
#             events_list.append(cleaned_read_events_covert)
#             cleaned_read_events_overt = np.array([event for event in cleaned_read_events if is_overt_v2(event)])
#             events_list.append(cleaned_read_events_overt)
#
#             cleaned_produce_events_covert = [event for event in cleaned_produce_events if not is_overt_v2(event)]
#             events_list.append(cleaned_produce_events_covert)
#             cleaned_produce_events_overt = [event for event in cleaned_produce_events if is_overt_v2(event)]
#             events_list.append(cleaned_produce_events_overt)
#
#             # num_events_overt_clean.append('There is '+str(len(cleaned_read_events_overt))+' read and '+str(len(cleaned_produce_events_overt))+' produce in block '+str(i)+' for '+sub)
#             # num_events_overt_notclean.append('There is '+str(len(all_read_events_overt))+' read and '+str(len(all_produce_events_overt))+' produce in block '+str(i)+' for '+sub)
#
#             picks = mne.pick_types(cleaned_raw.info, meg=True, eeg=False, stim=False, eog=False, ecg=False, misc=False)
#
#             for idx in range(len(events_list)):
#                 evs = events_list[idx]
#                 if idx < 4:
#                     cleaning = 'WITH_BADS'
#                 elif idx > 3:
#                     cleaning = 'WITHOUT_BADS'
#                 if idx % 2 == 0:
#                     condition = 'COVERT'
#                 elif idx % 2 == 1:
#                     condition = 'OVERT'
#
#                 if len(evs) > 0:
#                     epochs_main = mne.Epochs(cleaned_raw, events=evs, reject=reject_dict, picks=picks, baseline=None,
#                                          tmin=epoch_tmin, tmax=epoch_tmax, preload=True)
#
#                     if len(epochs_main) != 0:
#                         ar = AutoReject(verbose=True, picks=picks, n_jobs=3)
#                         # ransac = Ransac(verbose=True, picks=picks, n_jobs=3)
#                         try:
#                             epochs_clean = ar.fit_transform(epochs_main)
#                             # epochs_clean = ransac.fit_transform(epochs_main)
#                         except:
#                             epochs_clean = epochs_main
#
#                         trigger_index = produce_triggers.index(produce_trigger)
#                         if len(epochs_clean) != 0:
#                             syll_label = epochs_main.events[0][2]
#                             epochs_clean.save(os.path.join(evo_output, cleaning, condition, sub+'_'+str(i)+'_'+syllables[trigger_index]+'_'+str(syll_label)+'-epo.fif'))
