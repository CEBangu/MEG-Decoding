import mne
import os
import time
import numpy as np
import os.path as op
import pandas as pd
from pymatreader import read_mat
from argparse import ArgumentParser
from autoreject import AutoReject
from pathlib import Path

from preprocessing import Epoching


def baseline_cropper(raw, events, max_length=30):
    """
    This function takes in the uncropped raw and its events array and returns the baseline section, as well as the times used to
    segment the baseline 
    """
    t1 = raw.times[events[list(events[:,2]).index(514), 0]]
    t2 = raw.times[events[list(events[:,2]).index(4), 0]]

    if (t2 - t1) > max_length:
        for t_index in range(events[list(events[:,2]).index(514), 0], events[list(events[:, 2]).index(4),0], 900):
                        # if the rounded time difference between the current time and t1 is 29, then we have found the right time
            if round(raw.times[t_index] - t1) == 29:
                t2 = raw.times[t_index] #set t2 to that index
                break
    
    baseline = raw.copy().crop(tmin=t1, tmax=t2)

    return baseline, (t1, t2)

bad_localization_channel = "MEG 173"

# epoch settings
epoch_tmin=-0.4 # start
epoch_tmax=0.8 # end
reject_dict = dict(mag=5e-12) # autoreject

def main():

    parser = ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="Subject name should be BCOM_XX/N")
    parser.add_argument("--save_baseline", action="store_true", required=False, help="use this flag if you want to save the baseline")
    parser.add_argument("--root", type=str, required=True, help="Root of the directory the data is stored in, i.e., if launched from the server or locally")
    args = parser.parse_args()

    # make sure subject is of right format
    parsed_subject = [c for c in args.subject]
    if parsed_subject[-2] != "/":
        raise ValueError("'/' not found in the right position - check subject help for proper formatting")
    ##########################
    root = args.root 
    raw_path = op.join(root, "BCOM/DATA_RAW")
    preprocessed_path = op.join(root, "ciprian_project/data_analyzed/preprocessed")
    non_normalized_epoch_path = op.join(root, "ciprian_project/data_analyzed/non_normalized/epoch_test") #TODO: change this back when debugging is done!
    # normalized_epoch_path = op.join(root, "ciprian_project/data_analyzed/normalized/data")
    baseline_path = op.join(root, "ciprian_project/data_analyzed/baselines")

    # triggers
    triggers_pd = pd.read_csv(os.path.join(root, "BCOM/PROTOCOL", 'trigger_labels.csv'), sep=';')
    produce_triggers = triggers_pd['produce_head'].to_list() #the csv already has names
    read_triggers = triggers_pd['read_head'].to_list()
    syllables = triggers_pd['syllable'].to_list()

    # get paths
    subject_raw_path = op.join(preprocessed_path, args.subject, "subject_cleaned_ica_raw.fif")
    events_path = op.join(preprocessed_path, args.subject, "resampled_events.npy")

    # read in subject path
    subject_raw = mne.io.read_raw_fif(
        fname=subject_raw_path,
        preload=True,
    )

    sub = args.subject.split("/")[0]
    block = args.subject.split("/")[1] #NB! this comes out as a str, so it gets coerced into an int sometimes later

    # the coordinates on this channel are super weird, just exclude it from the analysis
    subject_raw = subject_raw.interpolate_bads(exclude=[bad_localization_channel])

    # get the resampled events
    subject_events = np.load(events_path)

    # if you want to save the baseline as well. Need to do it at least once
    
    if args.save_baseline:

        baseline_dir = op.join(baseline_path, sub)
        os.makedirs(baseline_dir, exist_ok=True)

        baseline, _ = baseline_cropper(subject_raw, subject_events)

        baseline.save(op.join(baseline_dir, sub + '_' + block + '_baseline_raw.fif'), overwrite=True)

        del baseline # free memory, we have what we need


    for trigger in produce_triggers:
    
        start = time.time()
        
        path = os.path.join(raw_path, sub, "MEG", sub, "BCom")

        for file in os.listdir(path):
            if os.path.isdir(os.path.join(path, file)):
                if sub == 'BCOM_08': #something special about this one
                    mat_path = os.path.join(path, file, str(int(block) + 1))
                else:
                    mat_path = os.path.join(path, file, str(int(block)))
                break
        
        produce_trigger = int(trigger)
        read_trigger = int(trigger) - 100
        all_read_events = [event for event in subject_events if int(event[2]) == int(read_trigger)] 
        all_produce_events = [event for event in subject_events if int(event[2]) == int(produce_trigger)] 

        Epo = Epoching(mat_path, subject_events)

        bad_idx = Epo.get_bad_syll(raw_path, sub, read_trigger, read_triggers, syllables, int(block))

        cleaned_read_events = all_read_events.copy()
        cleaned_produce_events = all_produce_events.copy()

        for idx in bad_idx[::-1]:
            cleaned_read_events.pop(idx)
            cleaned_produce_events.pop(idx)

        all_events = {
            "OVERT":[
                np.array([event for event in all_produce_events if Epo.is_overt(event)])
            ],
            "COVERT": [
                np.array([event for event in all_read_events if not Epo.is_overt(event)]), 
                np.array([event for event in all_read_events if Epo.is_overt(event)]),
                np.array([event for event in all_produce_events if not Epo.is_overt(event)]),
            ],
        }

        clean_events = {
            "OVERT": [
                np.array([event for event in cleaned_produce_events if Epo.is_overt(event)])
            ],
            "COVERT": [
                np.array([event for event in cleaned_read_events if not Epo.is_overt(event)]),
                np.array([event for event in cleaned_read_events if Epo.is_overt(event)]),
                np.array([event for event in cleaned_produce_events if not Epo.is_overt(event)]),
            ],
        }


        picks = mne.pick_types(subject_raw.info, meg=True, eeg=False, stim=False, eog=False, ecg=False, misc=False) 

        # ok so let's try 2 loops then
        for condition in clean_events:

            for events in clean_events[condition]:

                if len(events) > 0:
                    epochs_main = mne.Epochs(
                        subject_raw, 
                        events=events, 
                        reject=reject_dict, 
                        picks=picks, 
                        baseline=None,
                        tmin=epoch_tmin, 
                        tmax=epoch_tmax, 
                        preload=True,
                    ) 

                    # epochs_main.plot_drop_log()

                    if len(epochs_main) != 0:
                        ar = AutoReject(
                            verbose=True, 
                            picks=picks, 
                            n_jobs=3
                        )

                        try:
                            epochs_clean = ar.fit_transform(epochs_main)
                        except:
                            epochs_clean = epochs_main 

                        trigger_index = produce_triggers.index(produce_trigger)

                        if len(epochs_clean) != 0:
                            syll_label = epochs_main.events[0][2]
                            
                            cleaned_epoch_path = os.path.join(non_normalized_epoch_path, "WITHOUT_BADS", condition)
                            os.makedirs(cleaned_epoch_path, exist_ok=True)

                            epochs_clean.save(
                                os.path.join(cleaned_epoch_path, sub+'_'+block +'_'+syllables[trigger_index]+'_'+str(syll_label)+'-epo.fif'),
                                overwrite=True,
                            )

        for condition in all_events:
            for events in all_events[condition]:

                if len(events) > 0:
                    epochs_main = mne.Epochs(
                        subject_raw, 
                        events=events, 
                        reject=reject_dict, 
                        picks=picks, 
                        baseline=None,
                        tmin=epoch_tmin, 
                        tmax=epoch_tmax, 
                        preload=True,
                    ) 

                    # epochs_main.plot_drop_log()

                    if len(epochs_main) != 0:
                        ar = AutoReject(
                            verbose=True, 
                            picks=picks, 
                            n_jobs=3
                        )

                        try:
                            epochs_clean = ar.fit_transform(epochs_main)
                        except:
                            epochs_clean = epochs_main 

                        trigger_index = produce_triggers.index(produce_trigger)
                        if len(epochs_clean) != 0:
                            syll_label = epochs_main.events[0][2]
                            
                            cleaned_epoch_path = os.path.join(non_normalized_epoch_path, "WITH_BADS", condition)
                            os.makedirs(cleaned_epoch_path, exist_ok=True)

                            epochs_clean.save(
                                os.path.join(cleaned_epoch_path, sub+'_'+block +'_'+syllables[trigger_index]+'_'+str(syll_label)+'-epo.fif'),
                                overwrite=True,
                            )

        end = time.time()
        print(f"Iteration duration: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()