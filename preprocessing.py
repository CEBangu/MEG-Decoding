import mne
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser


def main():

    # get the target subject from SLURM

    parser = ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    args = parser.parse_args()

    # helper functions

    def find_recordings(path):
        """
        This function makes it easy to find the right recording subdirectories
        """
        subdirs = os.listdir(path)
        for subdir in subdirs:
            # the recording folders have messy names, but they all start with 04
            if "04" in subdir: 
                return subdir
            else:
                continue


    # code from Remy
    def get_flats(raw, flat_criteria, duration, id, start):
        """
        This function finds and flat channels and returns them as a list
        """
        events_tmp = mne.make_fixed_length_events(raw, duration=duration, id=id, start=start)
        epochs_tmp = mne.Epochs(raw, events=events_tmp, event_id=id, flat=flat_criteria, verbose=False)
        epochs_tmp.load_data()
        
        flat_channels = []
        
        if epochs_tmp.drop_log_stats()>0:
            flat_channels=list(set([ch for chs in epochs_tmp.drop_log for ch in chs]))
        del epochs_tmp

        return flat_channels

    def path_parser(path, location: str):
        """
        This function returns the relevant parts of the path
        to make saving things easier
        """
        allowed_locations = ["server", "local", "lab"]
        
        if location not in allowed_locations:
            raise ValueError("location must be one of 'server', 'local', or 'lab'")
        
        if location == "server":
            subject_index = 8
        elif location == "local":
            subject_index = 5
        elif location == "lab":
            subject_index = 6
            
        path = Path(path) 
        
        path_parts = path.parts
        
        subject = path_parts[subject_index] # on home mac this is 5! on lab pc it is 6! on server it is 8
        block = path_parts[-1]
        
        subject_block = os.path.join(subject, block)
        
        return subject_block

    def write_to_csv(file_path, strings):
        """
        Writes a list of strings to a CSV file, each string in its own column, and adds a timestamp at the end.

        Parameters:
        - file_path: str, path to the CSV file.
        - strings: list of str, strings to write to the file.
        """
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow(strings + [timestamp])
        
    ##############################################################################################################

    # Path Parsing

    old_pp_path = "/pasteur/zeus/projets/p02/BCOM/BCOM/DATA_ANALYZED/PREPROCESSED"   #"/Volumes/BCOM/BCOM/DATA_ANALYZED/PREPROCESSED" #"/home/ciprian/BCOM/BCOM/DATA_ANALYZED/PREPROCESSED"
    raw_path = "/pasteur/zeus/projets/p02/BCOM/BCOM/DATA_RAW" #"/Volumes/BCOM/BCOM/DATA_RAW" #"/home/ciprian/BCOM/BCOM/DATA_RAW" 
    processed_path = "/pasteur/zeus/projets/p02/BCOM/ciprian_project/data_analyzed/preprocessed" #"/Volumes/BCOM/ciprian_project/data_analyzed/preprocessed" #"/home/ciprian/BCOM/ciprian_project/data_analyzed/preprocessed" #
    os.makedirs(processed_path, exist_ok=True)


    subjects = [name for name in os.listdir(old_pp_path) if "BCOM" in name] # only the subjects that were deemed good to look at
    blocks = [2, 3, 4] # block names
    print(subjects)

    pdf_suffix = "c,rfDC"
    config_suffix = "config"
    header_suffix = "hs_file"


    # get all of the block recordings            
    raw_paths = [os.path.join(raw_path, f"{subject}/MEG/{subject}/BCom/") for subject in subjects]
    raw_paths = [os.path.join(raw_path, f"{find_recordings(raw_path)}") for raw_path in raw_paths]
    parsed_paths = [os.path.join(raw_path, str(block)) for raw_path in raw_paths for block in blocks]

    # just in case
    assert len(parsed_paths) == 63
    assert all(".DS_Store" not in path for path in parsed_paths)

    # get all of the empty room recordings
    empty_room_paths = [os.path.join(raw_path, f"{subject}/MEG/{subject}/emp_sup_v1") for subject in subjects]
    empty_room_paths = [os.path.join(raw_path, f"{find_recordings(raw_path)}/2") for raw_path in empty_room_paths]

    assert len(empty_room_paths) == 21

    # some of them are in different folders
    empty_room_folder_dict = {
        "BCOM_26" : "4",
        "BCOM_22" : "3",
        "BCOM_21": "1",
        "BCOM_18": "1",
        "BCOM_14": "3" 
    }

    for i, path in enumerate(empty_room_paths):
        for folder in empty_room_folder_dict:
            if folder in path:
                path = path.rsplit('/', 1)[0] + f"/{empty_room_folder_dict[folder]}"
                print(path)
                empty_room_paths[i] = path


    # Group together empty rooms and subject blocks so that we can apply the transformations properly
    raw_empty_paired = [
        (parsed_paths[i:i+3], empty_room_paths[i // 3]) for i in range(0, len(parsed_paths), 3)
    ]

    # example
    print(raw_empty_paired[0])
    print("\n")
    print(raw_empty_paired[0][0])
    print(f"length:{len(raw_empty_paired[0][0])}")
    print("\n")
    print(raw_empty_paired[0][1])


    # there is no "2" for subject 19, so let's replace it with the other one subject 18 
    raw_empty_paired[13] = (raw_empty_paired[13][0], raw_empty_paired[5][1])
    raw_empty_paired[13]

    for i, j in enumerate(raw_empty_paired):
        print(i, j)

    ############################################################

    # MNE params

    # filtering params, from MNE best practices
    lowpass_filter = 150.0 #to get freq up to 150
    highpass_filter = 0.5
    sampling_rate = 500

    # detecting flat channels
    flat_criteria= dict(mag=1e-13)
    duration=3
    id=1
    start=2

    # ICA setup
    ica_method = 'fastica'
    n_components = 0.97
    decim = 3
    random_state = 23
    reject = dict(mag=2e-11)

    # noth filtering freqs
    notch_freqs=(50, 100, 150)

    # path parsing location - set this based on where you are running the notebook
    location = "server"

    #############################################################

    #preprocessing loop

    target_subject = args.subject
    target_raw_empty_paired = []
    for i, pair in enumerate(raw_empty_paired):
        if target_subject in pair[0][0]:
            target_raw_empty_paired.append(pair)


    for i, pairs in enumerate(target_raw_empty_paired): #NB! make sure you change the index when you want to start from a different place
        
        print(len(pairs))
        subject_raw_paths = pairs[0]
        empty_room_path = pairs[1]    

        print(empty_room_path)
        
        empty_room_pdf_name=os.path.join(empty_room_path, pdf_suffix)
        empty_room_config_name=os.path.join(empty_room_path, config_suffix)
        
        empty_room_raw = mne.io.read_raw_bti(
            pdf_fname=empty_room_pdf_name,
            config_fname=empty_room_config_name,
            head_shape_fname=None,
            rename_channels=True,
            sort_by_ch_name=True,
            ecg_ch="ECG",
            eog_ch=("EOGv", "EOGh"),    
            preload=True,
        )
        
        noisy_channels_empty_room = mne.preprocessing.find_bad_channels_lof(
            empty_room_raw
            .copy()
            .pick("meg")
            .filter(1,100) # the idea here is to remove slow drift to keep channels that might be eroneously removed
            )
        
        flat_channels_empty_room = get_flats(
            raw=empty_room_raw,
            flat_criteria=flat_criteria,
            duration=duration,
            id=id,
            start=start
        )
        
        bad_channels_empty_room = noisy_channels_empty_room + flat_channels_empty_room
        
        print(bad_channels_empty_room)
        
        empty_room_raw.info['bads'] = bad_channels_empty_room

        empty_room_raw.notch_filter(freqs=notch_freqs) # check about that weird thing in 46
        
        filtered_empty_room_raw = empty_room_raw.copy().filter(l_freq=highpass_filter, h_freq=lowpass_filter)
        
        resampled_empty_room_raw = filtered_empty_room_raw.resample(sfreq=sampling_rate)
        
        for j, block in enumerate(subject_raw_paths):
            
            # path parsing for saving later
            subject_block = path_parser(block, location=location)
            data_path = os.path.join(processed_path, subject_block)
            ica_path = os.path.join(data_path, "ICA")
            
            for path in [data_path, ica_path]:
                os.makedirs(path, exist_ok=True)
                
            
            # get file names
            subject_pdf_fname=os.path.join(block, pdf_suffix)
            subject_config_fname=os.path.join(block, config_suffix)
            subject_head_shape_fname=os.path.join(block, header_suffix)
            
            subject_raw = mne.io.read_raw_bti(
                pdf_fname=subject_pdf_fname,
                config_fname=subject_config_fname,
                head_shape_fname=subject_head_shape_fname,
                rename_channels=True,
                sort_by_ch_name=True,
                ecg_ch="ECG",
                eog_ch=("EOGv", "EOGh"),    
                preload=True,
            )
            
            noisy_channels_subject = mne.preprocessing.find_bad_channels_lof(
                subject_raw
                .copy()
                .pick("meg")
                .filter(1,100) # the idea here is to remove slow drift to keep channels that might be eroneously removed
            )
            
            flat_channels_subject = get_flats(
                raw=subject_raw, 
                flat_criteria=flat_criteria,
                duration=duration,
                id=id,
                start=start
            )
            
            bad_channels_subject = noisy_channels_subject + flat_channels_subject
            
            print(bad_channels_subject)

            # get the union of the bads so that the .info['bads'] are the same for both 
            bads_union = list(set(bad_channels_subject).union(set(bad_channels_empty_room)))

            print(bads_union)

            # apply
            subject_raw.info['bads'] = bads_union


            subject_raw.notch_filter(freqs=notch_freqs)
            
            filtered_subject_raw = subject_raw.copy().filter(l_freq=highpass_filter, h_freq=lowpass_filter)
            
            subject_events = mne.find_events(filtered_subject_raw, shortest_event=1)
            
            resampled_subject_raw, resampled_subject_events = filtered_subject_raw.resample(
                sfreq=sampling_rate, 
                events=subject_events
            )
            
            ica_subject = ICA(n_components=n_components, # can use None for better explanation but that takes forever
            method=ica_method,
            random_state=random_state,
            max_iter=20000 # used to be 2000 - then 5000 - now 10000; gunna try 19 again at 20k
            )
            
            # highpass filter at 1 so that the drift doesnt take all the variance - see MNE docs
            subject_raw_ica = resampled_subject_raw.copy().filter(1,None)
            
            ica_subject.fit(
                subject_raw_ica,
                picks='meg',
                decim=decim,
                reject=reject
            ) 
                
            # save ICA solution, with the exclude property populated
            subject_ica_path = os.path.join(ica_path, "solution_ica.fif")
            ica_subject.save(subject_ica_path, overwrite=True)
            
            # IMPORTANT! we want a different solution applied each time, so copy in the filtered empty room
            resampled_empty_room_raw_copy = resampled_empty_room_raw.copy()

            # make sure the bads are the same
            resampled_empty_room_raw_copy.info['bads'] = bads_union        
            # save the objects
            # NB! note there is no interpolation here. Better to interpolate during the epoching stage
            # since that is analysis based

            resampled_subject_raw_path = os.path.join(data_path, "subject_cleaned_no_ICA_raw.fif")
            resampled_empty_room_raw_copy_path = os.path.join(data_path, "empty_room_cleaned_no_ICA_raw.fif")
            
            # save everything, including the resample events
            resampled_subject_raw.save(resampled_subject_raw_path, overwrite=True)
            resampled_empty_room_raw_copy.save(resampled_empty_room_raw_copy_path, overwrite=True)
            np.save(os.path.join(data_path, "resampled_events.npy"), resampled_subject_events)

            #just in case the 
            print(f"{subject_block} done")
            write_to_csv("/pasteur/zeus/projets/p02/BCOM/ciprian_project/MEG-Decoding/processing_records.csv", [str(subject_block)])
            
            del resampled_subject_raw
            del subject_raw_ica
            del resampled_empty_room_raw_copy

if __name__ == "__main__":
    main()
        