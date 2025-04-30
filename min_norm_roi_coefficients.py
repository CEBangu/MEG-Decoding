import os
import mne
import pywt
import numpy as np
from argparse import ArgumentParser
from datahandling import BcomMEG
from wavelets import process_channel, save_coefficient_results
from mne import set_config

# NB! 55 minutes to run completely for covert producing


def main():

    parser = ArgumentParser()

    parser.add_argument('--subject_list', nargs='+', type=str, required=True, help='The subject_block(s) you want the coefficients for')
    parser.add_argument('--avoid_reading', action='store_true', help="do you want to avoid the reading epochs?")
    parser.add_argument('--avoid_producing', action='store_true', help="do you want to avoid producing epochs?")
    parser.add_argument('--speech_type', type=str, required=True, help="Covert or Overt?")
    parser.add_argument('--data_dir', type=str, help="Directory where the data is stored")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the coefficients")
    parser.add_argument('--empty_room_dir', type=str, required=True, help="Directory where the baselines are stored")
    parser.add_argument('--mne_dir', type=str, help="where fsaverage is located")

    args = parser.parse_args()

    ############################
            # CWT setup #
    ############################

    sampling_rate = 300 # data already downsampled to 300 at this point
    log_samples = 100 # we want 100 coefficients
    cwt_wavelet_name = 'cmor' # reconstruction wavelet
    B = 1.0 # wavelet bandwith (higher means more frequencies at each scale, but less precision in peak timing)
    C = 1.0 # central frequency (higher means more oscialltions per time window, meaning higher frequency features per scale)
    cwt_wavelet = f'{cwt_wavelet_name}{B}-{C}'
    frequencies = np.logspace(np.log10(1), np.log10(sampling_rate/2), log_samples)
    sampling_period = 1/sampling_rate
    scales = pywt.central_frequency(wavelet=cwt_wavelet)/ (frequencies * sampling_period)
    dwt_wavelet_name='db4' # denoizing wavelet 
    level=5 # level of decomposition. NB in Dash et al. they use 7, but our signal is shorter, so 5 is max

    ###########################
            # Labels #
    ###########################

    # get fsaverage
    subjects_dir = args.mne_dir if args.mne_dir is not None else os.path.dirname(mne.datasets.fetch_fsaverage(verbose=True))

    set_config("SUBJECTS_DIR", subjects_dir, set_env=True)

    #get labels
    labels_hcp = mne.read_labels_from_annot(
        subject='fsaverage',
        parc='HCPMMP1',
        hemi='lh',
        subjects_dir=subjects_dir
    )

    labels_aparc = mne.read_labels_from_annot(
        subject='fsaverage',
        parc='aparc',
        hemi='lh',
        subjects_dir=subjects_dir
    )

    # from from Sheets et al. 2021 
    sma_labels = [label for label in labels_hcp if 'L_6ma_ROI-lh' in label.name][0] +\
            [label for label in labels_hcp if 'L_6mp_ROI-lh' in label.name][0] +\
            [label for label in labels_hcp if 'L_SCEF_ROI-lh' in label.name][0] +\
            [label for label in labels_hcp if 'L_SFL_ROI-lh' in label.name][0]


    # commonly attirbuted
    broca_labels = [label for label in labels_hcp if "L_44_ROI-lh" in label.name][0] +\
            [label for label in labels_hcp if "L_45_ROI-lh" in label.name][0]


    # in the name
    stg_labels = [label for label in labels_aparc if 'superiortemporal-lh' in label.name][0]

    # in the name
    mtg_labels = [label for label in labels_aparc if 'middletemporal-lh' in label.name][0]

    # from Eckert et al. 2021
    spt_labels = [label for label in labels_hcp if 'PSL' in label.name][0]


    label_dictionary = {
        "sma": sma_labels,
        "broca": broca_labels,
        "stg": stg_labels,
        "mtg": mtg_labels,
        "spt": spt_labels,
    }

    ##################################
    # Setting up fsaverage source space #
    ##################################

    fs_average_source_space = mne.setup_source_space(
        subject='fsaverage', # only once, no co-registration
        spacing='ico5',
        add_dist=False,
    )

    ##############################
    # Block-wise forward solution #
    ##############################

    speech_type = args.speech_type.upper()
    data_dir = args.data_dir
    data_dir = os.path.join(data_dir, speech_type)
    subjects = args.subject_list
    avoid_reading = args.avoid_reading
    avoid_producing = args.avoid_producing
    empty_room_dir = args.empty_room_dir

    save_dir = args.save_dir

    bad_localization_channel = "MEG 173"
    
    data = BcomMEG(dir=data_dir,
                subjects=subjects,
                avoid_reading=avoid_reading,
                avoid_producing=avoid_producing,
                )
    
    scaled = "scaled_fsaverage"

    for subject in data.data: # loop through the subjects (blocks, really)
        # we have the morphed transformations, so we will load them for the subject
        morphed_trans = f"/pasteur/appa/scratch/cbangu/trans/{subject}-trans.fif"
        morphed_source = f"/pasteur/appa/scratch/cbangu/MNE-fsaverage-data/{scaled}_{subject}/bem/{scaled}_{subject}-ico-5-src.fif"
        morphed_bem = f"/pasteur/appa/scratch/cbangu/MNE-fsaverage-data/{scaled}_{subject}/bem/{scaled}_{subject}-5120-5120-5120-bem-sol.fif"
        
        # forward solution by block
        first_epoch_name = list(data.data[subject].keys())[0]
        fwd_solution_epoch = data.data[subject][first_epoch_name]


        fwd_solution = mne.make_forward_solution(
            fwd_solution_epoch.info,
            trans=morphed_trans,
            src=morphed_source,
            bem=morphed_bem,
            meg=True,
            eeg=False,
        )
        
        # need to get the covariance matrix from the empty_room recording 
        subject_empty_room = os.path.join(empty_room_dir, subject[:7], subject[-1], subject + "baseline_raw.fif")

        empty_room_raw = mne.io.read_raw_fif(
            subject_empty_room,
            preload=True,
        )
        
        empty_room_raw.interpolate_bads(exclude=[bad_localization_channel], origin=(0., 0., 0.))
        
        noise_cov = mne.compute_raw_covariance(
            empty_room_raw,
            method="auto",
            rank=None,
            picks="meg",
        )


        inverse_operator = mne.minimum_norm.make_inverse_operator(
            fwd_solution_epoch.info,
            fwd_solution,
            noise_cov,
            loose=0.2,
            depth=0.8,
        ) 

        morph = mne.compute_source_morph(
            src=inverse_operator['src'],
            subject_from=f"{scaled}_{subject}",
            subject_to='fsaverage',
            subjects_dir=subjects_dir,
            src_to=fs_average_source_space,
        )

        snr = 3.0
        lambda2 = 1.0/snr**2

        for syllable in data.data[subject]:  # loop through the syllables to get the inverse solution for each set
            
            data.data[subject][syllable].crop(tmin=-0.2, tmax=0.6)

            stc = mne.minimum_norm.apply_inverse_epochs(
                data.data[subject][syllable],  # these are all epoch.fif objects
                inverse_operator=inverse_operator,
                lambda2=lambda2,
                method='eLORETA',
            )

            # we need to morph it back to fsaverage to be able to use the parcelation

            morphed_stc = morph.apply(stc)

            # set up array to store the different ROIs
            n_time_points = data.data[subject][syllable][0].copy().get_data().shape[-1]
            roi_array = np.zeros([len(label_dictionary), len(data.data[subject][syllable]), log_samples, n_time_points]) # num labels x num epochs x coefficient array
            print(roi_array.shape)

            for i, label in enumerate(label_dictionary): #for each ROI get the timecourse
                print(f"getting timecourse for {label}")

                label_time_courses = mne.extract_label_time_course(
                    morphed_stc,
                    label_dictionary[label],
                    src=fs_average_source_space,
                    mode='mean_flip',
                    return_generator=False,
                )
                # print(len(syllable))
                # print(len(label_time_courses))
                for j, tc in enumerate(label_time_courses): # process the time course to get the transform
                    result = process_channel(
                        signal=tc,
                        cwt_wavelet=cwt_wavelet,
                        scales=scales,
                        sampling_period=sampling_period,
                        dwt_wavelet_name=dwt_wavelet_name,
                        level=level,
                    )
                    # print(result.shape)
                    reshaped_result = np.transpose(result, (0, 2, 1))
                    # print(reshaped_result.shape)
                    reshaped_result = reshaped_result.squeeze()
                    # print(reshaped_result.shape)
                    # print(tc.shape[1]) # scales x timepoints 
                    reshaped_result = reshaped_result[:, :tc.shape[1]] # getting rid of aliasing
                    # print(reshaped_result.shape)
                    roi_array[i, j] = reshaped_result

            # print(roi_array.shape)

            save_coefficient_results( # ROI x Epochs x Coefficient Array
                subject=subject,
                syllable=syllable,
                all_coefficients=roi_array,
                save_dir=save_dir
                )

                

if __name__ == "__main__":
    main()