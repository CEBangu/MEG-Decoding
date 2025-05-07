import mne
import os
import pywt
import numpy as np
from wavelets import process_channel, save_coefficient_results
from mne import set_config
from argparse import ArgumentParser
from datahandling import BcomMEG

def main():
    parser = ArgumentParser()

    parser.add_argument('--subject_list', nargs='+', type=str, required=True, help='The subject_block(s) you want the coefficients for')
    parser.add_argument('--avoid_reading', action='store_true', help="do you want to avoid the reading epochs?")
    parser.add_argument('--avoid_producing', action='store_true', help="do you want to avoid producing epochs?")
    parser.add_argument('--speech_type', type=str, required=True, help="Covert or Overt?")
    parser.add_argument('--data_dir', type=str, help="Directory where the data is stored")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the coefficients")
    parser.add_argument('--tc_save_dir', type=str, required=True, help="Directory to save the normalized time courses in")
    parser.add_argument('--empty_room_dir', type=str, required=True, help="Directory where the empty room recordings are stored")
    parser.add_argument('--baseline_dir', type=str, required=True, help='directory where baselines are stored')
    parser.add_argument('--trans_dir', type=str, help="directory where the MRI transformations are stored")
    parser.add_argument('--raw_dir', type=str, help="where the raw files are stored")
    parser.add_argument('--mne_dir', type=str, help="where fsaverage is located")

    args = parser.parse_args()


    ############################
            # CWT setup #
    ############################

    sampling_rate = 500 # data already downsampled to 500 at this point
    log_samples = 100 # we want 100 coefficients
    cwt_wavelet_name = 'cmor' # reconstruction wavelet
    B = 1.0 # wavelet bandwith (higher means more frequencies at each scale, but less precision in peak timing)
    C = 1.0 # central frequency (higher means more oscialltions per time window, meaning higher frequency features per scale)
    cwt_wavelet = f'{cwt_wavelet_name}{B}-{C}'
    frequencies = np.logspace(np.log10(1), np.log10(sampling_rate/2), log_samples)
    sampling_period = 1/sampling_rate
    scales = pywt.central_frequency(wavelet=cwt_wavelet)/ (frequencies * sampling_period)
    dwt_wavelet_name='db4' # denoizing wavelet 
    level=5

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
    covert_dir = os.path.join(data_dir, "COVERT")
    overt_dir = os.path.join(data_dir, "OVERT")
    data_dir = os.path.join(data_dir, speech_type)
    subjects = args.subject_list
    avoid_reading = args.avoid_reading
    avoid_producing = args.avoid_producing
    empty_room_dir = args.empty_room_dir
    mne_dir = args.mne_dir
    trans_dir = args.trans_dir
    raw_dir = args.raw_dir
    tc_save_dir = args.tc_save_dir

    save_dir = args.save_dir

    bad_localization_channel = "MEG 173"
    
    data = BcomMEG(dir=data_dir, # we need all of the data so that we can compute the data_covariance matrix
                subjects=subjects,
                avoid_reading=avoid_reading,
                avoid_producing=avoid_producing,
                )
    
    scaled = "scaled_fsaverage"

    for subject in data.data: # loop through the subjects (blocks, really)
            
        # we have the morphed transformations, so we will load them for the subject
        morphed_trans = os.path.join(trans_dir, f"{subject}-trans.fif")
        morphed_source = os.path.join(mne_dir, f"{scaled}_{subject}/bem/{scaled}_{subject}-ico-5-src.fif")
        morphed_bem = os.path.join(mne_dir, f"{scaled}_{subject}/bem/{scaled}_{subject}-5120-5120-5120-bem-sol.fif")
        
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

        #####################
        # noise covariance #
        ####################
        
        sub = subject[:7]
        block = subject[-1]

        # need to get the covariance matrix from the empty_room recording 
        subject_empty_room = os.path.join(empty_room_dir, sub, block, "empty_room_cleaned_ICA_raw.fif")

        empty_room_raw = mne.io.read_raw_fif(
            subject_empty_room,
            preload=True,
        )
        
        empty_room_raw.interpolate_bads(exclude=[bad_localization_channel], origin=(0., 0., 0.))
        
        noise_cov = mne.compute_raw_covariance(
            empty_room_raw,
            method="auto",
            rank="info",
            picks="meg",
        )

        #################
        # get baseline #
        ################

        subject_baseline = os.path.join(args.baseline_dir, sub, f"{subject}_baseline_raw.fif")
        baseline = mne.io.read_raw_fif(subject_baseline, preload=True)
        baseline.pick_types(meg=True)


        ###################
        # data covariance #
        ###################
        
        raw = mne.io.read_raw_fif(fname=os.path.join(raw_dir, f"{sub}/{block}/subject_cleaned_ica_raw.fif"))
        
        epochs_array = []
        dropped_epochs = []
        

        # get all of the data to compute the data covariance matrix
        for file in os.listdir(covert_dir):
            if subject in file:
                file_path = os.path.join(covert_dir, file)
                epoch = mne.read_epochs(file_path)
                if epoch.info['dev_head_t'] != raw.info['dev_head_t']:
                        dropped_epochs.append(("Covert", file, len(epoch.events)))
                else:
                        epochs_array.append(epoch)

        for file in os.listdir(overt_dir):
            if subject in file:
                file_path = os.path.join(covert_dir, file)
                epoch = mne.read_epochs(file_path)
                if epoch.info['dev_head_t'] != raw.info['dev_head_t']:
                    dropped_epochs.append(("Overt", file, len(epoch.events)))
                else:
                    epochs_array.append(epoch)                 
                
        epochs_array = mne.concatenate_epochs(epochs_array)


        data_cov = mne.compute_covariance(
                epochs_array.crop(tmin=-0.2, tmax=0.6), 
                method="auto", 
                rank="info"
            )


        ####################
        #    beamforming   #
        ####################

        filters = mne.beamformer.make_lcmv(
                epochs_array.info,
                fwd_solution,
                data_cov,
                reg=0.05,
                noise_cov=noise_cov,
                pick_ori="max-power",
                weight_norm="unit-noise-gain",
                rank="info",
        )


        del epochs_array
        del raw

        ###########################
        # now on the actual data # 
        ###########################

        recon_baseline = mne.beamformer.apply_lcmv_raw(baseline, filters)

        morph = mne.compute_source_morph(
            src=morphed_source,
            subject_from=f"{scaled}_{subject}",
            subject_to="fsaverage",
            subjects_dir=subjects_dir,
        )

        morphed_baseline = morph.apply(recon_baseline)


        for syllable in data.data[subject]:
            data.data[subject][syllable].crop(tmin=-0.2, tmax=0.6)

            recon_task = mne.beamformer.apply_lcmv_epochs(data.data[subject][syllable], filters)

            morphed_time_courses = [morph.apply(stc) for stc in recon_task]

            n_time_points = data.data[subject][syllable][0].copy().get_data().shape[-1]
        
            roi_array = np.zeros([len(label_dictionary), len(data.data[subject][syllable]), log_samples, n_time_points])
            
            for i, label in enumerate(label_dictionary):
                print(f"getting timecourse for {label}")

                label_time_courses_condition = mne.extract_label_time_course(
                    morphed_time_courses,
                    label_dictionary[label],
                    src=fs_average_source_space,
                    mode="mean_flip",
                    return_generator=False,
                )

                label_time_course_baseline = mne.extract_label_time_course(
                    morphed_baseline,
                    label_dictionary[label],
                    src=fs_average_source_space,
                    mode="mean_flip",
                    return_generator=False,
                )

                # mean substraction normalization for the time-course
                normalized_time_courses = [condition_time_course - label_time_course_baseline[0].mean() for condition_time_course in label_time_courses_condition]
                
                # save the time course
                np.save(
                    os.path.join(tc_save_dir, f"{subject}_{syllable}_{label}_normalized_time_courses.npy"),
                    normalized_time_course
                )
                
                baseline_tf = process_channel(
                    signal=label_time_course_baseline,
                    cwt_wavelet=cwt_wavelet,
                    scales=scales,
                    sampling_period=sampling_period,
                    dwt_wavelet_name=dwt_wavelet_name,
                    level=level,
                )

                reshaped_baseline = np.transpose(baseline_tf, (0, 2, 1))
                reshaped_baseline = reshaped_baseline.squeeze()
                reshaped_baseline = reshaped_baseline[:, :label_time_course_baseline.shape[1]]

                baseline_row_mean = reshaped_baseline.mean(axis=1, keepdims=True)
                baseline_row_std = reshaped_baseline.std(axis=1, keepdims=True)

                
                for j, tc in enumerate(label_time_courses_condition):
                    result_condition = process_channel(
                        signal=tc,
                        cwt_wavelet=cwt_wavelet,
                        scales=scales,
                        sampling_period=sampling_period,
                        dwt_wavelet_name=dwt_wavelet_name,
                        level=level,
                    )

                    reshaped_result = np.transpose(result_condition, (0, 2, 1))
                    reshaped_result = reshaped_result.squeeze()
                    reshaped_result = reshaped_result[:, :tc.shape[1]] 

                    normalized_result = (reshaped_result - baseline_row_mean) / baseline_row_std

                    roi_array[i, j] = normalized_result

                save_coefficient_results(
                    subject=subject,
                    syllable=syllable,
                    all_coefficients=roi_array,
                    save_dir=save_dir
                    )


if __name__ == "__main__":
    main()











                 

