import os
import mne
import pywt
import numpy as np
from argparse import ArgumentParser
from datahandling import BcomMEG
from experiment import process_channel, save_coefficient_results



def main():

    parser = ArgumentParser()

    parser.add_argument('--subject_list', nargs='+', type=str, required=True, help='The subject_block(s) you want the coefficients for')
    parser.add_argument('--avoid_reading', action='store_true', help="do you want to avoid the reading epochs?")
    parser.add_argument('--avoid_producing', action='store_true', help="do you want to avoid producing epochs?")
    parser.add_argument('--speech_type', type=str, required=True, help="Covert or Overt?")
    parser.add_argument('--data_dir', type=str, help="Directory where the data is stored")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the coefficients")

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
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
    subjects_dir = os.path.dirname(fs_dir) 

    #get labels
    labels = mne.read_labels_from_annot(
        subject='fsaverage',
        parc='HCPMMP1',
        hemi='lh',
        subjects_dir=subjects_dir
    )

    ba6_labels = [label for label in labels if 'L_6ma_ROI-lh' in label.name][0] +\
                [label for label in labels if 'L_6mp_ROI-lh' in label.name][0]


    broca_labels = [label for label in labels if "L_44_ROI-lh" in label.name][0] +\
            [label for label in labels if "L_45_ROI-lh" in label.name][0]


    stg_labels = [label for label in labels if "L_A1_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_MBelt_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_LBelt_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_TA2_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_STGa_ROI-lh" in label.name][0]

    mtg_labels = [label for label in labels if "L_MT_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_TPOJ1_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_TPOJ2_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_STSdp_ROI-lh" in label.name][0]

    stp_labels = [label for label in labels if "L_A1_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_MBelt_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_LBelt_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_TA2_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_RI_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_PBelt_ROI-lh" in label.name][0] 

    ba10_labels = [label for label in labels if "L_10d_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_10v_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_a10p_ROI-lh" in label.name][0] +\
                [label for label in labels if "L_p10p_ROI-lh" in label.name][0]


    label_dictionary = {
        "ba6": ba6_labels,
        "broca": broca_labels,
        "stg": stg_labels,
        "mtg": mtg_labels,
        "stp": stp_labels,
        "ba10": ba10_labels
    }

    ##################################
    # Setting up common source space #
    ##################################

    common_subject = 'fsaverage'
    source = mne.setup_source_space(
        subject=common_subject, # only once, no co-registration
        spacing='oct6',
        add_dist=False,
    )

    bem = mne.make_bem_solution(
        mne.make_bem_model(
        subject=common_subject,
        ico=4,
        conductivity=(0.3, ),
        subjects_dir=subjects_dir
        )
    )

    ##############################
    # Block-wise forward solution#
    ##############################

    dir = args.data_dir
    subjects = args.subject_list
    avoid_reading = args.avoid_reading
    avoid_producing = args.avoid_producing

    save_dir = args.save_dir


    data = BcomMEG(dir=dir,
                subjects=subjects,
                avoid_reading=avoid_reading,
                avoid_producing=avoid_producing,
                )


    for subject in data.data:
        # forward solution by block
        first_epoch_name = list(data.data[subject].keys())[0]
        fwd_solution_epoch = data.data[subject][first_epoch_name]
        fwd_solution = mne.make_forward_solution(
            fwd_solution_epoch.info,
            trans=common_subject,
            src=source,
            bem=bem,
            meg=True,
            eeg=False,
        )
        
        # covariance matrix
        cov = mne.compute_covariance(
            fwd_solution_epoch,
            tmin=-0.3,
            tmax=-0.2,
            method='empirical',
        )

        inverse_operator = mne.minimum_norm.make_inverse_operator(
            fwd_solution_epoch.info,
            fwd_solution,
            cov,
            loose=0.2,
            depth=0.8,
        )

        snr = 2.0
        lambda2 = 1.0/snr**2

        for syllable in data.data[subject]: 
            stc = mne.minimum_norm.apply_inverse_epochs(
                data.data[subject][syllable],  # these are all epoch.fif objects
                inverse_operator=inverse_operator,
                lambda2=lambda2,
                method='eLORETA',
            )

            for label in label_dictionary:
                label_time_courses = mne.extract_label_time_course(
                    stc,
                    label_dictionary[label],
                    src=source,
                    mode='mean_flip',
                    return_generator=False,
                )

                for i, tc in enumerate(label_time_courses):
                    result = process_channel(
                        signal=tc,
                        cwt_wavelet=cwt_wavelet,
                        scales=scales,
                        sampling_period=sampling_period,
                        dwt_wavelet_name=dwt_wavelet_name,
                        level=level,
                    )
                    reshaped_result = np.transpose(result, (0, 2, 1)).squeeze() # scales x timepoints 

                    save_coefficient_results(
                        subject=subject,
                        syllable=syllable,
                        all_coefficients=reshaped_result,
                        save_dir=save_dir
                        )
                    
                    break
                break
            break
        break

if __name__ == "__main__":
    main()