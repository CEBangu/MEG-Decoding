import argparse
import pywt
import os
import numpy as np
from datahandling import BcomMEG
import time
from joblib import Parallel, delayed


def save_results(subject, syllable, all_coefficients, save_dir):
    """ Save computed coefficients to the correct directory """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    output_file = os.path.join(save_dir, f"{subject}_{syllable}_coefficients.npy")
    np.save(output_file, all_coefficients)
    print(f"Saved results to {output_file}")

def scalogram_de_reconstruction(data, wavelet='db4', level=5):
    # First decompose
    coefficients = pywt.wavedec(data, wavelet, level=level)
    coefficients[-1] = np.zeros_like(coefficients[-1]) #d1 #get rid of these two as in Dash et al 2020.
    coefficients[-2] = np.zeros_like(coefficients[-2]) #d2
    # Reconstruct
    reconstructed_signal = pywt.waverec(coefficients, wavelet)[:len(data)]
    return reconstructed_signal

def scalogram_cwt(processed_data, wavelet, scales, sampling_period):
    coefficients, _ = pywt.cwt(data=processed_data, scales=scales, wavelet=wavelet, sampling_period=sampling_period)
    return coefficients

def process_channel(signal, cwt_wavelet, scales, sampling_period, dwt_wavelet_name, level):
    """Function to parallelize the channel computation"""
    processed = scalogram_de_reconstruction(signal, wavelet=dwt_wavelet_name, level=level)
    coefficients = scalogram_cwt(processed_data=processed, wavelet=cwt_wavelet, scales=scales, sampling_period=sampling_period)
    return np.abs(coefficients)


def main():
    parser = argparse.ArgumentParser(description="This script computes the Continuous Wavelet Transform coefficients for the Scalograms")

    parser.add_argument('--subject_list', nargs='+', type=str, required=True, help='The subject_block(s) you want the coefficients for')
    parser.add_argument('--avoid_reading', action='store_true', help="do you want to avoid the reading epochs?")
    parser.add_argument('--avoid_producing', action='store_true', help="do you want to avoid producing epochs?")
    parser.add_argument('--speech_type', type=str, required=True, help="Covert or Overt?")
    parser.add_argument('--data_dir', type=str, help="Directory where the data is stored")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the coefficients")



    args = parser.parse_args()

    
    speech_type = args.speech_type.upper() # to standardize input
    data_dir = args.data_dir
    data_dir = os.path.join(data_dir, speech_type)
    save_dir = args.save_dir
    # directory = "/Users/ciprianbangu/Cogmaster/M2 Internship/BCI code/Data_Sample"
    subject_list = args.subject_list
    subject_list = [subject.upper() for subject in subject_list] # to standardize input, just in case

    avoid_reading = args.avoid_reading
    avoid_producing = args.avoid_producing

    data = BcomMEG(subjects=subject_list, # load data
                   dir=data_dir,
                   avoid_producing=avoid_producing,
                   avoid_reading=avoid_reading
                   )


    data.get_raw_data() # extract the data from the EPO objects

    # DWT Denoize variables
    dwt_wavelet_name='db4' # denoizing wavelet 
    level=5 # level of decomposition. NB in Dash et al. they use 7, but our signal is shorter, so 5 is max
    
    
    # CWT Reconstruction variables - better to compute as much outside the loop b/c lots of repetitions
    sampling_rate = 300 # data already downsampled to 300 at this point
    log_samples = 100 # we want 100 coefficients
    cwt_wavelet_name = 'cmor' # reconstruction wavelet
    B = 1.0 # wavelet bandwith (higher means more frequencies at each scale, but less precision in peak timing)
    C = 1.0 # central frequency (higher means more oscialltions per time window, meaning higher frequency features per scale)
    cwt_wavelet = f'{cwt_wavelet_name}{B}-{C}'
    frequencies = np.logspace(np.log10(1), np.log10(sampling_rate/2), log_samples)
    sampling_period = 1/sampling_rate
    scales = pywt.central_frequency(wavelet=cwt_wavelet)/ (frequencies * sampling_period)

    for subject in data.data:
        for syllable in data.data[subject]:
            all_coefficients = np.zeros((data.data[subject][syllable].shape[0], data.data[subject][syllable].shape[1], log_samples, data.data[subject][syllable].shape[2]))
            for epoch in range(data.data[subject][syllable].shape[0]):
                start_time = time.time()
                
                results = Parallel(n_jobs=-1)(delayed(process_channel)(
                    signal=data.data[subject][syllable][epoch][channel],
                    cwt_wavelet=cwt_wavelet, 
                    scales=scales, 
                    sampling_period=sampling_period,
                    dwt_wavelet_name=dwt_wavelet_name, 
                    level=level
                    ) for channel in range(data.data[subject][syllable].shape[1])
                )
                
                for channel, coefficients in enumerate(results):
                    all_coefficients[epoch, channel] = coefficients

                # Old code - just in case this doesn't work lol
                # for channel in range(data.data[subject][syllable].shape[1]):
                    

                #     signal = data.data[subject][syllable][epoch][channel]
                #     processed = scalogram_de_reconstruction(signal, wavelet=dwt_wavelet_name, level=level)
                #     coefficients = scalogram_cwt(processed_data=processed, wavelet=cwt_wavelet, scales=scales, sampling_period=sampling_period)
                #     # need one that is 1xchannelxcoefficientsxtime
                #     all_coefficients[epoch, channel] = np.abs(coefficients)

                end_time = time.time()    
                print(f"Processing time for subject {subject}, syllable {syllable}, epoch {epoch}: {end_time - start_time} seconds")
                
            save_results(
                subject=subject,
                syllable=syllable,
                all_coefficients=all_coefficients,
                save_dir=save_dir)

if __name__ == "__main__":
    main()