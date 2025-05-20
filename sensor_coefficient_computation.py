import argparse
import pywt
import os
import mne
import numpy as np
from datahandling import BcomMEG
import time
from joblib import Parallel, delayed
from wavelets import process_channel, save_coefficient_results

def main():
    parser = argparse.ArgumentParser(description="This script computes the Continuous Wavelet Transform coefficients for the Scalograms")

    parser.add_argument('--subject_list', nargs='+', type=str, required=True, help='The subject_block(s) you want the coefficients for')
    parser.add_argument('--avoid_reading', action='store_true', help="do you want to avoid the reading epochs?")
    parser.add_argument('--avoid_producing', action='store_true', help="do you want to avoid producing epochs?")
    parser.add_argument('--speech_type', type=str, required=True, help="Covert or Overt?")
    parser.add_argument('--data_dir', type=str, help="Directory where the data is stored")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the coefficients")
    parser.add_argument('--baseline_dir', type=str, required=True, help="Directory where baselines are located")



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

    # DWT Denoize variables
    dwt_wavelet_name='db4' # denoizing wavelet 
    level=5 # level of decomposition. NB in Dash et al. they use 7, but our signal is shorter, so 5 is max
    
    
    # CWT Reconstruction variables - better to compute as much outside the loop b/c lots of repetitions
    sampling_rate = 500 # data already downsampled to 500 at this point
    log_samples = 100 # we want 100 coefficients
    cwt_wavelet_name = 'cmor' # reconstruction wavelet
    B = 1.0 # wavelet bandwith (higher means more frequencies at each scale, but less precision in peak timing)
    C = 1.0 # central frequency (higher means more oscialltions per time window, meaning higher frequency features per scale)
    cwt_wavelet = f'{cwt_wavelet_name}{B}-{C}'
    frequencies = np.logspace(np.log10(1), np.log10(sampling_rate/2), log_samples)
    sampling_period = 1/sampling_rate
    scales = pywt.central_frequency(wavelet=cwt_wavelet)/ (frequencies * sampling_period)

    # loop through the subjects - although in the new cluster code this ends up being one subject per node
    for subject in data.data:
        
        # name parsing
        sub = subject[:7]
        
        # get subject's baseline
        subject_baseline = os.path.join(args.baseline_dir, sub, f"{subject}_baseline_raw.fif")
        baseline = mne.io.read_raw_fif(subject_baseline, preload=True)
        baseline.pick_types(meg=True)

        # compute baseline coefficients
        baseline_coefficients = np.zeros((baseline._data.shape[0], log_samples, baseline._data.shape[1]))
        results =  Parallel(n_jobs=-1)(delayed(process_channel)(
            signal=data,
            cwt_wavelet=cwt_wavelet,
            scales=scales,
            sampling_period=sampling_period,
            dwt_wavelet_name=dwt_wavelet_name,
            level = level
            ) for data in baseline._data
        )

        # store the baseline 
        for channel, result in enumerate(results):
            baseline_coefficients[channel] = result

        # get the mean and std for the z-scoring
        baseline_mean = baseline_coefficients.mean(axis=2, keepdims=True)
        baseline_std = baseline_coefficients.std(axis=2, keepdims=True)
        print(baseline_mean.min())
        # loop through the available syllables
        for syllable in data.data[subject]:
            
            # first crop the samples
            data.data[subject][syllable].crop(tmin=-0.2, tmax=0.6)

            # then get the np arrays
            data.data[subject][syllable] = data.data[subject][syllable].get_data() #replace the epoch.fif with an np array

            # set up coefficient array
            all_coefficients = np.zeros(
                (data.data[subject][syllable].shape[0], # epochs 
                 data.data[subject][syllable].shape[1], # sensors
                 log_samples, # scales
                 data.data[subject][syllable].shape[2]) # timepoints
            )

            for epoch in range(data.data[subject][syllable].shape[0]):
                start_time = time.time()
                
                # same thing for the data, just paralelize over sensors since there are a lot of them
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
                    # baseline correction
                    all_coefficients[epoch, channel] = (coefficients - baseline_mean[channel]) / baseline_std[channel] 


                end_time = time.time()    
                print(f"Processing time for subject {subject}, syllable {syllable}, epoch {epoch}: {end_time - start_time} seconds")
            
            # save the results
            save_coefficient_results(
                subject=subject,
                syllable=syllable,
                all_coefficients=all_coefficients,
                save_dir=save_dir)

if __name__ == "__main__":
    main()