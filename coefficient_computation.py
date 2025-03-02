import argparse
import pywt
import numpy as np
from datahandling import BcomMEG
import time

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


def main():
    parser = argparse.ArgumentParser(description="This script computes the Continuous Wavelet Transform coefficients for the Scalograms")

    parser.add_argument('--subject_list', nargs='+', type=str, required=True, help='The subject_block(s) you want the coefficients for')
    parser.add_argument('--avoid_reading', action='store_true', help="do you want to avoid the reading epochs?")
    parser.add_argument('--avoid_producing', action='store_true', help="do you want to avoid producing epochs?")
    parser.add_argument('--speech_type', type=str, required=True, help="Covert or Overt?")
    # parser.add_argument('--output_path', type=str, required=True, default='output.txt', help="Output file path")



    args = parser.parse_args()

    speech_type = args.speech_type.upper()
    # directory = f'/Volumes/@neurospeech/PROJECTS/BCI/BCOM/DATA_ANALYZED/EVOKED/DATA/WITHOUT_BADS/{speech_type}' # change to Zeus?
    directory = "/Users/ciprianbangu/Cogmaster/M2 Internship/BCI code/Data_Sample"
    subject_list = args.subject_list
    avoid_reading = args.avoid_reading
    avoid_producing = args.avoid_producing

    data = BcomMEG(subjects=subject_list,
                   dir=directory,
                   avoid_producing=avoid_producing,
                   avoid_reading=avoid_reading
                   )


    data.get_raw_data()
    sampling_rate = 300
    log_samples = 100
    wavelet_name = 'cmor'
    B = 1.0
    C = 1.0 
    wavelet = f'{wavelet_name}{B}-{C}'
    frequencies = np.logspace(np.log10(1), np.log10(sampling_rate/2), log_samples)
    sampling_period = 1/sampling_rate
    scales = pywt.central_frequency(wavelet=wavelet)/ (frequencies * sampling_period)

    for subject in data.data:
        for syllable in data.data[subject]:
            all_coefficients = np.zeros((data.data[subject][syllable].shape[0], data.data[subject][syllable].shape[1], log_samples, data.data[subject][syllable].shape[2]))
            for epoch in range(data.data[subject][syllable].shape[0]):
                start_time = time.time()
                for channel in range(data.data[subject][syllable].shape[1]):
                    

                    signal = data.data[subject][syllable][epoch][channel]
                    processed = scalogram_de_reconstruction(signal, wavelet='db4', level=5)
                    coefficients = scalogram_cwt(processed_data=processed, B=B, C=C, wavelet=wavelet, sampling_rate=sampling_rate, log_samples=log_samples)
                    # need one that is 1xchannelxcoefficientsxtime
                    all_coefficients[epoch, channel] = np.abs(coefficients)

                end_time = time.time()    
                print(f"Processing time for subject {subject}, syllable {syllable}, epoch {epoch}: {end_time - start_time} seconds")
                
            np.save(f"{subject}_{syllable}_coefficients.npy", all_coefficients)

if __name__ == "__main__":
    main()