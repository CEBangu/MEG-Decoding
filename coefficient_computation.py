import argparse
import os
import pywt
import numpy as np
from datahandling import BcomMEG

def scalogram_de_reconstruction(data, wavelet='db4', level=5):
    # First decompose
    coefficients = pywt.wavedec(data, wavelet, level=level)
    coefficients[-1] = np.zeros_like(coefficients[-1]) #d1 #get rid of these two as in Dash et al 2020.
    coefficients[-2] = np.zeros_like(coefficients[-2]) #d2
    # Reconstruct
    reconstructed_signal = pywt.waverec(coefficients, wavelet)[:len(data)]
    return reconstructed_signal

def scalogram_cwt(processed_data, wavelet, B, C, sampling_rate, log_samples):
    wavelet = f'{wavelet}{B}-{C}'
    sampling_period = 1/sampling_rate
    frequencies = np.logspace(np.log10(1), np.log10(sampling_rate/2), log_samples)
    scales = pywt.central_frequency(wavelet=wavelet)/ (frequencies * sampling_period)
    coefficients, _ = pywt.cwt(data=processed_data, scales=scales, wavelet=wavelet, sampling_period=sampling_period)
    return coefficients


def main():
    parser = argparse.ArgumentParser(description="This script computes the Continuous Wavelet Transform coefficients for the Scalograms")

    parser.add_argument('--subject_list', nargs='+', type=str, required=True, help='The subject_block(s) you want the coefficients for')
    parser.add_argument('--avoid_reading', action='store_true', help="do you want to avoid the reading epochs?")
    parser.add_argument('--avoid_producing', action='store_true', help="do you want to avoid producing epochs?")
    parser.add_argument('--speech_type', type=str, required=True, help="Covert or Overt?")
    parser.add_argument('--output_path', type=str, required=True, default='output.txt', help="Output file path")



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

    for subject in data.data:
        for syllable in data.data[subject]:




if __name__ == "__main__":
    main()