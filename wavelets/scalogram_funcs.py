import pywt
import os
import numpy as np



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

def save_coefficient_results_sensors(subject, syllable, all_coefficients, save_dir):
    """ Save computed coefficients to the correct directory """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    for epoch, sensors in enumerate(all_coefficients):
        output_file = os.path.join(save_dir, f"{subject}_{syllable}_coefficients_{epoch}.npy")
        np.save(output_file, sensors)
        print(f"Saved results to {output_file}")

def save_coefficient_results_roi(subject, syllable, all_coefficients, save_dir):
    index2roi = {
        0: "sma",
        1: "broca",
        2: "stg",
        3: "mtg",
        4: "spt",
    }
    os.makedirs(save_dir, exist_ok=True)
    for roi, roi_block in enumerate(all_coefficients):
        for epoch_index, epoch_data in enumerate(roi_block):
            output_file = os.path.join(save_dir, f"{subject}_{syllable}_coefficients_{index2roi[roi]}_{epoch_index}.npy")
            np.save(output_file, epoch_data)
            print(f"Saved results to {output_file}")


