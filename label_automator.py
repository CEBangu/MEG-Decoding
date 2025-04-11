import os
import subprocess

def main():

    # make the vowel labels
    subprocess.run([
        'python3', 
        'label_maker_vowels.py', 
        '--data_dir', labels_data_path, 
        '--save_dir', labels_save_path, 
        '--res_dims', res_dims,
        '--producing_only'    
    ])
    
    subprocess.run([
        'python3', 
        'label_maker_vowels.py', 
        '--data_dir', labels_data_path, 
        '--save_dir', labels_save_path, 
        '--res_dims', res_dims,   
    ])   

    # train test_split for vowels
    subprocess.run([
        'python3', 
        'train_test_split_vowels.py', 
        '--data_dir', split_data_path,
        '--save_dir', split_save_path,
        '--dimensions', res_dims
        '--class_tag', class_tag
    ])







    subprocess.run(['python3', 'label_maker_covertcovert.py', x, y])

    subprocess.run(['python3', 'scalogram_mover.py', x, y])

    subprocess.run(['python3', 'label_maker_overtcovert.py', x, y])
