import os
import csv
import argparse

def main():


    parser= argparse.ArgumentParser(description="This script creates the dataset label csvs for the model training")

    parser.add_argument('--data_dir', type=str, required=True, help='directory the data is located in')
    parser.add_argument('--save_dir', type=str, required=True, help='directory you want to save the labels in')
    parser.add_argument('--res_dims', type=str, required=True, help='resolution and dimensions of the data contained, ie 224_16x16')

    args = parser.parse_args()
    
    # label dictionary - maybe it would be better to do a label2id but i set up all the scripts to work like this so i don't want to break anything now/
    vowels = {'a': 0, 'e': 1, 'i': 2}

    # combining vowels and consonants, but labeled by vowels
    scalogram_files = os.listdir(args.data_dir)
    csv_file = os.path.join(args.save_dir, f'combined_labels_{args.res_dims}_all.csv')
    

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['FileName', 'Label'])

        for file_name in scalogram_files:
            vowel_found = False
            label = file_name.split('_')[3]
            for vowel in vowels:
                if vowel in label:
                    label = vowels[vowel]
                    writer.writerow([file_name, label])
                    vowel_found = True
                    break

            if not vowel_found:
                raise ValueError(f"no vowel found in file name {file_name}!")

    # Vowels only
    # All first then vowlels only
    csv_file = os.path.join(args.save_dir, f'vowels_only_labels_{args.res_dims}.csv')

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['FileName', 'Label'])

        for file_name in scalogram_files:
            label = file_name.split('_')[3]
            if len(label) > 1:
                continue
            elif label in vowels:
                writer.writerow([file_name, vowels[label]])
            else:
                raise ValueError(f"no vowel found in {file_name}! or filename formatted incorrectly. ")


if __name__ == "__main__":
    main()
