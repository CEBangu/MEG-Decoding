import os
import csv
import argparse

def main():


    parser= argparse.ArgumentParser(description="This script creates the dataset label csvs for the model training in the vowel discrimintaion regime")

    parser.add_argument('--data_dir', type=str, required=True, help='directory the data is located in')
    parser.add_argument('--save_dir', type=str, required=True, help='directory you want to save the labels in')
    parser.add_argument('--res_dims', type=str, required=True, help='resolution and dimensions of the data contained, ie 224_16x16')

    args = parser.parse_args()
    
    # label dictionary - maybe it would be better to do a label2id but i set up all the scripts to work like this so i don't want to break anything now/
    covert_overt = {"overt.png" : 0, "covert.png": 1}

    csv_file = os.path.join(args.save_dir, f'combined_labels_{args.res_dims}_all.csv')
    
    scalogram_files = os.listdir(args.data_dir)

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['FileName', 'Label'])

        for file_name in scalogram_files:
            parsed_name = file_name.split("_")
            for speech_type in covert_overt:
                if speech_type in parsed_name:
                    label = covert_overt[speech_type]
                    writer.writerow([file_name, label])



if __name__ == "__main__":
    main()
