import os
import pandas as pd
import numpy as np
import os.path as op
from sklearn.model_selection import train_test_split

vowel2id = {
    'a': 0,
    'e': 1,
    'i': 2,
    }

speech2id = {
    3: 1, # producing
    2: 0, # reading
}

voweltype2id = {
    1: 'pure',
    2: 'composite',
}

def dir_to_dataframe(data_dir, vowel2id, speech2id, voweltype2id):
    """
    Convert a directory of files to a dataframe
    """
    df = pd.DataFrame({
        "FileName":[],
        "Label": [],
        "Vowel_Type": [],
        "Speech_Type": [],
    })
    
    for file in os.listdir(data_dir):
        if ".png" in file:
            file_path = op.join(data_dir, file)
            parsed_file = file.split("_")
            
            label = parsed_file[3][1] if len(parsed_file[3]) > 1 else parsed_file[3][0]

            if label in vowel2id:
                label = int(vowel2id[label])
            else:
                raise ValueError(f"Label {label} not in vowel2id")
            
            speech_type = int(len(parsed_file[4]))
            if speech_type in speech2id:
                speech_type = speech2id[speech_type]
            else:
                raise ValueError(f"Speech type {speech_type} not in speech2id")
            
            vowel_type = int(len(parsed_file[3]))
            if vowel_type in voweltype2id:
                vowel_type = voweltype2id[vowel_type]
            else:
                raise ValueError(f"Vowel type {vowel_type} not in voweltype2id")
            
            df = pd.concat([df, pd.DataFrame({"FileName": [file_path], "Label": [label], "Vowel_Type":[vowel_type], "Speech_Type": [speech_type]})], ignore_index=True)
    
    return df


def dataframe_to_train_test_covert(df_covert, prod=False, read=False, vowel2id=vowel2id, speech2id=speech2id):
    """
    Convert dataframes to train and test sets for covert production and reading, and covert read-prod tasks.
    """
    if prod:
        df = df_covert[df_covert['Speech_Type'] == 1]
        df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, stratify=df['Label'])

    elif read:
        df = df_covert[df_covert['Speech_Type'] == 0]
        df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, stratify=df['Label'])

    else:
        df = df_covert
        # one that just splits them:
        df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, stratify=df['Label'])

    df_train = df_train.drop(columns=['Vowel_Type', 'Speech_Type'])
    df_test = df_test.drop(columns=['Vowel_Type', 'Speech_Type'])

    df_train['Label'] = df_train['Label'].astype(int)
    df_test['Label'] = df_test['Label'].astype(int)


    return df_train, df_test


def dataframe_to_train_test_special_covert(df_covert, prod=False, read=True, vowel2id=vowel2id, speech2id=speech2id):
    """converts the datframes to training and testing when you want to train on reading and producing, but only test on one of them."""
    total_samples = len(df_covert)
    if prod:
        producing_indices = df_covert[df_covert['Speech_Type'] == 1].index
        random_prod_samples = np.random.choice(producing_indices, size=int(total_samples * 0.1), replace=False) # 10% of the total samples
        df_train = df_covert.drop(random_prod_samples)
        df_test = df_covert.loc[random_prod_samples]
    elif read:
        reading_indices = df_covert[df_covert['Speech_Type'] == 0].index
        random_read_samples = np.random.choice(reading_indices, size=int(total_samples * 0.1), replace=False)
        df_train = df_covert.drop(random_read_samples)
        df_test = df_covert.loc[random_read_samples]
    else:
        raise ValueError("Either prod or read must be True")
    df_train = df_train.drop(columns=['Vowel_Type', 'Speech_Type'])
    df_test = df_test.drop(columns=['Vowel_Type', 'Speech_Type'])
    df_train['Label'] = df_train['Label'].astype(int)
    df_test['Label'] = df_test['Label'].astype(int)
    
    return df_train, df_test


def main():
    sensor_root = '/pasteur/appa/scratch/cbangu/scalograms/covert/'
    roi_root = '/pasteur/appa/scratch/cbangu/scalograms_beamformer_roi/covert/'
    datasets_root = '/pasteur/appa/scratch/cbangu/datasets/'

    sensors = [
        "1x1", "2x2", "3x3", "4x4", "5x5", "6x6", 
        "7x7", "8x8", "9x9", "10x10", "11x11", "12x12", 
        "13x13", "14x14", "15x15", "16x16",
    ]
    rois = [
        "sma",
        "broca",
        "stg",
        "spt",
        "2x2"
    ]
    vowel2id = {
    'a': 0,
    'e': 1,
    'i': 2,
    }

    speech2id = {
        3: 1, # producing
        2: 0, # reading
    }

    voweltype2id = {
        1: 'pure',
        2: 'composite',
    }


    for sensor in sensors:
        path = op.join(sensor_root, sensor, "data")

        df = dir_to_dataframe(path, vowel2id, speech2id, voweltype2id)
        df_train, df_test = dataframe_to_train_test_covert(df, prod=True, read=False, vowel2id=vowel2id, speech2id=speech2id)
        df_train.to_csv(op.join(datasets_root, sensor, 'prod_train.csv'), index=False)
        df_test.to_csv(op.join(datasets_root, sensor, 'prod_test.csv'), index=False)

        df_train, df_test = dataframe_to_train_test_covert(df, prod=False, read=True, vowel2id=vowel2id, speech2id=speech2id)
        df_train.to_csv(op.join(datasets_root, sensor, 'read_train.csv'), index=False)
        df_test.to_csv(op.join(datasets_root, sensor, 'read_test.csv'), index=False)

        df_train, df_test = dataframe_to_train_test_covert(df, prod=False, read=False, vowel2id=vowel2id, speech2id=speech2id)
        df_train.to_csv(op.join(datasets_root, sensor, 'read-prod_train.csv'), index=False)
        df_test.to_csv(op.join(datasets_root, sensor, 'read-prod_test.csv'), index=False)

        df_train, df_test = dataframe_to_train_test_special_covert(df, prod=True, read=False, vowel2id=vowel2id, speech2id=speech2id)
        df_train.to_csv(op.join(datasets_root, sensor, 'read-prod-prod_train.csv'), index=False)
        df_test.to_csv(op.join(datasets_root, sensor, 'read-prod-prod_test.csv'), index=False)

        df_train, df_test = dataframe_to_train_test_special_covert(df, prod=False, read=True, vowel2id=vowel2id, speech2id=speech2id)
        df_train.to_csv(op.join(datasets_root, sensor, 'prod-read-read_train.csv'), index=False)
        df_test.to_csv(op.join(datasets_root, sensor, 'prod-read-read_test.csv'), index=False)


    for roi in rois:
        path = op.join(roi_root, roi, "data")

        if roi == "2x2":
            roi_name = "combo"
        else:
            roi_name = roi

        df = dir_to_dataframe(path, vowel2id, speech2id, voweltype2id)
        df_train, df_test = dataframe_to_train_test_covert(df, prod=True, read=False, vowel2id=vowel2id, speech2id=speech2id)
        df_train.to_csv(op.join(datasets_root, roi_name, 'prod_train.csv'), index=False)
        df_test.to_csv(op.join(datasets_root, roi_name, 'prod_test.csv'), index=False)

        df_train, df_test = dataframe_to_train_test_covert(df, prod=False, read=True, vowel2id=vowel2id, speech2id=speech2id)
        df_train.to_csv(op.join(datasets_root, roi_name, 'read_train.csv'), index=False)
        df_test.to_csv(op.join(datasets_root, roi_name, 'read_test.csv'), index=False)

        df_train, df_test = dataframe_to_train_test_covert(df, prod=False, read=False, vowel2id=vowel2id, speech2id=speech2id)
        df_train.to_csv(op.join(datasets_root, roi_name, 'read-prod_train.csv'), index=False)
        df_test.to_csv(op.join(datasets_root, roi_name, 'read-prod_test.csv'), index=False)

        df_train, df_test = dataframe_to_train_test_special_covert(df, prod=True, read=False, vowel2id=vowel2id, speech2id=speech2id)
        df_train.to_csv(op.join(datasets_root, roi_name, 'read-prod-prod_train.csv'), index=False)
        df_test.to_csv(op.join(datasets_root, roi_name, 'read-prod-prod_test.csv'), index=False)

        df_train, df_test = dataframe_to_train_test_special_covert(df, prod=False, read=True, vowel2id=vowel2id, speech2id=speech2id)
        df_train.to_csv(op.join(datasets_root, roi_name, 'prod-read-read_train.csv'), index=False)
        df_test.to_csv(op.join(datasets_root, roi_name, 'prod-read-read_test.csv'), index=False)

if __name__ == "__main__":
    main()





