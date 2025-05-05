import os
import pandas as pd
import os.path as op
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

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
            label = parsed_file[3]
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

def main():

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True, help="ROI or dimensions")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the data")
    
    args = parser.parse_args()
    root = args.root
    data_type = args.data_type

    covert_data_path = op.join(root, "covert", data_type, "data")
    overt_data_path = op.join(root, "overt", data_type, "data")
    covert_covert_path = op.join(root, "covert_covert", data_type, "train_test_split")
    covert_overt_path = op.join(root, "covert_overt", data_type, "train_test_split")

    ##############
    # tags #
    ##############
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

    ##############
    # dataframes #
    ##############
    covert_dataframe = pd.DataFrame({
        "FileName":[],
        "Label": [],
        "Vowel_Type": [],
        "Speech_Type": [],
    })

    overt_dataframe = pd.DataFrame({
        "FileName":[],
        "Label": [],
        "Vowel_Type": [],
        "Speech_Type": [],
    })

    #######################
    # populate dataframes #
    #######################
    # covert 
    covert_dataframe = dir_to_dataframe(covert_data_path, vowel2id, speech2id, voweltype2id)
    # overt
    overt_dataframe = dir_to_dataframe(overt_data_path, vowel2id, speech2id, voweltype2id)

    ######################
    # train test split #
    ######################
    # covert pure vowels
    covert_pure_all = covert_dataframe[covert_dataframe["Vowel_Type"] == 'pure']
    covert_pure_producing = covert_dataframe[(covert_dataframe["Vowel_Type"] == 'pure') & (covert_dataframe["Speech_Type"] == 1)]

    covert_pure_train_all, covert_pure_test_all = train_test_split(
        covert_pure_all,
        test_size=0.1, 
        stratify=covert_pure_all["Label"], 
        random_state=42,
        shuffle=True,
    )
    covert_pure_train_producing, covert_pure_test_producing = train_test_split(
        covert_pure_producing, 
        test_size=0.1, 
        stratify=covert_pure_producing["Label"],
        random_state=42,
        shuffle=True,
    )

    # clean up dataframes
    for df in [covert_pure_train_all, covert_pure_test_all, covert_pure_train_producing, covert_pure_test_producing]:
        df["Label"] = df["Label"].astype(int) # type enforcement
        df["FileName"] = df["FileName"].astype(str) # type enforcement
        df = df.drop_columns(["Speech_Type", "Vowel_Type"]) # drop unnecessary columns

    # save dataframes
    covert_pure_train_all.to_csv(op.join(covert_data_path, "train_test_split", "covert_pure_train_all.csv"), index=False)
    covert_pure_test_all.to_csv(op.join(covert_data_path, "train_test_split", "covert_pure_test_all.csv"), index=False)
    covert_pure_train_producing.to_csv(op.join(covert_data_path, "train_test_split", "covert_pure_train_producing.csv"), index=False)
    covert_pure_test_producing.to_csv(op.join(covert_data_path, "train_test_split", "covert_pure_test_producing.csv"), index=False)


    ##########################################################################################################################################

    # covert composite vowels
    covert_composite_all = covert_dataframe
    covert_composite_producing = covert_dataframe[(covert_dataframe["Speech_Type"] == 1)]

    covert_composite_train_all, covert_composite_test_all = train_test_split(
        covert_composite_all,
        test_size=0.1, 
        stratify=covert_composite_all["Label"],
        random_state=42,
        shuffle=True,
    )

    covert_composite_train_producing, covert_composite_test_producing = train_test_split(
        covert_composite_producing, 
        test_size=0.1, 
        stratify=covert_composite_producing["Label"],
        random_state=42,
        shuffle=True,
    )

    for df in [covert_composite_train_all, covert_composite_test_all, covert_composite_train_producing, covert_composite_test_producing]:
        df["Label"] = df["Label"].astype(int) # type enforcement
        df["FileName"] = df["FileName"].astype(str) # type enforcement
        df = df.drop_columns(["Speech_Type", "Vowel_Type"]) # drop unnecessary columns

    covert_composite_train_all.to_csv(op.join(covert_data_path, "train_test_split", "covert_composite_train_all.csv"), index=False)
    covert_composite_test_all.to_csv(op.join(covert_data_path, "train_test_split", "covert_composite_test_all.csv"), index=False)
    covert_composite_train_producing.to_csv(op.join(covert_data_path, "train_test_split", "covert_composite_train_producing.csv"), index=False)
    covert_composite_test_producing.to_csv(op.join(covert_data_path, "train_test_split", "covert_composite_test_producing.csv"), index=False)


    #############################################################################################################################################

    # overt vowels - there is no reading, so it's all producing by default
    # overt pure vowels
    overt_pure_producing = overt_dataframe[overt_dataframe["Vowel_Type"] == "pure"]

    overt_pure_train_producing, overt_pure_test_producing = train_test_split(
        overt_pure_producing, 
        test_size=0.1, 
        stratify=overt_pure_producing["Label"],
        random_state=42,
        shuffle=True,
    )

    for df in [overt_pure_train_producing, overt_pure_test_producing]:
        df["Label"] = df["Label"].astype(int) # type enforcement
        df["FileName"] = df["FileName"].astype(str) # type enforcement
        df = df.drop_columns(["Speech_Type", "Vowel_Type"]) # drop unnecessary columns

    overt_pure_train_producing.to_csv(op.join(overt_data_path, "train_test_split", "overt_pure_train_producing.csv"), index=False)
    overt_pure_test_producing.to_csv(op.join(overt_data_path, "train_test_split", "overt_pure_test_producing.csv"), index=False)

    #############################################################################################################################################

    # overt composite vowels
    overt_composite_producing = overt_dataframe

    overt_composite_train_producing, overt_composite_test_producing = train_test_split(
        overt_composite_producing, 
        test_size=0.1, 
        stratify=overt_composite_producing["Label"],
        random_state=42,
        shuffle=True,
    )

    for df in [overt_composite_train_producing, overt_composite_test_producing]:
        df["Label"] = df["Label"].astype(int) # type enforcement
        df["FileName"] = df["FileName"].astype(str) # type enforcement
        df = df.drop_columns(["Speech_Type", "Vowel_Type"]) # drop unnecessary columns

    overt_composite_train_producing.to_csv(op.join(overt_data_path, "train_test_split", "overt_composite_train_producing.csv"), index=False)
    overt_composite_test_producing.to_csv(op.join(overt_data_path, "train_test_split", "overt_composite_test_producing.csv"), index=False)

    #############################################################################################################################################

    # covert reading vs covert producing - to be split into pure vowels and consonents + vowels: 2 cases

    covert_covert_pure = covert_dataframe[covert_dataframe["Vowel_Type"] == "pure"]
    covert_covert_all = covert_dataframe

    covert_covert_pure_train, covert_covert_pure_test = train_test_split(
        covert_covert_pure,
        test_size=0.1, 
        stratify=covert_covert_pure["Speech_Type"],
        random_state=42,
        shuffle=True,
    )
    covert_covert_all_train, covert_covert_all_test = train_test_split(
        covert_covert_all, 
        test_size=0.1, 
        stratify=covert_covert_all["Speech_Type"],
        random_state=42,
        shuffle=True,
    )

    for df in [covert_covert_pure_train, covert_covert_pure_test, covert_covert_all_train, covert_covert_all_test]:
        df = df.drop_columns(["Label", "Vowel_Type"]) # drop unnecessary columns
        df = df.rename(columns={"Speech_Type": "Label"}) # we wnat the speech type to be the label in this case
        df["Label"] = df["Label"].astype(int) # type enforcement
        df["FileName"] = df["FileName"].astype(str) # type enforcement


    covert_covert_pure_train.to_csv(op.join(covert_covert_path, "train_test_split", "covert_covert_pure_train.csv"), index=False)
    covert_covert_pure_test.to_csv(op.join(covert_covert_path, "train_test_split", "covert_covert_pure_test.csv"), index=False)
    covert_covert_all_train.to_csv(op.join(covert_covert_path, "train_test_split", "covert_covert_all_train.csv"), index=False)
    covert_covert_all_test.to_csv(op.join(covert_covert_path, "train_test_split", "covert_covert_all_test.csv"), index=False)

    #############################################################################################################################################

    # covert vs overt - to be split into reading vs producing, and producing vs producing - to be split into pure vowels and consonants + vowels: 4 cases

    # covert producing vs overt producing

    covert_producing_pure = covert_dataframe[(covert_dataframe["Vowel_Type"] == "pure") & (covert_dataframe["Speech_Type"] == 1)]
    covert_producing_all = covert_dataframe[covert_dataframe["Speech_Type"] == 1]

    # make sure there is the same amount of samples for both overt and covert
    n_samples = len(overt_dataframe)

    covert_producing_pure = covert_producing_pure.sample(n_samples, random_state=42)
    covert_producing_pure = covert_producing_pure.reset_index(drop=True)

    covert_prod_overt_prod_pure = pd.concat([covert_producing_pure, overt_dataframe], ignore_index=True)
    covert_prod_overt_prod_pure_train, covert_prod_overt_prod_pure_test = train_test_split(
        covert_prod_overt_prod_pure,
        test_size=0.1,
        stratify=covert_prod_overt_prod_pure["Speech_Type"],
        random_state=42,
        shuffle=True,
    )

    covert_producing_all = covert_producing_all.sample(n_samples, random_state=42)
    covert_producing_all = covert_producing_all.reset_index(drop=True)

    covert_prod_overt_prod_all = pd.concat([covert_producing_all, overt_dataframe], ignore_index=True)

    covert_prod_overt_prod_train_all, covert_prod_overt_prod_test_all = train_test_split(
        covert_prod_overt_prod_all,
        test_size=0.1, 
        stratify=covert_prod_overt_prod_all["Speech_Type"],
        random_state=42,
        shuffle=True,
    )
    for df in [covert_prod_overt_prod_train_all, covert_prod_overt_prod_test_all, covert_prod_overt_prod_pure_train, covert_prod_overt_prod_pure_test]:
        df = df.drop_columns(["Label", "Vowel_Type"]) # drop unnecessary columns
        df = df.rename(columns={"Speech_Type": "Label"}) # we wnat the speech type to be the label in this case
        df["Label"] = df["Label"].astype(int) # type enforcement
        df["FileName"] = df["FileName"].astype(str) # type enforcement

    covert_prod_overt_prod_train_all.to_csv(op.join(covert_overt_path, "train_test_split", "covert_prod_overt_prod_train.csv"), index=False)
    covert_prod_overt_prod_test_all.to_csv(op.join(covert_overt_path, "train_test_split", "covert_prod_overt_prod_test.csv"), index=False)
    covert_prod_overt_prod_pure_train.to_csv(op.join(covert_overt_path, "train_test_split", "covert_prod_overt_prod_pure_train.csv"), index=False)
    covert_prod_overt_prod_pure_test.to_csv(op.join(covert_overt_path, "train_test_split", "covert_prod_overt_prod_pure_test.csv"), index=False)
    #############################################################################################################################################

    # covert reading vs overt producing

    covert_reading_pure = (
        covert_dataframe
        .query("Vowel_Type == 'pure' and Speech_Type == 'reading'")
        .sample(n=n_samples, random_state=42)
        .reset_index(drop=True)
    )
    covert_reading_all = (
        covert_dataframe
        .query("Speech_Type == 'reading'")
        .sample(n_samples, random_state=42)
        .reset_index(drop=True)
    )

    covert_read_overt_prod_pure = pd.concat([covert_reading_pure, overt_dataframe], ignore_index=True)

    covert_read_overt_prod_pure_train, covert_read_overt_prod_pure_test = train_test_split(
        covert_read_overt_prod_pure,
        test_size=0.1,
        stratify=covert_read_overt_prod_pure["Speech_Type"],
        random_state=42,
        shuffle=True,
    )

    covert_read_overt_prod_all = pd.concat([covert_reading_all, overt_dataframe], ignore_index=True)
    covert_read_overt_prod_train_all, covert_read_overt_prod_test_all = train_test_split(
        covert_read_overt_prod_all,
        test_size=0.1, 
        stratify=covert_read_overt_prod_all["Speech_Type"],
        random_state=42,
        shuffle=True,
    )
    for df in [covert_read_overt_prod_train_all, covert_read_overt_prod_test_all, covert_read_overt_prod_pure_train, covert_read_overt_prod_pure_test]:
        df = df.drop_columns(["Label", "Vowel_Type"]) # drop unnecessary columns
        df = df.rename(columns={"Speech_Type": "Label"}) # we wnat the speech type to be the label in this case
        df["Label"] = df["Label"].astype(int) # type enforcement
        df["FileName"] = df["FileName"].astype(str) # type enforcement

    covert_read_overt_prod_train_all.to_csv(op.join(covert_overt_path, "train_test_split", "covert_read_overt_prod_train.csv"), index=False)
    covert_read_overt_prod_test_all.to_csv(op.join(covert_overt_path, "train_test_split", "covert_read_overt_prod_test.csv"), index=False)

    covert_read_overt_prod_pure_train.to_csv(op.join(covert_overt_path, "train_test_split", "covert_read_overt_prod_pure_train.csv"), index=False)
    covert_read_overt_prod_pure_test.to_csv(op.join(covert_overt_path, "train_test_split", "covert_read_overt_prod_pure_test.csv"), index=False)

    #############################################################################################################################################

if __name__ == "__main__":
    main()
    # python train_test_split.py --data_type covert --root /path/to/root











