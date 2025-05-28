import os
import pandas as pd
import numpy as np
import os.path as op
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
        if ".npy" in file:
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

def dataframe_to_train_test_covert(df_covert_prod, df_covert_read, roi=None, pure=False, prod=False):
    if roi:
        df_covert_prod = df_covert_prod[df_covert_prod['FileName'].str.contains(roi)]
        df_covert_read = df_covert_read[df_covert_read['FileName'].str.contains(roi)]
    
    if prod:
        df = df_covert_prod
        
    else:    
        reading_sample = df_covert_read.sample(n=len(df_covert_prod), random_state=42)
        df = pd.concat([df_covert_prod, reading_sample])
    
    if pure:
        df = df[df['Vowel_Type']=="pure"]
    x, y = train_test_split(
        df, 
        test_size=0.1,
        stratify=df["Label"],
        random_state=42,
        shuffle=True,
    )
    
    dataframes = [x, y]
    for i, split in enumerate(dataframes):
        split["Label"] = split["Label"].astype(int) # type enforcement
        split["FileName"] = split["FileName"].astype(str) # type enforcement
        split = split.drop(columns=["Speech_Type", "Vowel_Type"]) # drop unnecessary columns
        split = split.reset_index(drop=True)
        dataframes[i] = split
        
    
    (
      x, y
    ) = dataframes
    
    return x, y

def dataframe_to_train_test_covert_covert(df_covert_prod, df_covert_read, roi=None, pure=False):
    if roi:
        df_covert_prod = df_covert_prod[df_covert_prod['FileName'].str.contains(roi)]
        df_covert_read = df_covert_read[df_covert_read['FileName'].str.contains(roi)]
        
        reading_sample = df_covert_read.sample(n=len(df_covert_prod), random_state=42)
        df = pd.concat([df_covert_prod, reading_sample])
    
    else:    
        reading_sample = df_covert_read.sample(n=len(df_covert_prod), random_state=42)
        df = pd.concat([df_covert_prod, reading_sample])
    
    if pure:
        df = df[df['Vowel_Type']=="pure"]
        
    x, y = train_test_split(
        df, 
        test_size=0.1,
        stratify=df["Speech_Type"],
        random_state=42,
        shuffle=True,
    )
    
    dataframes = [x, y]
    for i, split in enumerate(dataframes):
        split = split.drop(columns=["Label", "Vowel_Type"]) # drop unnecessary columns
        split = split.rename(columns={"Speech_Type": "Label"}) # drop unnecessary columns
        split["Label"] = split["Label"].astype(int) # type enforcement
        split["FileName"] = split["FileName"].astype(str) # type enforcement
        split = split.reset_index(drop=True)
        dataframes[i] = split
        
    
    (
      x, y
    ) = dataframes
    
    return x, y

def dataframe_to_train_test_covert_overt(df_covert_prod, df_covert_read, df_overt_prod, roi=None, pure=False, reading=False):
    if roi:
        df_covert_prod = df_covert_prod[df_covert_prod['FileName'].str.contains(roi)]
        df_covert_read = df_covert_read[df_covert_read['FileName'].str.contains(roi)]
        df_overt_prod = df_overt_prod[df_overt_prod['FileName'].str.contains(roi)]
    
    if reading:
        reading_sample = df_covert_read.sample(n=len(df_overt_prod), random_state=42)
        df = pd.concat([df_overt_prod, reading_sample])
    else:
        producing_sample = df_covert_prod.sample(n=len(df_overt_prod), random_state=42)
        df = pd.concat([df_overt_prod, producing_sample])
    
    if pure:
        df = df[df['Vowel_Type']=="pure"]
        
    df['covert_overt'] = df['FileName'].apply(lambda x: 1 if "covert" in x else 0)
        
    x, y = train_test_split(
        df, 
        test_size=0.1,
        stratify=df["covert_overt"],
        random_state=42,
        shuffle=True,
    )
    
    dataframes = [x, y]
    for i, split in enumerate(dataframes):
        split = split.drop(columns=["Label", "Vowel_Type", "Speech_Type"]) # drop unnecessary columns
        split = split.rename(columns={"covert_overt": "Label"}) # drop unnecessary columns
        split["Label"] = split["Label"].astype(int) # type enforcement
        split["FileName"] = split["FileName"].astype(str) # type enforcement
        split = split.reset_index(drop=True)
        dataframes[i] = split
        
    
    (
      x, y
    ) = dataframes
    
    return x, y

def dataframe_to_train_test_overt(df_overt_prod, roi=None, pure=False):
    if roi:
        df_overt_prod = df_overt_prod[df_overt_prod['FileName'].str.contains(roi)]
    
    df = df_overt_prod.copy()
    
    if pure:
        df = df[df['Vowel_Type']=="pure"]
        
    x, y = train_test_split(
        df, 
        test_size=0.1,
        stratify=df["Label"],
        random_state=42,
        shuffle=True,
    )
    
    dataframes = [x, y]
    for i, split in enumerate(dataframes):
        split["Label"] = split["Label"].astype(int) # type enforcement
        split["FileName"] = split["FileName"].astype(str) # type enforcement
        split = split.drop(columns=["Speech_Type", "Vowel_Type"]) # drop unnecessary columns
        split = split.reset_index(drop=True)
        dataframes[i] = split
        
    
    (
      x, y
    ) = dataframes
    
    return x, y


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

df_covert_prod = dir_to_dataframe(
    data_dir = "/pasteur/appa/scratch/cbangu/coefficients/covert_producing",
    vowel2id=vowel2id,
    speech2id=speech2id,
    voweltype2id=voweltype2id
)
df_covert_read = dir_to_dataframe(
    data_dir = "/pasteur/appa/scratch/cbangu/coefficients/covert_reading",
    vowel2id=vowel2id,
    speech2id=speech2id,
    voweltype2id=voweltype2id
)
df_overt_prod = dir_to_dataframe(
    data_dir = "/pasteur/appa/scratch/cbangu/coefficients/overt_producing",
    vowel2id=vowel2id,
    speech2id=speech2id,
    voweltype2id=voweltype2id
)
#######
# then just enumerate the cases - starting with sensor space

# covert vowels - reading and prodcuing
cpr_train, cpr_test = dataframe_to_train_test_covert(
    df_covert_prod=df_covert_prod, 
    df_covert_read=df_covert_read,
    roi=None,
    pure=False,
    prod=False,
)
cpr_train.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert/covert_composite_readprod_train.csv")
cpr_test.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert/covert_composite_readprod_test.csv")

del cpr_train, cpr_test

cpr_train_pure, cpr_test_pure = dataframe_to_train_test_covert(
    df_covert_prod=df_covert_prod, 
    df_covert_read=df_covert_read,
    roi=None,
    pure=True,
    prod=False,
)
cpr_train_pure.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert/covert_pure_readprod_train.csv")
cpr_test_pure.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert/covert_pure_readprod_test.csv")
del cpr_train_pure, cpr_test_pure

# covert vowels - prodcuing only
cp_train, cp_test = dataframe_to_train_test_covert(
    df_covert_prod=df_covert_prod, 
    df_covert_read=df_covert_read,
    roi=None,
    pure=False,
    prod=True,
)
cp_train.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert/covert_composite_producing_train.csv")
cp_test.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert/covert_composite_producing_test.csv")
del cp_train, cp_test

cp_train_pure, cp_test_pure = dataframe_to_train_test_covert(
    df_covert_prod=df_covert_prod, 
    df_covert_read=df_covert_read,
    roi=None,
    pure=True,
    prod=True,
)
cp_train_pure.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert/covert_pure_producing_train.csv")
cp_test_pure.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert/covert_pure_producing_test.csv")
del cp_train_pure, cp_test_pure

# covert - covert
cc_train, cc_test = dataframe_to_train_test_covert_covert(
    df_covert_prod=df_covert_prod, 
    df_covert_read=df_covert_read,
    roi=None,
    pure=False,
)
cc_train.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert-covert/covert_composite_readprod_train.csv")
cc_test.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert-covert/covert_composite_readprod_test.csv")
del cc_train, cc_test

cc_train_pure, cc_test_pure = dataframe_to_train_test_covert_covert(
    df_covert_prod=df_covert_prod, 
    df_covert_read=df_covert_read,
    roi=None,
    pure=True,
)
cc_train_pure.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert-covert/covert_pure_readprod_train.csv")
cc_test_pure.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert-covert/covert_pure_readprod_test.csv")
del cc_train_pure, cc_test_pure

# covert - overt
crop_train, crop_test = dataframe_to_train_test_covert_overt(
    df_covert_prod=df_covert_prod, 
    df_covert_read=df_covert_read,
    df_overt_prod=df_overt_prod,
    roi=None,
    pure=False,
    reading=True,
)
crop_train.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert-overt/covert_overt_composite_readprod_train.csv")
crop_test.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert-overt/covert_overt_composite_readprod_test.csv")
del crop_train, crop_test

# covert - overt - pure
crop_train_pure, crop_test_pure = dataframe_to_train_test_covert_overt(
    df_covert_prod=df_covert_prod, 
    df_covert_read=df_covert_read,
    df_overt_prod=df_overt_prod,
    roi=None,
    pure=True,
    reading=True,
)
crop_train_pure.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert-overt/covert_overt_pure_readprod_train.csv")
crop_test_pure.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert-overt/covert_overt_pure_readprod_test.csv")
del crop_train_pure, crop_test_pure

# covert - overt - producing only
cpop_train, cpop_test = dataframe_to_train_test_covert_overt(
    df_covert_prod=df_covert_prod, 
    df_covert_read=df_covert_read,
    df_overt_prod=df_overt_prod,
    roi=None,
    pure=False,
    reading=False,
)
cpop_train.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert-overt/covert_overt_composite_producing_train.csv")
cpop_test.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert-overt/covert_overt_composite_producing_test.csv")
del cpop_train, cpop_test

# covert - overt - producing only - pure
cpop_train_pure, cpop_test_pure = dataframe_to_train_test_covert_overt(
    df_covert_prod=df_covert_prod, 
    df_covert_read=df_covert_read,
    df_overt_prod=df_overt_prod,
    roi=None,
    pure=True,
    reading=False,
)
cpop_train_pure.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert-overt/covert_overt_pure_producing_train.csv")
cpop_test_pure.to_csv("/pasteur/appa/scratch/cbangu/datasets/covert-overt/covert_overt_pure_producing_test.csv")
del cpop_train_pure, cpop_test_pure

# overt vowels - producing only
op_train, op_test = dataframe_to_train_test_overt(
    df_overt_prod=df_overt_prod, 
    roi=None,
    pure=False,
)
op_train.to_csv("/pasteur/appa/scratch/cbangu/datasets/overt/overt_composite_producing_train.csv")
op_test.to_csv("/pasteur/appa/scratch/cbangu/datasets/overt/overt_composite_producing_test.csv")
del op_train, op_test

# overt vowels - producing only - pure
op_train_pure, op_test_pure = dataframe_to_train_test_overt(
    df_overt_prod=df_overt_prod, 
    roi=None,
    pure=True,
)
op_train_pure.to_csv("/pasteur/appa/scratch/cbangu/datasets/overt/overt_pure_producing_train.csv")
op_test_pure.to_csv("/pasteur/appa/scratch/cbangu/datasets/overt/overt_pure_producing_test.csv")
del op_train_pure, op_test_pure

del df_covert_prod, df_covert_read, df_overt_prod # delete to save memory

# roi space
df_covert_prod = dir_to_dataframe(
    data_dir = "/pasteur/appa/scratch/cbangu/coefficients/covert_producing_roi_beamforming",
    vowel2id=vowel2id,
    speech2id=speech2id,
    voweltype2id=voweltype2id
)
df_covert_read = dir_to_dataframe(
    data_dir = "/pasteur/appa/scratch/cbangu/coefficients/covert_reading_roi_beamforming",
    vowel2id=vowel2id,
    speech2id=speech2id,
    voweltype2id=voweltype2id
)
df_overt_prod = dir_to_dataframe(
    data_dir = "/pasteur/appa/scratch/cbangu/coefficients/overt_producing_roi_beamforming",
    vowel2id=vowel2id,
    speech2id=speech2id,
    voweltype2id=voweltype2id
)

for roi in ["broca", "sma", "stg", "mtg", "spt"]:
    # covert vowels - reading and prodcuing
    cpr_train, cpr_test = dataframe_to_train_test_covert(
        df_covert_prod=df_covert_prod, 
        df_covert_read=df_covert_read,
        roi=roi,
        pure=False,
        prod=False,
    )
    cpr_train.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-{roi}/covert_composite_readprod_train_{roi}.csv")
    cpr_test.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-{roi}/covert_composite_readprod_test_{roi}.csv")
    
    del cpr_train, cpr_test

    cpr_train_pure, cpr_test_pure = dataframe_to_train_test_covert(
        df_covert_prod=df_covert_prod, 
        df_covert_read=df_covert_read,
        roi=roi,
        pure=True,
        prod=False,
    )
    cpr_train_pure.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-{roi}/covert_pure_readprod_train_{roi}.csv")
    cpr_test_pure.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-{roi}/covert_pure_readprod_test_{roi}.csv")
    
    del cpr_train_pure, cpr_test_pure

    # covert vowels - producing only
    cp_train, cp_test = dataframe_to_train_test_covert(
        df_covert_prod=df_covert_prod, 
        df_covert_read=df_covert_read,
        roi=roi,
        pure=False,
        prod=True,
    )
    cp_train.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-{roi}/covert_composite_producing_train_{roi}.csv")
    cp_test.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-{roi}/covert_composite_producing_test_{roi}.csv")
    del cp_train, cp_test

    cp_train_pure, cp_test_pure = dataframe_to_train_test_covert(
        df_covert_prod=df_covert_prod, 
        df_covert_read=df_covert_read,
        roi=roi,
        pure=True,
        prod=True,
    )
    cp_train_pure.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-{roi}/covert_pure_producing_train_{roi}.csv")
    cp_test_pure.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-{roi}/covert_pure_producing_test_{roi}.csv")
    del cp_train_pure, cp_test_pure

    # covert - covert
    cc_train, cc_test = dataframe_to_train_test_covert_covert(
        df_covert_prod=df_covert_prod, 
        df_covert_read=df_covert_read,
        roi=roi,
        pure=False,
    )
    cc_train.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-covert-{roi}/covert_composite_reaprod_train_{roi}.csv")
    cc_test.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-covert-{roi}/covert_composite_reaprod_test_{roi}.csv")
    del cc_train, cc_test

    cc_train_pure, cc_test_pure = dataframe_to_train_test_covert_covert(
        df_covert_prod=df_covert_prod, 
        df_covert_read=df_covert_read,
        roi=roi,
        pure=True,
    )
    cc_train_pure.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-covert-{roi}/covert_pure_reaprod_train_{roi}.csv")
    cc_test_pure.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-covert-{roi}/covert_pure_reaprod_test_{roi}.csv")
    del cc_train_pure, cc_test_pure

    # covert read - overt prod
    crop_train, crop_test = dataframe_to_train_test_covert_overt(
        df_covert_prod=df_covert_prod, 
        df_covert_read=df_covert_read,
        df_overt_prod=df_overt_prod,
        roi=roi,
        pure=False,
        reading=True,
    )
    crop_train.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-overt-{roi}/covert_overt_composite_readprod_train_{roi}.csv")
    crop_test.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-overt-{roi}/covert_overt_composite_readprod_test_{roi}.csv")
    del crop_train, crop_test

    # covert read - overt prod - pure
    crop_train_pure, crop_test_pure = dataframe_to_train_test_covert_overt(
        df_covert_prod=df_covert_prod, 
        df_covert_read=df_covert_read,
        df_overt_prod=df_overt_prod,
        roi=roi,
        pure=True,
        reading=True,
    )
    crop_train_pure.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-overt-{roi}/covert_overt_pure_readprod_train_{roi}.csv")
    crop_test_pure.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-overt-{roi}/covert_overt_pure_readprod_test_{roi}.csv")
    del crop_train_pure, crop_test_pure
    
    # covert read - overt prod - producing only
    cpop_train, cpop_test = dataframe_to_train_test_covert_overt(
        df_covert_prod=df_covert_prod, 
        df_covert_read=df_covert_read,
        df_overt_prod=df_overt_prod,
        roi=roi,
        pure=False,
        reading=False,
    )
    cpop_train.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-overt-{roi}/covert_overt_composite_producing_train_{roi}.csv")
    cpop_test.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-overt-{roi}/covert_overt_composite_producing_test_{roi}.csv")
    del cpop_train, cpop_test

    # covert read - overt prod - producing only - pure
    cpop_train_pure, cpop_test_pure = dataframe_to_train_test_covert_overt(
        df_covert_prod=df_covert_prod, 
        df_covert_read=df_covert_read,
        df_overt_prod=df_overt_prod,
        roi=roi,
        pure=True,
        reading=False,
    )
    cpop_train_pure.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-overt-{roi}/covert_overt_pure_producing_train_{roi}.csv")
    cpop_test_pure.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/covert-overt-{roi}/covert_overt_pure_producing_test_{roi}.csv")
    del cpop_train_pure, cpop_test_pure

    # overt vowels - producing only
    op_train, op_test = dataframe_to_train_test_overt(
        df_overt_prod=df_overt_prod, 
        roi=roi,
        pure=False,
    )
    op_train.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/overt-{roi}/overt_composite_producing_train_{roi}.csv")
    op_test.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/overt-{roi}/overt_composite_producing_test_{roi}.csv")
    del op_train, op_test

    # overt vowels - producing only - pure
    op_train_pure, op_test_pure = dataframe_to_train_test_overt(
        df_overt_prod=df_overt_prod, 
        roi=roi,
        pure=True,
    )
    op_train_pure.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/overt-{roi}/overt_pure_producing_train_{roi}.csv")
    op_test_pure.to_csv(f"/pasteur/appa/scratch/cbangu/datasets/overt-{roi}/overt_pure_producing_test_{roi}.csv")
    del op_train_pure, op_test_pure
