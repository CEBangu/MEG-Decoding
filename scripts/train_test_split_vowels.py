import os
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

# I think this should still work if we have ONLY producing samples. 

def main():
    parser = ArgumentParser(description="This script takes the labels and makes train_validation_test splits for the data in quesiton")

    parser.add_argument('--data_dir', type=str, required=True, help="Path to where the label csvs are")
    parser.add_argument('--save_dir', type=str, required=True, help="path where the output csvs will be stored")
    parser.add_argument('--dimensions', type=str, required=True, help="the dimensions, formatted like they are in the labels file")
    
    
    args = parser.parse_args()

    label_files = os.listdir(args.data_dir)
    
    for f in label_files:

        print(f"processing {f}")
        df = pd.read_csv(os.path.join(args.data_dir, f)).reset_index(drop=True)

        total_samples = len(df)
        print(f"Total samples: {total_samples}")

        # we want 10% of total to be in the test set
        test_amount = total_samples // 10

        # but we need it only from the producing samples, so we need a lil df magic
        # so take all of the producing samples out of df and put them in a new one
        producing_df = pd.DataFrame() # empty df to hold producing 
        for index, row in df.iterrows():
            if len(row.iloc[0].split("_")[4]) >= 3:
                producing_df = pd.concat([producing_df, pd.DataFrame([row])], ignore_index=True)
                df.drop(index, inplace=True)
        print(f"producing samples: {len(producing_df)}")
        


        # splitting to right percentages
        percentage_for_test = test_amount/len(producing_df)
        print(f"Percentage for test (or dataloss): {percentage_for_test}")


        train_df, test_df = train_test_split(producing_df, test_size=percentage_for_test, random_state=42)
        
        df_identifiers = set(df.iloc[:, 0])  # first column is a unique identifier
        test_df_identifiers = set(test_df.iloc[:, 0])

        contamination = df_identifiers.intersection(test_df_identifiers)
        if contamination:
            print(f"WARNING: Data contamination detected! {len(contamination)} overlapping samples found between df and test_df.")
            print(contamination)
            raise ValueError("Data contamination detected between training and test sets!")

        # now we add the producing samples that we train on back to the dataframe
        df = pd.concat([df, train_df], ignore_index=True) 
        print(f"Total samples in the new train dataset: {len(df)}")


        # Saving to new CSV files
        unpacked_filename = f.split("_")
        if "producing" in unpacked_filename:
            filename_train = f"{unpacked_filename[0]}_producing_{args.dimensions}_Kfold_train.csv"
            filename_test = f"{unpacked_filename[0]}_producing_{args.dimensions}_test.csv"
        else:
            filename_train = f'{unpacked_filename[0]}_all_{args.dimensions}_Kfold_train.csv'
            filename_test = f'{unpacked_filename[0]}_all_{args.dimensions}_test.csv'

        df.to_csv(os.path.join(args.save_dir, filename_train), index=False)  # 90% of data
        test_df.to_csv(os.path.join(args.save_dir, filename_test), index=False)    # 10% of data

        print(f"Training set: {len(df)} samples")
        print(f"Test set: {len(test_df)} samples")


if __name__ == "__main__":
    main()