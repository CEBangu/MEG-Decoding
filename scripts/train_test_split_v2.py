import os
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

# I think this should still work if we have ONLY producing samples. 

def main():
    parser = ArgumentParser(description="This script takes the labels and makes train_validation_test splits for the data in quesiton")

    parser.add_argument('--dimensions', type=str, required=True, help="the dimensions, formatted like they are in the labels file")
    parser.add_argument('--data_path', type=str, required=True, help="path where the labels csv is stored")
    parser.add_argument('--output_path', type=str, required=True, help="path where the output csvs will be stored")
    
    args = parser.parse_args()

    df = pd.read_csv(args.data_path).reset_index(drop=True)

    total_samples = len(df)
    print(f"Total samples: {total_samples}")

    # we want 10% of total to be in the test set
    test_amount = total_samples // 10

    # but we need it only from the producing samples, so we need a lil df magic
    # so take all of the producing samples out of df and put them in a new one
    producing_df = pd.DataFrame() # empty df to hold producing 
    for index, row in df.iterrows():
        if len(row.iloc[0].split("_")[4]) >=3:
            producing_df = pd.concat([producing_df, pd.DataFrame([row])], ignore_index=True)
            df.drop(index, inplace=True)
    


    # splitting to right percentages
    percentage_for_test = test_amount/len(producing_df)
    print(f"Percentage for test (or dataloss): {percentage_for_test}")


    train_df, test_df = train_test_split(producing_df, test_size=percentage_for_test, random_state=42)
    print(f"Train: {len(train_df)}")
    print(f"Test: {len(test_df)}")
    print(test_amount)  

    # now we add the producing samples that we train on back to the dataframe
    df = pd.concat([df, train_df], ignore_index=True) 
    print(f"Total samples in the dataset: {len(df)}")


    # Saving to new CSV files
    filename_train_combined = f"combined_labels_{args.dimensions}_Kfold_train.csv"
    filename_test_combined = f"combined_labels_{args.dimensions}_test.csv"

    train_df.to_csv(os.path.join(args.output_path, filename_train_combined), index=False)  # 90% of data
    test_df.to_csv(os.path.join(args.output_path, filename_test_combined), index=False)    # 10% of data

    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")


if __name__ == "__main__":
    main()