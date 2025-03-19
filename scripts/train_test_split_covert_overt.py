import os
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser



def is_csv(filename):
    return filename.lower().endswith(".csv")


def main():
    parser = ArgumentParser(description="This script takes the labels and makes train_validation_test splits for the data in quesiton")

    parser.add_argument('--data_dir', type=str, required=True, help="Path to where the label csvs are")
    parser.add_argument('--save_dir', type=str, required=True, help="path where the output csvs will be stored")
    parser.add_argument('--dimensions', type=str, required=True, help="the dimensions, formatted like they are in the labels file")
    
    
    args = parser.parse_args()

    label_files = os.listdir(args.data_dir)
    
    for f in label_files:
        if is_csv(f):

            print(f"processing {f}")
            df = pd.read_csv(os.path.join(args.data_dir, f)).reset_index(drop=True)

            total_samples = len(df)
            print(f"Total samples: {total_samples}")

            #make a new dataframe
            new_df = pd.DataFrame()

            # get all of the overt samples
            new_df = pd.concat([new_df, df[df["Label"] == 0]], axis=0)

            # subsample the covert samples
            df_label_1 = df[df["Label"] == 1].sample(n=len(new_df), random_state=42).reset_index(drop=True)
            
            # add them to the new dataframe
            new_df = pd.concat([new_df, df_label_1], axis=0)

            new_df.to_csv(os.path.join(args.save_dir, f"covert_overt_{args.dimensions}_Kfold_train.csv"), index=False)


if __name__ == "__main__":
    main()