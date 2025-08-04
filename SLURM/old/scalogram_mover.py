from argparse import ArgumentParser
import os
import shutil

def main():
    parser = ArgumentParser()
    parser.add_argument("--overt_path", type=str, required=True)
    parser.add_argument("--covert_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--reading", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    for file in os.listdir(args.overt_path):
        new_file_name = file.split(".")[0] + "_overt." + file.split(".")[1]
        shutil.copy(os.path.join(args.overt_path, file), os.path.join(args.output_path, new_file_name))
        print(f"Copied {file} to {new_file_name}")
    
    for file in os.listdir(args.covert_path):
        if args.reading:
            if len(file.split("_")[4]) < 3:
                new_file_name = file.split(".")[0] + "_covert." + file.split(".")[1]
                shutil.copy(os.path.join(args.covert_path, file), os.path.join(args.output_path, new_file_name))
                print(f"Copied {file} to {new_file_name}")
        else:
            if len(file.split("_")[4]) >= 3:
                new_file_name =file.split(".")[0] + "_covert." + file.split(".")[1]
                shutil.copy(os.path.join(args.covert_path, file), os.path.join(args.output_path, new_file_name))
                print(f"Copied {file} to {new_file_name}")

if __name__ == "__main__":
    main()