import numpy as np
from plotting import ScalogramPlotter
from argparse import ArgumentParser
import os
import re
import time
from joblib import Parallel, delayed

def process_file(filename, args, plotter):
    file_path = os.path.join(args.data_dir, filename)
    
    try:
        data = np.load(file_path, mmap_mode="r")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return


    for epoch_index, epoch in enumerate(data):
        new_filename = re.sub(r'_coefficients.*', f'_scalogram_{epoch_index}', filename)
        start_time = time.time()

        if plotter.average:
            scalogram = plotter.plot_average_scalogram(epoch)        
        else:
            scalogram = plotter.plot_many(epoch)
    
        plotter.save_plot(filename=new_filename, fig=scalogram)
            
        end_time = time.time()

        print(f"Time taken for epoch {epoch_index}: {end_time - start_time} seconds")
        
    print(f"Finished plotting {filename}")

def main():
    parser = ArgumentParser(description="Plot scalograms from a directory of data")

    parser.add_argument("--data_dir", type=str, help="Directory of data to plot")
    parser.add_argument("--save_dir", type=str, help="Directory to save the plots")
    parser.add_argument("--dimensions", type=int, nargs=2, default=[16, 16], help="Dimensions of the plot")
    parser.add_argument("--cmap", type=str, default="turbo", help="Colormap to use")
    parser.add_argument("--index_list", type=str, default=",".join(map(str, range(247))), help="List of indices to plot")
    parser.add_argument("--resolution", type=int, default=224, help="Resolution of the plot")
    parser.add_argument("--average", action="store_true", help="Whether to average the data before plotting")
    args = parser.parse_args()

    # post parse
    args.index_list = list(map(int, args.index_list.split(",")))
    # instantiate the plotter
    plotter = ScalogramPlotter(
                dimensions=args.dimensions, 
                cmap=args.cmap, 
                data_dir=args.data_dir, 
                save_dir=args.save_dir, 
                index_list=args.index_list,
                resolution=args.resolution,
                average=args.average
                )

    files = [file for file in os.listdir(args.data_dir) if file.endswith(".npy")]

    num_workers = min(os.cpu_count(), len(files))

    Parallel(n_jobs=num_workers)(
        delayed(process_file)(file, args, plotter) for file in files
        )

    print("finished plotting all files")

if __name__ == "__main__":
    main()
    