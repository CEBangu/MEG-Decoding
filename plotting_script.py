import numpy as np
from plotting import ScalogramPlotter
from argparse import ArgumentParser
import os
import re
import time



def main():
    parser = ArgumentParser(description="Plot scalograms from a directory of data")

    parser.add_argument("--data_dir", type=str, help="Directory of data to plot")
    parser.add_argument("--save_dir", type=str, help="Directory to save the plots")
    parser.add_argument("--dimensions", type=int, nargs=2, default=[16, 16], help="Dimensions of the plot")
    parser.add_argument("--cmap", type=str, default="turbo", help="Colormap to use")
    parser.add_argument("--index_list", type=int, nargs="+", default=list(range(247)), help="List of indices to plot")
    parser.add_argument("--resolution", type=int, default=224, help="Resolution of the plot")
    parser.add_argument("--average", action="store_true", help="Whether to average the data before plotting")
    args = parser.parse_args()

    # instnatiate the plotter
    plotter = ScalogramPlotter(
                dimensions=args.dimensions, 
                cmap=args.cmap, 
                data_dir=args.data_dir, 
                save_dir=args.save_dir, 
                index_list=args.index_list,
                resolution=args.resolution,
                average=args.average
                )

    # loop through the files in the data directory
    for filename in os.listdir(args.data_dir):
        if filename.endswith(".npy"):
            file_path = os.path.join(args.data_dir, filename)
            data = np.load(file_path, mmap_mode="r")
            for epoch_index, epoch in enumerate(data):
                new_filename = re.sub(r'_coefficients.*', f'_scalogram_{epoch_index}', filename)
                if plotter.average:
                    start_time = time.time()
                    scalogram = plotter.plot_average_scalogram(epoch)
                    scalogram.save_plot(new_filename)
                    end_time = time.time()
                    print(f"Time taken for epoch {epoch_index}: {end_time - start_time} seconds")
                else:
                    start_time = time.time()
                    scalogram = plotter.plot_many(epoch)
                    plotter.save_plot(filename=new_filename, fig=scalogram)
                    end_time = time.time()
                    print(f"Time taken for epoch {epoch_index}: {end_time - start_time} seconds")
            print(f"Finished plotting {filename}")
    print("Finished plotting all files")

if __name__ == "__main__":
    main()
    