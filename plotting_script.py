import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotting import ScalogramPlotter
from argparse import ArgumentParser
import re
import time
from joblib import Parallel, delayed

def process_epoch_sensor(epoch_index, epoch, filename, args, plotter):
    new_filename = re.sub(r'_coefficients.*', f'_scalogram_{epoch_index}', filename)
    start_time = time.time()
    if plotter.average:
        scalogram = plotter.plot_average_scalogram(epoch)
    else:
        scalogram = plotter.plot_many(epoch)
    plotter.save_plot(filename=new_filename, fig=scalogram)
    end_time = time.time()
    print(f"Time taken for epoch {epoch_index} of file {filename}: {end_time - start_time} seconds")


def process_epoch_roi(epoch_index, epoch, filename, args, plotter):
    new_filename = re.sub(r'_coefficients.*', f'_scalogram_{epoch_index}', filename)
    start_time = time.time()
    epoch = epoch[:, epoch_index, :, :]
    if plotter.average:
        scalogram = plotter.plot_average_scalogram(epoch)
    else:
        scalogram = plotter.plot_roi(epoch)
    plotter.save_plot(filename=new_filename, fig=scalogram)
    end_time = time.time()
    print(f"Time taken for epoch {epoch_index} of file {filename}: {end_time - start_time} seconds")

def process_file(filename, args, plotter):
    file_path = os.path.join(args.data_dir, filename)
    if not os.path.exists(file_path):
        print(f"ERROR: File {file_path} does not exist!")
        return
    try:
        data = np.load(file_path, mmap_mode="r")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return
    if data.shape[1] == 247: # kind of hacky, but it should work - need a way of gating the two and I don't want to add another argument
    # but every sensor job will have 247 sensors in the np array
        Parallel(n_jobs=args.epoch_workers)(
            delayed(process_epoch_sensor)(epoch_index, epoch, filename, args, plotter)
            for epoch_index, epoch in enumerate(data)
        )
    else: #if it doesn't, then it's an ROI job 
        Parallel(n_jobs=args.epoch_workers)(
            delayed(process_epoch_roi)(i, data, filename, args, plotter)
            for i in range(data.shape[1])
        )
    print(f"Finished plotting {filename}")

def main():
    parser = ArgumentParser(description="Plot scalograms from a directory of data")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of data to plot")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the plots")
    parser.add_argument("--dimensions", type=int, nargs=2, default=[16, 16], help="Dimensions of the plot")
    parser.add_argument("--cmap", type=str, default="turbo", help="Colormap to use")
    parser.add_argument("--index_list", type=str, default=",".join(map(str, range(247))),
                        help="Comma-separated list of indices to plot")
    parser.add_argument("--resolution", type=int, default=224, help="Resolution of the plot")
    parser.add_argument("--average", action="store_true", help="Whether to average the data before plotting")
    parser.add_argument("--epoch_workers", type=int, default=1, help="Number of workers to process epochs in parallel")
    parser.add_argument("--task_id", type=int, default=0, help="SLURM array task id")
    args = parser.parse_args()

    # Convert comma-separated index list to a list of integers.
    args.index_list = list(map(int, args.index_list.split(",")))

    plotter = ScalogramPlotter(
        dimensions=args.dimensions,
        cmap=args.cmap,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        index_list=args.index_list,
        resolution=args.resolution,
        average=args.average
    )

    # Get a sorted list of all .npy files.
    files = sorted([file for file in os.listdir(args.data_dir) if file.endswith(".npy")])
    if not files:
        print("No .npy files found in the data directory.")
        return

    # Determine total number of array tasks from the environment variable.
    total_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    
    # Process every total_tasks-th file starting from task_id.
    for i in range(args.task_id, len(files), total_tasks):
        print(f"Task {args.task_id} processing file: {files[i]}")
        process_file(files[i], args, plotter)

if __name__ == "__main__":
    main()