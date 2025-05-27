# parallel_percentile_chunk_4d.py

import os
import sys
import numpy as np

def compute_file_percentiles(np_file):
    """
    Given a .npy file of shape (n_epochs, n_sensors, n_scales, n_times),
    return two lists:
      - p_min: the 1st percentile for each (epoch, sensor)
      - p_max: the 99th percentile for each (epoch, sensor)
    """
    # load in read-only mode
    data = np.load(np_file, mmap_mode="r")
    # data.shape == (n_epochs, n_sensors, n_scales, n_times)

    # vectorized version: compute percentiles over the last two axes
    # result shape will be (n_epochs, n_sensors)
    p_min = np.percentile(data, 1, axis=(2, 3))
    p_max = np.percentile(data, 99, axis=(2, 3))

    # flatten to 1D lists
    return p_min.ravel().tolist(), p_max.ravel().tolist()


if __name__ == "__main__":
    data_dir      = sys.argv[1]           # path to dir with .npy files
    chunk_index   = int(sys.argv[2])      # SLURM_ARRAY_TASK_ID (0-based)
    total_chunks  = int(sys.argv[3])      # total number of array jobs
    output_file   = sys.argv[4]           # e.g. percentiles_chunk_0.csv

    # gather & sort all .npy files
    all_files = sorted(
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith(".npy")
    )

    # split list of files into `total_chunks` roughly-equal chunks
    chunks = np.array_split(all_files, total_chunks)
    # pick just this job’s files
    files = chunks[chunk_index]

    global_min = []
    global_max = []

    for path in files:
        try:
            pmin, pmax = compute_file_percentiles(path)
            global_min.extend(pmin)
            global_max.extend(pmax)
        except Exception as e:
            # log the path that failed
            print(f"ERROR processing {path}: {e}", file=sys.stderr)

    # write out as CSV: one line per epoch×sensor
    with open(output_file, "w") as f:
        for lo, hi in zip(global_min, global_max):
            f.write(f"{lo:.6e},{hi:.6e}\n")