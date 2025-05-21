# parallel_percentile_chunk.py

import os
import sys
import numpy as np

def compute_file_percentiles(np_file):
    p_max, p_min = [], []
    epoch = np.load(np_file, mmap_mode="r")  # shape: (sensors, scales x time)
    for sensor in epoch:
        p_max.append(np.percentile(sensor, 99))
        p_min.append(np.percentile(sensor, 1))
    return p_min, p_max

if __name__ == "__main__":
    data_dir = sys.argv[1]         # path to dir with .npy files
    chunk_index = int(sys.argv[2]) # SLURM_ARRAY_TASK_ID
    total_chunks = int(sys.argv[3])# total number of array jobs
    output_file = sys.argv[4]      # where to save this job's output

    all_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".npy")
    ])

    chunks = np.array_split(all_files, total_chunks)
    files = chunks[chunk_index]

    global_min, global_max = [], []

    for path in files:
        try:
            pmin, pmax = compute_file_percentiles(path)
            global_min.extend(pmin)
            global_max.extend(pmax)
        except Exception as e:
            print(f"ERROR in {path}: {e}")

    with open(output_file, "w") as f:
        for min_val, max_val in zip(global_min, global_max):
            f.write(f"{min_val},{max_val}\n")