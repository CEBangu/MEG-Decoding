# parallel_percentile_chunk_roi.py

import os
import sys
import numpy as np
from collections import defaultdict

# 1) Define your index→ROI mapping here, in the order
#    that the first axis of your .npy files is laid out.
index2roi = {
    0: "sma",
    1: "broca",
    2: "stg",
    3: "mtg",
    4: "spt",
    # add more if you have them...
}

def compute_file_percentiles(np_file):
    """
    Given a .npy file of shape (n_rois, n_scales, n_times),
    return two dicts mapping roi_name→list_of_1st-percentiles
                                    and roi_name→list_of_99th-percentiles.
    """
    data = np.load(np_file, mmap_mode="r")
    # data.shape == (n_rois, n_scales, n_times)

    pmins = defaultdict(list)
    pmaxs = defaultdict(list)

    for roi_idx, arr in enumerate(data):
        roi_name = index2roi.get(roi_idx, f"roi{roi_idx}")
        # flatten over scales×times and get percentiles
        pmin = np.percentile(arr, 1)
        pmax = np.percentile(arr, 99)
        pmins[roi_name].append(pmin)
        pmaxs[roi_name].append(pmax)

    return pmins, pmaxs


if __name__ == "__main__":
    data_dir     = sys.argv[1]        # directory of .npy files
    chunk_index  = int(sys.argv[2])   # e.g. $SLURM_ARRAY_TASK_ID
    total_chunks = int(sys.argv[3])   # total array jobs
    output_dir   = sys.argv[4]        # where to write roi_{name}_chunk{idx}.csv

    os.makedirs(output_dir, exist_ok=True)

    # gather & chunk the file list
    all_files = sorted(
        os.path.join(data_dir, fn)
        for fn in os.listdir(data_dir)
        if fn.endswith(".npy")
    )
    chunks = np.array_split(all_files, total_chunks)
    my_files = chunks[chunk_index]

    # accumulate per‐ROI across all files in this chunk
    chunk_pmin = defaultdict(list)
    chunk_pmax = defaultdict(list)

    for path in my_files:
        try:
            pmins, pmaxs = compute_file_percentiles(path)
            # merge into our chunk aggregates
            for roi in pmins:
                chunk_pmin[roi].extend(pmins[roi])
                chunk_pmax[roi].extend(pmaxs[roi])
        except Exception as e:
            print(f"ERROR processing {path}: {e}", file=sys.stderr)

    # write one file per ROI for this chunk
    for roi, mins in chunk_pmin.items():
        maxs = chunk_pmax[roi]
        outpath = os.path.join(output_dir, f"roi_{roi}_chunk{chunk_index}.csv")
        with open(outpath, "w") as f:
            for lo, hi in zip(mins, maxs):
                f.write(f"{lo:.6e},{hi:.6e}\n")