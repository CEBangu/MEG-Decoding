import os
import sys
import numpy as np
from collections import defaultdict

def compute_file_percentiles(np_file):
    p_max, p_min = [], []
    data = np.load(np_file, mmap_mode="r")
    for sensor in data:
        p_max.append(np.percentile(sensor, 99))
        p_min.append(np.percentile(sensor, 1))
    return p_min, p_max

def extract_roi_name(filename):
    # Example: BCOM_01_coefficients_broca_0.npy → 'broca'
    parts = filename.split("_")
    for roi in ["broca", "sma", "stg", "mtg", "spt"]:
        if roi in parts:
            return roi
    return "unknown"

if __name__ == "__main__":
    data_dir = sys.argv[1]
    chunk_index = int(sys.argv[2])
    total_chunks = int(sys.argv[3])
    output_dir = sys.argv[4]

    all_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".npy")
    ])

    chunks = np.array_split(all_files, total_chunks)
    files = chunks[chunk_index]

    # Dictionary of roi → [values]
    roi_min = defaultdict(list)
    roi_max = defaultdict(list)

    for path in files:
        try:
            roi = extract_roi_name(os.path.basename(path))
            pmin, pmax = compute_file_percentiles(path)
            roi_min[roi].extend(pmin)
            roi_max[roi].extend(pmax)
        except Exception as e:
            print(f"ERROR in {path}: {e}")

    # Save per-ROI CSV
    for roi in roi_min:
        outfile = os.path.join(output_dir, f"roi_{roi}_chunk{chunk_index}.csv")
        with open(outfile, "w") as f:
            for min_val, max_val in zip(roi_min[roi], roi_max[roi]):
                f.write(f"{min_val},{max_val}\n")