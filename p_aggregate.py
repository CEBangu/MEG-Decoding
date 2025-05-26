# aggregate_percentiles.py
import pandas as pd
import glob

# Collect all per-job outputs
files = sorted(glob.glob("percentiles/percentiles_OP/percentile_*.csv"))
df = pd.concat([pd.read_csv(f, header=None) for f in files])

# Columns: 0 = 1st percentile, 1 = 99th percentile
global_1st = df[0].mean()
global_99th = df[1].mean()

print("Overt Producing")
print(f"Global 1st percentile (mean):  {global_1st:.4f}")
print(f"Global 99th percentile (mean): {global_99th:.4f}")

files = sorted(glob.glob("percentiles/percentiles_CP/percentile_*.csv"))
df = pd.concat([pd.read_csv(f, header=None) for f in files])

# Columns: 0 = 1st percentile, 1 = 99th percentile
global_1st = df[0].mean()
global_99th = df[1].mean()

print("Covert Producing")
print(f"Global 1st percentile (mean):  {global_1st:.4f}")
print(f"Global 99th percentile (mean): {global_99th:.4f}")

files = sorted(glob.glob("percentiles/percentiles_CR/percentile_*.csv"))
df = pd.concat([pd.read_csv(f, header=None) for f in files])

# Columns: 0 = 1st percentile, 1 = 99th percentile
global_1st = df[0].mean()
global_99th = df[1].mean()

print("Covert_Reading")
print(f"Global 1st percentile (mean):  {global_1st:.4f}")
print(f"Global 99th percentile (mean): {global_99th:.4f}")

rois = ["sma", "broca", "stg", "mtg", "spt"]

print("Covert Producing")
for roi in rois:
    files = sorted(glob.glob(f"percentiles/percentiles_roi_CP/roi_{roi}_*.csv"))
    if not files:
        continue
    df = pd.concat([pd.read_csv(f, header=None) for f in files])
    print(f"{roi.upper()}: min= {df[0].mean():.4f}, max= {df[1].mean():.4f}")

print("Covert Reading")
for roi in rois:
    files = sorted(glob.glob(f"percentiles/percentiles_roi_CR/roi_{roi}_*.csv"))
    if not files:
        continue
    df = pd.concat([pd.read_csv(f, header=None) for f in files])
    print(f"{roi.upper()}: min= {df[0].mean():.4f}, max= {df[1].mean():.4f}")

print("Overt Producing")
for roi in rois:
    files = sorted(glob.glob(f"percentiles/percentiles_roi_OP/roi_{roi}_*.csv"))
    if not files:
        continue
    df = pd.concat([pd.read_csv(f, header=None) for f in files])
    print(f"{roi.upper()}: min= {df[0].mean():.4f}, max= {df[1].mean():.4f}")