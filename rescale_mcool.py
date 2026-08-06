import cooler
import h5py
import numpy as np

mcool_path = "your.mcool"
resolution = 10000
uri = f"{mcool_path}::resolutions/{resolution}"

clr = cooler.Cooler(uri)

region = (chrom, start, end)

# 1) 
mat = clr.matrix(balance=True).fetch(region)

# 2) 
with h5py.File(mcool_path, "r") as f:
    scale = f[f"resolutions/{resolution}/bins/weight"].attrs["scale"]

# 3) 
mat = mat * float(scale)

# 4) 
mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
mat = np.maximum(mat, 0.0)
target = np.log1p(mat).astype(np.float32)