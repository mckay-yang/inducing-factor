import sys
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def plot_rainfall(nc_file: str, output_img: str = "rainfall_plot.png"):
    ds = xr.open_dataset(nc_file)
    data_vars = list(ds.data_vars)
    if not data_vars:
        raise ValueError("No data variables found in the dataset")
    rain_var = next((var for var in ["pre", "pr", "precip", "tp"] if var in data_vars), data_vars[0])
    da = ds[rain_var]
    da_plot = da.mean(dim="time") if "time" in da.dims else da
    plt.figure(figsize=(10, 6))
    da_plot.plot(cmap="Blues", cbar_kwargs={"label": f"{rain_var} units"})
    plt.title(f"Rainfall Distribution ({rain_var} - Mean)")
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    ds.close()


if __name__ == "__main__":
    plot_rainfall("/Volumes/T9/Datasets/rainfall/pre_2017.nc")
