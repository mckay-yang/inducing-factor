import os
from datetime import date

import numpy as np
import pandas as pd


def build_synthetic_panel(random_state=42):
    rng = np.random.default_rng(random_state)
    records = []
    years = list(range(2015, 2022))
    n_points = 160
    for point_id in range(n_points):
        x = 95.0 + rng.random()
        y = 29.0 + rng.random()
        elevation = rng.normal(3600, 250)
        slope = rng.uniform(5, 35)
        aspect = rng.uniform(0, 360)
        dist_to_road = rng.uniform(200, 8000)
        dist_to_river = rng.uniform(100, 9000)
        base_rain = rng.normal(4.5, 0.8)
        base_temp = rng.normal(11.0, 1.5)
        resident_level = rng.uniform(0, 2)
        landuse = 1 if rng.random() < 0.45 else 2
        for year in years:
            temperature = base_temp + rng.normal(0, 0.8)
            resident = resident_level + rng.normal(0, 0.3)
            disturbed = bool((dist_to_road < 2500 and rng.random() < 0.45) or (dist_to_road < 1200 and rng.random() < 0.2))
            disturbance_month = int(rng.integers(4, 13)) if disturbed else np.nan
            monthly_rainfall = {}
            for month in range(1, 13):
                rainfall = base_rain + rng.normal(0, 0.7)
                if dist_to_road < 2500:
                    rainfall += 0.6
                if disturbed and month in {disturbance_month - 1, disturbance_month - 2, disturbance_month - 3}:
                    rainfall += 1.8
                monthly_rainfall[f"rainfall_month_{month:02d}"] = rainfall
            if dist_to_road < 2500:
                resident += 1.0
                temperature += 0.4

            disturbance_value = 0.0
            disturbance_class = 0
            if disturbed:
                day_of_year = int((date(year, disturbance_month, 15) - date(year, 1, 1)).days + 1)
                days_in_year = 366 if date(year, 12, 31).toordinal() - date(year, 1, 1).toordinal() + 1 == 366 else 365
                disturbance_value = year + (day_of_year / days_in_year)
                disturbance_class = 4 if dist_to_road < 1200 else 3 if slope > 28 else 1

            records.append(
                {
                    "point_id": point_id,
                    "x": x,
                    "y": y,
                    "year": year,
                    "elevation": elevation,
                    "slope": slope,
                    "aspect": aspect,
                    "dist_to_road": dist_to_road,
                    "dist_to_river": dist_to_river,
                    "temperature": temperature,
                    "resident": resident,
                    "landuse": landuse,
                    "disturbance_time_value": disturbance_value,
                    "disturbance_type": disturbance_value,
                    "disturbance_flag": int(disturbance_value > 0),
                    "disturbance_class": disturbance_class,
                    "disturbance_month": disturbance_month,
                    **monthly_rainfall,
                }
            )
    return pd.DataFrame(records)


def save_as_npz_chunks(df, cache_dir):
    half = len(df) // 2
    for idx, part in enumerate([df.iloc[:half].copy(), df.iloc[half:].copy()]):
        np.savez_compressed(os.path.join(cache_dir, f"chunk_{idx}.npz"), **{col: part[col].to_numpy() for col in part.columns})
