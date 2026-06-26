"""
Process the downloaded MOD05_L2 HDF file into the CSV used by REX-SUB.

Reproduces the dataset described in Rios & Lee (2026), Section 5:
  - Total precipitable water at 1 km x 1 km resolution
  - Region: lon [-57.0, -24.5], lat [29.9, 51.3]
  - Log-transformed: log(Total_Precipitable_Water)
  - N = 2,748,620 valid (non-zero, non-fill) locations

Output
------
  data/modis_tpw.csv      — full dataset (N rows, 3 cols: lon, lat, log_tpw)
  data/modis_tpw_train.csv — 90% training set  (used by REX-SUB)
  data/modis_tpw_val.csv  — 10% validation set

Requirements
------------
    pip install pyhdf numpy pandas scipy
"""

import os
import numpy as np
import pandas as pd

HDF_FILE  = "data/raw/MOD05_L2.A2023173.0120.061.2023173131455.hdf"
OUT_DIR   = "data"

# Bounding box from the paper (Section 5)
LON_MIN, LON_MAX = -57.0, -24.5
LAT_MIN, LAT_MAX =  29.9,  51.3

# Random seed for train/val split (matches standard practice)
RANDOM_SEED = 2023


def read_hdf(hdf_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read Longitude, Latitude, and Water_Vapor_Near_Infrared from MOD05_L2 HDF.

    The near-infrared water vapor field is at 1 km resolution (daytime only).
    Scale factor: 0.001  (raw integer × 0.001 = precipitable water in cm)
    Fill value  : 65535
    Valid range : [0, 30000]
    """
    try:
        from pyhdf.SD import SD, SDC
    except ImportError:
        raise ImportError(
            "Install pyhdf:  pip install pyhdf\n"
            "  (On Windows you may need:  pip install python-hdf4)"
        )

    print(f"Reading: {hdf_path}")
    hdf = SD(hdf_path, SDC.READ)

    # List available scientific datasets
    datasets = hdf.datasets()
    print(f"Available SDS: {list(datasets.keys())}")

    # Longitude & Latitude (5 km arrays — we upscale to match 1 km NIR data)
    lon_5km = hdf.select("Longitude").get().astype(np.float32)
    lat_5km = hdf.select("Latitude").get().astype(np.float32)

    # Water vapor at 1 km (Near-Infrared algorithm)
    wv_sds   = hdf.select("Water_Vapor_Near_Infrared")
    wv_raw   = wv_sds.get()
    wv_attrs = wv_sds.attributes()

    scale  = wv_attrs.get("scale_factor", 0.001)
    offset = wv_attrs.get("add_offset",   0.0)
    fill   = wv_attrs.get("_FillValue",   65535)

    print(f"Water vapor shape (1 km): {wv_raw.shape}")
    print(f"Lat/Lon shape   (5 km):   {lon_5km.shape}")

    hdf.end()

    # ── Upscale 5 km lat/lon to 1 km via repeat ──────────────────────────────
    # MOD05_L2 geolocation is provided every 5 pixels; interpolate to 1 km
    from scipy.ndimage import zoom

    scale_factor = wv_raw.shape[0] / lon_5km.shape[0]
    lon_1km = zoom(lon_5km, scale_factor, order=1)
    lat_1km = zoom(lat_5km, scale_factor, order=1)

    # Trim to exact water-vapor array size if zoom introduces rounding
    rows, cols = wv_raw.shape
    lon_1km = lon_1km[:rows, :cols]
    lat_1km = lat_1km[:rows, :cols]

    # ── Apply scale and mask fill values ─────────────────────────────────────
    wv_physical = np.where(wv_raw == fill, np.nan, wv_raw * scale - offset)

    return lon_1km.ravel(), lat_1km.ravel(), wv_physical.ravel()


def apply_bounding_box(
    lon: np.ndarray, lat: np.ndarray, wv: np.ndarray
) -> pd.DataFrame:
    """Subset to the paper's geographic region and drop invalid pixels."""
    mask = (
        (lon >= LON_MIN) & (lon <= LON_MAX) &
        (lat >= LAT_MIN) & (lat <= LAT_MAX) &
        np.isfinite(wv)  &
        (wv > 0)          # zero = clear sky / no valid retrieval
    )
    df = pd.DataFrame({
        "lon": lon[mask],
        "lat": lat[mask],
        "tpw": wv[mask],
    })
    print(f"Valid pixels in bounding box: {len(df):,}")
    return df


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform the precipitable water values.
    The paper (Section 5) applies log transformation to ensure support
    on the real line for the Gaussian process model.
    """
    df = df.copy()
    df["log_tpw"] = np.log(df["tpw"])
    return df.drop(columns=["tpw"])


def train_val_split(df: pd.DataFrame, val_frac: float = 0.10) -> tuple:
    """
    Random 90/10 train/validation split.
    Matches the paper (Section 5): '10% of observations randomly for validation'.
    """
    rng    = np.random.default_rng(RANDOM_SEED)
    idx    = rng.permutation(len(df))
    n_val  = int(len(df) * val_frac)
    val_df = df.iloc[idx[:n_val]].reset_index(drop=True)
    trn_df = df.iloc[idx[n_val:]].reset_index(drop=True)
    return trn_df, val_df


def main():
    if not os.path.exists(HDF_FILE):
        print(
            f"[ERROR] HDF file not found: {HDF_FILE}\n"
            "Run download_modis.py first:\n"
            "  python download_modis.py --method earthaccess"
        )
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Read raw data
    lon, lat, wv = read_hdf(HDF_FILE)

    # 2. Spatial subset + drop invalid
    df = apply_bounding_box(lon, lat, wv)

    # 3. Log transform
    df = log_transform(df)
    print(f"\nLog-TPW summary:")
    print(df["log_tpw"].describe().round(4))

    # 4. Save full dataset
    full_path = os.path.join(OUT_DIR, "modis_tpw.csv")
    df.to_csv(full_path, index=False, float_format="%.6f")
    print(f"\nFull dataset  → {full_path}  ({len(df):,} rows)")

    # 5. Train / validation split
    train_df, val_df = train_val_split(df)
    trn_path = os.path.join(OUT_DIR, "modis_tpw_train.csv")
    val_path = os.path.join(OUT_DIR, "modis_tpw_val.csv")
    train_df.to_csv(trn_path, index=False, float_format="%.6f")
    val_df.to_csv(val_path,   index=False, float_format="%.6f")
    print(f"Training set  → {trn_path}  ({len(train_df):,} rows)")
    print(f"Validation set→ {val_path}  ({len(val_df):,} rows)")

    # 6. Quick sanity check
    assert len(train_df) + len(val_df) == len(df), "Split sizes don't add up"
    print("\nDone. Data ready for REX-SUB.")


if __name__ == "__main__":
    main()
