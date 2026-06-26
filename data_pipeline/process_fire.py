"""
Process NASA FIRMS VIIRS Collection 2 fire archive shapefile
into analysis-ready Parquet + summary CSV.

Input : D:/research/data/processed/fire_archive_SV-C2_764287.shp
Output: D:/research/data/processed/fire_viirs_2018.parquet
        D:/research/data/processed/fire_summary.csv
"""

import pyogrio
import pandas as pd
import numpy as np
import os

SHP   = "D:/research/data/processed/fire_archive_SV-C2_764287.shp"
PQOUT = "D:/research/data/processed/fire_viirs_2018.parquet"
CSVOUT= "D:/research/data/processed/fire_summary.csv"
CHUNK = 2_000_000   # rows per chunk


def process():
    total = 19_170_193
    chunks = []

    print(f"Reading {total:,} records in chunks of {CHUNK:,} ...")
    for start in range(0, total, CHUNK):
        n = min(CHUNK, total - start)
        df = pyogrio.read_dataframe(
            SHP,
            columns=["LATITUDE","LONGITUDE","BRIGHTNESS","SCAN","TRACK",
                     "ACQ_DATE","ACQ_TIME","CONFIDENCE","VERSION",
                     "BRIGHT_T31","FRP","DAYNIGHT","TYPE"],
            skip_features=start,
            max_features=n,
            use_arrow=True,
        )

        # Drop geometry (lat/lon already in columns)
        if hasattr(df, "geometry"):
            df = df.drop(columns="geometry", errors="ignore")

        # Cast types
        df["ACQ_DATE"]   = pd.to_datetime(df["ACQ_DATE"])
        df["ACQ_TIME"]   = df["ACQ_TIME"].astype(str).str.zfill(4)
        df["CONFIDENCE"] = df["CONFIDENCE"].map({"l":"low","n":"nominal","h":"high"}).fillna(df["CONFIDENCE"])
        df["DAYNIGHT"]   = df["DAYNIGHT"].map({"D":"day","N":"night"}).fillna(df["DAYNIGHT"])

        chunks.append(df)
        pct = min(start + CHUNK, total) / total * 100
        print(f"  {pct:5.1f}%  ({start + len(df):,} rows read)", end="\r")

    print()
    df = pd.concat(chunks, ignore_index=True)
    print(f"Total rows: {len(df):,}")

    # Save Parquet
    df.to_parquet(PQOUT, index=False, compression="snappy")
    size_mb = os.path.getsize(PQOUT) / 1e6
    print(f"Parquet saved: {PQOUT}  ({size_mb:.0f} MB)")

    # Summary
    summary = {
        "total_detections" : len(df),
        "date_min"         : str(df["ACQ_DATE"].min().date()),
        "date_max"         : str(df["ACQ_DATE"].max().date()),
        "lat_min"          : df["LATITUDE"].min(),
        "lat_max"          : df["LATITUDE"].max(),
        "lon_min"          : df["LONGITUDE"].min(),
        "lon_max"          : df["LONGITUDE"].max(),
        "frp_mean_mw"      : df["FRP"].mean(),
        "frp_max_mw"       : df["FRP"].max(),
        "brightness_mean_k": df["BRIGHTNESS"].mean(),
    }

    print("\n=== Dataset Summary ===")
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")

    # Monthly counts
    monthly = (
        df.groupby(df["ACQ_DATE"].dt.month)
          .size()
          .rename_axis("month")
          .reset_index(name="detections")
    )
    monthly["month_name"] = pd.to_datetime(monthly["month"], format="%m").dt.strftime("%b")
    print("\n=== Monthly Fire Detections ===")
    print(monthly[["month_name","detections"]].to_string(index=False))

    # Confidence distribution
    print("\n=== Confidence ===")
    print(df["CONFIDENCE"].value_counts().to_string())

    # Type distribution
    type_map = {0:"vegetation fire", 1:"volcano", 2:"other land", 3:"offshore"}
    df["TYPE_NAME"] = df["TYPE"].map(type_map)
    print("\n=== Fire Type ===")
    print(df["TYPE_NAME"].value_counts().to_string())

    # Save summary CSV
    pd.DataFrame([summary]).to_csv(CSVOUT, index=False)
    print(f"\nSummary saved: {CSVOUT}")

    return df


if __name__ == "__main__":
    process()
