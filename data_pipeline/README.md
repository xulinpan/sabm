# Getting the REX-SUB Dataset

**Product:** MOD05_L2 — MODIS/Terra Total Precipitable Water Vapor, Collection 6.1  
**Granule:** `MOD05_L2.A2023173.0120.061.2023173131455.hdf`  
**Date:** 2023-06-22 01:20 UTC (day-of-year 173)  
**Region:** lon [−57.0, −24.5], lat [29.9, 51.3] (western North Atlantic)  
**N:** 2,748,620 valid pixels at 1 km × 1 km  

---

## Step 1 — Create a NASA Earthdata account (free, 2 minutes)

1. Go to https://urs.earthdata.nasa.gov/users/new
2. Register and verify your email

---

## Step 2 — Get your LAADS App Token

1. Log in at https://ladsweb.modaps.eosdis.nasa.gov/profile/#app-keys
2. Click **Generate Token**
3. Copy the token string

---

## Step 3 — Download the granule

### Option A — Python (recommended)

```bash
pip install earthaccess h5py pyhdf numpy pandas scipy

# Download via earthaccess (reads ~/.netrc or prompts for credentials)
python download_modis.py --method earthaccess --outdir data/raw

# OR download with your LAADS token
python download_modis.py --method requests --token YOUR_TOKEN --outdir data/raw
```

### Option B — wget (one-liner)

```bash
wget -c \
  --header "Authorization: Bearer YOUR_TOKEN" \
  -O "data/raw/MOD05_L2.A2023173.0120.061.2023173131455.hdf" \
  "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD05_L2/2023/173/MOD05_L2.A2023173.0120.061.2023173131455.hdf"
```

### Option C — R

```r
source("rexsub_data.R")
download_granule(token = "YOUR_TOKEN")
```

---

## Step 4 — Process HDF → CSV

```bash
# Python
python process_modis.py
```

```r
# R
source("rexsub_data.R")
splits <- run_pipeline()
summarise_data(splits$train, splits$val)
plot_precipitable_water(splits$train)
```

Outputs:
| File | Description |
|------|-------------|
| `data/modis_tpw.csv` | Full dataset (N rows, cols: lon, lat, log_tpw) |
| `data/modis_tpw_train.csv` | 90% training split |
| `data/modis_tpw_val.csv` | 10% validation split |

---

## Dataset Details

| Field | Value |
|-------|-------|
| HDF SDS variable | `Water_Vapor_Near_Infrared` |
| Algorithm | Near-infrared (1 km, daytime) |
| Scale factor | 0.001 (raw × 0.001 = precipitable water in cm) |
| Fill value | 65535 |
| Transform | log(tpw) — applied before GP fitting |
| Geolocation | Upscaled from 5 km via bilinear interpolation |

---

## Direct Archive URL

```
https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD05_L2/2023/173/
```

Browse the full directory listing at that URL (requires login).
