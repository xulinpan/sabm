# =============================================================================
# REX-SUB Paper — Data Download & Preparation in R
# Rios & Lee (2026), Computational Statistics and Data Analysis 224, 108404
#
# This script reproduces the MODIS dataset used in Section 5 of the paper.
#
# Requirements:
#   install.packages(c("rhdf5", "tidyverse", "sf", "terra"))
#   BiocManager::install("rhdf5")   # or: install.packages("BiocManager")
#
# Alternatively use the Python scripts (download_modis.py + process_modis.py)
# and load the output CSV here.
# =============================================================================

# ── 0. Configuration ─────────────────────────────────────────────────────────

HDF_FILE   <- "data/raw/MOD05_L2.A2023173.0120.061.2023173131455.hdf"
CSV_FULL   <- "data/modis_tpw.csv"
CSV_TRAIN  <- "data/modis_tpw_train.csv"
CSV_VAL    <- "data/modis_tpw_val.csv"

# Bounding box — Section 5 of the paper
LON_MIN <- -57.0;  LON_MAX <- -24.5
LAT_MIN <-  29.9;  LAT_MAX <-  51.3

GRANULE_URL <- paste0(
  "https://ladsweb.modaps.eosdis.nasa.gov",
  "/archive/allData/61/MOD05_L2/2023/173",
  "/MOD05_L2.A2023173.0120.061.2023173131455.hdf"
)

set.seed(2023)


# ── 1. Download (token method) ────────────────────────────────────────────────
# Get your free LAADS App Token at:
#   https://ladsweb.modaps.eosdis.nasa.gov/profile/#app-keys
# Then replace "YOUR_TOKEN_HERE" below.

download_granule <- function(token, outdir = "data/raw") {
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  outfile <- file.path(outdir, basename(GRANULE_URL))

  if (file.exists(outfile)) {
    message("Already downloaded: ", outfile)
    return(invisible(outfile))
  }

  message("Downloading: ", basename(GRANULE_URL), " ...")
  res <- tryCatch(
    download.file(
      url      = GRANULE_URL,
      destfile = outfile,
      headers  = c("Authorization" = paste("Bearer", token)),
      mode     = "wb",
      method   = "libcurl"
    ),
    error = function(e) {
      message("Download failed: ", conditionMessage(e))
      message("\nManual alternative — run in terminal:")
      message(sprintf(
        'wget -c --header "Authorization: Bearer %s" -O "%s" "%s"',
        token, outfile, GRANULE_URL
      ))
      return(-1L)
    }
  )
  if (res == 0L) message("Saved: ", outfile) else message("Download error.")
  invisible(outfile)
}

# Uncomment and set your token to download:
# download_granule(token = "YOUR_TOKEN_HERE")


# ── 2. Read & process the HDF file ───────────────────────────────────────────

read_modis_hdf <- function(hdf_path) {
  if (!requireNamespace("rhdf5", quietly = TRUE)) {
    stop(
      "Package rhdf5 needed.\n",
      "  BiocManager::install('rhdf5')\n",
      "  or: install.packages('BiocManager'); BiocManager::install('rhdf5')"
    )
  }

  message("Reading: ", hdf_path)

  # List contents
  h5ls_result <- rhdf5::h5ls(hdf_path)
  message("Available datasets:")
  print(h5ls_result[, c("name", "otype", "dclass", "dim")])

  # Read fields (HDF-EOS group path)
  grp <- "//MODIS_SWATH_Type_L2/Data Fields/"

  wv_raw <- rhdf5::h5read(hdf_path, paste0(grp, "Water_Vapor_Near_Infrared"))
  lon_5k <- rhdf5::h5read(hdf_path, "//MODIS_SWATH_Type_L2/Geolocation Fields/Longitude")
  lat_5k <- rhdf5::h5read(hdf_path, "//MODIS_SWATH_Type_L2/Geolocation Fields/Latitude")

  # Scale factor and fill value (standard for MOD05_L2 Collection 6.1)
  scale_factor <- 0.001
  fill_value   <- 65535L

  message("WV shape (1km): ", paste(dim(wv_raw), collapse = " x "))
  message("Lat/Lon shape (5km): ", paste(dim(lon_5k), collapse = " x "))

  list(wv_raw = wv_raw, lon_5k = lon_5k, lat_5k = lat_5k,
       scale  = scale_factor, fill  = fill_value)
}


upscale_geolocation <- function(lon_5k, lat_5k, target_rows, target_cols) {
  # Upscale 5-km geolocation to 1-km grid via bilinear interpolation
  # using the terra package (fast raster operations)
  if (!requireNamespace("terra", quietly = TRUE)) {
    stop("Install terra:  install.packages('terra')")
  }

  row_factor <- target_rows / nrow(lon_5k)
  col_factor <- target_cols / ncol(lon_5k)

  lon_r <- terra::rast(lon_5k)
  lat_r <- terra::rast(lat_5k)

  lon_1k <- terra::disagg(lon_r, fact = c(row_factor, col_factor), method = "bilinear")
  lat_1k <- terra::disagg(lat_r, fact = c(row_factor, col_factor), method = "bilinear")

  # Crop to exact target size
  lon_out <- terra::values(lon_1k)[seq_len(target_rows * target_cols)]
  lat_out <- terra::values(lat_1k)[seq_len(target_rows * target_cols)]

  list(lon = lon_out, lat = lat_out)
}


process_modis <- function(hdf_path) {
  raw  <- read_modis_hdf(hdf_path)
  nrow <- nrow(raw$wv_raw)
  ncol <- ncol(raw$wv_raw)

  # Physical values
  wv <- as.numeric(raw$wv_raw)
  wv[wv == raw$fill] <- NA
  wv <- wv * raw$scale

  # Upscale geolocation
  geo <- upscale_geolocation(raw$lon_5k, raw$lat_5k, nrow, ncol)

  data.frame(
    lon = geo$lon,
    lat = geo$lat,
    tpw = wv
  )
}


# ── 3. Spatial subset + log transform ────────────────────────────────────────

prepare_dataset <- function(df) {
  # Bounding box filter
  df <- df[
    !is.na(df$tpw)         &
    df$tpw   > 0           &
    df$lon   >= LON_MIN    &
    df$lon   <= LON_MAX    &
    df$lat   >= LAT_MIN    &
    df$lat   <= LAT_MAX,
  ]
  df$log_tpw <- log(df$tpw)
  df$tpw     <- NULL
  message("Valid pixels in bounding box: ", format(nrow(df), big.mark = ","))
  df
}


# ── 4. Train / validation split (90/10) ──────────────────────────────────────

split_dataset <- function(df, val_frac = 0.10) {
  n      <- nrow(df)
  idx    <- sample(n)
  n_val  <- floor(n * val_frac)
  list(
    train = df[idx[(n_val + 1):n], ],
    val   = df[idx[seq_len(n_val)], ]
  )
}


# ── 5. Main pipeline ──────────────────────────────────────────────────────────

run_pipeline <- function() {
  dir.create("data", showWarnings = FALSE)

  # ── Option A: load from pre-processed CSV (after running Python scripts) ──
  if (file.exists(CSV_TRAIN) && file.exists(CSV_VAL)) {
    message("Loading pre-processed CSVs ...")
    train <- read.csv(CSV_TRAIN)
    val   <- read.csv(CSV_VAL)
    message("Train: ", format(nrow(train), big.mark = ","), " rows")
    message("Val:   ", format(nrow(val),   big.mark = ","), " rows")
    return(list(train = train, val = val))
  }

  # ── Option B: process HDF directly in R ───────────────────────────────────
  if (!file.exists(HDF_FILE)) {
    stop(
      "HDF file not found: ", HDF_FILE, "\n",
      "Run download_modis.py first, or set your token and call:\n",
      "  download_granule(token = 'YOUR_TOKEN')"
    )
  }

  df     <- process_modis(HDF_FILE)
  df     <- prepare_dataset(df)
  splits <- split_dataset(df)

  write.csv(df,            CSV_FULL,  row.names = FALSE)
  write.csv(splits$train,  CSV_TRAIN, row.names = FALSE)
  write.csv(splits$val,    CSV_VAL,   row.names = FALSE)

  message("Full dataset → ", CSV_FULL,  " (", format(nrow(df), big.mark=","), " rows)")
  message("Training set → ", CSV_TRAIN, " (", format(nrow(splits$train), big.mark=","), " rows)")
  message("Validation   → ", CSV_VAL,   " (", format(nrow(splits$val),   big.mark=","), " rows)")

  splits
}


# ── 6. Quick exploratory summary ─────────────────────────────────────────────

summarise_data <- function(train, val) {
  cat("\n=== Dataset Summary (matches REX-SUB paper, Section 5) ===\n\n")
  cat(sprintf("Training observations  : %s\n", format(nrow(train), big.mark=",")))
  cat(sprintf("Validation observations: %s\n", format(nrow(val),   big.mark=",")))
  cat(sprintf("Total N                : %s\n", format(nrow(train)+nrow(val), big.mark=",")))
  cat(sprintf("Longitude range        : [%.1f, %.1f]\n", LON_MIN, LON_MAX))
  cat(sprintf("Latitude  range        : [%.1f, %.1f]\n", LAT_MIN, LAT_MAX))
  cat("\nLog-TPW (training):\n")
  print(summary(train$log_tpw))
}


# ── 7. Reproduce Figure 2 from the paper ──────────────────────────────────────

plot_precipitable_water <- function(train) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    message("Install ggplot2 for plot:  install.packages('ggplot2')")
    return(invisible(NULL))
  }
  library(ggplot2)

  p <- ggplot(train[sample(nrow(train), min(50000, nrow(train))), ],
              aes(x = lon, y = lat, colour = log_tpw)) +
    geom_point(size = 0.1, alpha = 0.6) +
    scale_colour_gradientn(
      colours = c("#000080", "#0000FF", "#00FFFF", "#FFFF00", "#FF0000"),
      name    = "log(TPW)"
    ) +
    coord_fixed(ratio = 1) +
    labs(
      title    = "Total Precipitable Water — MODIS Terra, 2023-06-22",
      subtitle = "MOD05_L2 Collection 6.1 | log-transformed",
      x        = "Longitude",
      y        = "Latitude",
      caption  = "Rios & Lee (2026) Figure 2 reproduction"
    ) +
    theme_minimal(base_size = 11)

  ggsave("data/modis_tpw_plot.png", p, width = 8, height = 6, dpi = 200)
  message("Plot saved: data/modis_tpw_plot.png")
  print(p)
  invisible(p)
}


# ── Run ───────────────────────────────────────────────────────────────────────
# Uncomment to execute:
# splits <- run_pipeline()
# summarise_data(splits$train, splits$val)
# plot_precipitable_water(splits$train)
