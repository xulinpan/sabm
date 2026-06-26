# load_scabm_data.R
# Reads the processed NEON mosquito dataset into SC-ABM model structures.
#
# Returns a list with:
#   grid      -- spatial domain (m, coords, dist, neighbours)
#   X         -- (m x 4) habitat covariate matrix for GP prior alpha
#   W         -- (m x 2) covariate matrix for log(K) and log(psi) LDD
#   obs_C     -- data.frame: cell_idx, time_idx, y_C, raw_count,
#                            prop_identified, trap_hours, trap_type_co2
#   obs_P     -- data.frame: cell_idx, time_idx, y_P, trap_hours
#   time_idx  -- data.frame: time_idx, date_CDT, day_of_year
#   meta      -- list (site, species, m, T, season, channel_design, ...)

load_scabm_data <- function(dir = "D:/research/data/processed") {

  # ── Grid ────────────────────────────────────────────────────────────────────
  grid_df <- read.csv(file.path(dir, "mos_grid.csv"), stringsAsFactors = FALSE)
  m       <- nrow(grid_df)

  dist_raw <- read.csv(file.path(dir, "mos_dist_km.csv"),
                       row.names = 1, check.names = FALSE)
  dist_km  <- as.matrix(dist_raw)
  colnames(dist_km) <- rownames(dist_km) <- seq_len(m) - 1L

  neigh_df <- read.csv(file.path(dir, "mos_neighbors.csv"))
  # 0-indexed in file; convert to 1-indexed lists for R
  neighbours <- vector("list", m)
  for (i in seq_len(m)) {
    nb <- neigh_df$neighbor_idx[neigh_df$cell_idx == (i - 1L)] + 1L
    neighbours[[i]] <- sort(nb)
  }

  grid <- list(
    m          = m,
    plot_id    = grid_df$plot_id,
    lat        = grid_df$lat,
    lon        = grid_df$lon,
    elevation  = grid_df$elevation_m,
    nlcd       = grid_df$nlcd_class,
    is_wetland = grid_df$is_wetland,
    is_forest  = grid_df$is_forest,
    dist_km    = dist_km,
    neighbours = neighbours
  )

  # ── Covariate matrices ───────────────────────────────────────────────────────
  X <- as.matrix(grid_df[, c("X_intercept", "X_elev_z", "X_wetland", "X_forest")])
  W <- as.matrix(grid_df[, c("W_intercept", "W_elev_z")])
  rownames(X) <- rownames(W) <- grid_df$plot_id

  # ── Time index ───────────────────────────────────────────────────────────────
  time_idx <- read.csv(file.path(dir, "mos_time_index.csv"),
                       stringsAsFactors = FALSE)
  time_idx$date_CDT <- as.Date(time_idx$date_CDT)
  T <- nrow(time_idx)

  # ── Observations ─────────────────────────────────────────────────────────────
  obs_C <- read.csv(file.path(dir, "mos_obs_C.csv"), stringsAsFactors = FALSE)
  obs_P <- read.csv(file.path(dir, "mos_obs_P.csv"), stringsAsFactors = FALSE)

  # Convert to 1-indexed (R convention)
  obs_C$cell_idx <- obs_C$cell_idx + 1L
  obs_C$time_idx <- obs_C$time_idx + 1L
  obs_P$cell_idx <- obs_P$cell_idx + 1L
  obs_P$time_idx <- obs_P$time_idx + 1L

  # ── Observation matrices (m x T, NA = not surveyed) ─────────────────────────
  Y_C <- matrix(NA_integer_, nrow = m, ncol = T)
  for (k in seq_len(nrow(obs_C))) {
    Y_C[obs_C$cell_idx[k], obs_C$time_idx[k]] <- obs_C$y_C[k]
  }

  Y_P <- matrix(NA_integer_, nrow = m, ncol = T)
  for (k in seq_len(nrow(obs_P))) {
    Y_P[obs_P$cell_idx[k], obs_P$time_idx[k]] <- obs_P$y_P[k]
  }

  # Trap hours matrices (for detection offset log(rho) = b0 + b1*log(h))
  trap_hours_C <- matrix(NA_real_, nrow = m, ncol = T)
  for (k in seq_len(nrow(obs_C))) {
    trap_hours_C[obs_C$cell_idx[k], obs_C$time_idx[k]] <- obs_C$trap_hours[k]
  }
  trap_hours_P <- matrix(NA_real_, nrow = m, ncol = T)
  for (k in seq_len(nrow(obs_P))) {
    trap_hours_P[obs_P$cell_idx[k], obs_P$time_idx[k]] <- obs_P$trap_hours[k]
  }

  # ── Metadata ─────────────────────────────────────────────────────────────────
  meta <- jsonlite::fromJSON(file.path(dir, "mos_summary.json"))

  message(sprintf(
    "Loaded NEON SC-ABM data: site=%s, species=%s, m=%d, T=%d",
    meta$site, meta$focal_species, m, T
  ))
  message(sprintf(
    "  O_C: %d obs (%d pos / %d zeros),  coverage %.0f%%",
    nrow(obs_C),
    sum(obs_C$y_C > 0, na.rm = TRUE),
    sum(obs_C$y_C == 0, na.rm = TRUE),
    meta$observation$coverage_C_pct
  ))
  message(sprintf(
    "  O_P: %d obs (%d pres / %d abs),  coverage %.0f%%",
    nrow(obs_P),
    sum(obs_P$y_P == 1, na.rm = TRUE),
    sum(obs_P$y_P == 0, na.rm = TRUE),
    meta$observation$coverage_P_pct
  ))

  list(
    grid         = grid,
    X            = X,
    W            = W,
    time_idx     = time_idx,
    obs_C        = obs_C,
    obs_P        = obs_P,
    Y_C          = Y_C,
    Y_P          = Y_P,
    trap_hours_C = trap_hours_C,
    trap_hours_P = trap_hours_P,
    m            = m,
    T            = T,
    meta         = meta
  )
}

# ── Quick sanity check if run directly ──────────────────────────────────────
if (!interactive()) {
  library(jsonlite)
  dat <- load_scabm_data()

  cat("\n--- Grid ---\n")
  print(data.frame(cell = seq_len(dat$m),
                   plot = dat$grid$plot_id,
                   lat  = round(dat$grid$lat, 4),
                   lon  = round(dat$grid$lon, 4),
                   nlcd = dat$grid$nlcd,
                   n_nb = sapply(dat$grid$neighbours, length)))

  cat("\n--- Y_C matrix (overnight counts, m x T) ---\n")
  print(dat$Y_C)

  cat("\n--- Y_P matrix (daytime binary, m x T) ---\n")
  print(dat$Y_P)

  cat("\n--- X covariate matrix ---\n")
  print(round(dat$X, 3))
}
