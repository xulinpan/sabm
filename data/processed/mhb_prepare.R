# mhb_prepare.R
# Format SwissTits (AHMbook) for the count ABM dual-channel observation model.
#
# Y_C[i, t]  = Great Tit count from visit 1 in year t at quadrat i  (NegBin channel)
# Y_P[i, t]  = detection indicator from visit 2 in year t at quadrat i  (Bernoulli channel)
#
# Using visits 1 and 2 from the same annual survey gives two conditionally
# independent channels: observers are different on each visit, within-season
# abundance is approximately constant, and the count/detection are measured
# separately, satisfying the model's dual-channel independence assumption.
#
# Spatial neighbour graph: queen neighbourhood on the ~6 km MHB grid (Swiss
# LV03 metres).  Threshold set to 9 km to capture the diagonal neighbours
# (6 * sqrt(2) ≈ 8.5 km) while excluding second-order cells at 12 km.

library(AHMbook)
library(spdep)

# ── 1. Load SwissTits ─────────────────────────────────────────────────────────
data("SwissTits")

# counts: [267 sites, 3 visits, 10 years, 6 species]
# Species order: Great tit, Blue tit, Coal tit, Crested tit, Marsh tit, Willow tit
cnt   <- SwissTits$counts   # [267, 3, 10, 6]
sites <- SwissTits$sites    # data.frame: siteID, coordx, coordy, elev, rlength, forest

# Years 2004-2013 (10 years)
YEARS <- 2004:2013

# Species index: Great Tit = 1 ("PARMAJ" / "Great tit")
SP <- 1
sp_name <- SwissTits$species$name[SP]
cat("Species:", sp_name, "\n")
cat("Sites:", nrow(sites), "\n")
cat("Years:", paste(YEARS, collapse = "-"), "\n")

# ── 2. Extract Great Tit counts [m, visits, T] ────────────────────────────────
m  <- nrow(sites)   # 267
TT <- dim(cnt)[3]   # 10

gt <- cnt[, , , SP]   # [267, 3, 10]
dimnames(gt) <- list(
  site  = sites$siteID,
  visit = paste0("v", 1:3),
  year  = YEARS
)

# Missingness summary
cat("\nMissingness per visit:\n")
for (v in 1:3) {
  cat(sprintf("  Visit %d: %d NAs (%.1f%%)\n", v,
              sum(is.na(gt[, v, ])),
              100 * mean(is.na(gt[, v, ]))))
}

# ── 3. Build Y_C and Y_P ──────────────────────────────────────────────────────
#
# Y_C: count from visit 1  (NegBin channel)
# Y_P: binary detection from visit 2  (Bernoulli channel)
#       → 1 if count > 0, 0 if count = 0, NA if not surveyed
Y_C <- gt[, 1, ]   # [267, 10]
Y_P <- ifelse(is.na(gt[, 2, ]), NA, as.integer(gt[, 2, ] > 0))

cat("\nY_C summary (visit 1 counts):\n")
cat(sprintf("  Non-missing: %d / %d\n", sum(!is.na(Y_C)), length(Y_C)))
cat(sprintf("  Range: [%d, %d]\n", min(Y_C, na.rm = TRUE), max(Y_C, na.rm = TRUE)))
cat(sprintf("  Mean: %.2f\n", mean(Y_C, na.rm = TRUE)))

cat("\nY_P summary (visit 2 detection):\n")
cat(sprintf("  Non-missing: %d / %d\n", sum(!is.na(Y_P)), length(Y_P)))
cat(sprintf("  Detection rate: %.1f%%\n", 100 * mean(Y_P, na.rm = TRUE)))

# ── 4. Spatial neighbour graph ────────────────────────────────────────────────
coords_m <- cbind(sites$coordx, sites$coordy)   # [267, 2] in LV03 metres

# Distance matrix (km)
D_m <- as.matrix(dist(coords_m))          # metres
D   <- D_m / 1000                         # km

# The 267 MHB quadrats are a stratified sample across Switzerland:
# median 1st-NN distance = 10 km, 4th-NN = 16 km.  A fixed threshold
# leaves many quadrats with 0-1 neighbours, breaking the immigration model.
# Use k=4 nearest neighbours so every quadrat has exactly 4 immigration
# sources — ecologically appropriate for Great Tit regional dispersal.

K_NN    <- 4L
knn_obj <- knearneigh(coords_m, k = K_NN)
nb_list <- knn2nb(knn_obj, sym = FALSE)   # directed (each site draws from k NN)
nb_idx  <- lapply(nb_list, function(x) x)

cat(sprintf("\nNeighbourhood: k=%d nearest neighbours\n", K_NN))
cat(sprintf("  NN distance range: [%.1f, %.1f] km\n",
            min(D[D > 0]), max(sapply(seq_len(nrow(D)),
                                      function(i) sort(D[i, ])[K_NN + 1]))))
cat("Neighbour sizes: all", unique(lengths(nb_idx)), "(k-NN is exact)\n")

# Row-normalised weight matrix W  (sparse, [267, 267])
# W[i,j] = 1/|N_i| for j in N_i (uniform immigration from each neighbour)
W <- matrix(0.0, m, m)
for (i in seq_len(m)) {
  nbs <- nb_idx[[i]]
  if (length(nbs) > 0) {
    W[i, nbs] <- 1.0 / length(nbs)
  }
}

# ── 5. Standardise covariates ─────────────────────────────────────────────────
elev   <- sites$elev              # metres above sea level
forest <- sites$forest            # % forest cover

elev_z   <- (elev   - mean(elev))   / sd(elev)
forest_z <- (forest - mean(forest)) / sd(forest)

cat("\nCovariate ranges:\n")
cat(sprintf("  Elevation: [%d, %d] m  (mean %.0f)\n",
            min(elev), max(elev), mean(elev)))
cat(sprintf("  Forest:    [%d, %d] %%  (mean %.1f)\n",
            min(forest), max(forest), mean(forest)))

# ── 6. Pack and save ──────────────────────────────────────────────────────────
mhb <- list(
  Y_C      = Y_C,          # [m, T] count from visit 1
  Y_P      = Y_P,          # [m, T] binary from visit 2
  W        = W,            # [m, m] row-normalised neighbour weights
  D        = D,            # [m, m] inter-quadrat distance (km)
  nb_idx   = nb_idx,       # list of neighbour indices
  coords   = coords_m,     # [m, 2] Swiss LV03 coords (metres)
  elev     = elev,         # [m] elevation (m)
  elev_z   = elev_z,       # [m] standardised elevation
  forest   = forest,       # [m] forest cover (%)
  forest_z = forest_z,     # [m] standardised forest cover
  siteID   = sites$siteID, # [m] quadrat IDs
  species  = sp_name,
  years    = YEARS,
  m        = m,
  TT       = TT
)

out_path <- "D:/research/data/processed/mhb_great_tit.rds"
saveRDS(mhb, out_path)
cat("\nSaved:", out_path, "\n")
cat(sprintf("Data dimensions: m=%d quadrats, T=%d years\n", m, TT))
