# run_inference_mhb.R
# SMFVB inference for the count ABM on the Swiss MHB Great Tit dataset.
#
# Uses SwissTits (AHMbook) — 267 quadrats, 10 annual surveys (2004-2013).
# Y_C  = territory count from visit 1  (NegBin channel)
# Y_P  = detection indicator from visit 2  (Bernoulli channel)
#
# Parameters phi, K, psi, rho, r_nb, kappa are fixed at literature-informed
# values for Great Tit (Parus major).  The SMFVB algorithm then infers the
# full spatio-temporal field {mu_{i,t}} via closed-form CAVI.
#
# Outputs (written to D:/research/inference/):
#   mhb_smfvb_mu.csv       — posterior variational means [m x T]
#   mhb_smfvb_summary.csv  — site-level summary statistics
#   mhb_ppcheck.csv        — posterior predictive check statistics

source("D:/research/inference/smfvb.R")

# ── 1. Load prepared MHB data ─────────────────────────────────────────────────
dat_path <- "D:/research/data/processed/mhb_great_tit.rds"
if (!file.exists(dat_path)) {
  cat("Preparing MHB data...\n")
  source("D:/research/data/processed/mhb_prepare.R")
}
dat <- readRDS(dat_path)

Y_C <- dat$Y_C     # [267, 10]
Y_P <- dat$Y_P     # [267, 10]
W   <- dat$W       # [267, 267] row-normalised
m   <- dat$m       # 267
TT  <- dat$TT      # 10
YEARS <- dat$years

cat("=== MHB Great Tit SMFVB ===\n")
cat(sprintf("m = %d quadrats,  T = %d years (%d-%d)\n",
            m, TT, min(YEARS), max(YEARS)))
cat(sprintf("Y_C: %d non-missing (%.1f%%)\n",
            sum(!is.na(Y_C)), 100 * mean(!is.na(Y_C))))
cat(sprintf("Y_P: %d non-missing (%.1f%%)\n",
            sum(!is.na(Y_P)), 100 * mean(!is.na(Y_P))))

# ── 2. Parameter specification ────────────────────────────────────────────────
#
# Parameters calibrated to match MHB Great Tit observed moments:
#   E[Y_C]  = 8.15  =>  rho * N* = 0.72 * 11.3 = 8.1  ✓
#   P(Y_P=1) = 0.681 => 1 - exp(-kappa*N*) = 1-exp(-0.1*11.3) = 0.68  ✓
#
# phi=0.95: high territory-level persistence (re-occupancy rate, not individual
#   survival; Great Tit territories refill quickly via local breeders)
# K=100: weak density regulation at observed densities exp(-11/100)=0.895;
#   regulation becomes significant only at very high densities
# omega=0.05: 5% of neighbourhood abundance disperses as immigrants;
#   W is scaled by omega so Sigma_j W[i,j] = omega (not 1), preventing
#   divergent dynamics from row-normalised dispersal
# psi=1.0: ~1 LDD immigrant per quadrat per year (colonisation from outside)
# rho=0.72: detection probability in MHB point-count visits
# r=2.0: moderate NegBin overdispersion
# kappa=0.1: Bernoulli detection rate for visit-2 binary channel

phi_val   <- 0.95
K_val     <- 100.0
omega_val <- 0.05
psi_val   <- 1.0
rho_val   <- 0.72
r_val     <- 2.0
kappa_val <- 0.10

# Scale W by omega so total inter-site immigration fraction = omega
W_scaled <- dat$W * omega_val

# Covariate modulation: K and phi vary with elevation and forest cover
phi <- plogis(qlogis(phi_val) - 0.10 * dat$elev_z)
K   <- rep(K_val, m) * exp(0.20 * dat$forest_z)

psi   <- rep(psi_val,   m)
rho   <- rep(rho_val,   m)
r_nb  <- rep(r_val,     m)
kappa <- rep(kappa_val, m)

# Equilibrium check at mean parameters:
# N* solves N = phi*N*exp(-N/K) + omega*N + psi
# At N*=11, phi=0.95, K=100, omega=0.05, psi=1.0:
# RHS = 0.95*11*exp(-0.11) + 0.05*11 + 1.0 = 9.36 + 0.55 + 1.0 = 10.91 ≈ 11 ✓

cat("\nParameter ranges after covariate modulation:\n")
cat(sprintf("  phi:   [%.3f, %.3f]  (mean %.3f)\n",
            min(phi), max(phi), mean(phi)))
cat(sprintf("  K:     [%.1f, %.1f]  (mean %.1f)\n",
            min(K), max(K), mean(K)))
cat(sprintf("  omega: %.2f (W scaled; sum W[i,]=%.3f)\n",
            omega_val, mean(rowSums(W_scaled))))
cat(sprintf("  psi:   %.2f (constant)\n", psi_val))
cat(sprintf("  rho:   %.3f (constant)\n", rho_val))
cat(sprintf("  r_nb:  %.1f (constant)\n", r_val))
cat(sprintf("  kappa: %.2f (constant)\n", kappa_val))

# ── 3. Run SMFVB ──────────────────────────────────────────────────────────────
cat("\nRunning SMFVB...\n")
t_start <- proc.time()

set.seed(2026L)
fit <- smfvb(
  Y_C      = Y_C,
  Y_P      = Y_P,
  W        = W_scaled,
  phi      = phi,
  K        = K,
  psi      = psi,
  rho      = rho,
  r_nb     = r_nb,
  kappa    = kappa,
  max_iter = 300L,
  tol      = 1e-5,
  verbose  = TRUE
)

t_smfvb <- (proc.time() - t_start)["elapsed"]
cat(sprintf("\nSMFVB completed in %.1f seconds (%d iterations)\n",
            t_smfvb, fit$n_iter))
cat(sprintf("Final ELBO: %.3f\n", tail(fit$elbo_trace, 1)))

# ── 4. Posterior predictive check ─────────────────────────────────────────────
# Replicates are restricted to observed locations (the mask of non-NA Y_C /
# Y_P cells) so that replicated and observed summaries cover the same sites.
cat("\nGenerating posterior predictive replicates (R = 1000)...\n")
pp <- smfvb_ppcheck(fit, rho, r_nb, kappa, R = 1000L)

obs_mask_c <- !is.na(Y_C)   # [m, T] logical: surveyed count cells
obs_mask_p <- !is.na(Y_P)   # [m, T] logical: surveyed detection cells

# T1: mean Y_C at observed locations
ts1_obs <- mean(Y_C[obs_mask_c])
ts1_rep <- apply(pp$Y_C_rep, 3L, function(x) mean(x[obs_mask_c]))
p1 <- mean(ts1_rep >= ts1_obs)
cat(sprintf("  T1 (mean count @ obs locs):    obs=%.2f  rep=%.2f  p=%.3f\n",
            ts1_obs, mean(ts1_rep), p1))

# T2: detection rate at observed locations
ts2_obs <- mean(Y_P[obs_mask_p])
ts2_rep <- apply(pp$Y_P_rep, 3L, function(x) mean(x[obs_mask_p]))
p2 <- mean(ts2_rep >= ts2_obs)
cat(sprintf("  T2 (detection rate @ obs locs): obs=%.3f  rep=%.3f  p=%.3f\n",
            ts2_obs, mean(ts2_rep), p2))

# T3: temporal trend (slope of annual mean Y_C at observed locations)
obs_annual <- colMeans(Y_C, na.rm = TRUE)
ts3_obs <- coef(lm(obs_annual ~ seq_along(obs_annual)))[2]
ts3_rep <- apply(pp$Y_C_rep, 3L, function(x) {
  yr_means <- sapply(seq_len(ncol(x)), function(t) mean(x[obs_mask_c[, t], t]))
  coef(lm(yr_means ~ seq_along(yr_means)))[2]
})
p3 <- mean(ts3_rep >= ts3_obs)
cat(sprintf("  T3 (annual trend @ obs locs):   obs=%.3f  rep=%.3f  p=%.3f\n",
            ts3_obs, mean(ts3_rep), p3))

# ── 5. Results summary ────────────────────────────────────────────────────────
mu_hat <- fit$mu   # [267, 10] posterior variational means

cat("\n=== Posterior variational means (N_{i,t}) ===\n")
cat(sprintf("  Overall mean: %.2f territories / quadrat\n", mean(mu_hat)))
cat(sprintf("  Range: [%.2f, %.2f]\n", min(mu_hat), max(mu_hat)))

# Temporal trend in mean abundance
mean_N_year <- colMeans(mu_hat)
cat(sprintf("\n  Annual mean abundance (territories / quadrat):\n"))
for (t in seq_len(TT)) {
  cat(sprintf("    %d: %.2f\n", YEARS[t], mean_N_year[t]))
}

# Top 10 most abundant quadrats (average over years)
mean_N_site <- rowMeans(mu_hat)
top10 <- order(mean_N_site, decreasing = TRUE)[1:10]
cat("\n  Top 10 quadrats by mean abundance:\n")
cat(sprintf("  %s\n", paste(sprintf("%s (%.1f)",
                                    dat$siteID[top10], mean_N_site[top10]),
                              collapse = ", ")))

# ── 6. Save outputs ───────────────────────────────────────────────────────────
# Posterior means matrix
mu_df <- as.data.frame(mu_hat)
colnames(mu_df) <- paste0("year_", YEARS)
mu_df$siteID    <- dat$siteID
mu_df <- mu_df[, c("siteID", paste0("year_", YEARS))]
write.csv(mu_df, "D:/research/inference/mhb_smfvb_mu.csv", row.names = FALSE)

# Site-level summary
site_summary <- data.frame(
  siteID    = dat$siteID,
  coordx    = dat$coords[, 1],
  coordy    = dat$coords[, 2],
  elev      = dat$elev,
  forest    = dat$forest,
  mean_N    = rowMeans(mu_hat),
  sd_N      = apply(mu_hat, 1, sd),
  trend_N   = apply(mu_hat, 1, function(x) coef(lm(x ~ seq_along(x)))[2]),
  phi_i     = phi,
  K_i       = K
)
write.csv(site_summary, "D:/research/inference/mhb_smfvb_summary.csv",
          row.names = FALSE)

# ELBO trace
elbo_df <- data.frame(iter = seq_along(fit$elbo_trace),
                       elbo = fit$elbo_trace)
write.csv(elbo_df, "D:/research/inference/mhb_smfvb_elbo.csv",
          row.names = FALSE)

# PP-check statistics
pp_check_df <- data.frame(
  ts1_Y_C_mean = ts1_rep,
  ts2_Y_P_rate = ts2_rep,
  ts3_trend    = ts3_rep
)
write.csv(pp_check_df, "D:/research/inference/mhb_ppcheck.csv",
          row.names = FALSE)

cat("\nSaved:\n")
cat("  D:/research/inference/mhb_smfvb_mu.csv\n")
cat("  D:/research/inference/mhb_smfvb_summary.csv\n")
cat("  D:/research/inference/mhb_smfvb_elbo.csv\n")
cat("  D:/research/inference/mhb_ppcheck.csv\n")
cat(sprintf("\nTotal wall time: %.1f s\n", t_smfvb))
