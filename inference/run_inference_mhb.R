# run_inference_mhb.R
# SMFVB variational EM for the count ABM on the Swiss MHB Great Tit dataset.
#
# Y_C  = territory count from visit 1  (NegBin channel)
# Y_P  = detection indicator from visit 2  (Bernoulli channel)
#
# rho, kappa, psi are estimated jointly with the latent field via variational EM.
# phi, K (covariate-driven), omega, r are held fixed.

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

cat("=== MHB Great Tit SMFVB-VEM ===\n")
cat(sprintf("m = %d quadrats,  T = %d years (%d-%d)\n",
            m, TT, min(YEARS), max(YEARS)))

# ── 2. Fixed parameters ────────────────────────────────────────────────────────
# phi, K: ecologically constrained + covariate modulation from glutz1993
# omega:  inter-site dispersal fraction (fixed at 5%)
# r:      NegBin overdispersion (fixed at 2.0; identifiable from count variance)
#
# rho, kappa, psi: ESTIMATED by variational EM (M-step NR updates)

phi_base  <- 0.95
K_base    <- 100.0
omega_val <- 0.05
r_val     <- 2.0

phi <- plogis(qlogis(phi_base) - 0.10 * dat$elev_z)
K   <- rep(K_base, m) * exp(0.20 * dat$forest_z)

W_scaled <- dat$W * omega_val        # scale W so row sums = omega

r_nb <- rep(r_val, m)

cat("\nCovariate-modulated parameter ranges:\n")
cat(sprintf("  phi:  [%.3f, %.3f]  (mean %.3f)\n", min(phi), max(phi), mean(phi)))
cat(sprintf("  K:    [%.1f,  %.1f]  (mean %.1f)\n", min(K),   max(K),   mean(K)))
cat(sprintf("  omega: %.2f (W row sums = omega)\n", omega_val))
cat(sprintf("  r:     %.1f (fixed NegBin overdispersion)\n", r_val))

# ── 3. Priors for estimated parameters ────────────────────────────────────────
#   rho   ~ Beta(2, 2):      weakly informative, centres at 0.5
#   kappa ~ Gamma(1.5, 8):   centres at 0.12 (plausible detection rate)
#   psi   ~ Gamma(2, 2):     centres at 1.0 LDD immigrant per quadrat/yr
prior <- list(rho_a = 2.0, rho_b = 2.0, kappa_a = 1.5, kappa_b = 8.0,
              psi_a = 2.0, psi_b = 2.0)

cat("\nPriors for estimated parameters:\n")
cat("  rho   ~ Beta(2, 2)        [mean 0.50, weak]\n")
cat("  kappa ~ Gamma(1.5, 8)     [mean 0.12, sd 0.097]\n")
cat("  psi   ~ Gamma(2, 2)       [mean 1.00, sd 0.707]\n")

# ── 4. Initialise at moment-matched values ─────────────────────────────────────
# Warm start: match observed count mean and detection rate at process mean
mu_bar_init <- mean(Y_C, na.rm = TRUE) / 0.72   # initial rho guess
rho_init    <- 0.72
kappa_init  <- -log(1 - mean(Y_P, na.rm = TRUE)) / mu_bar_init
psi_init    <- 1.0

cat(sprintf("\nInitial values:  rho=%.3f  kappa=%.3f  psi=%.3f\n",
            rho_init, kappa_init, psi_init))

# ── 5. Run variational EM ──────────────────────────────────────────────────────
cat("\nRunning SMFVB-VEM...\n")
t_start <- proc.time()

set.seed(2026L)
fit <- smfvb_vem(
  Y_C        = Y_C,
  Y_P        = Y_P,
  W          = W_scaled,
  phi        = phi,
  K          = K,
  psi_init   = psi_init,
  rho_init   = rho_init,
  r_nb       = r_nb,
  kappa_init = kappa_init,
  estimate   = c("rho", "kappa", "psi"),
  prior      = prior,
  max_vem_iter = 30L,
  estep_max    = 200L,
  estep_tol    = 1e-6,
  vem_tol      = 1e-5,
  verbose      = TRUE
)

t_vem <- (proc.time() - t_start)["elapsed"]
cat(sprintf("\nVEM completed in %.1f seconds (%d outer iterations)\n",
            t_vem, fit$n_vem_iter))

# ── 6. Report estimated parameters ────────────────────────────────────────────
cat("\n=== Estimated parameters ===\n")
cat(sprintf("  rho   = %.4f  (SE = %.4f,  95%% CI [%.4f, %.4f])\n",
            fit$rho,   fit$se_rho,
            fit$rho   - 1.96 * fit$se_rho, fit$rho   + 1.96 * fit$se_rho))
cat(sprintf("  kappa = %.4f  (SE = %.4f,  95%% CI [%.4f, %.4f])\n",
            fit$kappa, fit$se_kappa,
            fit$kappa - 1.96 * fit$se_kappa, fit$kappa + 1.96 * fit$se_kappa))
cat(sprintf("  psi   = %.4f  (SE = %.4f,  95%% CI [%.4f, %.4f])\n",
            fit$psi,   fit$se_psi,
            fit$psi   - 1.96 * fit$se_psi,  fit$psi   + 1.96 * fit$se_psi))
cat(sprintf("  ELBO  = %.3f\n", fit$elbo))

# Implied model predictions at estimated parameters
mu_hat  <- fit$mu
rho_hat <- fit$rho
kappa_hat <- fit$kappa

cat("\n=== Implied model fit at estimated parameters ===\n")
obs_mask_c <- !is.na(Y_C)
obs_mask_p <- !is.na(Y_P)
cat(sprintf("  E[Y_C] = rho*mu = %.4f * %.4f = %.3f  (observed: %.3f)\n",
            rho_hat, mean(mu_hat[obs_mask_c]),
            rho_hat * mean(mu_hat[obs_mask_c]),
            mean(Y_C[obs_mask_c])))
cat(sprintf("  P(Y_P=1) = 1-exp(-kappa*mu) = %.3f  (observed: %.3f)\n",
            1 - exp(-kappa_hat * mean(mu_hat[obs_mask_p])),
            mean(Y_P[obs_mask_p])))

# ── 7. Posterior predictive checks ────────────────────────────────────────────
cat("\nRunning posterior predictive check (R = 1000)...\n")
pp <- smfvb_ppcheck(fit, rho = rho_hat, r_nb = r_nb[1L], kappa = kappa_hat,
                     R = 1000L)

# T1: mean Y_C at observed count locations
ts1_obs <- mean(Y_C[obs_mask_c])
ts1_rep <- apply(pp$Y_C_rep, 3L, function(x) mean(x[obs_mask_c]))
p1 <- mean(ts1_rep >= ts1_obs)
cat(sprintf("  T1 mean Y_C:     obs=%.2f  rep=[%.2f, %.2f]  p=%.3f\n",
            ts1_obs, quantile(ts1_rep, 0.025), quantile(ts1_rep, 0.975), p1))

# T2: detection rate at observed binary locations
ts2_obs <- mean(Y_P[obs_mask_p])
ts2_rep <- apply(pp$Y_P_rep, 3L, function(x) mean(x[obs_mask_p]))
p2 <- mean(ts2_rep >= ts2_obs)
cat(sprintf("  T2 detect rate:  obs=%.3f  rep=[%.3f, %.3f]  p=%.3f\n",
            ts2_obs, quantile(ts2_rep, 0.025), quantile(ts2_rep, 0.975), p2))

# T3: temporal trend (slope of annual mean Y_C at observed locations)
obs_annual <- colMeans(Y_C, na.rm = TRUE)
ts3_obs <- coef(lm(obs_annual ~ seq_along(obs_annual)))[2]
ts3_rep <- apply(pp$Y_C_rep, 3L, function(x) {
  yr_means <- sapply(seq_len(ncol(x)),
                     function(t) mean(x[obs_mask_c[, t], t]))
  coef(lm(yr_means ~ seq_along(yr_means)))[2]
})
p3 <- mean(ts3_rep >= ts3_obs)
cat(sprintf("  T3 trend:        obs=%.3f  rep=[%.3f, %.3f]  p=%.3f\n",
            ts3_obs, quantile(ts3_rep, 0.025), quantile(ts3_rep, 0.975), p3))

# ── 8. Temporal pattern ────────────────────────────────────────────────────────
cat("\n=== Temporal pattern ===\n")
annual_mu <- colMeans(mu_hat)
annual_yc <- colMeans(Y_C, na.rm = TRUE)
cat(sprintf("  %-6s  %8s  %8s  %8s\n", "Year", "mu_bar", "rho*mu", "obs Y_C"))
for (t in seq_len(TT))
  cat(sprintf("  %d  %8.2f  %8.2f  %8.2f\n",
              YEARS[t], annual_mu[t], rho_hat * annual_mu[t], annual_yc[t]))

# ── 9. Spatial pattern ────────────────────────────────────────────────────────
cat("\n=== Spatial recovery ===\n")
mu_flat  <- as.numeric(mu_hat)[as.numeric(obs_mask_c)]
yc_flat  <- as.numeric(Y_C)[as.numeric(obs_mask_c)]
cat(sprintf("  cor(mu, Y_C) at obs cells: r = %.3f\n", cor(mu_flat, yc_flat)))
cat(sprintf("  Elevation r:  %.3f\n", cor(rowMeans(mu_hat), dat$elev)))
cat(sprintf("  Forest r:     %.3f\n", cor(rowMeans(mu_hat), dat$forest)))

# ── 10. Save outputs ───────────────────────────────────────────────────────────
mu_df <- as.data.frame(mu_hat)
colnames(mu_df) <- paste0("year_", YEARS)
mu_df$siteID    <- dat$siteID
mu_df <- mu_df[, c("siteID", paste0("year_", YEARS))]
write.csv(mu_df, "D:/research/inference/mhb_smfvb_mu.csv", row.names = FALSE)

site_summary <- data.frame(
  siteID  = dat$siteID,
  coordx  = dat$coords[, 1L],
  coordy  = dat$coords[, 2L],
  elev    = dat$elev,
  forest  = dat$forest,
  mean_N  = rowMeans(mu_hat),
  sd_N    = apply(mu_hat, 1L, sd),
  trend_N = apply(mu_hat, 1L, function(x)
    coef(lm(x ~ seq_along(x)))[2L]),
  phi_i   = phi,
  K_i     = K
)
write.csv(site_summary, "D:/research/inference/mhb_smfvb_summary.csv",
          row.names = FALSE)

elbo_df <- data.frame(iter = seq_along(fit$elbo_trace), elbo = fit$elbo_trace)
write.csv(elbo_df, "D:/research/inference/mhb_smfvb_elbo.csv",
          row.names = FALSE)

pp_check_df <- data.frame(ts1_Y_C_mean = ts1_rep,
                           ts2_Y_P_rate = ts2_rep,
                           ts3_trend    = ts3_rep)
write.csv(pp_check_df, "D:/research/inference/mhb_ppcheck.csv",
          row.names = FALSE)

param_df <- data.frame(
  parameter = c("rho", "kappa", "psi"),
  estimate  = c(fit$rho, fit$kappa, fit$psi),
  se        = c(fit$se_rho, fit$se_kappa, fit$se_psi),
  ci_lo     = c(fit$rho   - 1.96 * fit$se_rho,
                fit$kappa - 1.96 * fit$se_kappa,
                fit$psi   - 1.96 * fit$se_psi),
  ci_hi     = c(fit$rho   + 1.96 * fit$se_rho,
                fit$kappa + 1.96 * fit$se_kappa,
                fit$psi   + 1.96 * fit$se_psi),
  prior     = c("Beta(2,2)", "Gamma(1.5,8)", "Gamma(2,2)")
)
write.csv(param_df, "D:/research/inference/mhb_vem_params.csv", row.names = FALSE)
write.csv(fit$param_trace, "D:/research/inference/mhb_vem_trace.csv",
          row.names = FALSE)

cat("\nSaved:\n")
cat("  mhb_smfvb_mu.csv       (posterior variational means)\n")
cat("  mhb_smfvb_summary.csv  (site-level summaries)\n")
cat("  mhb_smfvb_elbo.csv     (ELBO trace, final E-step)\n")
cat("  mhb_ppcheck.csv        (posterior predictive replicates)\n")
cat("  mhb_vem_params.csv     (estimated rho, kappa, psi + Laplace SE)\n")
cat("  mhb_vem_trace.csv      (VEM parameter trace per outer iteration)\n")
cat(sprintf("\nTotal wall time: %.1f s\n", t_vem))
