# run_inference_neon_v2.R
# ABC-SMC inference for SC-ABM on NEON UNDE, Version 2.
#
# Change from v1: s4 (occupancy) excluded from ABC distance.
# Reason: constant-psi model always produces s4=1.0 (LDD builds N from N0=0
# in step 1), while observed s4=0.771 due to seasonal zeros in April-May.
# This structural mismatch dominates the v1 distance (var_pilot[s4]~0 inflates
# that term ~52,000x), causing ESS=3.7%. Excluding s4 lets s1,s2,s3,s5 drive
# the posterior. The paper reports s4 mismatch as a known model limitation
# (constant-psi SC-ABM assumes steady-state; seasonality requires time-varying psi).
#
# Increase to 8 populations for better convergence.

stopifnot(requireNamespace("R6",       quietly = TRUE))
stopifnot(requireNamespace("jsonlite", quietly = TRUE))

PKG <- "D:/research/scabmnkd/R"
for (f in c("utils", "grid", "kernel", "model",
            "summaries", "prior", "abc_smc", "build_model")) {
  source(file.path(PKG, paste0(f, ".R")))
}
source("D:/research/data/processed/load_scabm_data.R")

dat <- load_scabm_data("D:/research/data/processed")
m   <- dat$m; T <- dat$T

grid_neon <- list(
  m          = m,
  dists      = dat$grid$dist_km,
  neighbours = dat$grid$neighbours,
  max_nn     = max(lengths(dat$grid$neighbours))
)

# Observation matrices (T+1 x m)
Y_C_obs <- rbind(rep(NA_real_, m), t(dat$Y_C))
Y_P_obs <- rbind(rep(NA_real_, m), t(dat$Y_P))

# Observed summaries
s_obs_full <- compute_summaries(Y_C_obs, Y_P_obs, grid_neon)

# ── Key change: set s4 to 1.0 (model's structural prediction) ───────────────
# ABC will not penalise proposals for producing s4=1.0.
s_obs_fit <- s_obs_full
s_obs_fit["s4"] <- 1.0
cat("Observed summaries (s4 set to 1.0 for fitting):\n")
print(round(s_obs_fit, 4))

# ── Adapted priors (same as v1) ───────────────────────────────────────────────
sample_prior <- function() {
  r     <- max(rgamma(1, shape = 2.0, rate = 0.5),  1e-3)
  kappa <- max(rgamma(1, shape = 1.0, rate = 1.0),  1e-4)
  gamma <- max(rgamma(1, shape = 2.0, rate = 0.2),  1e-4)
  list(
    phi_logit = rnorm(1, mean =  0.0, sd = 1.0),
    log_K     = rnorm(1, mean =  9.0, sd = 1.5),
    log_psi   = rnorm(1, mean =  6.0, sd = 1.5),
    mu_rho    = rnorm(1, mean = -2.0, sd = 1.5),
    r = r, kappa = kappa, gamma = gamma
  )
}
log_prior <- function(theta) {
  if (theta$r <= 0 || theta$kappa <= 0 || theta$gamma <= 0) return(-Inf)
  lp  <- -0.5 * theta$phi_logit^2
  lp  <- lp - 0.5 * (theta$log_K  - 9.0)^2 / (1.5^2)
  lp  <- lp - 0.5 * (theta$log_psi - 6.0)^2 / (1.5^2)
  lp  <- lp - 0.5 * (theta$mu_rho  + 2.0)^2 / (1.5^2)
  lp  <- lp + 1.0 * log(theta$r    ) - 2.0 * theta$r
  lp  <- lp + 0.0 * log(theta$kappa) - 1.0 * theta$kappa
  lp  <- lp + 1.0 * log(theta$gamma) - 5.0 * theta$gamma
  lp
}

# ── Simulator ────────────────────────────────────────────────────────────────
run_sim <- function(theta) {
  tryCatch({
    mdl  <- build_model(theta, grid_neon, dat$X, dat$W)
    traj <- mdl$simulate(rep(0L, m), T)
    rho_val <- sigmoid(theta$mu_rho)

    Y_C_s <- matrix(NA_real_, T + 1L, m)
    Y_P_s <- matrix(NA_real_, T + 1L, m)

    for (k in seq_len(nrow(dat$obs_C))) {
      ci <- dat$obs_C$cell_idx[k]; ti <- dat$obs_C$time_idx[k]
      N_it <- max(traj[ti + 1L, ci], 0L)
      Y_C_s[ti + 1L, ci] <- rnbinom(1L, size = theta$r,
                                     mu = rho_val * N_it + 1e-12)
    }
    for (k in seq_len(nrow(dat$obs_P))) {
      ci <- dat$obs_P$cell_idx[k]; ti <- dat$obs_P$time_idx[k]
      N_it <- max(traj[ti + 1L, ci], 0L)
      Y_P_s[ti + 1L, ci] <- rbinom(1L, 1L,
                                    pmin(1 - exp(-theta$kappa * N_it), 1-1e-12))
    }
    s <- compute_summaries(Y_C_s, Y_P_s, grid_neon)
    if (any(!is.finite(s))) NULL else s
  }, error = function(e) NULL)
}

# ── Run ABC-SMC with s_obs_fit (s4 neutralised) ─────────────────────────────
cat("\n=== Running ABC-SMC v2 (8 populations, s4 excluded) ===\n")
set.seed(2026L)
t0 <- proc.time()

result2 <- abc_smc(
  s_obs         = s_obs_fit,
  run_sim       = run_sim,
  n_particles   = 500L,
  n_populations = 8L,
  alpha_q       = 0.40,
  n_pilot       = 500L,
  verbose       = TRUE
)

cat(sprintf("\nCompleted in %.1f s\n", (proc.time() - t0)["elapsed"]))

# ── Posterior summaries ───────────────────────────────────────────────────────
post2 <- posterior_summary(result2)
wts2  <- result2$weights
ess2  <- 1 / sum(wts2^2)
cat(sprintf("\nESS: %.1f / 500  (%.1f%%)\n", ess2, 100*ess2/500))
cat(sprintf("Final epsilon: %.4f\n\n", result2$epsilon_final))

wq <- function(vals, wts, probs = c(0.025, 0.25, 0.5, 0.75, 0.975)) {
  ord <- order(vals); cw <- cumsum(wts[ord])
  vapply(probs, function(p) vals[ord][which(cw >= p)[1L]], numeric(1L))
}

parts2 <- do.call(rbind, lapply(seq_along(result2$particles), function(i) {
  th <- result2$particles[[i]]
  data.frame(i, w = wts2[i],
             phi_logit=th$phi_logit, phi=sigmoid(th$phi_logit),
             log_K=th$log_K, K=exp(th$log_K),
             log_psi=th$log_psi, psi=exp(th$log_psi),
             mu_rho=th$mu_rho, rho=sigmoid(th$mu_rho),
             r=th$r, kappa=th$kappa, gamma=th$gamma)
}))

cat("=== Posterior 95% credible intervals (natural scale) ===\n")
cat(sprintf("%-8s %10s %10s %10s %10s %10s\n",
            "param","2.5%","25%","median","75%","97.5%"))
cat(strrep("-",55),"\n")
nat <- list(phi=parts2$phi, K=parts2$K, psi=parts2$psi, rho=parts2$rho,
            r=parts2$r, kappa=parts2$kappa, gamma=parts2$gamma)
for (nm in names(nat)) {
  q <- wq(nat[[nm]], wts2)
  cat(sprintf("%-8s %10.4f %10.4f %10.4f %10.4f %10.4f\n",
              nm, q[1],q[2],q[3],q[4],q[5]))
}

# ── Posterior predictive: latent N trajectory ──────────────────────────────
set.seed(99L)
idx2  <- sample.int(nrow(parts2), 300L, replace=TRUE, prob=wts2)
pp_N  <- matrix(NA_real_, T, 300L)
for (ii in seq_len(300L)) {
  th <- as.list(parts2[idx2[ii], c("phi_logit","log_K","log_psi",
                                   "mu_rho","r","kappa","gamma")])
  tryCatch({
    mdl  <- build_model(th, grid_neon, dat$X, dat$W)
    traj <- mdl$simulate(rep(0L, m), T)
    pp_N[, ii] <- rowMeans(traj)[2:(T+1)]
  }, error = function(e) {})
}
rho_med2 <- median(sigmoid(parts2$mu_rho[idx2]))

obs_mean_C <- apply(dat$Y_C, 2, mean, na.rm=TRUE)  # length T

cat("\n=== Posterior predictive vs observed (E[Y_C] = rho_med x E[N]) ===\n")
cat(sprintf("%-14s %8s %10s %10s %10s\n","date","obs","pp_2.5","pp_50","pp_97.5"))
cat(strrep("-",50),"\n")
for (t in seq_len(T)) {
  v <- pp_N[t, ][!is.na(pp_N[t,])]
  q <- quantile(rho_med2 * v, c(0.025, 0.5, 0.975))
  cat(sprintf("%-14s %8.1f %10.1f %10.1f %10.1f\n",
              dat$time_idx$date_CDT[t], obs_mean_C[t], q[1], q[2], q[3]))
}

# ── Save ─────────────────────────────────────────────────────────────────────
saveRDS(result2, "D:/research/inference/abc_result_neon_v2.rds")
write.csv(parts2, "D:/research/inference/particles_neon_v2.csv", row.names=FALSE)

cat("\nSaved: abc_result_neon_v2.rds  particles_neon_v2.csv\n")
