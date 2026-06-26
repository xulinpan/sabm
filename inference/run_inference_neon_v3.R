# run_inference_neon_v3.R
# ABC-SMC inference for SC-ABM on NEON UNDE — seasonal (time-varying psi) model.
#
# Change from v1/v2: replaces constant LDD rate psi with a Gaussian seasonal
# kernel: psi_t = psi * exp(-0.5 * ((t - mu_s) / sigma_s)^2).
# Two new parameters: mu_s (peak biweekly step) and sigma_s (seasonal width).
# New summary statistic s6 = biweekly step of peak observed mean count, which
# gives the ABC-SMC a direct anchor on mu_s.
#
# This fixes the structural mismatch identified in v1/v2: the constant model
# cannot reproduce the seasonal bell-curve because psi drives monotone buildup
# from N0 = 0 with no mechanism for autumn decline.

# ── 0. Dependencies ──────────────────────────────────────────────────────────
stopifnot(requireNamespace("R6",       quietly = TRUE))
stopifnot(requireNamespace("jsonlite", quietly = TRUE))

PKG <- "D:/research/scabmnkd/R"
for (f in c("utils", "grid", "kernel", "model",
            "summaries", "prior", "abc_smc", "build_model")) {
  source(file.path(PKG, paste0(f, ".R")))
}
source("D:/research/data/processed/load_scabm_data.R")

# ── 1. Load data ──────────────────────────────────────────────────────────────
dat <- load_scabm_data("D:/research/data/processed")
m   <- dat$m   # 10
T   <- dat$T   # 16

grid_neon <- list(
  m          = m,
  dists      = dat$grid$dist_km,
  neighbours = dat$grid$neighbours,
  max_nn     = max(lengths(dat$grid$neighbours))
)

Y_C_obs <- rbind(rep(NA_real_, m), t(dat$Y_C))  # (17 x 10)
Y_P_obs <- rbind(rep(NA_real_, m), t(dat$Y_P))  # (17 x 10)

# ── 2. Observed summary statistics (6-vector) ─────────────────────────────────
s_obs <- compute_summaries_v3(Y_C_obs, Y_P_obs, grid_neon)
cat("Observed summary statistics (seasonal model, s1-s6):\n")
cat(sprintf("  s1 mean Y_C          : %.2f\n",  s_obs["s1"]))
cat(sprintf("  s2 temporal growth   : %.4f\n",  s_obs["s2"]))
cat(sprintf("  s3 Moran's I (t=T)   : %.4f\n",  s_obs["s3"]))
cat(sprintf("  s4 occupancy Y_P     : %.4f\n",  s_obs["s4"]))
cat(sprintf("  s5 lag-1 autocorr    : %.4f\n",  s_obs["s5"]))
cat(sprintf("  s6 peak timing step  : %.0f\n",  s_obs["s6"]))

# ── 3. Priors (9 parameters) ──────────────────────────────────────────────────
#
#   mu_s       ~ Uniform(3, 13)   peak step: May 9 to Sep 19
#   log_sigma_s ~ N(log(3), 0.7)  seasonal width: prior centred on 3 steps (6 wks)
#   Remaining 7 parameters: same as v1/v2.

sample_prior <- function() {
  r     <- max(rgamma(1, shape = 2.0, rate = 0.5),  1e-3)
  kappa <- max(rgamma(1, shape = 1.0, rate = 1.0),  1e-4)
  gamma <- max(rgamma(1, shape = 2.0, rate = 0.2),  1e-4)
  list(
    phi_logit   = rnorm(1, mean =  0.0, sd = 1.0),
    log_K       = rnorm(1, mean =  9.0, sd = 1.5),
    log_psi     = rnorm(1, mean =  6.0, sd = 1.5),
    mu_rho      = rnorm(1, mean = -2.0, sd = 1.5),
    r           = r,
    kappa       = kappa,
    gamma       = gamma,
    mu_s        = runif(1, min = 3.0, max = 13.0),
    log_sigma_s = rnorm(1, mean = log(3.0), sd = 0.7)
  )
}

log_prior <- function(theta) {
  if (theta$r <= 0 || theta$kappa <= 0 || theta$gamma <= 0) return(-Inf)
  if (theta$mu_s < 3.0 || theta$mu_s > 13.0)               return(-Inf)
  lp  <- -0.5 * theta$phi_logit^2
  lp  <- lp - 0.5 * (theta$log_K  - 9.0)^2 / (1.5^2)
  lp  <- lp - 0.5 * (theta$log_psi - 6.0)^2 / (1.5^2)
  lp  <- lp - 0.5 * (theta$mu_rho  + 2.0)^2 / (1.5^2)
  lp  <- lp + 1.0 * log(theta$r    ) - 2.0 * theta$r       # Gamma(2,0.5)
  lp  <- lp + 0.0 * log(theta$kappa) - 1.0 * theta$kappa   # Gamma(1,1)
  lp  <- lp + 1.0 * log(theta$gamma) - 5.0 * theta$gamma   # Gamma(2,0.2)
  # mu_s: log-Uniform(3,13) = -log(10) (constant, omitted)
  lp  <- lp - 0.5 * (theta$log_sigma_s - log(3.0))^2 / (0.7^2)
  lp
}

# ── 4. Seasonal simulator ─────────────────────────────────────────────────────
# The R6 SCABMNKD$step() reads self$psi at call time, so updating mdl$psi
# before each step implements time-varying LDD without modifying model.R.

run_sim <- function(theta) {
  tryCatch({
    mdl      <- build_model(theta, grid_neon, dat$X, dat$W)
    base_psi <- rep(max(exp(theta$log_psi), 1e-6), m)
    mu_s     <- theta$mu_s
    sigma_s  <- max(exp(theta$log_sigma_s), 0.5)
    rho_val  <- sigmoid(theta$mu_rho)

    # Simulate T steps with time-varying psi
    traj        <- matrix(0L, nrow = T + 1L, ncol = m)
    traj[1L, ]  <- 0L                   # N0 = 0 (start of season)
    for (t in seq_len(T)) {
      season_wt  <- exp(-0.5 * ((t - mu_s) / sigma_s)^2)
      mdl$psi    <- base_psi * season_wt
      traj[t + 1L, ] <- mdl$step(traj[t, ])
    }

    # Generate observations
    Y_C_s <- matrix(NA_real_, T + 1L, m)
    Y_P_s <- matrix(NA_real_, T + 1L, m)
    for (k in seq_len(nrow(dat$obs_C))) {
      ci   <- dat$obs_C$cell_idx[k]; ti <- dat$obs_C$time_idx[k]
      N_it <- max(traj[ti + 1L, ci], 0L)
      Y_C_s[ti + 1L, ci] <- rnbinom(1L, size = theta$r,
                                     mu   = rho_val * N_it + 1e-12)
    }
    for (k in seq_len(nrow(dat$obs_P))) {
      ci   <- dat$obs_P$cell_idx[k]; ti <- dat$obs_P$time_idx[k]
      N_it <- max(traj[ti + 1L, ci], 0L)
      Y_P_s[ti + 1L, ci] <- rbinom(1L, 1L,
                                    pmin(1 - exp(-theta$kappa * N_it), 1 - 1e-12))
    }

    s <- compute_summaries_v3(Y_C_s, Y_P_s, grid_neon)
    if (any(!is.finite(s))) NULL else s
  }, error = function(e) NULL)
}

# ── 5. Prior predictive check ─────────────────────────────────────────────────
cat("\nPrior predictive check (50 draws):\n")
set.seed(42L)
pp <- replicate(50L, {
  th <- sample_prior()
  s  <- run_sim(th)
  if (!is.null(s)) s else rep(NA_real_, 6L)
})
pp <- pp[, apply(pp, 2L, function(x) all(is.finite(x))), drop = FALSE]
cat(sprintf("  Valid draws: %d/50\n", ncol(pp)))
if (ncol(pp) > 0L) {
  cat(sprintf("  s1 (mean Y_C):     sim [%.1f, %.1f]  obs=%.1f\n",
              min(pp[1L, ]), max(pp[1L, ]), s_obs["s1"]))
  cat(sprintf("  s6 (peak step):    sim [%.1f, %.1f]  obs=%.0f\n",
              min(pp[6L, ]), max(pp[6L, ]), s_obs["s6"]))
}

# ── 6. ABC-SMC (9 parameters, 6 summary statistics) ──────────────────────────
v3_keys <- c("phi_logit", "log_K", "log_psi", "mu_rho",
             "r", "kappa", "gamma", "mu_s", "log_sigma_s")

cat("\n=== Running ABC-SMC v3 (seasonal model) ===\n")
cat(sprintf("Particles: 500  Populations: 8  alpha_q: 0.40\n"))

set.seed(2026L)
t_start <- proc.time()

result3 <- abc_smc(
  s_obs         = s_obs,
  run_sim       = run_sim,
  n_particles   = 500L,
  n_populations = 8L,
  alpha_q       = 0.40,
  n_pilot       = 500L,
  verbose       = TRUE,
  param_keys    = v3_keys
)

t_elapsed <- (proc.time() - t_start)["elapsed"]
cat(sprintf("\nABC-SMC v3 completed in %.1f seconds\n", t_elapsed))

# ── 7. Posterior summary ──────────────────────────────────────────────────────
wts3 <- result3$weights
ess3 <- 1 / sum(wts3^2)
cat(sprintf("ESS: %.1f / 500 (%.1f%%)\n", ess3, 100 * ess3 / 500))
cat(sprintf("Final epsilon: %.4f\n\n", result3$epsilon_final))

wq <- function(vals, wts, probs = c(0.025, 0.5, 0.975)) {
  ord <- order(vals); cw <- cumsum(wts[ord])
  vapply(probs, function(p) vals[ord][which(cw >= p)[1L]], numeric(1L))
}

parts3 <- do.call(rbind, lapply(seq_along(result3$particles), function(i) {
  th <- result3$particles[[i]]
  data.frame(
    particle    = i,
    weight      = wts3[i],
    phi_logit   = th$phi_logit,  phi   = sigmoid(th$phi_logit),
    log_K       = th$log_K,      K     = exp(th$log_K),
    log_psi     = th$log_psi,    psi   = exp(th$log_psi),
    mu_rho      = th$mu_rho,     rho   = sigmoid(th$mu_rho),
    r           = th$r,
    kappa       = th$kappa,
    gamma       = th$gamma,
    mu_s        = th$mu_s,
    log_sigma_s = th$log_sigma_s, sigma_s = exp(th$log_sigma_s)
  )
}))

nat_params <- list(
  phi     = parts3$phi,
  K       = parts3$K,
  psi     = parts3$psi,
  rho     = parts3$rho,
  r       = parts3$r,
  kappa   = parts3$kappa,
  gamma   = parts3$gamma,
  mu_s    = parts3$mu_s,
  sigma_s = parts3$sigma_s
)

cat("=== Posterior credible intervals (natural scale) ===\n")
cat(sprintf("%-10s %10s %10s %10s\n", "param", "2.5%", "median", "97.5%"))
cat(strrep("-", 44), "\n")
for (nm in names(nat_params)) {
  q <- wq(nat_params[[nm]], wts3)
  cat(sprintf("%-10s %10.4f %10.4f %10.4f\n", nm, q[1], q[2], q[3]))
}

# ── 8. Posterior predictive trajectory ───────────────────────────────────────
cat("\nGenerating posterior predictive trajectories (300 draws)...\n")
set.seed(99L)
pp_idx  <- sample.int(nrow(parts3), 300L, replace = TRUE, prob = wts3)
pp_N    <- matrix(NA_real_, T, 300L)

for (ii in seq_len(300L)) {
  th <- as.list(parts3[pp_idx[ii],
                        c("phi_logit", "log_K", "log_psi", "mu_rho",
                          "r", "kappa", "gamma", "mu_s", "log_sigma_s")])
  tryCatch({
    mdl      <- build_model(th, grid_neon, dat$X, dat$W)
    base_psi <- rep(max(exp(th$log_psi), 1e-6), m)
    mu_s_i   <- th$mu_s
    sigma_s_i <- max(exp(th$log_sigma_s), 0.5)

    traj       <- matrix(0L, T + 1L, m)
    for (t in seq_len(T)) {
      mdl$psi        <- base_psi * exp(-0.5 * ((t - mu_s_i) / sigma_s_i)^2)
      traj[t + 1L, ] <- mdl$step(traj[t, ])
    }
    pp_N[, ii] <- rowMeans(traj)[2:(T + 1L)]
  }, error = function(e) {})
}

rho_med3    <- median(sigmoid(parts3$mu_rho[pp_idx]))
obs_mean_C  <- apply(dat$Y_C, 2L, mean, na.rm = TRUE)

pp_traj_df <- data.frame(
  time_idx       = seq_len(T),
  date_CDT       = dat$time_idx$date_CDT,
  obs_mean_YC    = obs_mean_C,
  pp_mean_N_2.5  = apply(pp_N, 1L, quantile, 0.025, na.rm = TRUE),
  pp_mean_N_50   = apply(pp_N, 1L, quantile, 0.500, na.rm = TRUE),
  pp_mean_N_97.5 = apply(pp_N, 1L, quantile, 0.975, na.rm = TRUE)
)

cat(sprintf("\n%-14s %8s %10s %10s %10s\n",
            "date", "obs_YC", "pp_2.5", "pp_50", "pp_97.5"))
cat(strrep("-", 54), "\n")
for (t in seq_len(T)) {
  cat(sprintf("%-14s %8.1f %10.1f %10.1f %10.1f\n",
              dat$time_idx$date_CDT[t],
              obs_mean_C[t],
              rho_med3 * pp_traj_df$pp_mean_N_2.5[t],
              rho_med3 * pp_traj_df$pp_mean_N_50[t],
              rho_med3 * pp_traj_df$pp_mean_N_97.5[t]))
}

# ── 9. Save ───────────────────────────────────────────────────────────────────
saveRDS(result3, "D:/research/inference/abc_result_neon_v3.rds")
write.csv(parts3,     "D:/research/inference/particles_neon_v3.csv",
          row.names = FALSE)
write.csv(pp_traj_df, "D:/research/inference/pp_trajectory_v3.csv",
          row.names = FALSE)

cat("\nSaved:\n")
cat("  D:/research/inference/abc_result_neon_v3.rds\n")
cat("  D:/research/inference/particles_neon_v3.csv\n")
cat("  D:/research/inference/pp_trajectory_v3.csv\n")
