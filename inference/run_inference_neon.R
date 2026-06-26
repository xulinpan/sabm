# run_inference_neon.R
# ABC-SMC inference for SC-ABM on NEON UNDE Coquillettidia perturbans 2024.
#
# Design choices:
#   Grid    : irregular 10-plot NEON grid wrapped as a plain list mimicking
#             SpatialGrid fields (m, dists, neighbours, max_nn).
#   Y_C     : overnight trap scaled counts  (84% coverage, (T+1 x m) matrix)
#   Y_P     : daytime trap binary detections(90% coverage, (T+1 x m) matrix)
#   N0      : all zeros (start of season); psi drives emergence buildup.
#   T       : 16 biweekly steps (April 11 – October 30).
#   Priors  : recalibrated for NEON scale.  Original priors (K~20, psi~0.14)
#             are 3 orders of magnitude below observed counts (~760 mean).
#   Limitation: constant-parameter model cannot reproduce fall decline;
#             ABC fits season-average steady-state behaviour.

# ── 0. Dependencies ──────────────────────────────────────────────────────────
stopifnot(requireNamespace("R6",       quietly = TRUE))
stopifnot(requireNamespace("jsonlite", quietly = TRUE))

PKG <- "D:/research/scabmnkd/R"
for (f in c("utils", "grid", "kernel", "model",
            "summaries", "prior", "abc_smc", "build_model")) {
  source(file.path(PKG, paste0(f, ".R")))
}
source("D:/research/data/processed/load_scabm_data.R")

# ── 1. Load processed NEON data ──────────────────────────────────────────────
dat <- load_scabm_data("D:/research/data/processed")
m   <- dat$m      # 10
T   <- dat$T      # 16

# ── 2. Build NEON-compatible grid object (plain list, SpatialGrid interface) ─
# SpatialGrid fields used by SCABMNKD: m, dists, neighbours, max_nn
grid_neon <- list(
  m          = m,
  dists      = dat$grid$dist_km,                        # (m x m) km
  neighbours = dat$grid$neighbours,                     # list of length m, 1-indexed
  max_nn     = max(lengths(dat$grid$neighbours))        # 5
)

cat(sprintf("Grid: m=%d  max_nn=%d  dists range [%.2f, %.2f] km\n",
            grid_neon$m, grid_neon$max_nn,
            min(grid_neon$dists[grid_neon$dists > 0]),
            max(grid_neon$dists)))

# ── 3. Format observations as (T+1 x m) matrices for compute_summaries ──────
# Row 1 = t=0 (season start, no observations); rows 2..T+1 = time steps 1..T
Y_C_obs <- rbind(rep(NA_real_, m), t(dat$Y_C))  # (17 x 10)
Y_P_obs <- rbind(rep(NA_real_, m), t(dat$Y_P))  # (17 x 10)

# ── 4. Compute observed summary statistics ───────────────────────────────────
s_obs <- compute_summaries(Y_C_obs, Y_P_obs, grid_neon)
cat("\nObserved summary statistics:\n")
cat(sprintf("  s1 mean Y_C          : %.2f\n", s_obs["s1"]))
cat(sprintf("  s2 temporal growth   : %.4f\n", s_obs["s2"]))
cat(sprintf("  s3 Moran's I (t=T)   : %.4f\n", s_obs["s3"]))
cat(sprintf("  s4 occupancy Y_P     : %.4f\n", s_obs["s4"]))
cat(sprintf("  s5 lag-1 autocorr    : %.4f\n", s_obs["s5"]))

# ── 5. Recalibrated priors for NEON scale ────────────────────────────────────
#
# Calibration rationale (all means on natural scale):
#   Observed mean Y_C ≈ 414  →  E[Y_C] = rho × E[N]
#   Observed peak Y_C ≈ 4800 →  K >> 4800 / rho
#
#   phi_logit ~ N(0, 1)  :  rho = sigmoid(0) = 0.50 biweekly adult survival
#   log_K     ~ N(9,1.5) :  K = exp(9) = 8103, range [403, 162755]
#   log_psi   ~ N(6,1.5) :  psi = exp(6) = 403/step (emergence from lakes)
#   mu_rho    ~ N(-2,1.5):  rho = sigmoid(-2) = 0.12; trap catches ~1-50% of N
#   r         ~ Gamma(2,0.5) : NegBin overdispersion (unchanged)
#   kappa     ~ Gamma(1,1)   : binary detection rate (unchanged)
#   gamma     ~ Gamma(2,0.2) : half-saturation (unchanged)

sample_prior <- function() {
  r     <- max(rgamma(1, shape = 2.0, rate = 0.5),  1e-3)
  kappa <- max(rgamma(1, shape = 1.0, rate = 1.0),  1e-4)
  gamma <- max(rgamma(1, shape = 2.0, rate = 0.2),  1e-4)
  list(
    phi_logit = rnorm(1, mean =  0.0, sd = 1.0),
    log_K     = rnorm(1, mean =  9.0, sd = 1.5),
    log_psi   = rnorm(1, mean =  6.0, sd = 1.5),
    mu_rho    = rnorm(1, mean = -2.0, sd = 1.5),
    r         = r,
    kappa     = kappa,
    gamma     = gamma
  )
}

log_prior <- function(theta) {
  if (theta$r <= 0 || theta$kappa <= 0 || theta$gamma <= 0) return(-Inf)
  lp  <- 0.0
  lp  <- lp - 0.5 * (theta$phi_logit        )^2            # N(0,1)
  lp  <- lp - 0.5 * (theta$log_K  - 9.0)^2 / (1.5^2)      # N(9,1.5)
  lp  <- lp - 0.5 * (theta$log_psi - 6.0)^2 / (1.5^2)     # N(6,1.5)
  lp  <- lp - 0.5 * (theta$mu_rho  + 2.0)^2 / (1.5^2)     # N(-2,1.5)
  lp  <- lp + 1.0 * log(theta$r    ) - 2.0 * theta$r       # Gamma(2,0.5)
  lp  <- lp + 0.0 * log(theta$kappa) - 1.0 * theta$kappa   # Gamma(1,1)
  lp  <- lp + 1.0 * log(theta$gamma) - 5.0 * theta$gamma   # Gamma(2,0.2) [rate=0.2 -> rate*5=1, scaled]
  lp
}

# ── 6. Simulator function ────────────────────────────────────────────────────
# Returns compute_summaries output or NULL on failure.
run_sim <- function(theta) {
  tryCatch({
    # Build homogeneous model with this theta
    mdl <- build_model(theta, grid_neon, dat$X, dat$W)

    # Simulate T steps from N0=0 (season start; psi drives buildup)
    N0   <- rep(0L, m)
    traj <- mdl$simulate(N0, T)    # (T+1 x m) integer matrix, row 1 = t=0

    rho_val <- sigmoid(theta$mu_rho)

    # Build simulated observation matrices (T+1 x m), NA where not surveyed
    Y_C_sim <- matrix(NA_real_, nrow = T + 1L, ncol = m)
    Y_P_sim <- matrix(NA_real_, nrow = T + 1L, ncol = m)

    # Populate count channel at overnight-trap locations
    for (k in seq_len(nrow(dat$obs_C))) {
      ci   <- dat$obs_C$cell_idx[k]       # 1-indexed
      ti   <- dat$obs_C$time_idx[k]       # 1-indexed (1..T)
      N_it <- max(traj[ti + 1L, ci], 0L)  # row ti+1 because row 1 = t=0
      mu   <- rho_val * N_it + 1e-12
      Y_C_sim[ti + 1L, ci] <- rnbinom(1L, size = theta$r, mu = mu)
    }

    # Populate binary channel at daytime-trap locations
    for (k in seq_len(nrow(dat$obs_P))) {
      ci   <- dat$obs_P$cell_idx[k]
      ti   <- dat$obs_P$time_idx[k]
      N_it <- max(traj[ti + 1L, ci], 0L)
      p_P  <- pmin(1 - exp(-theta$kappa * N_it), 1 - 1e-12)
      Y_P_sim[ti + 1L, ci] <- rbinom(1L, 1L, p_P)
    }

    s <- compute_summaries(Y_C_sim, Y_P_sim, grid_neon)
    if (any(!is.finite(s))) NULL else s
  }, error = function(e) NULL)
}

# Quick prior predictive check before full ABC
cat("\nPrior predictive check (50 draws):\n")
set.seed(42L)
pp <- replicate(50L, {
  th <- sample_prior()
  s  <- run_sim(th)
  if (!is.null(s)) s else rep(NA_real_, 5L)
})
pp <- pp[, apply(pp, 2L, function(x) all(is.finite(x))), drop = FALSE]
cat(sprintf("  Valid draws: %d/50\n", ncol(pp)))
if (ncol(pp) > 0L) {
  cat(sprintf("  s1 (mean Y_C):  sim range [%.1f, %.1f]  obs=%.1f\n",
              min(pp[1L,]), max(pp[1L,]), s_obs["s1"]))
  cat(sprintf("  s4 (occupancy): sim range [%.3f, %.3f]  obs=%.3f\n",
              min(pp[4L,]), max(pp[4L,]), s_obs["s4"]))
}

# ── 7. Run ABC-SMC ───────────────────────────────────────────────────────────
cat("\n=== Running ABC-SMC ===\n")
cat(sprintf("Particles: %d  Populations: %d  alpha_q: %.2f\n",
            500L, 6L, 0.40))

set.seed(2025L)
t_start <- proc.time()

result <- abc_smc(
  s_obs         = s_obs,
  run_sim       = run_sim,
  n_particles   = 500L,
  n_populations = 6L,
  alpha_q       = 0.40,
  n_pilot       = 500L,
  verbose       = TRUE
)

t_elapsed <- proc.time() - t_start
cat(sprintf("\nABC-SMC completed in %.1f seconds\n", t_elapsed["elapsed"]))

# ── 8. Posterior summaries ───────────────────────────────────────────────────
post <- posterior_summary(result)

# Back-transform to natural scale
post_nat <- post
post_nat$param_natural <- c(
  sprintf("phi = sigmoid(%.3f) = %.3f",
          post$mean[post$parameter == "phi_logit"],
          sigmoid(post$mean[post$parameter == "phi_logit"])),
  sprintf("K = exp(%.3f) = %.1f",
          post$mean[post$parameter == "log_K"],
          exp(post$mean[post$parameter == "log_K"])),
  sprintf("psi = exp(%.3f) = %.1f",
          post$mean[post$parameter == "log_psi"],
          exp(post$mean[post$parameter == "log_psi"])),
  sprintf("rho = sigmoid(%.3f) = %.4f",
          post$mean[post$parameter == "mu_rho"],
          sigmoid(post$mean[post$parameter == "mu_rho"])),
  sprintf("r = %.3f",   post$mean[post$parameter == "r"]),
  sprintf("kappa = %.4f", post$mean[post$parameter == "kappa"]),
  sprintf("gamma = %.3f", post$mean[post$parameter == "gamma"])
)

cat("\n=== Posterior Summary ===\n")
cat(sprintf("Final epsilon: %.6f\n\n", result$epsilon_final))
print(post)
cat("\nNatural-scale means:\n")
for (s in post_nat$param_natural) cat(" ", s, "\n")

# ── 9. Weighted posterior quantiles ─────────────────────────────────────────
wq <- function(vals, wts, probs = c(0.025, 0.5, 0.975)) {
  ord <- order(vals)
  cw  <- cumsum(wts[ord])
  vapply(probs, function(p) vals[ord][which(cw >= p)[1L]], numeric(1L))
}

keys <- c("phi_logit", "log_K", "log_psi", "mu_rho", "r", "kappa", "gamma")
nat_fn <- list(
  phi_logit = sigmoid,
  log_K     = exp,
  log_psi   = exp,
  mu_rho    = sigmoid,
  r         = identity,
  kappa     = identity,
  gamma     = identity
)
nat_name <- c(phi_logit="phi", log_K="K", log_psi="psi",
              mu_rho="rho", r="r", kappa="kappa", gamma="gamma")

cat("\n=== Posterior Credible Intervals (natural scale) ===\n")
cat(sprintf("%-8s %10s %10s %10s\n", "Param", "2.5%", "Median", "97.5%"))
cat(strrep("-", 42), "\n")

for (k in keys) {
  vals <- vapply(result$particles, `[[`, numeric(1L), k)
  q    <- wq(vals, result$weights)
  fn   <- nat_fn[[k]]
  cat(sprintf("%-8s %10.4f %10.4f %10.4f\n",
              nat_name[k], fn(q[1L]), fn(q[2L]), fn(q[3L])))
}

# ── 10. Save results ─────────────────────────────────────────────────────────
saveRDS(result, "D:/research/inference/abc_result_neon.rds")
write.csv(post,  "D:/research/inference/posterior_summary.csv", row.names = FALSE)

# Save full particle table
particle_df <- do.call(rbind, lapply(seq_along(result$particles), function(i) {
  th <- result$particles[[i]]
  data.frame(
    particle  = i,
    weight    = result$weights[i],
    phi_logit = th$phi_logit,
    phi       = sigmoid(th$phi_logit),
    log_K     = th$log_K,
    K         = exp(th$log_K),
    log_psi   = th$log_psi,
    psi       = exp(th$log_psi),
    mu_rho    = th$mu_rho,
    rho       = sigmoid(th$mu_rho),
    r         = th$r,
    kappa     = th$kappa,
    gamma     = th$gamma
  )
}))
write.csv(particle_df, "D:/research/inference/particles_neon.csv", row.names = FALSE)

cat("\nResults saved:\n")
cat("  D:/research/inference/abc_result_neon.rds\n")
cat("  D:/research/inference/posterior_summary.csv\n")
cat("  D:/research/inference/particles_neon.csv\n")
