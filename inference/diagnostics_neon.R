# diagnostics_neon.R
# Posterior diagnostics and predictive checks for the NEON ABC-SMC fit.

stopifnot(requireNamespace("R6",       quietly = TRUE))
stopifnot(requireNamespace("jsonlite", quietly = TRUE))

PKG <- "D:/research/scabmnkd/R"
for (f in c("utils", "grid", "kernel", "model",
            "summaries", "prior", "abc_smc", "build_model")) {
  source(file.path(PKG, paste0(f, ".R")))
}
source("D:/research/data/processed/load_scabm_data.R")

# ── Load data and fitted particles ──────────────────────────────────────────
dat    <- load_scabm_data("D:/research/data/processed")
result <- readRDS("D:/research/inference/abc_result_neon.rds")
parts  <- read.csv("D:/research/inference/particles_neon.csv")

m <- dat$m; T <- dat$T
grid_neon <- list(
  m          = m,
  dists      = dat$grid$dist_km,
  neighbours = dat$grid$neighbours,
  max_nn     = max(lengths(dat$grid$neighbours))
)

# Re-define adapted priors (for run_sim)
sample_prior <- function() list()  # not used here
sigmoid_fn   <- sigmoid             # already loaded from utils.R

# ── 1. Console: weighted posterior quantiles ─────────────────────────────────
wq <- function(vals, wts, probs = c(0.025, 0.25, 0.5, 0.75, 0.975)) {
  ord <- order(vals); cw <- cumsum(wts[ord])
  vapply(probs, function(p) vals[ord][which(cw >= p)[1L]], numeric(1L))
}
wts <- result$weights

cat("=== Weighted posterior distribution (natural scale) ===\n")
cat(sprintf("%-8s %8s %8s %8s %8s %8s\n","param","2.5%","25%","50%","75%","97.5%"))
cat(strrep("-",50),"\n")
params_list <- list(
  phi   = sigmoid(parts$phi_logit),
  K     = exp(parts$log_K),
  psi   = exp(parts$log_psi),
  rho   = sigmoid(parts$mu_rho),
  r     = parts$r,
  kappa = parts$kappa,
  gamma = parts$gamma
)
for (nm in names(params_list)) {
  q <- wq(params_list[[nm]], wts)
  cat(sprintf("%-8s %8.4f %8.4f %8.4f %8.4f %8.4f\n",
              nm, q[1], q[2], q[3], q[4], q[5]))
}

# ── 2. Posterior predictive trajectories ─────────────────────────────────────
cat("\n=== Posterior predictive check ===\n")
N_PP   <- 200L
set.seed(42L)
idx    <- sample.int(nrow(parts), N_PP, replace = TRUE, prob = wts)

# Simulate trajectories from posterior draws
traj_list <- vector("list", N_PP)
for (ii in seq_len(N_PP)) {
  th <- as.list(parts[idx[ii], c("phi_logit","log_K","log_psi",
                                 "mu_rho","r","kappa","gamma")])
  tryCatch({
    mdl <- build_model(th, grid_neon, dat$X, dat$W)
    traj_list[[ii]] <- mdl$simulate(rep(0L, m), T)   # (T+1 x m)
  }, error = function(e) NULL)
}
traj_list <- Filter(Negate(is.null), traj_list)
cat(sprintf("Valid posterior-predictive trajectories: %d/%d\n",
            length(traj_list), N_PP))

# Site-level mean N per time step (rows 2..T+1 = steps 1..T)
pp_mean_N <- sapply(traj_list, function(tr) rowMeans(tr)[2:(T+1)])  # (T x n_valid)

# Observed cell means per time step (from Y_C)
obs_mean_C <- apply(dat$Y_C, 2L, function(col) mean(col, na.rm = TRUE))  # length T

cat(sprintf("%-12s %8s %8s %8s %8s\n","time_step","obs_YC","pp_2.5","pp_50","pp_97.5"))
cat(strrep("-", 48), "\n")
for (t in seq_len(T)) {
  pp_t <- pp_mean_N[t, ]
  q    <- quantile(pp_t, c(0.025, 0.5, 0.975))
  # Convert N to expected Y_C: mean rho × N
  rho_med <- median(sigmoid(parts$mu_rho[idx]))
  cat(sprintf("t=%2d (%s)  %8.1f %8.1f %8.1f %8.1f\n",
              t, dat$time_idx$date_CDT[t],
              obs_mean_C[t],
              rho_med * q[1], rho_med * q[2], rho_med * q[3]))
}

# ── 3. Summary statistics: observed vs posterior predictive ─────────────────
cat("\n=== Summary statistics: observed vs posterior predictive ===\n")

Y_C_obs <- rbind(rep(NA_real_, m), t(dat$Y_C))
Y_P_obs <- rbind(rep(NA_real_, m), t(dat$Y_P))
s_obs <- compute_summaries(Y_C_obs, Y_P_obs, grid_neon)

rho_draws <- sigmoid(parts$mu_rho[idx])
r_draws   <- parts$r[idx]
kappa_draws <- parts$kappa[idx]

pp_s <- matrix(NA_real_, nrow = length(traj_list), ncol = 5L)
for (ii in seq_along(traj_list)) {
  traj    <- traj_list[[ii]]
  rho_i   <- rho_draws[ii]
  r_i     <- r_draws[ii]
  kappa_i <- kappa_draws[ii]

  Y_C_s <- matrix(NA_real_, T + 1L, m)
  Y_P_s <- matrix(NA_real_, T + 1L, m)
  for (k in seq_len(nrow(dat$obs_C))) {
    ci <- dat$obs_C$cell_idx[k]; ti <- dat$obs_C$time_idx[k]
    N_it <- max(traj[ti+1L, ci], 0L)
    Y_C_s[ti+1L, ci] <- rnbinom(1L, size=max(r_i,1e-3), mu=rho_i*N_it+1e-12)
  }
  for (k in seq_len(nrow(dat$obs_P))) {
    ci <- dat$obs_P$cell_idx[k]; ti <- dat$obs_P$time_idx[k]
    N_it <- max(traj[ti+1L, ci], 0L)
    Y_P_s[ti+1L, ci] <- rbinom(1L, 1L, pmin(1-exp(-kappa_i*N_it), 1-1e-12))
  }
  sv <- tryCatch(compute_summaries(Y_C_s, Y_P_s, grid_neon), error=function(e) NULL)
  if (!is.null(sv) && all(is.finite(sv))) pp_s[ii, ] <- sv
}
pp_s <- pp_s[complete.cases(pp_s), , drop = FALSE]

stat_names <- c("s1 mean Y_C","s2 growth rate","s3 Moran's I","s4 occupancy","s5 lag-1 AC")
cat(sprintf("%-16s %10s %10s %10s %10s\n","stat","obs","pp_2.5","pp_50","pp_97.5"))
cat(strrep("-", 58), "\n")
for (j in seq_len(5L)) {
  q <- quantile(pp_s[, j], c(0.025, 0.5, 0.975), na.rm = TRUE)
  cat(sprintf("%-16s %10.4f %10.4f %10.4f %10.4f\n",
              stat_names[j], s_obs[j], q[1], q[2], q[3]))
}

# ── 4. Effective sample size ─────────────────────────────────────────────────
ess <- 1 / sum(wts^2)
cat(sprintf("\nEffective sample size (ESS): %.1f / %d particles (%.1f%%)\n",
            ess, nrow(parts), 100 * ess / nrow(parts)))

# ── 5. Save posterior predictive mean trajectory ──────────────────────────────
pp_traj_df <- data.frame(
  time_idx   = seq_len(T),
  date_CDT   = dat$time_idx$date_CDT,
  day_of_year = dat$time_idx$day_of_year,
  obs_mean_YC = obs_mean_C,
  pp_mean_N_2.5  = apply(pp_mean_N, 1, quantile, 0.025),
  pp_mean_N_50   = apply(pp_mean_N, 1, quantile, 0.500),
  pp_mean_N_97.5 = apply(pp_mean_N, 1, quantile, 0.975)
)
write.csv(pp_traj_df, "D:/research/inference/pp_trajectory.csv", row.names = FALSE)
cat("\nSaved: D:/research/inference/pp_trajectory.csv\n")
