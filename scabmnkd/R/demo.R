#' SC-ABM-NKD demonstration on a synthetic 8x8 grid
#'
#' @description
#' Generates ground-truth population trajectories on an 8x8 spatial grid,
#' produces synthetic count and binary observations, computes summary statistics,
#' then runs ABC-SMC to recover the scalar parameters.
#'
#' @param nrow       Grid rows (default 8).
#' @param ncol       Grid columns (default 8).
#' @param T          Number of time steps (default 10).
#' @param n_particles ABC-SMC particles per population (default 100).
#' @param n_populations ABC-SMC populations (default 3).
#' @param seed       Random seed (default 42).
#' @return Invisibly returns a list with: traj, Y_C, Y_P, s_obs, abc_result.
#' @export
run_demo <- function(
    nrow = 8L, ncol = 8L, T = 10L,
    n_particles = 100L, n_populations = 3L,
    seed = 42L
) {
  set.seed(seed)

  # ── Grid and covariates ──────────────────────────────────────────────────
  grid  <- SpatialGrid$new(nrow, ncol, cell_size = 1.0)
  m     <- grid$m
  p     <- 2L          # habitat covariate dimension
  q     <- 2L          # LDD / K covariate dimension

  X <- matrix(rnorm(m * p), nrow = m, ncol = p)   # environmental covariates
  W <- matrix(rnorm(m * q), nrow = m, ncol = q)   # LDD / K covariates

  z_dim <- p + 1L + grid$max_nn

  # ── True parameters ──────────────────────────────────────────────────────
  true_params <- ModelParams(
    beta       = c(0.4, -0.2),
    beta_K     = c(3.0,  0.2),
    beta_phi   = c(0.5, -0.1),
    gamma_ldd  = c(-2.0, 0.0),
    r          = 5.0,
    kappa      = 0.30,
    gamma      = 8.0,
    mu_rho     = 0.5,
    sigma2_rho = 0.30
  )
  true_kernel <- NeuralDispersalKernel$new(z_dim)
  true_model  <- SCABMNKD$new(grid, X, W, true_kernel, true_params)

  # ── Ground-truth trajectory ───────────────────────────────────────────────
  N0   <- rpois(m, lambda = 10)
  traj <- true_model$simulate(N0, T)          # (T+1, m)

  message(sprintf(
    "Ground-truth: mean N = %.2f, max N = %d",
    mean(traj), max(traj)
  ))

  # ── Generate observations ─────────────────────────────────────────────────
  Y_C <- matrix(NA_integer_, nrow = T + 1L, ncol = m)
  Y_P <- matrix(NA_integer_, nrow = T + 1L, ncol = m)
  for (t in seq_len(T + 1L)) {
    Y_C[t, ] <- true_model$observe_count(traj[t, ])
    Y_P[t, ] <- true_model$observe_binary(traj[t, ])
  }

  # ── Summary statistics ────────────────────────────────────────────────────
  s_obs <- compute_summaries(Y_C, Y_P, grid)
  message("\nObserved summary statistics:")
  labs <- c("mean count", "growth rate", "Moran's I", "occupancy", "lag-1 AC")
  for (i in seq_along(s_obs)) {
    message(sprintf("  %-16s: %8.4f", labs[i], s_obs[i]))
  }

  # ── ABC-SMC ───────────────────────────────────────────────────────────────
  run_sim <- function(theta) {
    tryCatch({
      mod    <- build_model(theta, grid, X, W)
      N0_s   <- rpois(m, lambda = 10)
      tr_s   <- mod$simulate(N0_s, T)
      YC_s   <- matrix(NA_real_, nrow = T + 1L, ncol = m)
      YP_s   <- matrix(NA_real_, nrow = T + 1L, ncol = m)
      for (t in seq_len(T + 1L)) {
        YC_s[t, ] <- mod$observe_count(tr_s[t, ])
        YP_s[t, ] <- mod$observe_binary(tr_s[t, ])
      }
      compute_summaries(YC_s, YP_s, grid)
    }, error = function(e) NULL)
  }

  message(sprintf(
    "\nRunning ABC-SMC (%d particles x %d populations) ...",
    n_particles, n_populations
  ))
  abc_result <- abc_smc(
    s_obs          = s_obs,
    run_sim        = run_sim,
    n_particles    = n_particles,
    n_populations  = n_populations,
    alpha_q        = 0.50,
    n_pilot        = 200L,
    verbose        = TRUE
  )

  # ── Posterior summary ─────────────────────────────────────────────────────
  message("\nPosterior summary (weighted mean +/- sd):")
  ps <- posterior_summary(abc_result)
  message(sprintf("  %-16s  %8s  %8s", "parameter", "mean", "sd"))
  for (i in seq_len(nrow(ps))) {
    message(sprintf("  %-16s  %8.3f  %8.3f", ps$parameter[i], ps$mean[i], ps$sd[i]))
  }

  invisible(list(
    traj       = traj,
    Y_C        = Y_C,
    Y_P        = Y_P,
    s_obs      = s_obs,
    abc_result = abc_result
  ))
}
