#' Weighted Euclidean ABC distance
#'
#' @description
#' Implements the standardised distance metric (Eq 18):
#'   rho(s, s') = sum_k (s_k - s'_k)^2 / Var_pilot(s_k)
#'
#' @param s_sim    Numeric vector (5,) from simulator.
#' @param s_obs    Numeric vector (5,) observed summary statistics.
#' @param var_pilot Numeric vector (5,) of pilot-estimated variances.
#' @return Scalar non-negative distance.
#' @export
abc_distance <- function(s_sim, s_obs, var_pilot) {
  sum((s_sim - s_obs)^2 / (var_pilot + 1e-12))
}


# ── Internal helpers ───────────────────────────────────────────────────────────

.gauss_kern <- function(x, bw) {
  exp(-0.5 * (x / (bw + 1e-15))^2) / (bw * sqrt(2 * pi) + 1e-15)
}

.keys <- c("phi_logit", "log_K", "log_psi", "mu_rho", "r", "kappa", "gamma")

.particle_val <- function(particles, key) {
  vapply(particles, `[[`, numeric(1L), key)
}


#' ABC-SMC with adaptive tolerance
#'
#' @description
#' Implements Algorithm 1 from the SC-ABM-NKD paper.
#' Runs L SMC populations with decreasing acceptance thresholds epsilon_l.
#' The Gaussian perturbation kernel uses Silverman's bandwidth rule.
#'
#' Note: This uses the full simulator as run_sim (the Neural Process emulator
#' of Section 6 can be substituted as a drop-in replacement for run_sim).
#'
#' @param s_obs         Named numeric vector of observed summaries.
#' @param run_sim       Function: theta_list -> numeric vector (same length as s_obs) or NULL on failure.
#' @param n_particles   Number of SMC particles (default 300).
#' @param n_populations Number of SMC populations L (default 5).
#' @param alpha_q       Quantile for adaptive tolerance schedule (default 0.40).
#' @param n_pilot       Number of pilot draws for variance estimation (default 300).
#' @param verbose       Print progress messages (default TRUE).
#' @param param_keys    Character vector of parameter names to perturb (default: the 7 standard keys).
#' @return A list with elements:
#'   \describe{
#'     \item{particles}{List of n_particles named parameter lists.}
#'     \item{weights}{Normalised importance weights (length n_particles).}
#'     \item{var_pilot}{Vector of pilot summary-statistic variances.}
#'     \item{epsilon_final}{Final ABC tolerance.}
#'     \item{param_keys}{Character vector of parameter names used.}
#'   }
#' @export
abc_smc <- function(
    s_obs,
    run_sim,
    n_particles    = 300L,
    n_populations  = 5L,
    alpha_q        = 0.40,
    n_pilot        = 300L,
    verbose        = TRUE,
    param_keys     = NULL
) {
  keys    <- if (is.null(param_keys)) .keys else param_keys
  n_stats <- length(s_obs)

  # ── Pilot phase: estimate summary-statistic variances ─────────────────────
  if (verbose) message("Pilot phase: ", n_pilot, " prior draws ...")
  pilot_s <- matrix(NA_real_, nrow = n_pilot, ncol = n_stats)
  n_valid <- 0L
  for (i in seq_len(n_pilot)) {
    theta <- sample_prior()
    s     <- tryCatch(run_sim(theta), error = function(e) NULL)
    if (!is.null(s) && length(s) == n_stats && all(is.finite(s))) {
      n_valid <- n_valid + 1L
      pilot_s[n_valid, ] <- s
    }
  }
  if (n_valid < 10L)
    stop("Too few valid pilot simulations; check run_sim.")
  pilot_s   <- pilot_s[seq_len(n_valid), , drop = FALSE]
  var_pilot <- apply(pilot_s, 2L, var) + 1e-12

  # ── Population 1: sample from prior ────────────────────────────────────────
  if (verbose) message("Population 1/", n_populations, " (from prior) ...")
  particles <- vector("list", n_particles)
  all_dists <- numeric(n_particles)
  n_acc <- 0L

  while (n_acc < n_particles) {
    theta <- sample_prior()
    s     <- tryCatch(run_sim(theta), error = function(e) NULL)
    if (is.null(s) || length(s) != n_stats || !all(is.finite(s))) next
    n_acc <- n_acc + 1L
    particles[[n_acc]] <- theta
    all_dists[n_acc]   <- abc_distance(s, s_obs, var_pilot)
  }

  epsilon <- quantile(all_dists, probs = alpha_q)
  weights <- rep(1.0 / n_particles, n_particles)
  if (verbose) message("  epsilon_1 = ", round(epsilon, 4L))

  # ── Populations 2 … L ──────────────────────────────────────────────────────
  for (ell in seq(2L, n_populations)) {
    if (verbose) message("Population ", ell, "/", n_populations,
                         "  epsilon = ", round(epsilon, 4L), " ...")

    # Silverman bandwidth per parameter
    bw <- setNames(
      vapply(keys, function(k) {
        v <- .particle_val(particles, k)
        max(1.06 * sd(v) * n_particles^(-0.2), 1e-8)
      }, numeric(1L)),
      keys
    )

    new_particles <- vector("list", n_particles)
    new_dists     <- numeric(n_particles)
    n_acc <- 0L

    while (n_acc < n_particles) {
      # Perturb a particle drawn proportional to weights
      idx     <- sample.int(n_particles, size = 1L, prob = weights)
      theta_n <- particles[[idx]]
      theta_star <- lapply(setNames(keys, keys), function(k) {
        theta_n[[k]] + rnorm(1L, sd = bw[k])
      })

      if (!is.finite(log_prior(theta_star))) next

      s <- tryCatch(run_sim(theta_star), error = function(e) NULL)
      if (is.null(s) || length(s) != n_stats || !all(is.finite(s))) next
      d <- abc_distance(s, s_obs, var_pilot)
      if (d > epsilon) next

      n_acc <- n_acc + 1L
      new_particles[[n_acc]] <- theta_star
      new_dists[n_acc]       <- d
    }

    # Importance weights  w_n ∝ pi(theta*) / sum_m w_m K(theta*|theta_m)
    new_w <- numeric(n_particles)
    for (n_idx in seq_len(n_particles)) {
      th <- new_particles[[n_idx]]
      kern_sum <- sum(vapply(seq_len(n_particles), function(m_idx) {
        weights[m_idx] * prod(vapply(keys, function(k) {
          .gauss_kern(th[[k]] - particles[[m_idx]][[k]], bw[k])
        }, numeric(1L)))
      }, numeric(1L)))
      new_w[n_idx] <- exp(log_prior(th)) / (kern_sum + 1e-300)
    }

    weights   <- new_w / (sum(new_w) + 1e-300)
    particles <- new_particles
    dists     <- new_dists
    epsilon   <- quantile(dists, probs = alpha_q)
  }

  list(
    particles     = particles,
    weights       = weights,
    var_pilot     = var_pilot,
    epsilon_final = as.numeric(epsilon),
    param_keys    = keys
  )
}


#' Posterior summary table
#'
#' @description
#' Computes weighted mean and weighted standard deviation for each parameter
#' from ABC-SMC output.
#'
#' @param result Output list from \code{abc_smc()}.
#' @return A data.frame with columns parameter, mean, sd.
#' @export
posterior_summary <- function(result) {
  particles <- result$particles
  weights   <- result$weights
  keys      <- if (!is.null(result$param_keys)) result$param_keys else .keys
  do.call(rbind, lapply(keys, function(k) {
    vals  <- .particle_val(particles, k)
    wmean <- sum(weights * vals)
    wsd   <- sqrt(sum(weights * (vals - wmean)^2))
    data.frame(parameter = k, mean = wmean, sd = wsd,
               stringsAsFactors = FALSE)
  }))
}
