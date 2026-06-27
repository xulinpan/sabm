# smfvb.R
# Structured Mean-Field Variational Bayes for the count ABM.
#
# Implements the closed-form CAVI algorithm from:
#   "Closed-form Variational Inference for Count Statistical Agent-Based Models"
#
# Variational family: q(N_{i,t}) = Poisson(mu_{i,t}), all (i,t) independent.
#
# Every ELBO term is evaluated analytically via the Poisson MGF.
# The CAVI update for each mu_{i,t} solves a 1-D optimisation problem whose
# gradient and Hessian are both in closed form (no quadrature or MC).
# Newton-Raphson with warm-start and backtracking line search is used;
# the concavity of the surrogate ELBO guarantees convergence.

# ── MGF helpers ───────────────────────────────────────────────────────────────

# E[e^{s*N}] for N ~ Poisson(mu)
mgf_pois <- function(mu, s) exp(mu * (exp(s) - 1))

# E[N * e^{s*N}] = d/ds M_N(s) = mu * e^s * M_N(s)
mgf_pois_deriv <- function(mu, s) mu * exp(s) * mgf_pois(mu, s)

# ── Process prediction ─────────────────────────────────────────────────────────

# E_q[S_{i,t}] = phi_i * E_q[N_{i,t-1} * e^{-N_{i,t-1}/K_i}]
#              = phi_i * mu * e^{-1/K} * exp(mu*(e^{-1/K} - 1))   (MGF trick)
survival_mean <- function(mu_prev, phi, K) {
  phi * mgf_pois_deriv(mu_prev, -1.0 / K)
}

# mu_hat_{i,t} = E_q[S_{i,t}] + sum_j W_{ij} mu_{j,t-1} + psi_i
predict_process <- function(mu_prev, W, phi, K, psi) {
  surv  <- survival_mean(mu_prev, phi, K)
  immig <- as.vector(W %*% mu_prev)
  pmax(surv + immig + psi, 1e-8)
}

# ── Surrogate ELBO terms ──────────────────────────────────────────────────────

# KL[Poisson(mu) || Poisson(mu_hat)] = mu*log(mu/mu_hat) - mu + mu_hat
kl_pois <- function(mu, mu_hat) {
  mu * log(pmax(mu, 1e-12) / pmax(mu_hat, 1e-12)) - mu + mu_hat
}

# NegBin log-likelihood (plug in E_q[N] = mu)
negbin_ll <- function(y, mu, rho, r_nb) {
  mu_obs <- rho * mu + 1e-12
  lgamma(r_nb + y) - lgamma(r_nb) - lgamma(y + 1) +
    r_nb * log(r_nb / (r_nb + mu_obs)) +
    y    * log(mu_obs / (r_nb + mu_obs))
}

# NegBin gradient d(log p)/d(mu)
negbin_grad <- function(y, mu, rho, r_nb) {
  mu_obs <- rho * mu + 1e-12
  y / mu - rho * (r_nb + y) / (r_nb + mu_obs)
}

# NegBin Hessian d^2(log p)/d(mu)^2  (always <= 0)
negbin_hess <- function(y, mu, rho, r_nb) {
  mu_obs <- rho * mu + 1e-12
  -y / mu^2 + rho^2 * (r_nb + y) / (r_nb + mu_obs)^2
}

# Bernoulli detection log-likelihood via Poisson MGF.
# p(Y=1|N) = 1 - exp(-kappa*N)
# E_q[e^{-kappa*N}] = M_N(-kappa) = exp(-mu * alpha), alpha = 1 - e^{-kappa}
# log p(Y=1) ~= log(1 - M_N(-kappa))   [Jensen, O(mu^{-1}) error]
# log p(Y=0)  = log M_N(-kappa) = -mu * alpha   [exact]
bern_ll <- function(y, mu, kappa) {
  alpha <- 1 - exp(-kappa)
  M_neg <- exp(-mu * alpha)
  if (y == 1L) log(pmax(1 - M_neg, 1e-12)) else -mu * alpha
}

# Bernoulli gradient d(log p)/d(mu)
bern_grad <- function(y, mu, kappa) {
  alpha <- 1 - exp(-kappa)
  M_neg <- exp(-mu * alpha)
  if (y == 1L) alpha * M_neg / pmax(1 - M_neg, 1e-12) else -alpha
}

# Bernoulli Hessian d^2(log p)/d(mu)^2  (Y=1: <= 0; Y=0: 0)
bern_hess <- function(y, mu, kappa) {
  if (y != 1L) return(0.0)
  alpha <- 1 - exp(-kappa)
  M_neg <- exp(-mu * alpha)
  -alpha^2 * M_neg / pmax(1 - M_neg, 1e-12)^2
}

# ── CAVI update for a single coordinate (i, t) ────────────────────────────────
#
# Solves d L_tilde / d mu = 0 via Newton-Raphson with backtracking.
#
# Gradient:   g(mu) = log(mu_hat/mu) + obs_grad(mu)
# Hessian:    h(mu) = -1/mu + obs_hess(mu)   [always < 0: objective is concave]
# NR step:    delta = -g/h
# Backtrack:  halve step until |g(mu + step*delta)| < |g(mu)|
#
# mu_start: warm-start from previous iterate (crucial for fast convergence when
# mu and mu_hat differ by orders of magnitude, e.g. at t=1 with psi=0.15).

cavi_update_single <- function(mu_hat, yc, yp, rho, r_nb, kappa,
                               mu_start = NULL, max_nr = 50L, tol = 1e-7) {
  mu <- if (!is.null(mu_start) && is.finite(mu_start) && mu_start > 1e-8) {
    mu_start
  } else {
    mu_hat
  }

  for (k in seq_len(max_nr)) {
    g <- log(mu_hat / mu)
    h <- -1.0 / mu

    if (!is.na(yc)) {
      g <- g + negbin_grad(yc, mu, rho, r_nb)
      h <- h + negbin_hess(yc, mu, rho, r_nb)
    }
    if (!is.na(yp)) {
      g <- g + bern_grad(yp, mu, kappa)
      h <- h + bern_hess(yp, mu, kappa)
    }

    if (abs(g) < tol) break

    delta <- -g / h    # h < 0 always: delta has same sign as g

    # Backtracking: halve step until |gradient| decreases
    step <- 1.0
    for (ls in seq_len(30L)) {
      mu_try <- mu + step * delta
      if (mu_try < 1e-8) { step <- step / 2.0; next }

      g_try <- log(mu_hat / mu_try)
      if (!is.na(yc)) g_try <- g_try + negbin_grad(yc, mu_try, rho, r_nb)
      if (!is.na(yp)) g_try <- g_try + bern_grad(yp, mu_try, kappa)

      if (abs(g_try) < abs(g)) break
      step <- step / 2.0
    }
    mu <- pmax(mu + step * delta, 1e-8)
  }
  mu
}

# ── Full surrogate ELBO ───────────────────────────────────────────────────────
compute_elbo <- function(mu, Y_C, Y_P, W, phi, K, psi, rho, r_nb, kappa) {
  m  <- nrow(mu)
  TT <- ncol(mu)
  elbo <- 0.0

  for (t in seq_len(TT)) {
    mu_hat <- if (t == 1L) pmax(psi, 1e-8) else
              predict_process(mu[, t - 1L], W, phi, K, psi)
    elbo <- elbo - sum(kl_pois(mu[, t], mu_hat))

    for (i in seq_len(m)) {
      if (!is.na(Y_C[i, t]))
        elbo <- elbo + negbin_ll(Y_C[i, t], mu[i, t], rho[i], r_nb[i])
      if (!is.na(Y_P[i, t]))
        elbo <- elbo + bern_ll(Y_P[i, t], mu[i, t], kappa[i])
    }
  }
  elbo
}

# ── Main SMFVB function ────────────────────────────────────────────────────────
#
# Arguments:
#   Y_C, Y_P  [m, T]  observed counts / binary detections (NA = missing)
#   W         [m, m]  immigration weight matrix (need NOT be row-normalised)
#   phi, K, psi, rho, r_nb, kappa  [m] cell-level parameters
#   mu0       [m]     initial population at t=0 (before the survey period).
#                     Used to compute the process prediction for t=1 via the
#                     full transition model. Defaults to the observation-implied
#                     site means: rowMeans(Y_C/rho).
#   mu_init   [m, T]  optional full warm-start matrix
#   max_iter, tol, verbose  convergence control

smfvb <- function(Y_C, Y_P, W, phi, K, psi, rho, r_nb, kappa,
                  mu0      = NULL,
                  mu_init  = NULL,
                  max_iter = 200L,
                  tol      = 1e-5,
                  verbose  = TRUE) {

  m  <- nrow(Y_C)
  TT <- ncol(Y_C)
  stopifnot(nrow(Y_P) == m, ncol(Y_P) == TT)

  # Initial population at t=0: use observation-implied site means if not given.
  if (is.null(mu0)) {
    obs_mean <- rowMeans(Y_C / matrix(rho, m, TT), na.rm = TRUE)
    obs_mean[!is.finite(obs_mean)] <- mean(psi) + mean(K) * mean(phi) / 4
    mu0 <- pmax(obs_mean, 0.5)
  }

  # Process prediction for t=1 from mu0 (proper boundary condition).
  mu_hat_t1 <- predict_process(mu0, W, phi, K, psi)

  # Initialise variational means from observation-implied values.
  if (is.null(mu_init)) {
    init_obs <- Y_C / matrix(rho, m, TT)
    init_row <- rowMeans(init_obs, na.rm = TRUE)
    init_row[!is.finite(init_row)] <- mean(mu0)
    mu <- matrix(pmax(init_row, 0.5), m, TT)
  } else {
    mu <- mu_init
  }

  elbo_prev  <- -Inf
  elbo_trace <- numeric(max_iter)

  for (iter in seq_len(max_iter)) {

    # Forward CAVI sweep: update mu_{i,t} in temporal order, warm-starting
    # each coordinate from the current iterate mu[i, t].
    for (t in seq_len(TT)) {
      mu_hat <- if (t == 1L) mu_hat_t1 else
                predict_process(mu[, t - 1L], W, phi, K, psi)

      for (i in seq_len(m)) {
        mu[i, t] <- cavi_update_single(
          mu_hat  = mu_hat[i],
          yc      = Y_C[i, t],
          yp      = Y_P[i, t],
          rho     = rho[i],
          r_nb    = r_nb[i],
          kappa   = kappa[i],
          mu_start = mu[i, t]    # warm start from current iterate
        )
      }
    }

    # Evaluate surrogate ELBO
    elbo <- compute_elbo(mu, Y_C, Y_P, W, phi, K, psi, rho, r_nb, kappa)
    elbo_trace[iter] <- elbo

    if (verbose && (iter <= 5L || iter %% 10L == 0L))
      cat(sprintf("  iter %3d  ELBO = %10.3f\n", iter, elbo))

    rel_change <- abs(elbo - elbo_prev) / (1.0 + abs(elbo_prev))
    if (iter > 1L && rel_change < tol) {
      if (verbose)
        cat(sprintf("  Converged at iter %d  (rel |dELBO| = %.2e)\n",
                    iter, rel_change))
      elbo_trace <- elbo_trace[seq_len(iter)]
      break
    }
    elbo_prev <- elbo
  }

  list(mu = mu, elbo_trace = elbo_trace, n_iter = length(elbo_trace))
}

# ── Posterior predictive replicates ──────────────────────────────────────────
smfvb_ppcheck <- function(fit, rho, r_nb, kappa, R = 500L) {
  mu  <- fit$mu
  m   <- nrow(mu)
  TT  <- ncol(mu)

  Y_C_rep <- array(NA_integer_, c(m, TT, R))
  Y_P_rep <- array(NA_integer_, c(m, TT, R))

  for (r in seq_len(R)) {
    N_draw <- matrix(rpois(m * TT, lambda = as.vector(mu)), m, TT)
    for (t in seq_len(TT)) {
      mu_obs          <- rho * N_draw[, t] + 1e-12
      Y_C_rep[, t, r] <- rnbinom(m, size = r_nb, mu = mu_obs)
      p_det           <- 1 - exp(-kappa * N_draw[, t])
      Y_P_rep[, t, r] <- rbinom(m, 1L, prob = p_det)
    }
  }
  list(Y_C_rep = Y_C_rep, Y_P_rep = Y_P_rep)
}
