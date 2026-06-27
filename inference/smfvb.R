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

# ── Variational EM: M-step parameter updates ─────────────────────────────────
#
# Each function maximises L~ w.r.t. one scalar parameter via Newton-Raphson,
# using the plug-in approximation E_q[N_{i,t}] = mu_{i,t}. The Hessian at the
# optimum provides a Laplace-approximation standard error (SE).
#
# Gradient and Hessian derivations follow directly from the surrogate
# variational objective (Theorem 4) evaluated at the variational mean.

#' M-step for rho: NegBin count-detection fraction, rho in (0,1).
#' Prior: Beta(prior_a, prior_b).  Default Beta(2,2) is weakly informative.
mstep_rho <- function(Y_C, mu, r_nb, rho_init = 0.5,
                       prior_a = 2.0, prior_b = 2.0,
                       max_iter = 80L, tol = 1e-9) {
  obs   <- which(!is.na(Y_C))
  y_v   <- as.numeric(Y_C)[obs]
  mu_v  <- as.numeric(mu)[obs]
  r_v   <- if (length(r_nb) == 1L) rep(r_nb, length(obs)) else {
              r_mat <- matrix(rep(r_nb, ncol(Y_C)), nrow(Y_C), ncol(Y_C))
              as.numeric(r_mat)[obs]
            }
  rho <- max(0.02, min(0.98, rho_init))
  h   <- NA_real_
  for (k in seq_len(max_iter)) {
    d <- r_v + rho * mu_v
    g <- sum(y_v / rho - (r_v + y_v) * mu_v / d) +
         (prior_a - 1) / rho - (prior_b - 1) / (1 - rho)
    h <- sum(-y_v / rho^2 + (r_v + y_v) * mu_v^2 / d^2) -
         (prior_a - 1) / rho^2 - (prior_b - 1) / (1 - rho)^2
    if (abs(g) < tol || h >= 0) break
    rho <- max(0.02, min(0.98, rho - g / h))
  }
  list(est = rho, se = if (!is.na(h) && h < 0) sqrt(-1 / h) else NA_real_)
}

#' M-step for kappa: Bernoulli detection rate, kappa > 0.
#' Prior: Gamma(prior_alpha, prior_beta) with mean alpha*beta.
#' Default Gamma(1.5, 8) centres near 0.12 with moderate spread.
mstep_kappa <- function(Y_P, mu, kappa_init = 0.1,
                         prior_alpha = 1.5, prior_beta = 8.0,
                         max_iter = 80L, tol = 1e-9) {
  obs1  <- which(!is.na(Y_P) & Y_P == 1L)
  obs0  <- which(!is.na(Y_P) & Y_P == 0L)
  mu1   <- as.numeric(mu)[obs1]
  mu0   <- as.numeric(mu)[obs0]
  kappa <- max(1e-4, kappa_init)
  h     <- NA_real_
  for (k in seq_len(max_iter)) {
    ek <- exp(-kappa)
    a  <- 1 - ek                        # alpha = 1 - e^{-kappa}
    M1 <- exp(-mu1 * a)                 # E_q[e^{-kappa N}] at Y_P=1 sites
    M0 <- exp(-mu0 * a)                 # same at Y_P=0 sites
    # gradient d L~/d kappa
    g  <- sum(mu1 * ek * M1 / pmax(1 - M1, 1e-12)) -
           sum(mu0 * ek) +
           (prior_alpha - 1) / kappa - 1 / prior_beta
    # hessian d^2 L~/d kappa^2
    # d/dk [mu*ek*M/(1-M)] = mu*ek*M/(1-M) * (-1 - mu*ek/(1-M))  [chain rule]
    # d/dk [-mu0*ek] = mu0*ek
    h1 <- sum(mu1 * ek * M1 / pmax(1 - M1, 1e-12) *
               (-1 - mu1 * ek / pmax(1 - M1, 1e-12)))
    h0 <- sum(mu0 * ek)
    h  <- h1 + h0 - (prior_alpha - 1) / kappa^2
    if (abs(g) < tol || h >= 0) break
    kappa <- max(1e-4, kappa - g / h)
  }
  list(est = kappa, se = if (!is.na(h) && h < 0) sqrt(-1 / h) else NA_real_)
}

#' M-step for psi: LDD immigration mean, psi > 0.
#' mu_hat_base: [m, T] matrix of process predictions with psi contribution
#'   removed (i.e., surv_mean + immigration).  Since mu_hat = base + psi,
#'   d(-KL)/d psi = sum(mu / (base + psi) - 1).
#' Prior: Gamma(prior_alpha, prior_beta).
mstep_psi <- function(mu, mu_hat_base, psi_init = 1.0,
                       prior_alpha = 2.0, prior_beta = 2.0,
                       max_iter = 80L, tol = 1e-8) {
  psi <- max(0.01, psi_init)
  h   <- NA_real_
  for (k in seq_len(max_iter)) {
    mh  <- pmax(mu_hat_base + psi, 1e-8)
    g   <- sum(mu / mh - 1) + (prior_alpha - 1) / psi - 1 / prior_beta
    h   <- -sum(mu / mh^2) - (prior_alpha - 1) / psi^2
    if (abs(g) < tol || h >= 0) break
    psi <- max(0.01, psi - g / h)
  }
  list(est = psi, se = if (!is.na(h) && h < 0) sqrt(-1 / h) else NA_real_)
}

# ── Variational EM main loop ──────────────────────────────────────────────────
#
# Alternates:
#   E-step: CAVI sweeps over q(N_{1:T}) with fixed (rho, kappa, psi)
#   M-step: Newton-Raphson updates for the subset named in 'estimate'
#
# Parameters NOT in 'estimate' are held fixed at their initial values.

smfvb_vem <- function(Y_C, Y_P, W, phi, K,
                       psi_init   = 1.0,
                       rho_init   = 0.5,
                       r_nb       = 5.0,
                       kappa_init = 0.1,
                       estimate   = c("rho", "kappa", "psi"),
                       prior      = list(rho_a   = 2.0, rho_b   = 2.0,
                                         kappa_a = 1.5, kappa_b = 8.0,
                                         psi_a   = 2.0, psi_b   = 2.0),
                       max_vem_iter = 30L,
                       estep_max    = 100L,
                       estep_tol    = 1e-5,
                       vem_tol      = 1e-5,
                       verbose      = TRUE) {

  m  <- nrow(Y_C)
  TT <- ncol(Y_C)

  rho    <- rho_init
  kappa  <- kappa_init
  psi    <- psi_init
  r_nb_v <- if (length(r_nb) == 1L) rep(r_nb, m) else r_nb

  mu_cur    <- NULL
  elbo_prev <- -Inf

  # Storage for M-step results (for SE extraction after convergence)
  res_rho   <- list(est = rho,   se = NA_real_)
  res_kappa <- list(est = kappa, se = NA_real_)
  res_psi   <- list(est = psi,   se = NA_real_)

  param_rows <- vector("list", max_vem_iter)

  for (vi in seq_len(max_vem_iter)) {

    # ── E-step ──────────────────────────────────────────────────────────────
    fit_e <- smfvb(
      Y_C      = Y_C,     Y_P   = Y_P,
      W        = W,       phi   = phi,   K     = K,
      psi      = rep(psi,   m),
      rho      = rep(rho,   m),
      r_nb     = r_nb_v,
      kappa    = rep(kappa, m),
      mu_init  = mu_cur,
      max_iter = estep_max,
      tol      = estep_tol,
      verbose  = FALSE
    )
    mu_cur <- fit_e$mu
    elbo   <- tail(fit_e$elbo_trace, 1L)

    # ── M-step ──────────────────────────────────────────────────────────────
    if ("rho" %in% estimate) {
      res_rho <- mstep_rho(Y_C, mu_cur, r_nb_v,
                            rho_init = rho,
                            prior_a  = prior$rho_a,
                            prior_b  = prior$rho_b)
      rho <- res_rho$est
    }

    if ("kappa" %in% estimate) {
      res_kappa <- mstep_kappa(Y_P, mu_cur,
                               kappa_init  = kappa,
                               prior_alpha = prior$kappa_a,
                               prior_beta  = prior$kappa_b)
      kappa <- res_kappa$est
    }

    if ("psi" %in% estimate) {
      # mu_hat_base = process prediction with psi = 0 at each time step
      # mu_hat(psi) = base + psi  =>  d(-KL)/d psi = sum(mu/(base+psi) - 1)
      mu0_implied <- rowMeans(pmax(Y_C / matrix(rho, m, TT), 0.5), na.rm = TRUE)
      mu0_implied[!is.finite(mu0_implied)] <- mean(psi) + 1
      base <- matrix(0.0, m, TT)
      base[, 1L] <- predict_process(mu0_implied, W, phi, K, rep(0, m))
      for (tt in seq(2L, TT))
        base[, tt] <- predict_process(mu_cur[, tt - 1L], W, phi, K, rep(0, m))
      res_psi <- mstep_psi(mu_cur, base,
                           psi_init    = psi,
                           prior_alpha = prior$psi_a,
                           prior_beta  = prior$psi_b)
      psi <- res_psi$est
    }

    param_rows[[vi]] <- data.frame(iter  = vi,   rho   = rho,
                                    kappa = kappa, psi   = psi,
                                    elbo  = elbo)

    if (verbose)
      cat(sprintf("VEM %2d  rho=%.4f  kappa=%.4f  psi=%.4f  ELBO=%11.3f\n",
                  vi, rho, kappa, psi, elbo))

    rel <- abs(elbo - elbo_prev) / (1 + abs(elbo_prev))
    if (vi > 2L && rel < vem_tol) {
      if (verbose)
        cat(sprintf("VEM converged at iter %d  (rel dELBO = %.2e)\n", vi, rel))
      break
    }
    elbo_prev <- elbo
  }

  list(
    mu          = mu_cur,
    rho         = rho,     se_rho   = res_rho$se,
    kappa       = kappa,   se_kappa = res_kappa$se,
    psi         = psi,     se_psi   = res_psi$se,
    elbo        = elbo,
    elbo_trace  = fit_e$elbo_trace,
    param_trace = do.call(rbind,
                          param_rows[!vapply(param_rows, is.null, TRUE)]),
    n_vem_iter  = vi
  )
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
