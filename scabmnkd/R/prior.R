#' Sample one parameter set from the prior
#'
#' @description
#' Draws a named list of scalar parameters from the prior distributions
#' specified in Table 2 of the SC-ABM-NKD paper.
#'
#' @return A named list with elements: phi_logit, log_K, log_psi,
#'   mu_rho, r, kappa, gamma.
#' @export
sample_prior <- function() {
  r     <- rgamma(1, shape = 2.0, rate = 0.5)       # Gamma(2, 0.5) -> mean 4
  kappa <- rgamma(1, shape = 1.0, rate = 1.0)        # Gamma(1, 1)   -> mean 1
  gamma <- rgamma(1, shape = 2.0, rate = 0.2)        # Gamma(2, 0.2) -> mean 10

  list(
    phi_logit = rnorm(1, mean = 0.0, sd = 1.0),     # logit(phi) intercept
    log_K     = rnorm(1, mean = 3.0, sd = 1.0),     # log(K) intercept
    log_psi   = rnorm(1, mean = -2.0, sd = 1.0),    # log(psi) intercept
    mu_rho    = rnorm(1, mean = 0.0, sd = 1.0),     # logit(rho) intercept
    r         = max(r,     1e-3),
    kappa     = max(kappa, 1e-4),
    gamma     = max(gamma, 1e-4)
  )
}


#' Evaluate the log-prior density
#'
#' @description
#' Returns the log-prior density (up to an additive constant) for the
#' scalar ABC parameter set theta.
#'
#' @param theta A named list as returned by \code{sample_prior()}.
#' @return Scalar log-prior value (-Inf if parameters are out of support).
#' @export
log_prior <- function(theta) {
  r     <- theta$r
  kappa <- theta$kappa
  gamma <- theta$gamma

  if (r <= 0 || kappa <= 0 || gamma <= 0) return(-Inf)

  lp <- 0.0
  # Normal priors on logit / log-scale parameters
  lp <- lp - 0.5 * theta$phi_logit ^ 2                    # N(0, 1)
  lp <- lp - 0.5 * (theta$log_K  - 3.0) ^ 2              # N(3, 1)
  lp <- lp - 0.5 * (theta$log_psi + 2.0) ^ 2             # N(-2, 1)
  lp <- lp - 0.5 * theta$mu_rho ^ 2                       # N(0, 1)

  # Gamma priors: log p(x) proportional to (a-1)*log(x) - rate*x
  lp <- lp + 1.0 * log(r)     - 2.0 * r       # Gamma(2, 0.5): a-1=1, rate=2
  lp <- lp + 0.0 * log(kappa) - 1.0 * kappa   # Gamma(1, 1):   a-1=0, rate=1
  lp <- lp + 1.0 * log(gamma) - 5.0 * gamma   # Gamma(2, 0.2): a-1=1, rate=5

  lp
}
