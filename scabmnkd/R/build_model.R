#' Build a SCABMNKD model from a flat ABC parameter dict
#'
#' @description
#' Constructs a SCABMNKD simulator from the scalar ABC parameter list returned
#' by \code{sample_prior()}. Uses homogeneous (cell-constant) phi, K, psi, rho
#' across all cells, which is appropriate for the ABC inner loop where regression
#' covariates are not varied.
#'
#' @param theta  Named list as returned by \code{sample_prior()}.
#' @param grid   A SpatialGrid object.
#' @param X      (m x p) environmental covariate matrix.
#' @param W      (m x q) LDD / K covariate matrix.
#' @param kernel Optional NeuralDispersalKernel (new random kernel if NULL).
#' @return A SCABMNKD object.
#' @export
build_model <- function(theta, grid, X, W, kernel = NULL) {
  m <- grid$m
  p <- ncol(X)
  q <- ncol(W)
  z_dim <- p + 1L + grid$max_nn

  params <- ModelParams(
    beta       = rep(0.0, p),
    beta_K     = rep(0.0, q),
    beta_phi   = rep(0.0, p),
    gamma_ldd  = rep(0.0, q),
    sigma2_alpha = 0.3,
    theta_alpha  = 2.0,
    sigma2_K     = 0.01,    # near-zero: K determined by intercept only
    mu_rho      = theta$mu_rho,
    sigma2_rho  = 0.01,    # near-zero: rho determined by intercept only
    r     = max(theta$r,     1e-3),
    kappa = max(theta$kappa, 1e-4),
    gamma = max(theta$gamma, 1e-4)
  )

  if (is.null(kernel)) {
    kernel <- NeuralDispersalKernel$new(z_dim)
  }

  model <- SCABMNKD$new(grid, X, W, kernel, params)

  # Override derived cell params with scalar (homogeneous) values from theta
  model$phi   <- rep(sigmoid(theta$phi_logit), m)
  model$K     <- rep(max(exp(theta$log_K),  1.0),  m)
  model$psi   <- rep(max(exp(theta$log_psi), 1e-6), m)
  model$rho   <- rep(sigmoid(theta$mu_rho), m)

  # Refresh both feat and rev_feat with the (randomly sampled) alpha
  model$.__enclos_env__$private$precompute_features()

  model
}
