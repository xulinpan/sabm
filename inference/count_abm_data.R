# count_abm_data.R
# Abstract data format for the dual-channel count ABM.
#
# A count_abm_data object holds:
#   Y_C[m, T]  -- integer count observations (NA = not surveyed)
#   Y_P[m, T]  -- binary detection observations (NA = not surveyed)
#   W[m, m]    -- row-normalised spatial weight matrix
#   covariates -- named list of site-level numeric vectors
#
# On construction, empirical summaries are computed that drive
# data-driven parameter initialisation in count_abm_model().

new_count_abm_data <- function(Y_C, Y_P, W,
                                covariates = list(),
                                site_ids   = NULL,
                                time_ids   = NULL) {

  stopifnot(is.matrix(Y_C), is.matrix(Y_P), is.matrix(W))
  m  <- nrow(Y_C)
  TT <- ncol(Y_C)
  stopifnot(identical(dim(Y_P), c(m, TT)))
  stopifnot(identical(dim(W),   c(m, m)))
  stopifnot(all(is.na(Y_C) | (Y_C >= 0L & Y_C == floor(Y_C))))
  stopifnot(all(is.na(Y_P) | Y_P %in% c(0L, 1L)))
  if (length(covariates)) {
    stopifnot(is.list(covariates),
              all(sapply(covariates, length) == m))
  }

  yc  <- Y_C[!is.na(Y_C)]
  yp  <- Y_P[!is.na(Y_P)]

  # NegBin overdispersion via method of moments:
  # E[Y_C] = rho*mu,  Var[Y_C|N] + Var[rho*N] ≈ rho*mu + rho^2*mu/r
  # so r_hat = mu_hat^2 / max(s^2 - mu_hat, eps)
  mu_hat  <- mean(yc)
  s2_hat  <- var(yc)
  r_mom   <- mu_hat^2 / max(s2_hat - mu_hat, 0.5)
  r_mom   <- min(max(r_mom, 0.5), 20.0)

  # Lag-1 annual correlation of site-mean counts (proxy for phi)
  annual  <- colMeans(Y_C, na.rm = TRUE)
  lag1    <- if (TT >= 3) cor(annual[-TT], annual[-1L]) else 0.85
  lag1    <- min(max(lag1, 0.20), 0.98)

  empirical <- list(
    count_mean    = mu_hat,
    count_sd      = sqrt(s2_hat),
    detect_rate   = mean(yp),
    frac_C_obs    = mean(!is.na(Y_C)),   # fraction of cell-years with count
    frac_P_obs    = mean(!is.na(Y_P)),   # fraction with binary obs
    zero_frac     = mean(yc == 0L),      # fraction of observed zeros
    r_moment      = r_mom,
    phi_lag1_est  = lag1,                # lag-1 temporal correlation
    annual_means  = annual
  )

  structure(
    list(
      Y_C        = Y_C,
      Y_P        = Y_P,
      W          = W,
      covariates = covariates,
      m          = m,
      TT         = TT,
      site_ids   = if (is.null(site_ids)) seq_len(m)  else site_ids,
      time_ids   = if (is.null(time_ids)) seq_len(TT) else time_ids,
      empirical  = empirical
    ),
    class = "count_abm_data"
  )
}

print.count_abm_data <- function(x, ...) {
  e <- x$empirical
  cat("count_abm_data\n")
  cat(sprintf("  Sites: %d   Time steps: %d\n", x$m, x$TT))
  cat(sprintf("  Count  channel: mean=%.2f  sd=%.2f  zeros=%.1f%%  obs=%.1f%%\n",
              e$count_mean, e$count_sd,
              100 * e$zero_frac, 100 * e$frac_C_obs))
  cat(sprintf("  Binary channel: detection=%.3f  obs=%.1f%%\n",
              e$detect_rate, 100 * e$frac_P_obs))
  cat(sprintf("  NegBin r (moments): %.2f\n", e$r_moment))
  cat(sprintf("  Phi lag-1 estimate: %.3f\n", e$phi_lag1_est))
  if (length(x$covariates))
    cat(sprintf("  Covariates: %s\n", paste(names(x$covariates), collapse = ", ")))
  invisible(x)
}

summary.count_abm_data <- function(object, ...) {
  print(object)
  cat("\nAnnual means (Y_C):\n")
  print(round(object$empirical$annual_means, 2))
}
