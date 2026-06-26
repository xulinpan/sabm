#' Compute ABC summary statistics s(Y) = [s1, …, s5]
#'
#' @description
#' Implements the five summary statistics from Eqs (19)–(23):
#' \describe{
#'   \item{s1}{Mean observed count.}
#'   \item{s2}{Relative temporal growth rate of mean counts.}
#'   \item{s3}{Moran's I spatial autocorrelation at the last time step.}
#'   \item{s4}{Remote-sensing occupancy rate.}
#'   \item{s5}{Lag-1 temporal autocorrelation of mean counts.}
#' }
#'
#' @param Y_C  Numeric matrix (T+1 x m) of count observations; NA = unobserved.
#' @param Y_P  Numeric matrix (T+1 x m) of binary observations; NA = unobserved.
#' @param grid A SpatialGrid object (used for Moran's I adjacency).
#' @return Named numeric vector of length 5.
#' @export
compute_summaries <- function(Y_C, Y_P, grid) {

  T_steps <- nrow(Y_C) - 1L

  # s1 — mean observed count  (Eq 19)
  s1 <- mean(Y_C, na.rm = TRUE)

  # s2 — relative temporal growth rate  (Eq 20)
  Y_bar <- rowMeans(Y_C, na.rm = TRUE)   # (T+1,)
  growth_terms <- numeric(0)
  for (t in seq_len(T_steps)) {
    if (!is.na(Y_bar[t + 1L]) && !is.na(Y_bar[t])) {
      growth_terms <- c(growth_terms,
                        (Y_bar[t + 1L] - Y_bar[t]) / (Y_bar[t] + 1.0))
    }
  }
  s2 <- if (length(growth_terms) > 0L) mean(growth_terms) else 0.0

  # s3 — Moran's I at last time step  (Eq 21)
  y_last  <- Y_C[nrow(Y_C), ]
  obs_idx <- which(!is.na(y_last))
  if (length(obs_idx) > 1L) {
    y_obs  <- y_last[obs_idx]
    y_mean <- mean(y_obs)
    y_dev  <- y_obs - y_mean
    sigma2 <- var(y_obs)
    if (is.na(sigma2) || sigma2 < 1e-12) sigma2 <- 1e-12
    # Build adjacency set for observed cells
    nb_sets <- lapply(obs_idx, function(ia) grid$neighbours[[ia]])
    numer   <- 0.0
    n_pairs <- 0L
    for (a in seq_along(obs_idx)) {
      ia <- obs_idx[a]
      for (b in seq_along(obs_idx)) {
        ib <- obs_idx[b]
        if (ib %in% nb_sets[[a]]) {
          numer   <- numer + y_dev[a] * y_dev[b]
          n_pairs <- n_pairs + 1L
        }
      }
    }
    s3 <- if (n_pairs > 0L) numer / (n_pairs * sigma2) else 0.0
  } else {
    s3 <- 0.0
  }

  # s4 — remote-sensing occupancy rate  (Eq 22)
  s4 <- mean(Y_P, na.rm = TRUE)

  # s5 — lag-1 temporal autocorrelation  (Eq 23)
  valid <- !is.na(Y_bar)
  if (sum(valid) > 2L) {
    idx_t   <- which(valid[-1L] & valid[-length(valid)]) + 1L
    idx_tm1 <- idx_t - 1L
    y_t   <- Y_bar[idx_t]
    y_tm1 <- Y_bar[idx_tm1]
    if (length(y_t) > 1L) {
      cov_val <- cov(y_t, y_tm1)
      var_val <- var(y_t)
      s5 <- if (!is.na(var_val) && var_val > 1e-12) cov_val / var_val else 0.0
    } else {
      s5 <- 0.0
    }
  } else {
    s5 <- 0.0
  }

  c(s1 = s1, s2 = s2, s3 = s3, s4 = s4, s5 = s5)
}


#' Compute ABC summary statistics for the seasonal model (s1 – s6)
#'
#' @description
#' Extends \code{compute_summaries} with a sixth statistic:
#' \describe{
#'   \item{s6}{Biweekly step of peak mean observed count (peak-timing anchor for mu_s).}
#' }
#'
#' @param Y_C  Numeric matrix (T+1 x m) of count observations; NA = unobserved.
#' @param Y_P  Numeric matrix (T+1 x m) of binary observations; NA = unobserved.
#' @param grid A SpatialGrid object (used for Moran's I adjacency).
#' @return Named numeric vector of length 6.
#' @export
compute_summaries_v3 <- function(Y_C, Y_P, grid) {
  s15 <- compute_summaries(Y_C, Y_P, grid)

  T_steps <- nrow(Y_C) - 1L
  Y_bar   <- rowMeans(Y_C, na.rm = TRUE)          # (T+1,)
  steps   <- Y_bar[seq(2L, T_steps + 1L)]         # steps 1..T (drop t=0 row)
  steps[is.na(steps)] <- -Inf
  s6 <- as.numeric(which.max(steps))
  if (all(!is.finite(steps))) s6 <- as.numeric(T_steps %/% 2L)

  c(s15, s6 = s6)
}
