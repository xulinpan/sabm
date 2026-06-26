#' Construct a ModelParams list
#'
#' @description
#' Collects all named parameters theta (Eq 24) into a single list.
#' Regression coefficients control cell-level phi, K, psi, alpha, rho.
#'
#' @param beta     (p,) habitat-GP mean regression coefficients.
#' @param beta_K   (q,) log-K regression coefficients.
#' @param beta_phi (p,) logit-phi regression coefficients.
#' @param gamma_ldd (q,) log-psi (LDD) regression coefficients.
#' @param sigma2_alpha GP marginal variance (default 1.0).
#' @param theta_alpha  GP range parameter (default 3.0).
#' @param sigma2_K     Residual log-K variance (default 0.25).
#' @param mu_rho       Logit-scale detection intercept (default 0.5).
#' @param sigma2_rho   Site-level detection heterogeneity (default 0.5).
#' @param r            NegBin overdispersion parameter (default 5.0).
#' @param kappa        Remote-sensing detection rate (default 0.3).
#' @param gamma        Half-saturation constant for g(N) (default 8.0).
#' @return A named list of model parameters.
#' @export
ModelParams <- function(
    beta,
    beta_K,
    beta_phi,
    gamma_ldd,
    sigma2_alpha = 1.0,
    theta_alpha  = 3.0,
    sigma2_K     = 0.25,
    mu_rho       = 0.5,
    sigma2_rho   = 0.5,
    r            = 5.0,
    kappa        = 0.3,
    gamma        = 8.0
) {
  list(
    beta         = as.numeric(beta),
    beta_K       = as.numeric(beta_K),
    beta_phi     = as.numeric(beta_phi),
    gamma_ldd    = as.numeric(gamma_ldd),
    sigma2_alpha = sigma2_alpha,
    theta_alpha  = theta_alpha,
    sigma2_K     = sigma2_K,
    mu_rho       = mu_rho,
    sigma2_rho   = sigma2_rho,
    r            = r,
    kappa        = kappa,
    gamma        = gamma
  )
}


#' SC-ABM-NKD Simulator
#'
#' @description
#' Full simulator for the SC-ABM-NKD model.
#'
#' At each time step the latent count decomposes as:
#'   N_{i,t} = S_{i,t} + I_{i,t} + L_{i,t}
#'
#' where:
#'   S_{i,t} ~ Binomial(N_{i,t-1}, phi_i * exp(-N_{i,t-1}/K_i))   [Eqs 4-5]
#'   I_{i,t} ~ Poisson(Lambda_{i,t})                                [Eq 13]
#'   L_{i,t} ~ Poisson(psi_i)                                       [Eq 14]
#'
#' @export
SCABMNKD <- R6::R6Class("SCABMNKD",
  public = list(

    #' @field grid A SpatialGrid object.
    grid   = NULL,
    #' @field X (m x p) environmental covariate matrix.
    X      = NULL,
    #' @field W (m x q) LDD / carrying-capacity covariate matrix.
    W      = NULL,
    #' @field kernel A NeuralDispersalKernel object.
    kernel = NULL,
    #' @field params A ModelParams list.
    params = NULL,

    # Derived cell-level quantities (set by derive_cell_params)
    #' @field alpha (m,) latent habitat suitability.
    alpha  = NULL,
    #' @field phi   (m,) baseline persistence probability.
    phi    = NULL,
    #' @field K     (m,) carrying capacity.
    K      = NULL,
    #' @field psi   (m,) LDD rate.
    psi    = NULL,
    #' @field rho   (m,) count-survey detection probability.
    rho    = NULL,

    #' @description Create a new SCABMNKD simulator.
    #' @param grid    A SpatialGrid object.
    #' @param X       (m x p) environmental covariates.
    #' @param W       (m x q) LDD / K covariates.
    #' @param kernel  A NeuralDispersalKernel.
    #' @param params  A ModelParams list.
    initialize = function(grid, X, W, kernel, params) {
      self$grid   <- grid
      self$X      <- X
      self$W      <- W
      self$kernel <- kernel
      self$params <- params
      private$derive_cell_params()
      private$precompute_features()
    },

    # ── Process model ───────────────────────────────────────────────────────

    #' @description Advance one time step.
    #' @param N_prev Integer vector of length m (counts at t-1).
    #' @return Integer vector of length m (counts at t).
    step = function(N_prev) {
      m     <- self$grid$m
      N_f   <- as.numeric(N_prev)
      gamma <- self$params$gamma

      # (A) Survival  S_i ~ Binomial(N_{i,t-1}, phi_i exp(-N/K_i))  [Eqs 4-5]
      N_int <- pmin(as.integer(N_prev), .Machine$integer.max %/% 2L)
      N_f   <- as.numeric(N_int)
      p_s   <- self$phi * exp(-N_f / (self$K + 1e-12))
      p_s   <- pmin(pmax(p_s, 0.0), 1.0)
      S     <- rbinom(m, size = N_int, prob = p_s)

      # (B) Immigration  I_i ~ Poisson(Lambda_i)                     [Eq 13]
      #
      # Loop over SOURCES j (not targets), normalising outgoing weights so
      # Σ_i w_{j→i} ≤ 1.  This ensures total outgoing flux ≤ g(N_j)·N_j ≤ N_j
      # and prevents runaway population growth with random kernel weights.
      g      <- private$density_scale(N_f, gamma)
      Lambda <- numeric(m)
      for (j in seq_len(m)) {
        if (N_f[j] == 0) next
        targets <- private$rev_nb[[j]]
        if (length(targets) == 0L) next
        w_raw <- self$kernel$forward(private$rev_feat[[j]])   # (|N_j|,) in (0,1)
        w_sum <- sum(w_raw)
        if (w_sum > 1.0) w_raw <- w_raw / w_sum              # normalise
        Lambda[targets] <- Lambda[targets] + w_raw * g[j] * N_f[j]
      }
      I <- rpois(m, lambda = pmax(Lambda, 0.0))

      # (C) LDD  L_i ~ Poisson(psi_i)                                [Eq 14]
      L <- rpois(m, lambda = self$psi)

      result        <- S + I + L
      result[is.na(result)] <- 0L
      as.integer(pmin(result, .Machine$integer.max %/% 2L))
    },

    #' @description Simulate T time steps from initial counts.
    #' @param N0 Integer vector of length m (initial population counts).
    #' @param T  Number of time steps.
    #' @return Integer matrix of shape (T+1, m); row 1 is N0.
    simulate = function(N0, T) {
      traj      <- matrix(0L, nrow = T + 1L, ncol = self$grid$m)
      traj[1L, ] <- as.integer(N0)
      for (t in seq_len(T)) {
        traj[t + 1L, ] <- self$step(traj[t, ])
      }
      traj
    },

    # ── Observation models ──────────────────────────────────────────────────

    #' @description Count-survey observations (Eq 1).
    #'   Y^(C) | N, rho, r ~ NegBin(mu = rho*N, r)
    #' @param N   Integer vector of length m (latent counts).
    #' @param rho Numeric vector of detection probabilities (default: self$rho).
    #' @return Integer vector of observed counts.
    observe_count = function(N, rho = NULL) {
      if (is.null(rho)) rho <- self$rho
      mu <- rho * as.numeric(N) + 1e-12
      rnbinom(length(N), size = self$params$r, mu = mu)
    },

    #' @description Binary remote-sensing observations (Eq 3).
    #'   Y^(P) | N, kappa ~ Bernoulli(1 - exp(-kappa * N))
    #' @param N Integer vector of length m (latent counts).
    #' @return Integer vector of 0/1 observations.
    observe_binary = function(N) {
      prob <- 1.0 - exp(-self$params$kappa * as.numeric(N))
      rbinom(length(N), size = 1L, prob = prob)
    },

    #' @description Joint observation log-likelihood (Eq 6).
    #' @param Y_C   Observed counts at O_C locations.
    #' @param Y_P   Observed binary values at O_P locations.
    #' @param N_C   Latent counts at O_C locations.
    #' @param N_P   Latent counts at O_P locations.
    #' @param rho_C Detection probabilities at O_C locations.
    #' @return Scalar log-likelihood value.
    log_likelihood = function(Y_C, Y_P, N_C, N_P, rho_C) {
      r     <- self$params$r
      kappa <- self$params$kappa

      # NegBin log-pmf for count data
      mu_C <- rho_C * as.numeric(N_C) + 1e-12
      ll_C <- sum(dnbinom(Y_C, size = r, mu = mu_C, log = TRUE))

      # Bernoulli log-pmf for binary data
      prob_P <- pmin(pmax(1.0 - exp(-kappa * as.numeric(N_P)), 1e-12), 1 - 1e-12)
      ll_P   <- sum(Y_P * log(prob_P) + (1 - Y_P) * log(1 - prob_P))

      ll_C + ll_P
    }
  ),

  private = list(

    # Target-perspective features: feat[[i]] is (|N_i| x z_dim), source j → target i
    feat   = NULL,
    nb_idx = NULL,
    # Source-perspective features: rev_feat[[j]] is (|N_j| x z_dim), source j → target i
    # Used in step() for normalised outgoing dispersal
    rev_feat = NULL,
    rev_nb   = NULL,

    # ── Parameter derivation ───────────────────────────────────────────────

    derive_cell_params = function() {
      p   <- self$params
      m   <- self$grid$m
      X   <- self$X
      W   <- self$W
      D   <- self$grid$dists

      # Habitat suitability: GP draw  alpha = X beta + eps  (Eq 21)
      alpha_mu <- as.vector(X %*% p$beta)
      C        <- exp(-D / (p$theta_alpha + 1e-12))
      diag(C)  <- diag(C) + 1e-6                      # jitter
      L_chol   <- t(chol(p$sigma2_alpha * C))          # lower triangular
      self$alpha <- alpha_mu + as.vector(L_chol %*% rnorm(m))

      # Carrying capacity  log K_i = w_i' beta_K + u_i  (Eq 22b)
      log_K  <- as.vector(W %*% p$beta_K) +
                rnorm(m, sd = sqrt(p$sigma2_K))
      self$K <- exp(log_K)

      # Baseline persistence  logit(phi_i) = x_i' beta_phi  (Eq 22c)
      self$phi <- sigmoid(as.vector(X %*% p$beta_phi))

      # LDD rate  log psi_i = w_i' gamma_ldd  (Eq 14)
      self$psi <- exp(as.vector(W %*% p$gamma_ldd))

      # Detection probability  logit(rho_i) = mu_rho + v_i  (Eq 3)
      logit_rho <- p$mu_rho + rnorm(m, sd = sqrt(p$sigma2_rho))
      self$rho  <- sigmoid(logit_rho)
    },

    # ── Feature pre-computation ────────────────────────────────────────────

    precompute_features = function() {
      #
      # For each target cell i and each source j in N_i, build:
      #   z_{ji} = [x_i - x_j,  d_{ji},  ã_{ji,1}, …, ã_{ji,d_N}]  (Eq 9)
      # where ã_{ji,k} = (alpha_{N_k(j)} - alpha_j) / d_{N_k(j),j}  (Eq 10).
      #
      grid  <- self$grid
      alpha <- self$alpha
      X     <- self$X
      p_dim <- ncol(X)
      z_dim <- p_dim + 1L + grid$max_nn

      private$feat   <- vector("list", grid$m)
      private$nb_idx <- grid$neighbours

      for (i in seq_len(grid$m)) {
        nbs  <- grid$neighbours[[i]]
        n_nb <- length(nbs)
        Z    <- matrix(0.0, nrow = n_nb, ncol = z_dim)

        for (k in seq_len(n_nb)) {
          j <- nbs[k]
          # Habitat covariate difference
          Z[k, seq_len(p_dim)]  <- X[i, ] - X[j, ]
          # Euclidean distance j → i
          Z[k, p_dim + 1L]      <- grid$dists[j, i]
          # Directional habitat gradients of SOURCE cell j
          j_nbs <- grid$neighbours[[j]]
          for (k2 in seq_along(j_nbs)) {
            if (k2 > grid$max_nn) break
            jn   <- j_nbs[k2]
            d_jn <- grid$dists[j, jn]
            Z[k, p_dim + 1L + k2] <-
              if (d_jn > 0) (alpha[jn] - alpha[j]) / d_jn else 0.0
          }
        }
        private$feat[[i]] <- Z
      }

      # Source-perspective features: rev_feat[[j]][k, :] = z_{j → targets[k]}
      # Directional gradients of SOURCE j are the same for all targets of j,
      # so we compute them once per source cell.
      private$rev_nb   <- grid$neighbours   # symmetric: targets of j == neighbours of j
      private$rev_feat <- vector("list", grid$m)

      for (j in seq_len(grid$m)) {
        targets <- grid$neighbours[[j]]
        n_tgt   <- length(targets)
        Z_rev   <- matrix(0.0, nrow = n_tgt, ncol = z_dim)

        # Directional gradients of SOURCE j (shared across all rows)
        j_nbs  <- grid$neighbours[[j]]
        j_grad <- numeric(grid$max_nn)
        for (k2 in seq_along(j_nbs)) {
          if (k2 > grid$max_nn) break
          jn   <- j_nbs[k2]
          d_jn <- grid$dists[j, jn]
          j_grad[k2] <- if (d_jn > 0) (alpha[jn] - alpha[j]) / d_jn else 0.0
        }

        for (k in seq_len(n_tgt)) {
          i <- targets[k]
          Z_rev[k, seq_len(p_dim)]                        <- X[i, ] - X[j, ]
          Z_rev[k, p_dim + 1L]                            <- grid$dists[j, i]
          Z_rev[k, (p_dim + 2L):(p_dim + 1L + grid$max_nn)] <- j_grad
        }
        private$rev_feat[[j]] <- Z_rev
      }
    },

    # ── Helpers ────────────────────────────────────────────────────────────

    density_scale = function(N, gamma) {
      # g(N; gamma) = N / (N + gamma)   [Eq 11]
      N / (N + gamma + 1e-12)
    }
  )
)
