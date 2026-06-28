# mhb_validation.R
# Three supplementary analyses for the CSDA revision:
#  (1) Temporal holdout: train 2004-2012, predict 2013
#  (2) Fixed-parameter sensitivity: vary phi, K, omega, r
#  (3) False-positive sensitivity: vary eta in Bernoulli channel
#
# All analyses use fixed-parameter SMFVB (E-step only).
# rho=0.72, kappa=0.10, psi=1.0 calibrated from ecological literature.

source("D:/research/inference/smfvb.R")

raw <- readRDS("D:/research/data/processed/mhb_great_tit.rds")
m   <- raw$m;  TT <- raw$TT

# Default ecological parameters
PHI_BASE  <- 0.95;  K_BASE <- 100.0;  OMEGA <- 0.05
RHO_VAL   <- 0.72;  KAPPA_VAL <- 0.10;  PSI_VAL <- 1.0;  R_VAL <- 2.0

make_params <- function(phi_base = PHI_BASE,
                        K_base   = K_BASE,
                        omega    = OMEGA,
                        r        = R_VAL) {
  list(
    phi   = plogis(qlogis(phi_base) - 0.10 * raw$elev_z),
    K     = rep(K_base, m) * exp(0.20 * raw$forest_z),
    W_eff = raw$W * omega,
    r_nb  = rep(r, m)
  )
}

run_smfvb <- function(Y_C, Y_P, p, seed = 2026L) {
  set.seed(seed)
  smfvb(Y_C = Y_C, Y_P = Y_P,
        W     = p$W_eff,
        phi   = p$phi,
        K     = p$K,
        psi   = rep(PSI_VAL, m),
        rho   = rep(RHO_VAL, m),
        r_nb  = p$r_nb,
        kappa = rep(KAPPA_VAL, m),
        max_iter = 300L, tol = 1e-5, verbose = FALSE)
}

# NegBin predictive log-likelihood for a held-out vector
negbin_pll <- function(y_obs, mu_pred, rho, r) {
  mask  <- !is.na(y_obs)
  mu_o  <- rho * mu_pred[mask] + 1e-12
  sum(lgamma(r + y_obs[mask]) - lgamma(r) - lgamma(y_obs[mask] + 1) +
      r * log(r / (r + mu_o)) + y_obs[mask] * log(mu_o / (r + mu_o)))
}

cat("==========================================================\n")
cat("(1) TEMPORAL HOLDOUT VALIDATION\n")
cat("    Train: 2004-2012 (T=9)   Test: 2013 (T=10)\n")
cat("==========================================================\n\n")

p_base <- make_params()
Y_C_tr <- raw$Y_C[, 1:9];  Y_P_tr <- raw$Y_P[, 1:9]
fit_tr  <- run_smfvb(Y_C_tr, Y_P_tr, p_base)
mu_tr   <- fit_tr$mu     # [m, 9]

# One-step-ahead process prediction for 2013
mu9        <- mu_tr[, 9]
mu_pred_10 <- predict_process(mu9, p_base$W_eff,
                               p_base$phi, p_base$K, rep(PSI_VAL, m))

Y_C_test  <- raw$Y_C[, 10]
obs_2013  <- !is.na(Y_C_test)
n_2013    <- sum(obs_2013)

pll_ho   <- negbin_pll(Y_C_test, mu_pred_10, RHO_VAL, R_VAL)
rmse_ho  <- sqrt(mean((Y_C_test[obs_2013] -
                        RHO_VAL * mu_pred_10[obs_2013])^2))
r_cor    <- cor(RHO_VAL * mu_pred_10[obs_2013], Y_C_test[obs_2013])

# 95% NegBin predictive interval coverage
lo_95 <- qnbinom(0.025, size = R_VAL,
                  mu = RHO_VAL * mu_pred_10[obs_2013])
hi_95 <- qnbinom(0.975, size = R_VAL,
                  mu = RHO_VAL * mu_pred_10[obs_2013])
cov_95 <- mean(Y_C_test[obs_2013] >= lo_95 &
               Y_C_test[obs_2013] <= hi_95)

obs_mean_2013  <- mean(Y_C_test, na.rm = TRUE)
pred_mean_2013 <- mean(RHO_VAL * mu_pred_10[obs_2013])

cat(sprintf("Training fit (2004-2012, %d site-years):\n", 9 * m))
cat(sprintf("  Surrogate L~ = %.1f\n\n", tail(fit_tr$elbo_trace, 1L)))

cat(sprintf("One-step-ahead holdout (2013, n=%d sites):\n", n_2013))
cat(sprintf("  Observed mean Y_C_2013    = %.3f\n", obs_mean_2013))
cat(sprintf("  Predicted mean rho*mu_10  = %.3f\n", pred_mean_2013))
cat(sprintf("  Holdout PLL               = %.1f\n", pll_ho))
cat(sprintf("  Holdout RMSE              = %.3f  (count units)\n", rmse_ho))
cat(sprintf("  Pearson r(pred, obs)      = %.3f\n", r_cor))
cat(sprintf("  95%% NegBin PI coverage   = %.3f  (n=%d)\n\n", cov_95, n_2013))

# Null comparison: predict 2013 from training mean (count-scale → abundance-scale)
mu_null_obs <- rowMeans(Y_C_tr, na.rm = TRUE)
global_mean <- mean(Y_C_tr, na.rm = TRUE) / RHO_VAL
mu_null_obs[!is.finite(mu_null_obs)] <- global_mean   # impute never-observed sites
mu_null_abund <- mu_null_obs / RHO_VAL                # convert to abundance scale
pll_null <- negbin_pll(Y_C_test, mu_null_abund, RHO_VAL, R_VAL)
cat(sprintf("  Null (train-mean) PLL     = %.1f  (delta = %+.1f)\n\n",
            pll_null, pll_ho - pll_null))

cat("==========================================================\n")
cat("(2) FIXED-PARAMETER SENSITIVITY\n")
cat("==========================================================\n\n")

obs_all <- !is.na(raw$Y_C)
run_full <- function(p) {
  fit  <- run_smfvb(raw$Y_C, raw$Y_P, p)
  mu   <- fit$mu
  rcor <- cor(as.numeric(mu)[obs_all],
              as.numeric(raw$Y_C)[obs_all])
  list(mean_N = mean(mu),
       sd_N   = mean(apply(mu, 1L, sd)),
       r_cor  = rcor,
       L      = tail(fit$elbo_trace, 1L))
}

fmt_row_phi  <- "  %.2f    %8.3f  %8.3f  %6.3f  %10.1f\n"
fmt_row_int  <- "  %-6d  %8.3f  %8.3f  %6.3f  %10.1f\n"
fmt_row_flt  <- "  %.2f    %8.3f  %8.3f  %6.3f  %10.1f\n"
hdr <- sprintf("  %-6s  %8s  %8s  %6s  %10s\n", "param", "E[N]", "sd(N)", "r", "L~")

cat("phi sensitivity (K=100, omega=0.05, r=2.0):\n"); cat(hdr)
for (ph in c(0.85, 0.90, 0.95)) {
  rr <- run_full(make_params(phi_base = ph))
  cat(sprintf(fmt_row_phi, ph, rr$mean_N, rr$sd_N, rr$r_cor, rr$L))
}
cat("\n")

cat("K sensitivity (phi=0.95, omega=0.05, r=2.0):\n"); cat(hdr)
for (kv in c(50, 100, 200)) {
  rr <- run_full(make_params(K_base = kv))
  cat(sprintf(fmt_row_int, kv, rr$mean_N, rr$sd_N, rr$r_cor, rr$L))
}
cat("\n")

cat("omega sensitivity (phi=0.95, K=100, r=2.0):\n"); cat(hdr)
for (om in c(0.02, 0.05, 0.10)) {
  rr <- run_full(make_params(omega = om))
  cat(sprintf(fmt_row_flt, om, rr$mean_N, rr$sd_N, rr$r_cor, rr$L))
}
cat("\n")

cat("r sensitivity (phi=0.95, K=100, omega=0.05):\n"); cat(hdr)
for (rv in c(1.0, 2.0, 5.0)) {
  rr <- run_full(make_params(r = rv))
  cat(sprintf("  %.1f     %8.3f  %8.3f  %6.3f  %10.1f\n", rv, rr$mean_N, rr$sd_N, rr$r_cor, rr$L))
}
cat("\n")

cat("==========================================================\n")
cat("(3) FALSE-POSITIVE SENSITIVITY (eta)\n")
cat("==========================================================\n\n")

# The baseline SMFVB uses Bernoulli P(Y_P=1|N)=1-exp(-kappa*N), with eta=0.
# With false positives eta>0: P(Y_P=1|N)=1-(1-eta)*exp(-kappa*N).
# We evaluate the baseline spatial field {mu} under each eta and compute
# the modified Bernoulli log-likelihood, reporting the shift relative to eta=0.

fit_base <- run_smfvb(raw$Y_C, raw$Y_P, p_base)
mu_base  <- fit_base$mu

bern_ll_fp <- function(y, mu, kappa, eta) {
  # P(Y_P=1|N) = 1 - (1-eta)*exp(-kappa*N)
  # E_q[Y_P=1] approx using MGF:
  # log P(Y_P=1) ≈ log(1-(1-eta)*exp(-mu*(1-exp(-kappa))))
  # log P(Y_P=0) ≈ log((1-eta)*exp(-mu*(1-exp(-kappa))))
  alpha  <- 1 - exp(-kappa)
  M_neg  <- exp(-mu * alpha)                   # E_q[exp(-kappa*N)]
  if (y == 1L) {
    log(pmax(1 - (1 - eta) * M_neg, 1e-12))
  } else {
    log(pmax((1 - eta) * M_neg, 1e-12))
  }
}

cat("Bernoulli log-likelihood of Y_P under baseline mu and varying eta:\n")
cat(sprintf("  %-6s  %12s  %12s\n", "eta", "Bern log-lik", "delta vs eta=0"))
ll0 <- NULL
for (eta_val in c(0.00, 0.01, 0.05)) {
  ll_yp <- 0.0
  for (i in seq_len(m)) {
    for (tt in seq_len(TT)) {
      yp <- raw$Y_P[i, tt]
      if (!is.na(yp)) {
        ll_yp <- ll_yp + bern_ll_fp(as.integer(yp), mu_base[i, tt],
                                      KAPPA_VAL, eta_val)
      }
    }
  }
  if (is.null(ll0)) ll0 <- ll_yp
  cat(sprintf("  %.2f    %12.1f  %+12.1f\n",
              eta_val, ll_yp, ll_yp - ll0))
}

cat("\n")
cat("Implied shift in detection probability at mean abundance:\n")
mu_bar <- mean(mu_base)
for (eta_val in c(0.00, 0.01, 0.05)) {
  p_det <- 1 - (1 - eta_val) * exp(-KAPPA_VAL * mu_bar)
  cat(sprintf("  eta=%.2f:  P(Y_P=1|N=%.1f) = %.4f  (delta=%+.4f)\n",
              eta_val, mu_bar, p_det,
              p_det - (1 - exp(-KAPPA_VAL * mu_bar))))
}
