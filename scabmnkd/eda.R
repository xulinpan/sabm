# ─────────────────────────────────────────────────────────────────────────────
# SC-ABM-NKD  |  Simulated Dataset + Exploratory Data Analysis
# Run from the scabmnkd/ package root:
#   Rscript eda.R
# Output: eda_report.pdf  (multi-page)
# ─────────────────────────────────────────────────────────────────────────────

# ── 0. Load package files ────────────────────────────────────────────────────
if (!requireNamespace("R6", quietly = TRUE)) install.packages("R6")

for (f in c("utils", "grid", "kernel", "model",
            "summaries", "prior", "abc_smc", "build_model", "demo")) {
  source(file.path("R", paste0(f, ".R")))
}

# ── 1. Simulate Data ─────────────────────────────────────────────────────────
set.seed(2025)

NROW    <- 15L
NCOL    <- 15L
T_STEPS <- 20L

grid  <- SpatialGrid$new(NROW, NCOL, cell_size = 1.0)
m     <- grid$m          # 225 cells
p     <- 3L              # habitat covariate dimension
q     <- 2L              # LDD / K covariate dimension

# Environmental covariates: two gradients + one random field
row_idx <- ((seq_len(m) - 1L) %/% NCOL + 1L) / NROW   # row gradient
col_idx <- ((seq_len(m) - 1L) %% NCOL  + 1L) / NCOL   # col gradient
X <- cbind(row_idx, col_idx, rnorm(m, sd = 0.5))       # (m, 3)
W <- cbind(rep(1, m), col_idx)                          # (m, 2)

z_dim <- p + 1L + grid$max_nn

# True model parameters
# beta_phi intercept ~4 -> logit(phi)~3.5 at mean covariates -> phi~0.97,
# which keeps the population stable when dispersal is normalised.
true_params <- ModelParams(
  beta       = c(0.8,  -0.5,  0.3),
  beta_K     = c(3.2,   0.3),           # log(K) ~ 3.2 -> K ~ 25
  beta_phi   = c(4.0,  -0.5, 0.2),      # high phi so S dominates; ~0.96 mean
  gamma_ldd  = c(-2.0,  0.2),
  sigma2_alpha = 1.2,
  theta_alpha  = 4.0,
  sigma2_K     = 0.3,
  r            = 4.0,
  kappa        = 0.25,
  gamma        = 10.0,
  mu_rho       = 0.8,
  sigma2_rho   = 0.6
)

true_kernel <- NeuralDispersalKernel$new(z_dim)
model       <- SCABMNKD$new(grid, X, W, true_kernel, true_params)

# Start near equilibrium (~10 per cell); sigmoid(alpha) centres around 0.5
lambda_init <- 10 * (0.4 + 0.6 * sigmoid(model$alpha))
N0          <- rpois(m, lambda = lambda_init)

# Simulate trajectory
cat("Simulating", T_STEPS, "steps on", m, "cells ...\n")
traj <- model$simulate(N0, T_STEPS)      # (T_STEPS+1, m)  integer matrix

# Generate observations (all cells, all times)
Y_C <- matrix(NA_real_, T_STEPS + 1L, m)
Y_P <- matrix(NA_real_, T_STEPS + 1L, m)
for (t in seq_len(T_STEPS + 1L)) {
  Y_C[t, ] <- model$observe_count(traj[t, ])
  Y_P[t, ] <- model$observe_binary(traj[t, ])
}

# Summary statistics
s_obs <- compute_summaries(Y_C, Y_P, grid)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Console report
# ─────────────────────────────────────────────────────────────────────────────
cat("\n═══════════ SC-ABM-NKD  Simulated Dataset Summary ═══════════\n")
cat(sprintf("Grid          : %d × %d = %d cells\n", NROW, NCOL, m))
cat(sprintf("Time steps    : %d  (indices 1 … %d)\n", T_STEPS, T_STEPS + 1L))
cat(sprintf("Covariates    : p = %d (habitat),  q = %d (LDD/K)\n", p, q))
cat(sprintf("Neural kernel : z_dim = %d,  H1 = %d,  H2 = %d\n",
            z_dim, true_kernel$H1, true_kernel$H2))
cat("\n── True scalar parameters ──\n")
cat(sprintf("  r (NegBin overdispersion) : %.2f\n", true_params$r))
cat(sprintf("  kappa (remote-sensing)    : %.2f\n", true_params$kappa))
cat(sprintf("  gamma (half-saturation)   : %.2f\n", true_params$gamma))
cat(sprintf("  mu_rho (detection logit)  : %.2f  -> rho ~ %.2f\n",
            true_params$mu_rho, sigmoid(true_params$mu_rho)))
cat("\n── Derived cell parameters ──\n")
cat(sprintf("  alpha  : mean = %.2f, sd = %.2f, range [%.2f, %.2f]\n",
            mean(model$alpha), sd(model$alpha), min(model$alpha), max(model$alpha)))
cat(sprintf("  phi    : mean = %.3f, sd = %.3f, range [%.3f, %.3f]\n",
            mean(model$phi),  sd(model$phi),  min(model$phi),  max(model$phi)))
cat(sprintf("  K      : mean = %.1f,  sd = %.1f,  range [%.1f, %.1f]\n",
            mean(model$K),   sd(model$K),   min(model$K),   max(model$K)))
cat(sprintf("  psi    : mean = %.3f, sd = %.3f\n",
            mean(model$psi), sd(model$psi)))
cat(sprintf("  rho    : mean = %.3f, sd = %.3f, range [%.3f, %.3f]\n",
            mean(model$rho), sd(model$rho), min(model$rho), max(model$rho)))
cat("\n── Latent population N ──\n")
N_total <- rowSums(traj)
cat(sprintf("  Total N : t=1 %.0f  ->  t=%d %.0f  (change %.1f%%)\n",
            N_total[1], T_STEPS + 1L, N_total[T_STEPS + 1L],
            100 * (N_total[T_STEPS + 1L] / N_total[1] - 1)))
cat(sprintf("  Mean N per cell : %.2f (t=1)  ->  %.2f (t=%d)\n",
            mean(traj[1L, ], na.rm = TRUE),
            mean(traj[T_STEPS + 1L, ], na.rm = TRUE), T_STEPS + 1L))
cat(sprintf("  Max N any cell  : %.0f\n", max(traj, na.rm = TRUE)))
cat(sprintf("  Zero cells (t=%d): %.0f / %d  (%.1f%%)\n",
            T_STEPS + 1L,
            sum(traj[T_STEPS + 1L, ] == 0L, na.rm = TRUE), m,
            100 * mean(traj[T_STEPS + 1L, ] == 0L, na.rm = TRUE)))
cat("\n── Observations ──\n")
cat(sprintf("  Y_C : mean = %.2f, sd = %.2f, max = %.0f\n",
            mean(Y_C, na.rm = TRUE), sd(Y_C, na.rm = TRUE), max(Y_C, na.rm = TRUE)))
cat(sprintf("  Y_P : occupancy rate = %.3f\n", mean(Y_P, na.rm = TRUE)))
cat("\n── Summary statistics s1-s5 ──\n")
labs5 <- c("s1 mean count", "s2 growth rate", "s3 Moran's I",
           "s4 occupancy", "s5 lag-1 AC")
for (i in seq_along(s_obs))
  cat(sprintf("  %-18s: %8.4f\n", labs5[i], s_obs[i]))
cat("\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

# Reshape flat (m,) vector to (NROW, NCOL) for image()
to_grid <- function(v) matrix(v, nrow = NROW, ncol = NCOL, byrow = FALSE)

# Spatial heatmap using image() (origin at bottom-left, row=x, col=y)
plot_heatmap <- function(v, title = "", col = NULL, zlim = NULL) {
  mat <- to_grid(v)
  if (is.null(col))  col  <- rev(heat.colors(64))
  if (is.null(zlim)) zlim <- range(v, na.rm = TRUE)
  image(seq_len(NROW), seq_len(NCOL), mat,
        col = col, zlim = zlim,
        xlab = "row", ylab = "col", main = title, asp = 1)
}

# Colour palette: blue -> white -> red (diverging)
div_cols <- function(n = 64) {
  c(colorRampPalette(c("#2166AC", "#F7F7F7"))(n %/% 2),
    colorRampPalette(c("#F7F7F7", "#D6604D"))(n - n %/% 2))
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Open PDF device
# ─────────────────────────────────────────────────────────────────────────────
# cairo_pdf handles Unicode labels; fall back to pdf() if unavailable
use_cairo <- tryCatch({ grDevices::cairo_pdf; TRUE }, error = function(e) FALSE)
if (use_cairo) {
  cairo_pdf("eda_report.pdf", width = 14, height = 9)
} else {
  pdf("eda_report.pdf", width = 14, height = 9)
}
par(family = "sans")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: Temporal Dynamics
# ════════════════════════════════════════════════════════════════════════════
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
time_axis <- seq_len(T_STEPS + 1L)

# 1a  Total population over time
plot(time_axis, N_total, type = "b", pch = 19, cex = 0.7,
     col = "#2166AC", lwd = 2,
     xlab = "Time step", ylab = "Total N  (Σ cells)",
     main = "Total population over time")
abline(h = sum(model$K), lty = 2, col = "grey50")
legend("topright", c("Σ N_t", "Σ K_i"), lty = c(1, 2),
       col = c("#2166AC", "grey50"), lwd = c(2, 1), bty = "n")

# 1b  Mean ± SD per cell over time
cell_mean <- rowMeans(traj)
cell_sd   <- apply(traj, 1, sd)
plot(time_axis, cell_mean, type = "l", lwd = 2, col = "#2166AC",
     ylim = range(c(cell_mean - cell_sd, cell_mean + cell_sd)),
     xlab = "Time step", ylab = "N per cell",
     main = "Mean ± SD (per cell)")
polygon(c(time_axis, rev(time_axis)),
        c(cell_mean + cell_sd, rev(cell_mean - cell_sd)),
        col = adjustcolor("#2166AC", 0.2), border = NA)
lines(time_axis, cell_mean, lwd = 2, col = "#2166AC")

# 1c  Sample of 20 individual cell trajectories
sample_cells <- sample.int(m, size = min(20L, m))
matplot(time_axis, traj[, sample_cells], type = "l", lty = 1,
        col = adjustcolor("grey30", 0.4),
        xlab = "Time step", ylab = "N_i,t",
        main = "Sample cell trajectories (n=20)")
lines(time_axis, cell_mean, col = "#D6604D", lwd = 2)
legend("topright", c("Individual cells", "Mean"),
       col = c("grey50", "#D6604D"), lty = 1, lwd = c(1, 2), bty = "n")

# 1d  Mean observed count Y_C over time vs detection-corrected
obs_mean_C  <- rowMeans(Y_C, na.rm = TRUE)
corrected   <- obs_mean_C / mean(model$rho)
plot(time_axis, obs_mean_C, type = "b", pch = 19, cex = 0.7,
     col = "#4DAC26", lwd = 2,
     xlab = "Time step", ylab = "Mean observation",
     main = "Mean Y_C vs detection-corrected",
     ylim = range(c(obs_mean_C, corrected, cell_mean)))
lines(time_axis, corrected, col = "#762A83", lwd = 2, lty = 2)
lines(time_axis, cell_mean, col = "#2166AC", lwd = 2, lty = 3)
legend("topright", c("Mean Y_C", "Y_C / ρ̄ (corrected)", "True mean N"),
       col = c("#4DAC26", "#762A83", "#2166AC"),
       lty = c(1, 2, 3), lwd = 2, bty = "n", cex = 0.8)

# 1e  Occupancy rate over time
occ_rate <- rowMeans(Y_P, na.rm = TRUE)
det_prob  <- 1 - exp(-true_params$kappa * cell_mean)
plot(time_axis, occ_rate, type = "b", pch = 19, cex = 0.7,
     col = "#E08214", lwd = 2,
     ylim = c(0, 1),
     xlab = "Time step", ylab = "Rate",
     main = "Remote-sensing occupancy rate")
lines(time_axis, det_prob, col = "#8E0152", lwd = 2, lty = 2)
legend("topright", c("Observed occupancy", "E[1−exp(−κN̄)]"),
       col = c("#E08214", "#8E0152"), lty = c(1, 2), lwd = 2, bty = "n", cex = 0.8)

# 1f  Mean-variance plot (Taylor's power law) per cell
cell_mn  <- colMeans(traj)
cell_vr  <- apply(traj, 2, var)
ok       <- cell_mn > 0 & cell_vr > 0
lm_tv    <- lm(log(cell_vr[ok]) ~ log(cell_mn[ok]))
plot(log(cell_mn[ok]), log(cell_vr[ok]), pch = 16, cex = 0.5,
     col = adjustcolor("#2166AC", 0.5),
     xlab = "log(mean N per cell)", ylab = "log(var N per cell)",
     main = sprintf("Taylor's power law  (slope = %.2f)",
                    coef(lm_tv)[2]))
abline(lm_tv, col = "#D6604D", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "grey50")
legend("topleft", c("Fitted", "slope = 1 (Poisson)"),
       col = c("#D6604D", "grey50"), lty = 1:2, lwd = 2, bty = "n")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: Spatial Heatmaps — Latent Population N
# ════════════════════════════════════════════════════════════════════════════
t_snapshots <- c(1L, ceiling((T_STEPS + 1L) / 2), T_STEPS + 1L)
t_labels    <- c("t = 1 (initial)", sprintf("t = %d (mid)", t_snapshots[2]),
                 sprintf("t = %d (final)", T_STEPS + 1L))

N_zlim <- range(traj[t_snapshots, ])

par(mfrow = c(2, 3), mar = c(3, 3, 3, 2))
for (k in seq_along(t_snapshots)) {
  plot_heatmap(traj[t_snapshots[k], ],
               title = paste("Latent N:", t_labels[k]),
               zlim  = N_zlim)
}

# Spatial heatmaps of mean N, variance N, and habitat suitability alpha
plot_heatmap(colMeans(traj), title = "Mean N (over time)",
             col = rev(heat.colors(64)))
plot_heatmap(apply(traj, 2, var), title = "Var(N) (over time)",
             col = rev(heat.colors(64)))
plot_heatmap(model$alpha, title = "Habitat suitability  α_i",
             col = div_cols())

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: Spatial Heatmaps — Observations
# ════════════════════════════════════════════════════════════════════════════
par(mfrow = c(2, 3), mar = c(3, 3, 3, 2))

YC_zlim <- range(Y_C, na.rm = TRUE)
for (k in seq_along(t_snapshots)) {
  plot_heatmap(Y_C[t_snapshots[k], ],
               title = paste("Observed count Y_C:", t_labels[k]),
               zlim  = YC_zlim)
}

for (k in seq_along(t_snapshots)) {
  plot_heatmap(Y_P[t_snapshots[k], ],
               title = paste("Binary Y_P:", t_labels[k]),
               col   = colorRampPalette(c("white", "#E08214"))(8),
               zlim  = c(0, 1))
}

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4: Spatial Maps of Derived Cell Parameters
# ════════════════════════════════════════════════════════════════════════════
par(mfrow = c(2, 3), mar = c(3, 3, 3, 2))

plot_heatmap(model$phi, title = "Baseline persistence  φ_i",
             col = colorRampPalette(c("#FFFFCC", "#006837"))(64))
plot_heatmap(model$K, title = "Carrying capacity  K_i",
             col = colorRampPalette(c("#EFF3FF", "#084594"))(64))
plot_heatmap(model$psi, title = "LDD rate  ψ_i",
             col = colorRampPalette(c("#FFF7EC", "#7F0000"))(64))
plot_heatmap(model$rho, title = "Detection probability  ρ_i",
             col = colorRampPalette(c("#F7F7F7", "#4D004B"))(64))
plot_heatmap(colMeans(Y_C, na.rm = TRUE), title = "Mean Y_C (over time)",
             col = rev(heat.colors(64)))
plot_heatmap(colMeans(Y_P, na.rm = TRUE), title = "Mean occupancy (over time)",
             col = colorRampPalette(c("white", "#E08214"))(64))

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5: Marginal Distributions
# ════════════════════════════════════════════════════════════════════════════
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

# 5a  Histogram of all latent N values (excluding 0)
N_all <- as.vector(traj)
hist(N_all[N_all > 0], breaks = 40, freq = FALSE,
     col = "#9ECAE1", border = "white",
     xlab = "N_{i,t}  (positive cells only)",
     main = "Latent count distribution")
lines(density(N_all[N_all > 0], bw = "SJ"), col = "#08306B", lwd = 2)

# 5b  Histogram of Y_C with NegBin MoM fit overlay
YC_all <- as.vector(Y_C)
YC_all <- YC_all[!is.na(YC_all)]
mu_hat <- mean(YC_all)
v_hat  <- var(YC_all)
r_hat  <- mu_hat^2 / max(v_hat - mu_hat, 0.01)   # MoM estimate

hist(YC_all, breaks = 50, freq = FALSE,
     col = "#A1D99B", border = "white",
     xlab = "Y_C observed count",
     main = sprintf("Y_C distribution  (r_hat = %.1f, r_true = %.1f)",
                    r_hat, true_params$r))
x_nb  <- seq(0, max(YC_all))
y_nb  <- dnbinom(x_nb, size = r_hat, mu = mu_hat)
lines(x_nb, y_nb, col = "#006D2C", lwd = 2, type = "b", pch = 16, cex = 0.4)
legend("topright", "NegBin MoM fit", col = "#006D2C",
       lty = 1, lwd = 2, bty = "n")

# 5c  Y_P proportion
barplot(table(as.vector(Y_P)) / length(Y_P),
        col    = c("#DEEBF7", "#E08214"),
        names.arg = c("0 (absent)", "1 (detected)"),
        ylab   = "Proportion",
        main   = sprintf("Remote-sensing Y_P  (occupancy = %.3f)",
                         mean(Y_P, na.rm = TRUE)))

# 5d  Detection probability rho distribution
hist(model$rho, breaks = 20, freq = FALSE,
     col = "#DADAEB", border = "white",
     xlab = "ρ_i (detection probability)",
     main = "Detection probability distribution")
abline(v = mean(model$rho), col = "#54278F", lwd = 2, lty = 2)
legend("topright", sprintf("Mean ρ = %.3f", mean(model$rho)),
       col = "#54278F", lty = 2, lwd = 2, bty = "n")

# 5e  Carrying capacity K distribution (log scale)
hist(log(model$K), breaks = 20, freq = FALSE,
     col = "#FDD0A2", border = "white",
     xlab = "log(K_i)", main = "Carrying capacity  (log scale)")
abline(v = mean(log(model$K)), col = "#8C2D04", lwd = 2, lty = 2)

# 5f  Cell-level mean N vs K scatter
plot(model$K, colMeans(traj),
     pch = 16, cex = 0.6,
     col = adjustcolor("#2166AC", 0.5),
     xlab = "Carrying capacity K_i",
     ylab = "Mean N_i (over time)",
     main = "Mean N vs K  (cell level)")
abline(a = 0, b = 1, lty = 2, col = "grey50")
lm_nk <- lm(colMeans(traj) ~ model$K)
abline(lm_nk, col = "#D6604D", lwd = 2)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 6: Latent vs Observed — Detection Analysis
# ════════════════════════════════════════════════════════════════════════════
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

# 6a  Scatter: latent N vs observed Y_C
N_vec   <- as.vector(traj)
YC_vec  <- as.vector(Y_C)
rho_rep <- rep(model$rho, each = T_STEPS + 1L)   # repeat rho for each time
set.seed(1)
idx_samp <- sample(length(N_vec), min(3000L, length(N_vec)))

plot(N_vec[idx_samp], YC_vec[idx_samp],
     pch = 16, cex = 0.3, col = adjustcolor("#2166AC", 0.3),
     xlab = "Latent N_{i,t}", ylab = "Observed Y_C_{i,t}",
     main = "Count observation vs truth")
abline(a = 0, b = mean(model$rho), col = "#D6604D", lwd = 2)
legend("topleft", sprintf("slope = ρ̄ = %.3f", mean(model$rho)),
       col = "#D6604D", lty = 1, lwd = 2, bty = "n")

# 6b  Ratio Y_C / N (observed detection rate) by N quantile
breaks_q <- unique(quantile(N_vec, probs = seq(0, 1, 0.1)))
if (length(breaks_q) < 3L) breaks_q <- c(0, max(N_vec, 1) / 2, max(N_vec, 1))
N_cut    <- cut(N_vec, breaks = breaks_q, include.lowest = TRUE)
ratio    <- ifelse(N_vec > 0, YC_vec / N_vec, NA)
ratio_by_q <- tapply(ratio, N_cut, mean, na.rm = TRUE)
midpts     <- (breaks_q[-length(breaks_q)] + breaks_q[-1L]) / 2
plot(midpts, ratio_by_q, type = "b", pch = 19, col = "#2166AC", lwd = 2,
     xlab = "N decile midpoint", ylab = "Mean Y_C / N",
     main = "Effective detection rate by abundance",
     ylim = c(0, max(ratio_by_q, na.rm = TRUE) * 1.1))
abline(h = mean(model$rho), col = "#D6604D", lty = 2, lwd = 2)
legend("topright", sprintf("True ρ̄ = %.3f", mean(model$rho)),
       col = "#D6604D", lty = 2, lwd = 2, bty = "n")

# 6c  NegBin standardised residuals
fitted_mu <- rho_rep * N_vec + 1e-12
var_nb    <- fitted_mu * (1 + fitted_mu / true_params$r)
std_resid <- (YC_vec - fitted_mu) / sqrt(var_nb)
std_resid_s <- std_resid[idx_samp]
hist(std_resid_s[is.finite(std_resid_s)], breaks = 50, freq = FALSE,
     col = "#FCBBA1", border = "white",
     xlab = "Standardised NegBin residual",
     main = sprintf("NegBin residuals  (sd = %.3f)",
                    sd(std_resid_s, na.rm = TRUE)))
curve(dnorm(x), add = TRUE, col = "#A50F15", lwd = 2)
legend("topright", "N(0,1)", col = "#A50F15", lty = 1, lwd = 2, bty = "n")

# 6d  Bernoulli: N vs P(Y_P=1) — theoretical curve vs empirical
kappa_t   <- true_params$kappa
YP_vec    <- as.vector(Y_P)
N_breaks  <- unique(quantile(N_vec, probs = seq(0, 1, length.out = 30)))
if (length(N_breaks) < 3L) N_breaks <- c(0, max(N_vec, 1) / 2, max(N_vec, 1))
N_bin     <- cut(N_vec, breaks = N_breaks, include.lowest = TRUE)
emp_occ   <- tapply(YP_vec, N_bin, mean, na.rm = TRUE)
N_mid     <- (N_breaks[-length(N_breaks)] + N_breaks[-1L]) / 2
plot(N_mid, emp_occ, pch = 16, cex = 0.7, col = "#E08214",
     xlab = "Latent N_{i,t}", ylab = "Empirical P(Y_P = 1)",
     main = "Remote-sensing detection curve",
     ylim = c(0, 1))
N_seq <- seq(0, max(N_vec), length.out = 200)
lines(N_seq, 1 - exp(-kappa_t * N_seq), col = "#8E0152", lwd = 2)
legend("bottomright",
       c("Empirical", sprintf("1−exp(−%.2f·N)", kappa_t)),
       col = c("#E08214", "#8E0152"), pch = c(16, NA),
       lty = c(NA, 1), lwd = c(NA, 2), bty = "n")

# 6e  QQ-plot of NegBin residuals
qqnorm(std_resid_s[is.finite(std_resid_s)],
       pch = 16, cex = 0.3, col = adjustcolor("#2166AC", 0.4),
       main = "QQ-plot of NegBin standardised residuals")
qqline(std_resid_s[is.finite(std_resid_s)], col = "#D6604D", lwd = 2)

# 6f  Scatter: rho_i vs mean(Y_C_i) / mean(N_i)
mean_N_cell  <- colMeans(traj)
mean_YC_cell <- colMeans(Y_C, na.rm = TRUE)
eff_rho      <- ifelse(mean_N_cell > 0, mean_YC_cell / mean_N_cell, NA)
plot(model$rho, eff_rho,
     pch = 16, cex = 0.6, col = adjustcolor("#2166AC", 0.5),
     xlab = "True ρ_i", ylab = "Empirical Ȳ_C / N̄  per cell",
     main = "Detection probability: true vs empirical")
abline(a = 0, b = 1, col = "#D6604D", lwd = 2, lty = 2)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 7: Spatial Autocorrelation
# ════════════════════════════════════════════════════════════════════════════
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

# 7a  Moran's I of latent N over time
morans_I <- function(v, nbrs) {
  v   <- as.numeric(v)
  mu  <- mean(v)
  dev <- v - mu
  s2  <- mean(dev^2) + 1e-12
  numer <- denom <- 0
  W_sum <- 0
  for (i in seq_along(v)) {
    for (j in nbrs[[i]]) {
      numer <- numer + dev[i] * dev[j]
      W_sum <- W_sum + 1
    }
  }
  (length(v) / W_sum) * (numer / (length(v) * s2))
}

cat("Computing Moran's I over time ...\n")
mi_N  <- sapply(seq_len(T_STEPS + 1L), function(t) morans_I(traj[t, ], grid$neighbours))
mi_YC <- sapply(seq_len(T_STEPS + 1L), function(t) morans_I(Y_C[t, ], grid$neighbours))

plot(time_axis, mi_N, type = "b", pch = 19, cex = 0.7,
     col = "#2166AC", lwd = 2,
     xlab = "Time step", ylab = "Moran's I",
     main = "Spatial autocorrelation over time",
     ylim = range(c(mi_N, mi_YC)))
lines(time_axis, mi_YC, type = "b", pch = 17, cex = 0.7,
      col = "#4DAC26", lwd = 2, lty = 2)
abline(h = 0, col = "grey60", lty = 3)
legend("topright", c("Latent N", "Observed Y_C"),
       col = c("#2166AC", "#4DAC26"), lty = 1:2, lwd = 2,
       pch = c(19, 17), bty = "n")

# 7b  Empirical variogram of N at final time (binned)
N_final  <- traj[T_STEPS + 1L, ]
D_flat   <- as.vector(grid$dists)
diff_sq  <- outer(N_final, N_final, function(a, b) (a - b)^2 / 2)
d_vec    <- D_flat
g_vec    <- as.vector(diff_sq)
ok       <- d_vec > 0
d_ok     <- d_vec[ok]; g_ok <- g_vec[ok]
d_cuts   <- cut(d_ok, breaks = 15)
vario_d  <- tapply(d_ok, d_cuts, mean)
vario_g  <- tapply(g_ok, d_cuts, mean)
plot(vario_d, vario_g, type = "b", pch = 16, col = "#2166AC", lwd = 2,
     xlab = "Distance (cells)", ylab = "Semivariance  γ(h)",
     main = "Empirical variogram of N (t = final)")
abline(h = var(N_final), col = "grey50", lty = 2)
legend("bottomright", "Sill = Var(N)", col = "grey50", lty = 2, bty = "n")

# 7c  Correlation of N between adjacent cells (lagged correlogram)
max_lag <- min(10L, NROW - 1L)
lag_cor <- numeric(max_lag)
for (lag in seq_len(max_lag)) {
  pairs <- which(abs(row(grid$dists) - col(grid$dists)) == lag &
                 row(grid$dists) < col(grid$dists), arr.ind = FALSE)
  if (length(pairs) == 0L) { lag_cor[lag] <- NA; next }
  i_idx <- ((pairs - 1L) %% m) + 1L
  j_idx <- ((pairs - 1L) %/% m) + 1L
  lag_cor[lag] <- cor(N_final[i_idx], N_final[j_idx])
}
plot(seq_len(max_lag), lag_cor, type = "b", pch = 16, col = "#2166AC", lwd = 2,
     xlab = "Lag (cells)", ylab = "Correlation",
     main = "Correlogram of N (row direction)",
     ylim = c(-0.2, 1))
abline(h = c(0, 1.96 / sqrt(m)), lty = c(1, 2), col = c("grey50", "#D6604D"))

# 7d  Scatter: alpha_i vs mean(N_i) — habitat-abundance relationship
plot(model$alpha, colMeans(traj),
     pch = 16, cex = 0.5, col = adjustcolor("#2166AC", 0.5),
     xlab = "Habitat suitability  α_i",
     ylab = "Mean N_i (over time)",
     main = "Habitat suitability vs abundance")
lm_an <- lm(colMeans(traj) ~ model$alpha)
abline(lm_an, col = "#D6604D", lwd = 2)
legend("topleft", sprintf("R² = %.3f", summary(lm_an)$r.squared),
       bty = "n")

# 7e  Spatial map of N coefficient of variation
cv_N <- apply(traj, 2, sd) / (colMeans(traj) + 1e-12)
plot_heatmap(cv_N, title = "CV of N_i (temporal)",
             col = colorRampPalette(c("#FFFFB2", "#BD0026"))(64))

# 7f  Scatter: persistence phi_i vs temporal CV of N
plot(model$phi, cv_N,
     pch = 16, cex = 0.5, col = adjustcolor("#2166AC", 0.5),
     xlab = "Baseline persistence  φ_i",
     ylab = "Temporal CV of N_i",
     main = "Persistence vs temporal variability")
lm_pv <- lm(cv_N ~ model$phi)
abline(lm_pv, col = "#D6604D", lwd = 2)
legend("topright", sprintf("R² = %.3f", summary(lm_pv)$r.squared), bty = "n")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 8: Overdispersion & Summary Statistics
# ════════════════════════════════════════════════════════════════════════════
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

# 8a  Mean-variance per time step (check overdispersion)
step_mean <- rowMeans(traj)
step_var  <- apply(traj, 1, var)
plot(step_mean, step_var, pch = 16, cex = 0.7,
     col = adjustcolor("#2166AC", 0.6),
     xlab = "Mean N (per time step)", ylab = "Var N (per time step)",
     main = "Mean-variance (time steps)\nCheck overdispersion")
abline(a = 0, b = 1, lty = 2, col = "grey50")
lm_mv <- lm(step_var ~ step_mean)
abline(lm_mv, col = "#D6604D", lwd = 2)
legend("topleft",
       c("Slope = 1 (Poisson)", sprintf("Slope = %.2f", coef(lm_mv)[2])),
       col = c("grey50", "#D6604D"), lty = c(2, 1), lwd = 2, bty = "n")

# 8b  Summary statistics s1-s5 as a bar chart
s_cols <- c("#2166AC","#4DAC26","#E08214","#D6604D","#762A83")
bp <- barplot(s_obs, names.arg = paste0("s", 1:5), col = s_cols,
              main = "Summary statistics  s(Y_obs)",
              ylab = "Value", border = NA)
text(bp, s_obs + sign(s_obs) * 0.01 * diff(range(s_obs)),
     sprintf("%.3f", s_obs), cex = 0.75, font = 2)
legend("topright", labs5, fill = s_cols, bty = "n", cex = 0.75)

# 8c  Variability of s1-s5 across replicate simulations (n=30 pilot draws)
cat("Running 30 pilot simulations for summary-stat variability ...\n")
n_pilot <- 30L
pilot_s <- matrix(NA_real_, n_pilot, 5L)
for (b in seq_len(n_pilot)) {
  theta_b <- sample_prior()
  mod_b   <- tryCatch(build_model(theta_b, grid, X, W), error = function(e) NULL)
  if (is.null(mod_b)) next
  tr_b    <- mod_b$simulate(rpois(m, 10), T_STEPS)
  YC_b    <- matrix(NA_real_, T_STEPS + 1L, m)
  YP_b    <- matrix(NA_real_, T_STEPS + 1L, m)
  for (t in seq_len(T_STEPS + 1L)) {
    YC_b[t, ] <- mod_b$observe_count(tr_b[t, ])
    YP_b[t, ] <- mod_b$observe_binary(tr_b[t, ])
  }
  pilot_s[b, ] <- tryCatch(compute_summaries(YC_b, YP_b, grid),
                            error = function(e) rep(NA_real_, 5L))
}

boxplot(pilot_s, names = paste0("s", 1:5), col = s_cols,
        main = "Summary statistic variability\n(30 prior-predictive draws)",
        ylab = "Value", outline = FALSE, border = "grey40")
for (k in seq_along(s_obs))
  points(k, s_obs[k], pch = 18, cex = 1.5, col = "black")
legend("topright", "Observed s(Y)", pch = 18, pt.cex = 1.5, bty = "n")

# 8d  Cumulative N over time as area chart (top-5 and rest)
total_by_cell <- colSums(traj)
top5          <- order(total_by_cell, decreasing = TRUE)[1:5]
rest_sum      <- rowSums(traj[, -top5])
plot_df       <- cbind(traj[, top5], rest_sum)
col_area <- c("#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00","grey70")
cumulative <- t(apply(plot_df, 1, cumsum))
plot(time_axis, cumulative[, 6], type = "n",
     ylim = c(0, max(cumulative)),
     xlab = "Time step", ylab = "Cumulative N",
     main = "Population composition\n(top-5 cells vs rest)")
for (k in rev(seq_len(6))) {
  base_k <- if (k == 1L) rep(0, T_STEPS + 1L) else cumulative[, k - 1L]
  polygon(c(time_axis, rev(time_axis)),
          c(cumulative[, k], rev(base_k)),
          col = col_area[k], border = NA)
}
legend("topright", c(paste0("Cell #", top5), "Other cells"),
       fill = col_area, bty = "n", cex = 0.7)

# 8e  Negative-binomial log-likelihood surface over r (MoM vs truth)
YC_all_valid <- as.vector(Y_C[!is.na(Y_C)])
mu_mle <- mean(YC_all_valid)
r_seq  <- seq(0.5, 15, length.out = 80)
ll_seq <- sapply(r_seq, function(rv) {
  sum(dnbinom(YC_all_valid[YC_all_valid <= 200], size = rv,
              mu = mu_mle, log = TRUE))
})
plot(r_seq, ll_seq, type = "l", lwd = 2, col = "#2166AC",
     xlab = "r (overdispersion)", ylab = "Log-likelihood",
     main = "NegBin profile log-likelihood\n(over r, mu fixed at MLE)")
abline(v = true_params$r, col = "#D6604D", lty = 2, lwd = 2)
abline(v = r_hat,         col = "#4DAC26", lty = 3, lwd = 2)
legend("bottomright",
       c(sprintf("True r = %.1f", true_params$r),
         sprintf("MoM  r = %.1f", r_hat)),
       col = c("#D6604D", "#4DAC26"), lty = c(2, 3), lwd = 2, bty = "n")

# 8f  Temporal autocorrelation function of mean count
acf_obj <- acf(rowMeans(Y_C, na.rm = TRUE), lag.max = min(10, T_STEPS),
               plot = FALSE)
plot(acf_obj, main = "ACF of mean Y_C over time",
     col = "#2166AC", lwd = 2)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 9: Survival Component Analysis
# ════════════════════════════════════════════════════════════════════════════
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

# Reconstruct S, I, L components for one representative step (t=2)
set.seed(99)
N_t1 <- traj[1L, ]
p_s  <- model$phi * exp(-N_t1 / (model$K + 1e-12))
p_s  <- pmin(pmax(p_s, 0), 1)
S_t2 <- rbinom(m, size = N_t1, prob = p_s)
g    <- N_t1 / (N_t1 + true_params$gamma + 1e-12)
L_t2 <- rpois(m, lambda = model$psi)
N_t2 <- traj[2L, ]
I_t2 <- pmax(N_t2 - S_t2 - L_t2, 0L)   # residual (approximate)

# 9a  Bar: mean S / I / L contributions
comp_means <- c(mean(S_t2), mean(I_t2), mean(L_t2))
barplot(comp_means, names.arg = c("Survival S", "Immigration I", "LDD L"),
        col = c("#4DAC26", "#2166AC", "#D6604D"),
        ylab = "Mean count contribution",
        main = sprintf("Mean S / I / L at t=2\n(S+I+L = %.1f, true N = %.1f)",
                       sum(comp_means), mean(N_t2)))

# 9b  Density-regulated survival probability vs N
N_range <- seq(0, max(N_t1), length.out = 200)
ps_curve <- mean(model$phi) * exp(-N_range / mean(model$K))
plot(N_range, ps_curve, type = "l", lwd = 2, col = "#4DAC26",
     xlab = "N_{i,t-1}", ylab = "p_s = φ_i exp(−N/K_i)",
     main = "Density-regulated survival probability\n(mean φ, mean K)")
abline(h = mean(model$phi),    col = "grey50", lty = 2)
abline(v = mean(model$K),      col = "#2166AC", lty = 3)
abline(h = mean(model$phi)/exp(1), col = "#D6604D", lty = 4)
legend("topright",
       c("p_s(N)", "φ̄ (N→0)", "K̄", "φ̄/e (at N=K)"),
       col = c("#4DAC26","grey50","#2166AC","#D6604D"),
       lty = c(1,2,3,4), lwd = 2, bty = "n", cex = 0.8)

# 9c  Density scaling function g(N; gamma)
g_curve <- N_range / (N_range + true_params$gamma)
plot(N_range, g_curve, type = "l", lwd = 2, col = "#2166AC",
     xlab = "Source N_{j,t-1}", ylab = "g(N; γ)",
     main = sprintf("Density-scaling  g(N; γ=%.0f)", true_params$gamma),
     ylim = c(0, 1))
abline(h = 0.5, v = true_params$gamma, col = "grey50", lty = 2)
legend("bottomright",
       sprintf("g(γ, γ) = 0.5  (γ=%.0f)", true_params$gamma),
       col = "grey50", lty = 2, bty = "n")

# 9d  Scatter: N_{t-1} vs S (survival counts)
plot(N_t1[N_t1 > 0], S_t2[N_t1 > 0],
     pch = 16, cex = 0.4, col = adjustcolor("#4DAC26", 0.5),
     xlab = "N_{i,t-1}", ylab = "S_{i,t}  (survivors)",
     main = "Survival count vs prior abundance")
abline(a = 0, b = mean(p_s), col = "#D6604D", lwd = 2)

# 9e  LDD rate psi distribution
hist(log(model$psi + 1e-4), breaks = 20, freq = FALSE,
     col = "#FDD0A2", border = "white",
     xlab = "log(ψ_i + 1e-4)", main = "LDD rate distribution  (log scale)")

# 9f  Persistence phi vs carrying capacity K (joint distribution)
plot(model$phi, log(model$K),
     pch = 16, cex = 0.5, col = adjustcolor("#2166AC", 0.5),
     xlab = "Baseline persistence  φ_i",
     ylab = "log(K_i)",
     main = "φ_i vs K_i  (cell-level parameters)")

dev.off()

cat("═══════════════════════════════════════════════════════════════\n")
cat("EDA complete.  Plots saved to:  eda_report.pdf\n")
cat("Pages:\n")
cat("  1. Temporal dynamics (6 panels)\n")
cat("  2. Spatial heatmaps: latent N + habitat suitability\n")
cat("  3. Spatial heatmaps: observations Y_C and Y_P\n")
cat("  4. Spatial maps of derived cell parameters\n")
cat("  5. Marginal distributions (N, Y_C, Y_P, rho, K)\n")
cat("  6. Detection analysis (N vs Y_C, NegBin residuals, Bernoulli curve)\n")
cat("  7. Spatial autocorrelation (Moran's I, variogram, correlogram)\n")
cat("  8. Overdispersion, summary statistics, NegBin profile likelihood\n")
cat("  9. Survival/immigration/LDD component analysis\n")
cat("═══════════════════════════════════════════════════════════════\n")
