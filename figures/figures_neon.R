# figures_neon.R
# Generates publication figures for the SC-ABM NEON application section.
# Output: PDF files in D:/research/figures/

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(RColorBrewer)
library(ggrepel)

OUT  <- "D:/research/figures"
DATA <- "D:/research/data/processed"
INF  <- "D:/research/inference"

has_v3 <- file.exists(file.path(INF, "pp_trajectory_v3.csv")) &&
          file.exists(file.path(INF, "particles_neon_v3.csv"))

sigmoid <- function(x) 1 / (1 + exp(-x))

# ── Shared theme ─────────────────────────────────────────────────────────────

theme_paper <- function() {
  theme_classic(base_size = 10) +
  theme(
    strip.background = element_blank(),
    strip.text       = element_text(size = 9, face = "bold"),
    axis.title       = element_text(size = 9),
    axis.text        = element_text(size = 8),
    legend.text      = element_text(size = 8),
    legend.title     = element_text(size = 8),
    plot.title       = element_text(size = 10, face = "bold", hjust = 0.5),
    panel.grid.major = element_line(colour = "grey92", linewidth = 0.3)
  )
}

# ── Load data ─────────────────────────────────────────────────────────────────

grid   <- read.csv(file.path(DATA, "mos_grid.csv"))
edges  <- read.csv(file.path(DATA, "mos_neighbors.csv"))
tidx   <- read.csv(file.path(DATA, "mos_time_index.csv"))
pp_tr  <- read.csv(file.path(INF,  "pp_trajectory.csv"))
parts  <- read.csv(file.path(INF,  "particles_neon.csv"))
obs_C  <- read.csv(file.path(DATA, "mos_obs_C.csv"))

# Convert 0-indexed cell/time to 1-indexed
obs_C$cell_idx <- obs_C$cell_idx + 1L
obs_C$time_idx <- obs_C$time_idx + 1L

# Observed mean Y_C per time step
obs_mean <- obs_C %>%
  group_by(time_idx) %>%
  summarise(obs = mean(y_C, na.rm = TRUE), .groups = "drop")

# Merge with time index dates
tidx$time_idx <- seq_len(nrow(tidx))
obs_mean <- left_join(obs_mean, tidx[, c("time_idx","date_CDT")], by = "time_idx")
obs_mean$date <- as.Date(obs_mean$date_CDT)

# Weighted median rho from v1 particles
rho_med <- with(parts, {
  ord <- order(rho); cw <- cumsum(weight[ord])
  rho[ord][which(cw >= 0.5)[1L]]
})

# Constant-model posterior predictive (latent N → expected Y_C)
pp_tr$date <- as.Date(pp_tr$date_CDT)
pp_tr <- pp_tr %>%
  mutate(
    pp_lo  = rho_med * pp_mean_N_2.5,
    pp_med = rho_med * pp_mean_N_50,
    pp_hi  = rho_med * pp_mean_N_97.5
  )

# Seasonal-model posterior predictive (loaded if v3 has been run)
if (has_v3) {
  pp_v3   <- read.csv(file.path(INF, "pp_trajectory_v3.csv"))
  parts3  <- read.csv(file.path(INF, "particles_neon_v3.csv"))
  pp_v3$date <- as.Date(pp_v3$date_CDT)
  rho_med3 <- with(parts3, {
    ord <- order(rho); cw <- cumsum(weight[ord])
    rho[ord][which(cw >= 0.5)[1L]]
  })
  pp_v3 <- pp_v3 %>%
    mutate(
      pp_lo  = rho_med3 * pp_mean_N_2.5,
      pp_med = rho_med3 * pp_mean_N_50,
      pp_hi  = rho_med3 * pp_mean_N_97.5
    )
}

# ── Figure 1: Study area map ──────────────────────────────────────────────────

# Build edge segments (undirected: keep each pair once)
edge_df <- edges %>%
  left_join(grid[, c("cell_idx","lon","lat")], by = "cell_idx") %>%
  rename(lon1 = lon, lat1 = lat) %>%
  left_join(grid[, c("cell_idx","lon","lat")],
            by = c("neighbor_idx" = "cell_idx")) %>%
  rename(lon2 = lon, lat2 = lat) %>%
  filter(cell_idx < neighbor_idx)          # undirected: show each edge once

grid$habitat <- recode(grid$nlcd_class,
  woodyWetlands   = "Woody Wetland",
  deciduousForest = "Deciduous Forest",
  mixedForest     = "Mixed Forest")

hab_colours <- c(
  "Woody Wetland"    = "#4393C3",
  "Deciduous Forest" = "#74C476",
  "Mixed Forest"     = "#A1D99B")

# Short label (last 3 chars of plot id)
grid$label <- sub("UNDE_", "", grid$plot_id)

fig1 <- ggplot() +
  geom_segment(data = edge_df,
               aes(x = lon1, y = lat1, xend = lon2, yend = lat2),
               colour = "grey60", linewidth = 0.5, linetype = "dashed") +
  geom_point(data = grid,
             aes(x = lon, y = lat, fill = habitat),
             shape = 21, size = 4, colour = "white", stroke = 0.5) +
  geom_text_repel(data = grid,
                  aes(x = lon, y = lat, label = label),
                  size = 2.5, colour = "grey25",
                  box.padding = 0.35, point.padding = 0.3,
                  seed = 42L) +
  scale_fill_manual(values = hab_colours, name = "Habitat") +
  coord_fixed(ratio = 1 / cos(46.24 * pi / 180)) +
  labs(x = "Longitude (°W)", y = "Latitude (°N)",
       title = "NEON UNDE monitoring plots (m = 10)") +
  theme_paper() +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.35, "cm"))

ggsave(file.path(OUT, "fig1_study_area.pdf"),
       fig1, width = 8.5, height = 7, units = "cm")
cat("Saved fig1_study_area.pdf\n")

# ── Figure 2: Posterior predictive — constant (failure) vs seasonal (fix) ─────

breaks_dates <- as.Date(c("2024-04-01","2024-05-01","2024-06-01",
                           "2024-07-01","2024-08-01","2024-09-01","2024-10-01"))
labels_dates <- format(breaks_dates, "%b")

# Shared theme additions for both panels
pp_colours <- c("Posterior predictive (median)" = "#2166AC",
                "Observed mean"                 = "#D6604D")

.pp_panel <- function(pp_data, obs_data, title_str) {
  pd <- pp_data %>% left_join(obs_data[, c("date","obs")], by = "date")
  ggplot(pd, aes(x = date)) +
    geom_ribbon(aes(ymin = pp_lo, ymax = pp_hi),
                fill = "#4393C3", alpha = 0.22) +
    geom_line(aes(y = pp_med,
                  colour = "Posterior predictive (median)"),
              linewidth = 0.75) +
    geom_point(aes(y = obs, colour = "Observed mean"),
               size = 1.6, shape = 16) +
    geom_line(aes(y = obs, colour = "Observed mean"),
              linewidth = 0.45) +
    scale_colour_manual(values = pp_colours, name = NULL) +
    scale_x_date(breaks = breaks_dates, labels = labels_dates,
                 expand = expansion(mult = 0.02)) +
    scale_y_continuous(labels = comma,
                       expand = expansion(mult = c(0, 0.06))) +
    labs(x = NULL,
         y = expression(paste("Mean ", Y[it]^{(C)}, " per cell")),
         title = title_str) +
    theme_paper() +
    theme(legend.position      = c(0.72, 0.88),
          legend.background    = element_rect(fill = "white", colour = NA),
          legend.key.size      = unit(0.38, "cm"))
}

panel_const <- .pp_panel(
  pp_tr, obs_mean,
  "(a) Constant-ψ model: posterior predictive miss"
)

if (has_v3) {
  panel_seas <- .pp_panel(
    pp_v3, obs_mean,
    "(b) Seasonal-ψ model: bell-curve recovered"
  )
  library(patchwork)
  fig2 <- panel_const / panel_seas +
    plot_annotation(
      title   = "NEON UNDE 2024 — posterior predictive checks",
      caption = paste("Shading: 95% PP band.",
                      "Constant model cannot produce autumn decline;",
                      "seasonal model (psi_t ~ Gaussian) recovers the trajectory."),
      theme = theme(
        plot.title   = element_text(size = 10, face = "bold", hjust = 0.5),
        plot.caption = element_text(size = 7,  colour = "grey40", hjust = 0.5))
    )
  ggsave(file.path(OUT, "fig2_pp_trajectory.pdf"),
         fig2, width = 12, height = 14, units = "cm")
  cat("Saved fig2_pp_trajectory.pdf (two-panel: constant vs seasonal)\n")
} else {
  ggsave(file.path(OUT, "fig2_pp_trajectory.pdf"),
         panel_const, width = 12, height = 7, units = "cm")
  cat("Saved fig2_pp_trajectory.pdf (constant model only; run v3 for two-panel)\n")
}

# ── Figure 3: Posterior parameter distributions ───────────────────────────────

wkde <- function(x, w, n = 512) {
  bw <- bw.SJ(x) * 1.2
  xs <- seq(min(x), max(x), length.out = n)
  dens <- sapply(xs, function(xi)
    sum(w * dnorm(xi, mean = x, sd = bw)))
  data.frame(x = xs, density = dens / sum(dens * diff(c(xs[1], xs[2]))))
}

# Use v3 particles if available, otherwise v1
fig3_parts  <- if (has_v3) parts3  else parts
fig3_wtname <- if (has_v3) "weight" else "weight"
fig3_title  <- if (has_v3) {
  "ABC-SMC posterior — seasonal model (NEON UNDE 2024)"
} else {
  "ABC-SMC posterior — constant model (NEON UNDE 2024)"
}

params_info <- list(
  list(key = "phi",   label = expression(phi),
       pri_med = plogis(0)),
  list(key = "K",     label = expression(K),
       pri_med = exp(9)),
  list(key = "psi",   label = expression(psi),
       pri_med = exp(6)),
  list(key = "rho",   label = expression(rho),
       pri_med = plogis(-2)),
  list(key = "r",     label = expression(r),
       pri_med = 2 / 0.5),
  list(key = "kappa", label = expression(kappa),
       pri_med = log(2)),
  list(key = "gamma", label = expression(h),
       pri_med = 2 / 0.2 * log(2))
)

# Append seasonal parameters when v3 particles are present
if (has_v3) {
  params_info <- c(params_info, list(
    list(key = "mu_s",    label = expression(mu[s]),
         pri_med = 8.0),               # Uniform(3,13) median
    list(key = "sigma_s", label = expression(sigma[s]),
         pri_med = 3.0)                # exp(log(3)) prior median
  ))
}

log_params <- c("K", "psi")

.make_panel <- function(p, pdata) {
  key  <- p$key
  vals <- pdata[[key]]
  wts  <- pdata$weight / sum(pdata$weight)
  d    <- wkde(vals, wts)

  ord <- order(vals); cw <- cumsum(wts[ord])
  lo  <- vals[ord][which(cw >= 0.025)[1L]]
  med <- vals[ord][which(cw >= 0.500)[1L]]
  hi  <- vals[ord][which(cw >= 0.975)[1L]]

  g <- ggplot(d, aes(x = x, y = density)) +
    geom_ribbon(data = d[d$x >= lo & d$x <= hi, ],
                aes(ymin = 0, ymax = density),
                fill = "#4393C3", alpha = 0.35) +
    geom_line(linewidth = 0.7, colour = "#1F4E79") +
    geom_vline(xintercept = med, linetype = "solid",
               colour = "#D6604D", linewidth = 0.6) +
    geom_vline(xintercept = p$pri_med, linetype = "dashed",
               colour = "grey50", linewidth = 0.5) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
    labs(x = p$label, y = NULL) +
    theme_paper() +
    theme(axis.text.y  = element_blank(),
          axis.ticks.y = element_blank(),
          axis.line.y  = element_blank(),
          plot.margin  = margin(4, 6, 4, 6))

  if (key %in% log_params)
    g <- g + scale_x_continuous(labels = comma,
                                 limits = quantile(vals, c(0.001, 0.999)))
  g
}

library(patchwork)
panel_list <- lapply(params_info, .make_panel, pdata = fig3_parts)
n_panels   <- length(panel_list)

# Fill to even count for 2-col layout
if (n_panels %% 2L != 0L) panel_list <- c(panel_list, list(plot_spacer()))

rows <- lapply(seq(1, length(panel_list), by = 2L), function(i)
  panel_list[[i]] | panel_list[[i + 1L]])

fig3 <- Reduce(`/`, rows) +
  plot_annotation(
    title   = fig3_title,
    caption = paste("Shading: 95% CI",
                    "  Red line: posterior median",
                    "  Dashed: prior median"),
    theme = theme(
      plot.title   = element_text(size = 10, face = "bold", hjust = 0.5),
      plot.caption = element_text(size = 7, colour = "grey40", hjust = 0.5)))

fig3_height <- if (has_v3) 18 else 14
ggsave(file.path(OUT, "fig3_posteriors.pdf"),
       fig3, width = 12, height = fig3_height, units = "cm")
cat(sprintf("Saved fig3_posteriors.pdf (%d parameters)\n", n_panels))

cat("\nAll figures written to", OUT, "\n")
