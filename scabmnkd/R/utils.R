# ── Internal math helpers ──────────────────────────────────────────────────────

sigmoid <- function(x) 1.0 / (1.0 + exp(-x))

relu <- function(x) pmax(x, 0.0)

# Row-wise addition of a bias vector to a matrix (sweeps over columns)
add_bias <- function(mat, bias) sweep(mat, 2L, bias, "+")
