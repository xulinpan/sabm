# Run this script from the package root to execute the demo without installing.
# Usage:  Rscript run_demo.R
#   or in an R session:  source("run_demo.R")

if (!requireNamespace("R6", quietly = TRUE))
  install.packages("R6")

# Source all package files in dependency order
for (f in c("utils", "grid", "kernel", "model",
            "summaries", "prior", "abc_smc", "build_model", "demo")) {
  source(file.path("R", paste0(f, ".R")))
}

result <- run_demo(
  nrow         = 8L,
  ncol         = 8L,
  T            = 10L,
  n_particles  = 100L,
  n_populations = 3L,
  seed         = 42L
)
