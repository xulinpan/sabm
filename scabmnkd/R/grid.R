#' Flat-indexed 2-D spatial grid with queen's neighbourhood
#'
#' @description
#' Cells are indexed 1 … m in row-major order.
#' Queen's neighbourhood includes up to 8 adjacent cells.
#'
#' @export
SpatialGrid <- R6::R6Class("SpatialGrid",
  public = list(

    #' @field nrow Number of rows.
    nrow = NULL,
    #' @field ncol Number of columns.
    ncol = NULL,
    #' @field m Total number of cells (nrow * ncol).
    m = NULL,
    #' @field cell_size Physical size of each cell edge.
    cell_size = NULL,
    #' @field coords (m x 2) matrix of cell centroid coordinates.
    coords = NULL,
    #' @field dists (m x m) matrix of Euclidean distances between centroids.
    dists = NULL,
    #' @field neighbours List of length m; each element is an integer vector
    #'   of 1-based neighbour indices (queen's neighbourhood, ≤ 8 cells).
    neighbours = NULL,
    #' @field max_nn Maximum neighbourhood size across all cells.
    max_nn = NULL,

    #' @description Create a new SpatialGrid.
    #' @param nrow Number of rows.
    #' @param ncol Number of columns.
    #' @param cell_size Physical cell-edge length (default 1.0).
    initialize = function(nrow, ncol, cell_size = 1.0) {
      self$nrow      <- as.integer(nrow)
      self$ncol      <- as.integer(ncol)
      self$m         <- nrow * ncol
      self$cell_size <- cell_size

      # Centroid coordinates (row-major, 1-based)
      row_idx <- rep(seq_len(nrow), each = ncol)
      col_idx <- rep(seq_len(ncol), times = nrow)
      self$coords <- cbind(
        (row_idx - 1L) * cell_size,
        (col_idx - 1L) * cell_size
      )                                                # (m, 2)

      # Pairwise Euclidean distances
      self$dists <- as.matrix(dist(self$coords))       # (m, m)

      # Queen's neighbourhood lists
      self$neighbours <- private$build_queen_neighbours(nrow, ncol)
      self$max_nn     <- max(lengths(self$neighbours))
    }
  ),

  private = list(
    build_queen_neighbours = function(nrow, ncol) {
      m  <- nrow * ncol
      nb <- vector("list", m)
      for (idx in seq_len(m)) {
        r  <- (idx - 1L) %/% ncol + 1L   # 1-based row
        c_ <- (idx - 1L) %%  ncol + 1L   # 1-based col
        candidates <- integer(0)
        for (dr in -1L:1L) {
          for (dc in -1L:1L) {
            if (dr == 0L && dc == 0L) next
            rr <- r + dr
            cc <- c_ + dc
            if (rr >= 1L && rr <= nrow && cc >= 1L && cc <= ncol) {
              candidates <- c(candidates, (rr - 1L) * ncol + cc)
            }
          }
        }
        nb[[idx]] <- sort(candidates)
      }
      nb
    }
  )
)
