#' Neural Dispersal Kernel: 3-layer ReLU MLP
#'
#' @description
#' Maps pairwise habitat-feature vector z_{ji} to a dispersal weight in (0, 1).
#' Architecture: z_dim → H1 (ReLU) → H2 (ReLU) → 1 (sigmoid).
#' Implements Eqs (10)–(12) of the SC-ABM-NKD paper using pure R matrix ops.
#' Weights are initialised with He (Kaiming) normal scaling.
#'
#' @export
NeuralDispersalKernel <- R6::R6Class("NeuralDispersalKernel",
  public = list(

    #' @field z_dim Input feature dimension.
    z_dim = NULL,
    #' @field H1 Width of first hidden layer.
    H1 = NULL,
    #' @field H2 Width of second hidden layer.
    H2 = NULL,

    #' @description Initialise kernel with random He-scaled weights.
    #' @param z_dim Input dimension (p + 1 + max_nn from the grid).
    #' @param H1 First hidden-layer width (default 32).
    #' @param H2 Second hidden-layer width (default 16).
    initialize = function(z_dim, H1 = 32L, H2 = 16L) {
      self$z_dim <- as.integer(z_dim)
      self$H1    <- as.integer(H1)
      self$H2    <- as.integer(H2)
      private$W1 <- matrix(rnorm(H1 * z_dim, sd = sqrt(2.0 / z_dim)), H1, z_dim)
      private$b1 <- rep(0.0, H1)
      private$W2 <- matrix(rnorm(H2 * H1,    sd = sqrt(2.0 / H1)),    H2, H1)
      private$b2 <- rep(0.0, H2)
      private$w3 <- rnorm(H2, sd = sqrt(2.0 / H2))
      private$b3 <- 0.0
    },

    #' @description Forward pass.
    #' @param Z Numeric matrix of shape (n_pairs, z_dim), or a single row vector.
    #' @return Numeric vector of length n_pairs, values in (0, 1).
    forward = function(Z) {
      if (is.vector(Z)) Z <- matrix(Z, nrow = 1L)
      # Layer 1: (n, z_dim) %*% (z_dim, H1) -> (n, H1)
      H1_out <- relu(add_bias(tcrossprod(Z, private$W1), private$b1))
      # Layer 2: (n, H1) %*% (H1, H2) -> (n, H2)
      H2_out <- relu(add_bias(tcrossprod(H1_out, private$W2), private$b2))
      # Output: (n, H2) %*% (H2, 1) -> (n,) in (0,1)
      sigmoid(as.vector(H2_out %*% private$w3) + private$b3)
    },

    #' @description Return all weight matrices and bias vectors as a named list.
    get_weights = function() {
      list(W1 = private$W1, b1 = private$b1,
           W2 = private$W2, b2 = private$b2,
           w3 = private$w3, b3 = private$b3)
    },

    #' @description Set weights from a named list (e.g. after training).
    #' @param wts Named list with elements W1, b1, W2, b2, w3, b3.
    set_weights = function(wts) {
      private$W1 <- wts$W1; private$b1 <- wts$b1
      private$W2 <- wts$W2; private$b2 <- wts$b2
      private$w3 <- wts$w3; private$b3 <- wts$b3
      invisible(self)
    }
  ),

  private = list(
    W1 = NULL, b1 = NULL,
    W2 = NULL, b2 = NULL,
    w3 = NULL, b3 = NULL
  )
)
