#' Stochastic gradient descent
#'
#' @param X Feature matrix.
#' @param y Vector containing training labels.
#' @param eta Learning rate.
#' @param n_iter Maximum number of iterations.
#' @param w_init Initial parameter values.
#' @param save_steps Boolean checking if coefficients should be saved at each step.
#'
#' @return
#' @export
#'
#' @author Patrick Altmeyer
linear_classifier <- function(X,y,eta=0.001,n_iter=1000,w_init=NULL,save_steps=FALSE) {
  # Initialization: ----
  n <- nrow(X) # number of observations
  d <- ncol(X) # number of dimensions
  if (is.null(w_init)) {
    w <- matrix(rep(0,d)) # initialize coefficients as zero...
  } else {
    w <- matrix(w_init) # ...unless initial values have been provided.
  }
  w_avg <- 1/n_iter * w # initialize average coefficients
  iter <- 1 # iteration count
  if (save_steps) {
    steps <- data.table(iter=0, w=c(w), d=1:d) # if desired, save coefficient at each step
  } else {
    steps <- NA
  }
  feasible_w <- TRUE # to check if coefficients are finite, non-nan, ...
  # Surrogate loss:
  l <- function(X,y,w) {
    x <- (-1) * crossprod(X,w) * y
    pmax(0,1 + x) # Hinge loss
  }
  grad <- function(X,y,w) {
    X %*% ifelse(crossprod(X,w) * y<=1,-y,0) # Gradient of Hinge loss
  }
  # Stochastic gradient descent: ----
  while (feasible_w & iter<n_iter) {
    t <- sample(1:n,1) # random draw
    X_t <- matrix(X[t,])
    y_t <- matrix(y[t])
    v_t <- grad(X_t,y_t,w) # compute estimate of gradient
    # Update:
    w <- w - eta * v_t # update coefficient vector
    feasible_w <- all(sapply(w, function(i) !is.na(i) & is.finite(i))) # check if feasible
    if (feasible_w) {
      w_avg <- w_avg + 1/n_iter * w # update average
    }
    if (save_steps) {
      steps <- rbind(steps, data.table(iter=iter, w=c(w), d=1:d))
    }
    iter <- iter + 1 # increase counter
  }
  # Output: ----
  output <- list(
    X = X,
    y = matrix(y),
    coefficients = w_avg,
    eta = eta,
    n_iter = n_iter,
    steps = steps
  )
  class(output) <- "classifier" # assign S3 class
  return(output)
}

# Methods: ----
print.classifier <- function(classifier) {
  print("Coefficients:")
  print(classifier$coefficients)
}
print <- function(classifier) {
  UseMethod("print")
}

predict.classifier <- function(classifier, newdata=NULL, discrete=TRUE) {
  if (!is.null(newdata)) {
    fitted <- newdata %*% classifier$coefficients # out-of-sampple prediction
  } else {
    fitted <- classifier$X %*% classifier$coefficients # in-sample fit
  }
  if (discrete) {
    fitted <- sign(fitted) # map to {-1,1}
  }
  return(fitted)
}
predict <- function(classifier, newdata=NULL, discrete=TRUE) {
  UseMethod("predict")
}
