#' REVISE algoritm - a simplified version
#'
#' @param classifier The fitted classifier.
#' @param x_star Attributes of individual seeking individual recourse.
#' @param eta Learning rate.
#' @param lambda Regularization parameter.
#' @param n_iter Maximum number of operations.
#' @param save_steps Boolean indicating if intermediate steps should be saved.
#'
#' @return
#' @export
#'
#' @author Patrick Altmeyer
revise.classifier <- function(classifier,x_star,eta=1,lambda=0.01,n_iter=1000,save_steps=FALSE) {
  # Initialization: ----
  d <- length(x_star) # number of dimensions
  if (!is.null(names(x_star))) {
    d_names <- names(x_star) # names of attributes, if provided
  } else {
    d_names <- sprintf("X%i", 1:d)
  }
  w <- classifier$coefficients # coefficient vector
  x <- x_star # initialization of revised attributes
  distance <- 0 # initial distance from starting point
  converged <- predict(classifier, newdata = x)[1,1]==1 # positive outcome?
  iter <- 1 # counter
  if (save_steps) {
    steps <- data.table(iter=1, x=x, d=d_names) # save intermediate steps, if desired
  } else {
    steps <- NA
  }
  # Gradients:
  grad <- function(x,y,w) {
    w %*% ifelse(crossprod(x,w) * y<=1,-y,0) # gradient of Hinge loss with respect to X
  }
  grad_dist <- function(x,x_star) {
    d <- length(x_star)
    distance <- dist(matrix(cbind(x_star,x),nrow=d,byrow = T))
    matrix((x-x_star) / distance) # gradient of Euclidean distance with respect to X
  }
  # Gradient descent: ----
  while(!converged & iter<n_iter) {
    if (distance!=0) {
      x <- c(x - eta * (grad(x=matrix(x),y=1,w) + lambda * grad_dist(x,x_star))) # gradient descent step
    } else {
      x <- c(x - eta * grad(x=matrix(x),y=1,w)) # gradient with respect to distance not defined at zero
    }
    converged <- predict(classifier, newdata = x)[1,1]==1 # positive outcome?
    iter <- iter + 1 # update counter
    if (save_steps) {
      steps <- rbind(steps, data.table(iter=iter, x=x, d=d_names))
    }
    distance <- dist(matrix(cbind(x_star,x),nrow=d,byrow = T)) # update distance
  }
  # Output: ----
  if (converged) {
    revise <- x - x_star
  } else {
    revise <- NA
  }
  output <- list(
    x_star = x_star,
    revise = revise,
    classifier = classifier,
    steps = steps,
    lambda = lambda,
    distance = distance,
    mean_distance = mean(abs(revise))
  )
  return(output)
}

revise <- function(classifier,x_star,eta=1,lambda=0.01,n_iter=1000,save_steps=FALSE) {
  UseMethod("revise")
}
