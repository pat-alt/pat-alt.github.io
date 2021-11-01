// The input data.
data {
  int<lower=0> N;   // number of samples
  int<lower=0> K;   // number of predictors
  matrix[N, K] X;   // predictor matrix
  int<lower=0,upper=1> y[N]; // labels
}

// The parameters accepted by the model.
parameters {
  real alpha;       // intercept
  vector[K] beta;   // coefficients for predictors
}

// The model to be estimated.
model {

  y ~ bernoulli_logit(alpha + X * beta);

  // // Prior models for the unobserved parameters
  // alpha ~ normal(0, 1);
  // beta ~ normal(1, 1);
}
