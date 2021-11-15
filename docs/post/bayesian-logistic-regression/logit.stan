// The input data.
data {
  int<lower=0> N;   // number of samples
  int<lower=0> K;   // number of predictors
  matrix[N, K] X;   // predictor matrix
  int<lower=0,upper=1> y[N]; // labels
  real<lower=0> sigma; // standard deviation of prior
}

// The parameters accepted by the model.
parameters {
  real alpha;       // intercept
  vector[K] beta;   // coefficients for predictors
}

// The model to be estimated.
model {

  y ~ bernoulli_logit(alpha + X * beta);

  // Prior models for the unobserved parameters
  alpha ~ normal(0, 1);
  for (k in 1:K)
    beta[k] ~ normal(0, sigma);
}

// Other quantities.
generated quantities {

  // Using the fitted model for probabilistic predicition
  vector[N] y_hat = inv_logit(alpha + X * beta);

}

