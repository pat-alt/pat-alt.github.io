# Packages
using LinearAlgebra

# Helper functions:
∑(vector)=sum(vector)

# Compute cartesian product over two vectors:
function expandgrid(x,y)
    N = length(x) * length(y)
    grid = Iterators.product(x,y) |>
        Iterators.flatten |>
        collect |>
        z -> reshape(z, (2,N)) |>
        transpose |>
        Matrix
    return grid
end

# Sigmoid function:
function sigmoid(w,X)
    trunc = 8.0 # truncation to avoid numerical over/underflow
    if !isa(X, Matrix)
        if length(size(X))>1
            X = X'
        end
        z = X'w
    else
        z = X*w
    end
    z = clamp.(z,-trunc,trunc)
    p = exp.(z)
    p = p ./ (1 .+ p)
    return p
end

# Negative log likelihood
function nll(w,w_0,X,y,H_0)
    N = length(y)
    D = size(X)[2]
    μ = sigmoid(w,X)
    Δw = w-w_0
    return ∑( y[n] * log(μ[n]) + (1-y[n]) * log(1-μ[n]) for n=1:N) + 1/2 * Δw'H_0*Δw
end

# Gradient:
function ∇(w,w_0,X,y,H_0)
    N = length(y)
    μ = sigmoid(w,X)
    Δw = w-w_0
    return ∑((μ[n]-y[n]) * X[n,:] for n=1:N) + H_0*Δw
end

# Hessian:
function ∇∇(X,y,w,H_0)
    N = length(y)
    μ = sigmoid(w,X)
    return ∑(μ[n] * (1-μ[n]) * X[n,:] * X[n,:]' for n=1:N) + H_0
end

# Stochastic Gradient Descent:
function sgd(X,y,∇,w_0,H_0,ρ_0=1.0,T=10000,ε=0.001)
    # Initialization:
    N = length(y)
    w_t = w_0 # initial parameters (prior mode)
    w_map = 1/T * w_t # iterate averaging
    ρ = ρ_0 # initial step size
    t = 0 # iteration count
    while t<T
        n_t = rand(1:N) # sample minibatch (single sample)
        ρ = ρ * exp(-ε*t) # exponential decay 
        w_t = w_t - ρ .* ∇(w_t,w_0,X[n_t,:]',y[n_t],H_0) # update mode
        w_map += 1/T * w_t # iterate averaging
        t += 1 # update count
    end
    return w_map
end

# Main function:
struct BayesLogreg
    μ::Vector{Float64}
    Σ::Matrix{Float64}
end
function bayes_logreg(X,y,w_0,H_0,sgd_options...)
    # Model:
    w_map = sgd(X,y,∇,w_0,H_0,sgd_options...) # fit the model (find mode of posterior distribution)
    Σ_map = inv(∇∇(X,y,w_map,H_0)) # inverse Hessian at the mode
    Σ_map = Symmetric(Σ_map) # to ensure matrix is Hermitian (i.e. avoid rounding issues)
    # Output:
    mod = BayesLogreg(w_map, Σ_map)
    return mod
end

# Methods:
μ(mod::BayesLogreg) = mod.μ
Σ(mod::BayesLogreg) = mod.Σ
using Distributions
# Sampling from posterior distribution:
function sample_posterior(mod::BayesLogreg, n)
    rand(MvNormal(mod.μ, mod.Σ),n)
end
# Posterior predictions:
function posterior_predictive(mod::BayesLogreg, X)
    μ = mod.μ # MAP mean vector
    Σ = mod.Σ # MAP covariance matrix
    # Inner product:
    if !isa(X, Matrix)
        if length(size(X))>1
            X = X'
        end
        z = X'μ
    else
        z = X*μ
    end
    # Probit approximation
    v = [X[n,:]'Σ*X[n,:] for n=1:size(X)[1]]
    κ = 1 ./ sqrt.(1 .+ π/8 .* v) 
    z = κ .* z
    # Truncation to avoid numerical over/underflow:
    trunc = 8.0 
    z = clamp.(z,-trunc,trunc)
    p = exp.(z)
    p = p ./ (1 .+ p)
    return p
end
