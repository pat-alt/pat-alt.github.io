# Packages
using LinearAlgebra

# Helper functions:
∑(vector)=sum(vector)

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

# Gradient:
function ∇(w,w_0,X,y,H_0)
    N = length(y)
    μ = sigmoid(w,X)
    Δw = w-w_0
    g = 1/N * ∑((μ[n]-y[n]) * X[n,:] for n=1:N) .+ H_0'Δw 
    # Normalalize gradient to length 1:
    # g = g / sqrt(g'g)
    return g
end

# Hessian:
function ∇∇(X,y,w,H_0)
    N = length(y)
    μ = sigmoid(w,X)
    return 1/N .* ∑(μ[n] * (1-μ[n]) * X[n,:] * X[n,:]' for n=1:N) + H_0 
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
function bayes_logreg(X,y,w_0,H_0)
    # Model:
    w_map = sgd(X,y,∇,w_0,H_0) # fit the model (find mode of posterior distribution)
    Σ_map = inv(∇∇(X,y,w_map,H_0)) # inverse Hessian at the mode
    Σ_map = Symmetric(Σ_map)
    # Output:
    mod = BayesLogreg(w_map, Σ_map)
    return mod
end

# Methods:
μ(mod::BayesLogreg) = mod.μ
Σ(mod::BayesLogreg) = mod.Σ
using Distributions
function sample_posterior(mod::BayesLogreg, n)
    rand(MvNormal(mod.μ, mod.Σ),n)
end
function posterior_predictive(mod::BayesLogreg, X)
end
