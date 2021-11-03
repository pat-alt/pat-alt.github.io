# Packages
using LinearAlgebra

# Helper functions:
∑(vector)=sum(vector)

# Sigmoid function:
function sigmoid(w,X)
    return 1 ./ (1 .+ exp.(-X'w))
end

# Gradient:
function ∇(w,w_0,X,y,H_0,λ)
    N = length(y)
    μ = sigmoid(w,X)
    Δw = w-w_0
    return 1/N * ∑((μ[n]-y[n]) * X[n,:] for n=1:N) .+ H_0'Δw .+ 2 * λ * w
end

# Hessian:
function ∇∇(X,y,H_0,λ)
    N = length(y)
    μ = sigmoid(w,X)
    Δw = w-w_0
    return 1/N .* ∑(μ[n] * (1-μ[n]) * X[n,:] * X[n,:]' for n=1:N) + H_0 + UniformScaling(2 * λ)
end

# Stochastic Gradient Descent:
function sgd(X,y,∇,∇∇,w_0,H_0,ρ_0=1.0,T=1000,λ=1.0,ε=0.0001)
    # Initialization:
    N = length(y)
    w = w_0
    ρ = ρ_0
    # w_avg <- 1/n_iter * w # initialize average coefficients
    t = 0 # iteration count
    while t<T
        n_t = rand(1:N)
        println(n_t)
        ρ = ρ * exp(-ε*t) # exponential decay
        println(∇(w,w_0,X[n_t,:]',y[n_t],H_0,λ))
        w = w - ρ .* ∇(w,w_0,X[n_t,:]',y[n_t],H_0,λ)
        t += 1
    end
    return w
end

