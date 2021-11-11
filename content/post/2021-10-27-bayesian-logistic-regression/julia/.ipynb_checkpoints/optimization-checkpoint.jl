# Stochastic Gradient Descent:
function sgd(X,y,∇,w_0,H_0,ρ_0=1.0,T=10000,ε=0.001)
    # Initialization:
    N = length(y)
    w_t = w_0 # initial parameters (prior mode)
    w_hat = 1/T * w_t # iterate averaging
    ρ = ρ_0 # initial step size
    t = 0 # iteration count
    while t<T
        n_t = rand(1:N) # sample minibatch (single sample)
        ρ = ρ * exp(-ε*t) # exponential decay 
        w_t = w_t - ρ .* ∇(w_t,w_0,X[n_t,:]',y[n_t],H_0) # update mode
        w_hat += 1/T * w_t # iterate averaging
        t += 1 # update count
    end
    return w_hat
end