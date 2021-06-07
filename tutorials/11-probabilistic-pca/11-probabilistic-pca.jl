# Load Turing.
using Turing
# Load other dependencies
using Distributions, RDatasets

# Example data set - Fisher's Iris data set
data = dataset("datasets", "iris")
species = data[!, "Species"]
dat = data[!, 1:4]

@model pPCA(x, ::Type{T} = Float64) where {T} = begin

  # Dimensionality of the problem.
  N, D = size(x)

  # latent variable z
  z = Matrix{T}(undef, D, N)
  for d in 1:D
    z[d, :] ~ MvNormal(N, 1.)
  end

  # weights/loadings w
  w = Matrix{T}(undef, D, D)
  for d in 1:D
    w[d, :] ~ MvNormal(D, 1.)
  end

  mu = w * z

  for d in 1:D
    x[:,d] ~ MvNormal(mu[d,:], 1.)
  end

end

ppca = pPCA(dat)

# Hamiltonian Monte Carlo (HMC) sampler parameters
n_iterations = 10000
ϵ = 0.05
τ = 10

chain = sample(ppca, HMC(ϵ, τ), n_iterations)

# Extract paramter estimates for plotting.
w = reshape(mean(group(chain, :w))[:,2], (4,4))
z = reshape(mean(group(chain, :z))[:,2], size(dat))

#  plot(z[1, :], z[2,:])


## Extend to ARD
## Original paper
#
#  #  for (d in 1:D){
    #  w[d] ~ normal(0, sigma * alpha);
  #  }
  #  sigma ~ lognormal(0, 1);
  #  alpha ~ inv_gamma(1, 1);

