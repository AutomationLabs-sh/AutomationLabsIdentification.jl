# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
    Rbf(; neuron=5, layer=3, σ=Flux.relu, rng=GLOBAL_RNG)
MLJFlux like builder that constructs a radial basis function network, with one hidden layer and neurons and gaussian as activation function:
gaussian(x) = exp(-x^2). Each layers are initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a
`MersenneTwister`.
"""
mutable struct Rbf <: MLJFlux.Builder
    neuron::Int
end

Rbf(; neuron = 5) = Rbf(neuron)

function Base.show(io::IO, l::Rbf)
    print(io, "Radial basis function network(neuron: ", l.neuron)
    print(io, ")")
end

function MLJFlux.build(nn::Rbf, rng, n_in, n_out)

    init = Flux.glorot_uniform(rng) #weight initialisation
    gaussian(x) = exp(-x^2) #activiation fct

    return Flux.Chain(
        Flux.Dense(n_in, nn.neuron, bias = false, init = init),
        Flux.Chain(Flux.Dense(nn.neuron, nn.neuron, gaussian, init = init, bias = false)),
        Flux.Dense(nn.neuron, n_out, bias = false, init = init),
    )
end
