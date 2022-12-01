# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
Lstm(; neuron=5, layer=3, σ=Flux.relu, rng=GLOBAL_RNG)
MLJFlux like builder that constructs a Long short term memory recurrent neural network, with three-layers and five neurons
using `layer` nodes in the hidden layer and the specified `neuron`. An activation function `σ` is applied between the
hidden and final layers. Each layers are initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a
`MersenneTwister`.
"""
mutable struct Lstm <: MLJFlux.Builder
    neuron::Int
    layer::Int
    σ::Function
end

Lstm(; neuron = 5, layer = 3, σ = Flux.relu) = Lstm(neuron, layer, σ)

function Base.show(io::IO, l::Lstm)
    print(
        io,
        "Long short term memory recurrent neural network(layer: ",
        l.layer,
        ", neuron: ",
        l.neuron,
    )
    print(io, ", activation function: ", l.σ)
    print(io, ")")
end

function MLJFlux.build(nn::Lstm, rng, n_in, n_out)
    init = Flux.glorot_uniform(rng)
    inner_layer =
        Array{typeof(Flux.LSTM(nn.neuron, nn.neuron, init = init)),2}(undef, nn.layer, 1)

    for i = 1:1:nn.layer
        inner_layer[i, 1] = Flux.LSTM(nn.neuron, nn.neuron, init = init)
    end

    return Flux.Chain(
        Flux.Dense(n_in, nn.neuron, bias = false, init = init),
        Flux.Chain(inner_layer...),
        Flux.Dense(nn.neuron, n_out, bias = false, init = init),
    )
end
