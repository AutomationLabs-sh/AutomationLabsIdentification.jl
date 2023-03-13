
# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
DenseNet(; neuron=2, layer=10, σ=Flux.relu, rng=GLOBAL_RNG)
MLJFlux like builder that constructs a denselly connected network, with ten-layers and two neurons
using `layer` nodes in the hidden layer and the specified `neuron`. An activation function `σ` is applied between the
hidden and final layers. Each layers are initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a
`MersenneTwister`.
"""
mutable struct DenseNet <: MLJFlux.Builder
    neuron::Int
    layer::Int
    σ::Function
end

DenseNet(; neuron = 2, layer = 10, σ = Flux.relu) = DenseNet(neuron, layer, σ)

function Base.show(io::IO, l::DenseNet)
    print(io, "Densely connected network(layer: ", l.layer, ", neuron: ", l.neuron)
    print(io, ", activation function: ", l.σ)
    print(io, ")")
end

function MLJFlux.build(nn::DenseNet, rng, n_in, n_out)
    init = Flux.glorot_uniform(rng)
    inner_layer = Array{
        typeof(
            Flux.SkipConnection(
                Flux.Chain(Flux.Dense(nn.neuron * 1, nn.neuron, nn.σ, init = init)),
                vcat,
            ),
        ),
        2,
    }(
        undef,
        nn.layer,
        1,
    )

    for i = 1:1:nn.layer
        inner_layer[i, 1] = Flux.SkipConnection(
            Flux.Chain(Flux.Dense(nn.neuron * i, nn.neuron, nn.σ, init = init)),
            vcat,
        )
    end

    return Flux.Chain(
        Flux.Dense(n_in, nn.neuron, bias = false, init = init),
        Flux.Chain(inner_layer...),
        Flux.Dense(nn.neuron * (nn.layer + 1), n_out, bias = false, init = init),
    )
end
