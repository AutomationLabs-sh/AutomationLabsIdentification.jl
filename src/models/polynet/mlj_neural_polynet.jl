
# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
PolyNet(; neuron=2, layer=10, σ=Flux.relu, rng=GLOBAL_RNG)
MLJFlux like builder that constructs a poly-inception layer network, with ten-layers and two neurons
using `layer` nodes in the hidden layer and the specified `neuron`. An activation function `σ` is applied between the
hidden and final layers. Each layers are initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a `MersenneTwister`.
"""
mutable struct PolyNet <: MLJFlux.Builder
    neuron::Int
    layer::Int
    σ::Function
end

PolyNet(; neuron = 2, layer = 10, σ = Flux.relu) = PolyNet(neuron, layer, σ)

function Base.show(io::IO, l::PolyNet)
    print(io, "Poly-inception neural network(layer: ", l.layer, ", neuron: ", l.neuron)
    print(io, ", activation function: ", l.σ)
    print(io, ")")
end

function MLJFlux.build(nn::PolyNet, rng, n_in, n_out)
    init = Flux.glorot_uniform(rng)

    inner_layer = Array{
        typeof(
            Flux.SkipConnection(
                Flux.Parallel(
                    +,
                    Flux.Dense(nn.neuron, nn.neuron, nn.σ, init = init),
                    Flux.Chain(
                        Flux.Dense(nn.neuron, nn.neuron, nn.σ, init = init),
                        Flux.Dense(nn.neuron, nn.neuron, nn.σ, init = init),
                    ),
                ),
                +,
            ),
        ),
        2,
    }(
        undef,
        nn.layer,
        1,
    )

    for i = 1:1:nn.layer
        inner_dense = Flux.Dense(nn.neuron, nn.neuron, nn.σ, init = init)
        inner_layer[i, 1] = Flux.SkipConnection(
            Flux.Parallel(+, inner_dense, Flux.Chain(inner_dense, inner_dense)),
            +,
        )
    end

    return Flux.Chain(
        Flux.Dense(n_in, nn.neuron, bias = false, init = init),
        Flux.Chain(inner_layer...),
        Flux.Dense(nn.neuron, n_out, bias = false, init = init),
    )
end
