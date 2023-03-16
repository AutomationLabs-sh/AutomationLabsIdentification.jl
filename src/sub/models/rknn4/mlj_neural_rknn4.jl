# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
Rknn4(; neuron=2, layer=10, σ=Flux.relu, rng=GLOBAL_RNG, sample_time=1.0)
MLJFlux like builder that constructs a neural network ODE with sample time, ten-layers and two neurons
using `layer` nodes in the hidden layer and the specified `neuron`. An activation function `σ` is applied between the
hidden and final layers. Each layers are initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a `MersenneTwister`.
"""
mutable struct Rknn4 <: MLJFlux.Builder
    neuron::Int
    layer::Int
    σ::Function
    sample_time::Float64
end

Rknn4(; neuron = 2, layer = 10, σ = Flux.relu, sample_time = 1.0) =
    Rknn4(neuron, layer, σ, sample_time)

function Base.show(io::IO, l::Rknn4)
    print(io, "Runge Kutta neural network 4 (layer: ", l.layer, ", neuron: ", l.neuron)
    print(io, ", activation function: ", l.σ)
    print(io, ")")
end

function MLJFlux.build(nn::Rknn4, rng, n_in, n_out)

    init = Flux.glorot_uniform(rng)
    inner_layer = Array{typeof(Flux.Dense(nn.neuron, nn.neuron, nn.σ, init = init)),2}(
        undef,
        nn.layer,
        1,
    )

    for i = 1:1:nn.layer
        inner_layer[i, 1] = Flux.Dense(nn.neuron, nn.neuron, nn.σ, init = init)
    end

    fnn_inner = Flux.Chain(
        Flux.Dense(n_in, nn.neuron, bias = false, init = init),
        Flux.Chain(inner_layer...),
        Flux.Dense(nn.neuron, n_out, bias = false, init = init),
    )

    k1h = Flux.Chain(DenseSampleTime(nn.sample_time), fnn_inner)

    k12 = Flux.Chain(Dense_over_z(2), k1h)

    k2 = Flux.Chain(
        Flux.Parallel(
            vcat,
            Flux.Parallel(+, DenseIdentityOut(n_out), k12),
            DenseIdentityIn(n_out),
        ),
        fnn_inner,
    )

    k2h = Flux.Chain(DenseSampleTime(nn.sample_time), k2)

    k13 = Flux.Chain(Dense_over_z(2), k2h)

    k3 = Flux.Chain(
        Flux.Parallel(
            vcat,
            Flux.Parallel(+, DenseIdentityOut(n_out), k13),
            DenseIdentityIn(n_out),
        ),
        fnn_inner,
    )

    k3h = Flux.Chain(DenseSampleTime(nn.sample_time), k3)

    k4 = Flux.Chain(
        Flux.Parallel(
            vcat,
            Flux.Parallel(+, DenseIdentityOut(n_out), k3h),
            DenseIdentityIn(n_out),
        ),
        fnn_inner,
    )

    k4h = Flux.Chain(DenseSampleTime(nn.sample_time), k4)

    y = Flux.Parallel(
        +,
        rknn4_identity = DenseIdentityOut(n_out),
        rknn4_k1 = Flux.Chain(Dense_over_z(6), k1h),
        rknn4_k2 = Flux.Chain(Dense_over_z(3), k2h),
        rknn4_k3 = Flux.Chain(Dense_over_z(3), k3h),
        rknn4_k4 = Flux.Chain(Dense_over_z(6), k4h),
    )

    return y
end
