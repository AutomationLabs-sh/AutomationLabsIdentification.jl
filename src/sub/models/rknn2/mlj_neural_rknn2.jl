# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
Rknn2(; neuron=2, layer=10, σ=Flux.relu, rng=GLOBAL_RNG, sample_time=1.0)
MLJFlux like builder that constructs a neural network ODE with sample time, ten-layers and two neurons
using `layer` nodes in the hidden layer and the specified `neuron`. An activation function `σ` is applied between the
hidden and final layers. Each layers are initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a `MersenneTwister`.
"""
mutable struct Rknn2 <: MLJFlux.Builder
    neuron::Int
    layer::Int
    σ::Function
    sample_time::Float64
end

Rknn2(; neuron = 2, layer = 10, σ = Flux.relu, sample_time = 1.0) =
    Rknn2(neuron, layer, σ, sample_time)

function Base.show(io::IO, l::Rknn2)
    print(
        io,
        "Runge Kutta neural network 2 (layer: ",
        l.layer,
        ", neuron: ",
        l.neuron,
        ", sample time: ",
        l.sample_time,
    )
    print(io, ", activation function: ", l.σ)
    print(io, ")")
end

function MLJFlux.build(nn::Rknn2, rng, n_in, n_out)

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

    k10 = Flux.Chain(DenseSampleTime(nn.sample_time), fnn_inner)

    k1h = Flux.Chain(Dense_over_z(2), k10)

    k20 = Flux.Chain(
        Flux.Parallel(
            vcat,
            Flux.Parallel(+, DenseIdentityOut(n_out), k10),
            DenseIdentityIn(n_out),
        ),
        fnn_inner,
    )

    k2h = Flux.Chain(Dense_over_z(2), k20)

    y = Flux.Parallel(
        +, 
        rknn2_identity = DenseIdentityOut(n_out), 
        rknn2_k1 = k1h, 
        rknn2_k2 = k2h)

    return y
end

### Declaration de Sample time 
struct Dense_over_z
    z::Any
end

function (a::Dense_over_z)(x::AbstractVecOrMat)
    return x ./ a.z
end

(a::Dense_over_z)(x::AbstractArray) =
    reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, l::Dense_over_z)
    print(io, "Dense_over_z( 1 over ", l.z, "")
    print(io, ")")
end
