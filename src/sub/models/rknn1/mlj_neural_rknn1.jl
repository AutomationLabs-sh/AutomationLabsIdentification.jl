# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
Rknn1(; neuron=2, layer=10, σ=Flux.relu, rng=GLOBAL_RNG, sample_time=1.0)
MLJFlux like builder that constructs a neural network ODE with sample time, ten-layers and two neurons
using `layer` nodes in the hidden layer and the specified `neuron`. An activation function `σ` is applied between the
hidden and final layers. Each layers are initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a `MersenneTwister`.
"""
mutable struct Rknn1 <: MLJFlux.Builder
    neuron::Int
    layer::Int
    σ::Function
    sample_time::Float64
end

Rknn1(; neuron = 2, layer = 10, σ = Flux.relu, sample_time = 1.0) =
    Rknn1(neuron, layer, σ, sample_time)

function Base.show(io::IO, l::Rknn1)
    print(
        io,
        "Runge Kutta neural network 1 (layer: ",
        l.layer,
        ", neuron: ",
        l.neuron,
        ", sample time: ",
        l.sample_time,
    )
    print(io, ", activation function: ", l.σ)
    print(io, ")")
end

function MLJFlux.build(nn::Rknn1, rng, n_in, n_out)

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

    deltaT = DenseSampleTime(nn.sample_time)
    inner_euler = Flux.Chain(deltaT, fnn_inner)

    y = Flux.Parallel(
        +, 
        rknn1_identity = DenseIdentityOut(n_out), 
        rknn1_k1 = inner_euler)

    return y
end


### Declaration de Identity outputs
struct DenseIdentityOut
    n_out::Integer
end

function (a::DenseIdentityOut)(x::AbstractVecOrMat)
    return x[1:a.n_out, :]
end

(a::DenseIdentityOut)(x::AbstractArray) =
    reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, l::DenseIdentityOut)
    print(io, "DenseIdentityOut(", l.n_out, "")
    print(io, ")")
end

### Declaration de Identity Input
struct DenseIdentityIn
    n_out::Integer
end

function (a::DenseIdentityIn)(x::AbstractVecOrMat)
    return x[a.n_out+1:end, :]
end

(a::DenseIdentityIn)(x::AbstractArray) =
    reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, l::DenseIdentityIn)
    print(io, "DenseIdentityIn(", l.n_out, "")
    print(io, ")")
end

### Sample time dense matrix
struct DenseSampleTime
    sample_time::Any
end

function (a::DenseSampleTime)(x::AbstractVecOrMat)
    return a.sample_time .* x
end

(a::DenseSampleTime)(x::AbstractArray) =
    reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, l::DenseSampleTime)
    print(io, "DenseSampleTime(", l.sample_time, " Second")
    print(io, ")")
end
