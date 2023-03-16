# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
NeuralODE(; neuron=2, layer=10, σ=Flux.relu, rng=GLOBAL_RNG, sample_time=1.0)
MLJFlux like builder that constructs a neural network ODE with sample time, ten-layers and two neurons
using `layer` nodes in the hidden layer and the specified `neuron`. An activation function `σ` is applied between the
hidden and final layers. Each layers are initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a `MersenneTwister`.
"""
mutable struct NeuralODE <: MLJFlux.Builder
    neuron::Int
    layer::Int
    σ::Function
    sample_time::Float64
end

NeuralODE(; neuron = 2, layer = 10, σ = Flux.relu, sample_time = 1.0) =
    NeuralODE(neuron, layer, σ, sample_time)

function Base.show(io::IO, l::NeuralODE)
    print(
        io,
        "Neural ordinary differential equations (layer: ",
        l.layer,
        ", neuron: ",
        l.neuron,
    )
    print(io, ", activation function: ", l.σ)
    print(io, ")")
end

function MLJFlux.build(nn::NeuralODE, rng, n_in, n_out)

    #First declare a Fnn
    init = Flux.glorot_uniform(rng)
    inner_nn = Array{typeof(Flux.Dense(nn.neuron, nn.neuron, nn.σ, init = init)),2}(
        undef,
        nn.layer,
        1,
    )

    for i = 1:1:nn.layer
        inner_nn[i, 1] = Flux.Dense(nn.neuron, nn.neuron, nn.σ) #, init = init)
    end

    inner_layer = Flux.Chain(
        Flux.Dense(n_in, nn.neuron, bias = false, init = init),
        Flux.Chain(inner_nn...),
        Flux.Dense(nn.neuron, n_in, bias = false, init = init),
    )

    #Then discretised the Fnn at the sample time
    tspan = (0.0f0, Float32.(nn.sample_time)) #(0, nn.sample_time)# (0.0f0, Float32.(nn.sample_time))

    inner_ode = DiffEqFlux.NeuralODE(
        inner_layer,
        tspan,
        DifferentialEquations.BS3(),
        save_everystep = false,
        reltol = 1e-6,
        abstol = 1e-6,
        save_start = false,
    )
    #to do mettre un guard if NaN
    y = Flux.Chain(
        neuralode_inner = inner_ode, 
        neuralode_output = DenseDiffEqFlux(n_out),
    )

    return y
end

### Declaration de Identity 
struct DenseDiffEqFlux
    n_out::Integer
end

function (a::DenseDiffEqFlux)(x)#::Union{AbstractVecOrMat, AbstractArray})
    xarr = Array(x)# gpu(x)#Array(x) #to do deal with cpu
    rslt = reshape(xarr, size(xarr)[1:2])
    return rslt[1:a.n_out, :]
end
#=
(a::DenseDiffEqFlux)(x::AbstractArray) =
    reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)
=#
function Base.show(io::IO, l::DenseDiffEqFlux)
    print(io, "DenseDiffEqFlux(", l.n_out, "")
    print(io, ")")
end
