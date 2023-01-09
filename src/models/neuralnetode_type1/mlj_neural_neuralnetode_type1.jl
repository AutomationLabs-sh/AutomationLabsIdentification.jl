# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
NeuralNetODE_type1(; neuron=2, layer=10, σ=Flux.relu, rng=GLOBAL_RNG)
MLJFlux like builder that constructs a neural network ODE, with ten-layers and two neurons
using `layer` nodes in the hidden layer and the specified `neuron`. An activation function `σ` is applied between the
hidden and final layers. Each layers are initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a
`MersenneTwister`.
"""
mutable struct NeuralNetODE_type1 <: MLJFlux.Builder
    neuron::Int
    layer::Int
    σ::Function
end

NeuralNetODE_type1(; neuron = 2, layer = 10, σ = Flux.relu) =
    NeuralNetODE_type1(neuron, layer, σ)

function Base.show(io::IO, l::NeuralNetODE_type1)
    print(
        io,
        "Neural ordinary differential equations type 1 (layer: ",
        l.layer,
        ", neuron: ",
        l.neuron,
    )
    print(io, ", activation function: ", l.σ)
    print(io, ")")
end

function MLJFlux.build(nn::NeuralNetODE_type1, rng, n_in, n_out)

    init = Flux.glorot_uniform(rng)
    inner_layer = Array{typeof(Flux.Dense(nn.neuron, nn.neuron, nn.σ, init = init)),2}(
        undef,
        nn.layer,
        1,
    )
    
    for i = 1:1:nn.layer
        inner_layer[i, 1] = Flux.Dense(nn.neuron, nn.neuron, nn.σ, init = init) 
    end

    function DiffEqArray_to_Array_cpu(x)
        xarr = Array(x)# gpu(x)#Array(x) #to do deal with cpu
        return reshape(xarr, size(xarr)[1:2])
    end
    
    inner_ode = DiffEqFlux.NeuralODE(
        Flux.Chain(inner_layer...),
        (0.0f0, 1.0f0),
        DifferentialEquations.BS3(),
        save_everystep = false,
        reltol = 1e-6,
        abstol = 1e-6,
        save_start = false,
    ) 
    #to do mettre un guard if NaN
    return Flux.Chain(
        Flux.Dense(n_in, nn.neuron, bias = false, init = init), 
        inner_ode,
        DiffEqArray_to_Array_cpu,
        Flux.Dense(nn.neuron, n_out, bias = false, init = init), 
    ) 
end


