
# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
Icnn(; neuron=5, layer=3, σ=Flux.relu, rng=GLOBAL_RNG)
MLJFlux like builder that constructs a input convex neural network, with three-layers and five neurons
using `layer` nodes in the hidden layer and the specified `neuron`. An activation function `σ` is applied between the
hidden and final layers. Each layers are initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a
`MersenneTwister`.
"""
mutable struct Icnn <: MLJFlux.Builder
    neuron::Int
    layer::Int
    σ::Function
end

Icnn(; neuron = 5, layer = 3, σ = Flux.relu) = Icnn(neuron, layer, σ)

function Base.show(io::IO, l::Icnn)
    print(io, "Input convex neural network(layer: ", l.layer, ", neuron: ", l.neuron)
    print(io, ", activation function: ", l.σ)
    print(io, ")")
end

function MLJFlux.build(nn::Icnn, rng, n_in, n_out)
    init = Flux.glorot_uniform(rng)
    inner_layer = Array{typeof(DenseIcnn(nn.neuron, nn.neuron, nn.σ, init = init)),2}(
        undef,
        nn.layer,
        1,
    )

    for i = 1:1:nn.layer
        inner_layer[i, 1] = DenseIcnn(nn.neuron, nn.neuron, nn.σ, init = init) #only relu as activation fct
    end

    return Flux.Chain(
        DenseIcnn(n_in, nn.neuron, identity, bias = false, init = init),
        Flux.Chain(inner_layer...),
        DenseIcnn(nn.neuron, n_out, identity, bias = false, init = init),
    )
end

### Declaration de ICNN 

import Flux: glorot_uniform
import Flux: @functor
import Flux: create_bias
import Flux: Zeros

struct DenseIcnn{F,M<:AbstractMatrix,B}
    weight::M
    bias::B
    σ::F
    function DenseIcnn(W::M, bias = true, σ::F = identity) where {M<:AbstractMatrix,F}
        b = create_bias(W, bias, size(W, 1))
        new{F,M,typeof(b)}(W, b, σ)
    end
end

"""
DenseIcnn(in::Integer, out::Integer, σ = Flux.relu, init = glorot_uniform)
Flux like builder that constructs a linear convex Dense neuron. W and b are initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a `MersenneTwister`.

The following variables are mendatories:
* `in`: The number of inputs of the cell. 
* `out`: The number of outputs of the cell.

The following variable is optional:
* `init`: The initialization method, default is glorot_uniform.

The following variable cannot be modified:
* `σ`: the convexity need relu as activation function
"""
function DenseIcnn(
    in::Integer,
    out::Integer,
    σ = Flux.relu;
    initW = nothing,
    initb = nothing,
    init = glorot_uniform,
    bias = true,
)

    W = if initW !== nothing
        Base.depwarn(
            "keyword initW is deprecated, please use init (which similarly accepts a funtion like randn)",
            :Dense,
        )
        Flux.relu.(initW(out, in))
    else
        Flux.relu.(init(out, in))
    end

    b = if bias === true && initb !== nothing
        Base.depwarn(
            "keyword initb is deprecated, please simply supply the bias vector, bias=initb(out)",
            :Dense,
        )
        initb(out)
    else
        bias
    end

    return DenseIcnn(W, b, σ)
end

@functor DenseIcnn

function (a::DenseIcnn)(x::AbstractVecOrMat)
    W, b, σ = a.weight, a.bias, a.σ
    return σ.(σ.(W) * x .+ b)
end

(a::DenseIcnn)(x::AbstractArray) =
    reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, l::DenseIcnn)
    print(io, "DenseIcnn(", size(l.weight, 2), ", ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == Zeros() && print(io, "; bias=false")
    print(io, ")")
end

### Declaration de ICNN comme FLux
