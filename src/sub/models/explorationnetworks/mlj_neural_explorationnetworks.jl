# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

### exploration neural network
"""
  NetworkExploration(;network = 1, neuron=2, layer=10, σ=Flux.relu, rng=GLOBAL_RNG)
MLJFlux like builder that constructs a network exploration, with ten-layers and two neurons
using `layer` nodes in the hidden layer and the specified `neuron`. The target are a sub neural network such as Fnn or ResNet, the multi target is used for hyperparameters optimization were the network is a hyperparameter. 
An activation function `σ` is applied between the hidden and final layers. Each layers are initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a `MersenneTwister`.
"""
mutable struct ExplorationOfNetworks <: MLJFlux.Builder
    network::String
    neuron::Int
    layer::Int
    σ::Function
end

ExplorationOfNetworks(; network = 1, neuron = 2, layer = 10, σ = Flux.relu) =
    ExplorationOfNetworks(network, neuron, layer, σ)

#NamedTuple definition
const ExplorationOfNetworksList = (
    #Not Linear because it is not iterable
    Fnn = Fnn(),
    Rbf = Rbf(),
    Icnn = Icnn(),
    ResNet = ResNet(),
    PolyNet = PolyNet(),
    DenseNet = DenseNet(),
    NeuralODE = NeuralODE(),
    Rknn1 = Rknn1(),
    Rknn2 = Rknn2(),
    Rknn4 = Rknn4(),
    Rnn = Rnn(),
    Lstm = Lstm(),
    Gru = Gru(),
)

const DEFAULT_EXPLORATION_NETWORKS =
    ["Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet", "NeuralODE", "Rknn1", "Rknn2", "Rknn4"]

function MLJFlux.build(nn_mtn::ExplorationOfNetworks, rng, n_in, n_out)

    sub_nn = ExplorationOfNetworksList[Symbol(nn_mtn.network)]

    # Evaluate if the struct has neuron field
    if isdefined(sub_nn, :neuron) == true
        sub_nn.neuron = nn_mtn.neuron
    end

    # Evaluate if the struct has layer field
    if isdefined(sub_nn, :layer) == true
        sub_nn.layer = nn_mtn.layer
    end

    # Evaluate if the struct has activation function field
    if isdefined(sub_nn, :σ) == true
        sub_nn.σ = nn_mtn.σ
    end

    return MLJFlux.build(sub_nn, rng, n_in, n_out) #call with dispatched function the neural network
end
