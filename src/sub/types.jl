# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

# Struct for neural architecture
"""
    AbstractNeuralArchitecture
An abstract type that should be subtyped for neural architecture
"""
abstract type AbstractNeuralArchitecture end

"""
    fnn
An feedforward neural network architecture type for dynamical system identification problem [ref].
"""
struct fnn <: AbstractNeuralArchitecture end

"""
    rbf
An radial basis neural network architecture type for dynamical system identification problem [ref].
"""
struct rbf <: AbstractNeuralArchitecture end

"""
    icnn
An input convex neural network architecture type for dynamical system identification problem [ref].
"""
struct icnn <: AbstractNeuralArchitecture end

"""
    resnet
An residual layer network architecture type for dynamical system identification problem [ref].
"""
struct resnet <: AbstractNeuralArchitecture end

"""
    polynet
An poly-inception network architecture type for dynamical system identification problem [ref].
"""
struct polynet <: AbstractNeuralArchitecture end

"""
    densenet
An densely connected network architecture type for dynamical system identification problem [ref].
"""
struct densenet <: AbstractNeuralArchitecture end

"""
    neuralode
An neural neural network ODE architecture type for dynamical system identification problem [ref].
"""
struct neuralode <: AbstractNeuralArchitecture end

"""
    rknn1
A runge-kutta neural neural network 1 architecture type for dynamical system identification problem [ref].
"""
struct rknn1 <: AbstractNeuralArchitecture end

"""
    rknn2
A runge-kutta neural neural network 2 architecture type for dynamical system identification problem [ref].
"""
struct rknn2 <: AbstractNeuralArchitecture end

"""
    rknn4
A runge-kutta neural neural network 4 architecture type for dynamical system identification problem [ref].
"""
struct rknn4 <: AbstractNeuralArchitecture end

"""
    linear
An linear (Wv --> Ax + Bu) architecture type for dynamical system identification problem [ref].
"""
struct linear <: AbstractNeuralArchitecture end

"""
    rnn
A recurrent neural network architecture type for dynamical system identification problem [ref].
"""
struct rnn <: AbstractNeuralArchitecture end

"""
    lstm
A long short-term memory recurrent neural network architecture type for dynamical system identification problem [ref].
"""
struct lstm <: AbstractNeuralArchitecture end

"""
    gru
A gated recurrent unit recurrent neural network architecture type for dynamical system identification problem [ref].
"""
struct gru <: AbstractNeuralArchitecture end

""" 
    ExplorationModels
An multi target neural network architecture type for dynamical system identification problem.
"""
struct ExplorationModels <: AbstractNeuralArchitecture end

# struct for algorithm selection #
"""
    AbstractAlgorithm
An abstract type that should be subtyped for algorithm
"""
abstract type AbstractAlgorithm end

"""
    Adam
An adam algorithm selection for training neural networks.
"""
struct Adam <: AbstractAlgorithm end

"""
    Radam
A radam algorithm selection for training neural networks.
"""
struct Radam <: AbstractAlgorithm end

"""
    Nadam
A nadam algorithm selection for training neural networks.
"""
struct Nadam <: AbstractAlgorithm end

"""
    Oadam
A oadam algorithm selection for training neural networks.
"""
struct Oadam <: AbstractAlgorithm end

"""
    Lbfgs
A lbfgs algorithm selection for training neural networks.
"""
struct Lbfgs <: AbstractAlgorithm end

"""
    Pso
A pso algorithm selection for training neural networks.
"""
struct Pso <: AbstractAlgorithm end

"""
    Oaccel
A oaccel algorithm selection for training neural networks.
"""
struct Oaccel <: AbstractAlgorithm end

""" 
    Lls
A linear least squares for the least squares approximation of linear functions, (Wv --> Ax + Bu).
"""
struct Lls <: AbstractAlgorithm end

# NamedTuple const definition
const ARCHITECTURE_LIST = (
    linear = linear(),
    fnn = fnn(),
    rbf = rbf(),
    icnn = icnn(),
    resnet = resnet(),
    polynet = polynet(),
    densenet = densenet(),
    neuralode = neuralode(),
    rknn1 = rknn1(),
    rknn2 = rknn2(),
    rknn4 = rknn4(),
    rnn = rnn(),
    lstm = lstm(),
    gru = gru(),
    exploration_models = ExplorationModels(),
)

const PROCESSOR_LIST =
    (cpu_threads = MLJ.CPUThreads(), cpu_processes = MLJ.CPUProcesses(), cpu_1 = MLJ.CPU1())

const ALGORITHM_LIST = (
    adam = Adam(),
    radam = Radam(),
    nadam = Nadam(),
    oadam = Oadam(),
    lbfgs = Lbfgs(),
    pso = Pso(),
    oaccel = Oaccel(),
    lls = Lls(),
)

const ACTIVATION_FUNCTION_LIST = (
    relu = Flux.relu,
    sigmoid = Flux.sigmoid,
    swish = Flux.swish,
    tanh = Flux.tanh,
    identity = Flux.identity,
)

const NEURALNET_DEFAULT_PARAMETERS = (
    minimum_epochs = 50,
    maximum_epochs = 500,
    minimum_layers = 1,
    maximum_layers = 6,
    minimum_neuron = 3,
    maximum_neuron = 10,
    batch_size = 512,
    activation_function = "relu",
    loss_function = "mae",
)

const PROCESSOR_COMPUTATION_DEFAULT = (processor = "cpu_1",)

### MAE loss
mae_multi = function (yhat, y)
    return Flux.Losses.mae(Matrix(hcat(yhat...)), Matrix(y))
end

mae_losses = function (x, y)
    return Flux.Losses.mae(x, y)
end

### MSE loss 
mse_multi = function (yhat, y)
    return Flux.Losses.mse(Matrix(hcat(yhat...)), Matrix(y))
end

mse_losses = function (x, y)
    return Flux.Losses.mse(x, y)
end

### RMSE loss 
rmse_multi = function (yhat, y)
    return sqrt(Flux.Losses.mse(Matrix(hcat(yhat...)), Matrix(y)))
end

rmse_losses = function (x, y)
    return sqrt(Flux.Losses.mse(x, y))
end

### MAPE loss
mape_multi = function (yhat, y)
    return Statistics.mean(abs.((Matrix(hcat(yhat...)) .- Matrix(y)) ./ Matrix(y)))
end

mape_losses = function (x, y)
    return Statistics.mean(abs.((x .- y) ./ y))
end

const LOSS_FUNCTION_MULTI_LIST =
    (mae = mae_multi, mse = mse_multi, rmse = rmse_multi, mape = mape_multi)

const LOSS_FUNCTION_LIST =
    (mae = mae_losses, mse = mse_losses, rmse = rmse_losses, mape = mape_losses)



#struct for physics-informed
"""
AbstractPhysicsInformedArchitecture
An abstract type that should be subtyped for physics informed.
"""
abstract type AbstractPhysicsInformedArchitecture end

"""
physicsinformed
An physics informed architecture type for dynamical system identification problem [ref].
"""
struct physicsinformed <: AbstractNeuralArchitecture end

"""
physics_informed_oracle
An physics informed oracle architecture type for dynamical system identification problem [ref].
"""
struct physicsinformedoracle <: AbstractNeuralArchitecture end

#NamedTuple definition
PHYSICS_INFORMED_LIST = (
    physics_informed = physicsinformed(),
    physics_informed_oracle = physicsinformedoracle(),
)
