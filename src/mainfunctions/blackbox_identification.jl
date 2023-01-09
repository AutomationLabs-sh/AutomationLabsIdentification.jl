# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
"""
    blackbox_identification
A blackbox model for dynamical system identification problem with recurrent equation of the form : x(k+1) = f(x(k), u(k)).
The model learn with appropriate algorihtm the f. All the Identification.jl package blackbox models are available.

The following variables are mendatories:
* `train_dfin`: a DataFrame data table from measure data from dynamical system inputs.
* `train_dfout`: a DataFrame data table from measure data from dynamical system outputs.
* `algorithm`: an algorithm selection for neural network training, such as backpropagation algorithm or gradient based optimization LBFGS. 
* `architecture`: an architecture selection for model. 
* `maximum_time` : a time limit for the hyperparameters optimisation.
* `kws...` : optional parameters.

The selection of the architecture depends on the targeted model. The following selections are possible:
* `linear`: A linear model: x(k+1) = W(x(k), u(k))
* `fnn`: A feedforward neural network: x(k+1) = fnn(x(k), u(k)).
* `rbf`: A radial basis neural network function: x(k+1) = rbf(x(k), u(k)).
* `icnn`: An input convex neural network: x(k+1) = icnn(x(k), u(k)).
* `resnet`: A residual neural network: x(k+1) = resnet(x(k), u(k)).
* `polynet` : A poly-inception neural network: x(k+1) = polynet(x(k), u(k)).
* `densenet`: A densely connected neural network: x(k+1) = desenet(x(k), u(k)).
* `neuralnet_ode_type1`: A differential equation neural network: x(k+1) = neuralnet_ode_type1(x(k), u(k)).
* `neuralnet_ode_type2`: A differential equation neural network: x(t)' = neuralnet_ode_type1(x(t), u(t)).
* `rnn` : A recurrent neural network: x(k+1) = rnn(x(k), u(k)).
* `lstm`: A recurrent neural network : x(k+1) = lstm(x(k), u(k)).
* `gru`: A recurrent neural network : x(k+1) = gru(x(k), u(k)).
* `eploration_models`: A network selection as hyperparameter, where the algorithm can evaluate all the neural networks.


The training and hyperparamters optimization are finished when time exceeds maximum time. 
The neural network training is non linear and global minimal is not know until all local minima are evaluated, which is difficult.

It is possible to define optional variables, to have custom model parameters and training, as such:
* `neuralnet_activation_function`: Set the activation function, default value is Flux.relu.
* `neuralnet_minimum_epochs`: The hyperparameter minimum epochs when neural training, default value is 50.
* `neuralnet_maximum_epochs`: The hyperparameter maximum epochs when neural training, default value is 500.
* `neuralnet_minimum_layers`: The hyperparameter minimum layer, default value is 1.
* `neuralnet_maximum_layers`: The hyperparamter maximum layer, default value is 6.
* `neuralnet_minimum_neuron`: The hyperparameter minimum neuron, default value is 3.
* `neuralnet_maximum_neuron`: The hyperparameter maximum neuron, default value is 10.
* `neuralnet_batch_size`: The hyperparameter batch size when neural training, default value is 512.
* `computation_verbosity`: Set verbosity REPL informations, default value is 0.

## Example
For a dynamical system identification with fnn and backpropagation algorithm adam:

```
    # Load data 
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:50000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:50000, :]

    max_time = Dates.Minutes(15)
    m_fnn_1 = blackbox_identification(
        dfin, 
        dfout,
        "adam", 
        "fnn", 
        max_time,
    )
```
"""
function blackbox_identification(
    train_dfin::DataFrames.DataFrame,
    train_dfout::DataFrames.DataFrame,
    algorithm::String,
    architecture::String,
    maximum_time::Dates.TimePeriod;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    # Get parameters from kws for neural networks
    processor = get(kws, :computation_processor, PROCESSOR_COMPUTATION_DEFAULT.processor)

    # Get parameters from kws for computation informations
    verbosity = get(kws, :computation_verbosity, 0)

    # Architecture selection, fnn, resnet, densenet, icnn, ...
    tuned_model = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        kws,
    ) #dispatched function

    # Mach the model with the data
    mach_nn = MLJ.machine(tuned_model, train_dfin, train_dfout)

    # Train the neural network
    MLJ.fit!(mach_nn, verbosity = verbosity)

    return mach_nn

end

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
    neuralnetODE_type1
An neural neural network ODE architecture type for dynamical system identification problem [ref].
"""
struct neuralnetODE_type1 <: AbstractNeuralArchitecture end
"""
    neuralnetODE_type2
An neural neural network ODE architecture type for dynamical system identification problem [ref].
"""
struct neuralnetODE_type2 <: AbstractNeuralArchitecture end

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
    neuralnet_ode_type1 = neuralnetODE_type1(),
    neuralnet_ode_type2 = neuralnetODE_type2(),
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
    return Flux.Losses.mae(Matrix(hcat(yhat...)) , Matrix(y))
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
    return Statistics.mean(abs.( (Matrix(hcat(yhat...)) .- Matrix(y)) ./ Matrix(y)))
end

mape_losses = function (x, y)
    return Statistics.mean(abs.( (x .- y) ./ y))
end

const LOSS_FUNCTION_MULTI_LIST = (
    mae = mae_multi,
    mse = mse_multi,
    rmse = rmse_multi,
    mape = mape_multi,
)

const LOSS_FUNCTION_LIST = (
    mae = mae_losses,
    mse = mse_losses,
    rmse = rmse_losses,
    mape = mape_losses,
)
