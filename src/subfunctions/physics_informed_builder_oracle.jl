# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

### Physic informed with oracle ###
"""
physics_informed_builder(
    architecture::physicsinformedoracle,
    f::Function,
    solver::Lbfgs,
    init_t_p::Vector,
    nbr_inputs::Int,
    nbr_states::Int,
    sample_time::Float64,
    maximum_time::Union{Float64,Dates.TimePeriod};
    kws...,
)

An artificial neural network builder with hyperparameters optimization, it returns a tuned model wich can be match with data and trained.
The function is multiple dispatched according to args type.

The following variables are mendatories:
* `architecture`: a physics informed architecture.
* `f`: a mathematical function of a dynamical system.
* `solver`: an algorithm selection for optimisation. 
* `init_t_p`: an initialization of the trainable parameters.
* `nbr_inputs`: the number of inputs of the dynamical system.
* `nbr_states`: the number of states of the dynamical system.
* `sample_time`: the sample time of discretization.
* `maximum_time` : a maximum time for training.

The following variables are optinals:
* `kws...`: optional variables.

"""
function physics_informed_builder(
    architecture::physicsinformedoracle,
    f::Function,
    solver::Lbfgs,
    init_t_p::Vector,
    nbr_inputs::Int,
    nbr_states::Int,
    sample_time::Float64,
    maximum_time::Union{Float64,Dates.TimePeriod};
    kws...,
)

    # Get optional parameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(dict_kws, :neuralnet_activation_function, "relu"),
    )]

    # Model builder
    model_oracle = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = physics_informed_oracle(
            f,
            nbr_states,
            nbr_inputs,
            sample_time,
            Fnn(neuron = 5, layer = 2, σ = activation_function),
        ),
        batch_size = 512,
        optimiser = Optim.LBFGS(),
        epochs = 50,
        loss = Flux.Losses.mae,
    )

    # Iterator     
    iterated_model = MLJ.IteratedModel(
        model = model_oracle,
        resampling = nothing,
        control = [
            MLJ.Step(n = 1),
            MLJ.TimeLimit(t = maximum_time),
            MLJ.NumberSinceBest(n = 50),
            MLJ.Patience(n = 5),
        ],
        iteration_parameter = nothing,# :(epochs),
    )

    return iterated_model
end

function physics_informed_builder(
    architecture::physicsinformedoracle,
    f::Function,
    solver::Pso,
    init_t_p::Vector,
    nbr_inputs::Int,
    nbr_states::Int,
    sample_time::Float64, #or sample time
    maximum_time::Union{Float64,Dates.TimePeriod};
    kws...,
)

    # Get optional parameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(dict_kws, :neuralnet_activation_function, "relu"),
    )]

    # Model builder
    model_oracle = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = physics_informed_oracle(
            f,
            nbr_states,
            nbr_inputs,
            sample_time,
            Fnn(neuron = 5, layer = 2, σ = activation_function),
        ),
        batch_size = 512,
        optimiser = Optim.ParticleSwarm(),
        epochs = 50,
        loss = Flux.Losses.mae,
    )

    # Iterator     
    iterated_model = MLJ.IteratedModel(
        model = model_oracle,
        resampling = nothing,
        control = [
            MLJ.Step(n = 1),
            MLJ.TimeLimit(t = maximum_time),
            MLJ.NumberSinceBest(n = 50),
            MLJ.Patience(n = 5),
        ],
        iteration_parameter = nothing,#:(epochs),
    )

    return iterated_model
end

function physics_informed_builder(
    architecture::physicsinformedoracle,
    f::Function,
    solver::Oaccel,
    init_t_p::Vector,
    nbr_inputs::Int,
    nbr_states::Int,
    sample_time::Float64,
    maximum_time::Union{Float64,Dates.TimePeriod};
    kws...,
)

    # Get optional parameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(dict_kws, :neuralnet_activation_function, "relu"),
    )]

    # Model builder
    model_oracle = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = physics_informed_oracle(
            f,
            nbr_states,
            nbr_inputs,
            sample_time,
            Fnn(neuron = 5, layer = 2, σ = activation_function),
        ),
        batch_size = 512,
        optimiser = Optim.OACCEL(),
        epochs = 50,
        loss = Flux.Losses.mae,
    )

    # Iterator     
    iterated_model = MLJ.IteratedModel(
        model = model_oracle,
        resampling = nothing,
        control = [
            MLJ.Step(n = 1),
            MLJ.TimeLimit(t = maximum_time),
            MLJ.NumberSinceBest(n = 50),
            MLJ.Patience(n = 5),
        ],
        iteration_parameter = nothing,#:(epochs),
    )

    return iterated_model
end
