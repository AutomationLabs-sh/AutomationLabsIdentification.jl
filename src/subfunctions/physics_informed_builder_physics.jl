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
    architecture::physicsinformed, 
    f::Function, 
    solver::Lbfgs,
    init_t_p::Vector,
    nbr_inputs::Int, 
    nbr_states::Int,
    sample_time::Float64,
    maximum_time::Union{Float64, Dates.TimePeriod};
    kws...,
)
An physics informed with hyperparameters optimization, it returns a tuned model wich can be match with data and trained.
The function is multiple dispatched according to args type.

The following variables are mendatories:
* `architecture`: a physics informed architecture.
* `f`: a function of the continuous dynamical system.
* `solver`: an algorithm selection for neural network training. 
* `init_t_p` : a initial vector of the trainable parameters.
* `nbr_inputs` : the dynamical system inputs number.
* `nbr_outputs` : the dynamical system outputs number.
* `sample_time`: the sample time of the dynamical system discretisation.
* `maximum_time` : a maximum time for training.

The following variables are optinals:
* `kws...`: optinal variables.
"""
function physics_informed_builder(
    architecture::physicsinformed,
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
    lower_params = get(dict_kws, :lower_params, [-Inf])
    upper_params = get(dict_kws, :upper_params, [Inf])

    # f physical definition
    model_f = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Identification.physics_informed(
            f,
            init_t_p,
            nbr_states,
            nbr_inputs,
            sample_time;
            lower_p = lower_params,
            upper_p = upper_params,
        ),
        batch_size = 512,
        optimiser = Optim.LBFGS(),
        epochs = 50,
        loss = Flux.Losses.mae,
    )

    iterated_model = MLJ.IteratedModel(
        model = model_f,
        resampling = nothing,
        control = [
            MLJ.Step(n = 1),
            MLJ.TimeLimit(t = maximum_time),
            MLJ.NumberSinceBest(n = 50),
            MLJ.Patience(n = 5),
        ],
        #iteration_parameter = nothing# :(epochs),
    )

    return iterated_model
end

function physics_informed_builder(
    architecture::physicsinformed,
    f::Function,
    solver::Pso,
    init_t_p::Vector,
    nbr_inputs::Int,
    nbr_states::Int,
    sample_time::Float64,
    maximum_time::Union{Float64,Dates.TimePeriod};
    kws...,
)

    # Get optional parameters
    dict_kws = Dict{Symbol,Any}(kws)
    lower_params = get(dict_kws, :lower_params, [-Inf])
    upper_params = get(dict_kws, :upper_params, [Inf])

    # f physical definition
    model_f = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Identification.physics_informed(
            f,
            init_t_p,
            nbr_states,
            nbr_inputs,
            sample_time;
            lower_p = lower_params,
            upper_p = upper_params,
        ),
        batch_size = 512,
        optimiser = Optim.ParticleSwarm(),
        epochs = 50,
        loss = Flux.Losses.mae,
    )

    iterated_model = MLJ.IteratedModel(
        model = model_f,
        resampling = nothing,
        control = [
            MLJ.Step(n = 1),
            MLJ.TimeLimit(t = maximum_time),
            MLJ.NumberSinceBest(n = 50),
            MLJ.Patience(n = 5),
        ],
        #iteration_parameter = nothing#:(epochs),
    )

    return iterated_model
end

#=
#ooaccel solver cannot have constraint on trainable parameters

function physics_informed_builder(
    architecture::physicsinformed,
    f::Function,
    solver::oaccel,
    init_t_p::Vector,
    nbr_inputs::Int,
    nbr_states::Int,
    sample_time::Float64,
    maximum_time::Union{Float64,Dates.TimePeriod};
    kws...,
)

    # Get optional parameters
    dict_kws = Dict{Symbol,Any}(kws)
    lower_params = get(dict_kws, :lower_params, [-Inf])
    upper_params = get(dict_kws, :upper_params, [Inf])

    # f physical definition
    model_f = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Identification.physics_informed(
            f,
            init_t_p,
            nbr_states,
            nbr_inputs,
            sample_time;
            lower_p = lower_params,
            upper_p = upper_params,
        ),
        batch_size = 512,
        optimiser = Optim.OACCEL(),
        epochs = 50,
        loss = Flux.Losses.mae,
    )

    iterated_model = MLJ.IteratedModel(
        model = model_f,
        resampling = nothing,
        control = [MLJ.Step(n = 10), MLJ.TimeLimit(t = maximum_time)],
        iteration_parameter = :(epochs),
    )

    return iterated_model
end

=#
