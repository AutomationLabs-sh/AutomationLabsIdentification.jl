# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

##### Rnn #####
"""
    _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUProcesses,
    algorithm::Adam,
    max_time::Dates.TimePeriod;
    kws_...
)
An artificial neural network builder with hyperparameters optimization, it returns a tuned model wich can be match with data and trained.
The function is multiple dispatched according to args type.

The following variables are mendatories:
* `nn`: a neural network architecture.
* `processor`: a processor selection for training.
* `algorithm`: an algorithm selection for neural network training. 
* `max_time` : a maximum time for training.

The following variables are optinals:
* `kws_...`: optional variables.

"""
function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUProcesses,
    algorithm::Adam,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Flux.ADAM(),
        epochs = 1000,
        loss = loss_fct,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Distributed.nworkers(),
        acceleration = MLJ.CPUProcesses(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Distributed.nworkers()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUProcesses,
    algorithm::Radam,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Flux.RADAM(),
        epochs = 1000,
        loss = loss_fct,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Distributed.nworkers(),
        acceleration = MLJ.CPUProcesses(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Distributed.nworkers()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUProcesses,
    algorithm::Nadam,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Flux.NADAM(),
        epochs = 1000,
        loss = loss_fct,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Distributed.nworkers(),
        acceleration = MLJ.CPUProcesses(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Distributed.nworkers()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUProcesses,
    algorithm::Oadam,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Flux.OADAM(),
        epochs = 1000,
        loss = loss_fct,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Distributed.nworkers(),
        acceleration = MLJ.CPUProcesses(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Distributed.nworkers()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

### Other solvers ###
function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUProcesses,
    algorithm::Pso,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Optim.ParticleSwarm(),
        epochs = 1000,
        loss = loss_fct,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Distributed.nworkers(),
        acceleration = MLJ.CPUProcesses(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Distributed.nworkers()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUProcesses,
    algorithm::Oaccel,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Optim.OACCEL(),
        epochs = 1000,
        loss = loss_fct,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Distributed.nworkers(),
        acceleration = MLJ.CPUProcesses(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Distributed.nworkers()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUProcesses,
    algorithm::Lbfgs,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Optim.LBFGS(),
        epochs = 1000,
        loss = loss_fct,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),# Distributed.nworkers(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Distributed.nworkers(),# Distributed.nworkers(),
        acceleration = MLJ.CPUProcesses(),#MLJ.CPUProcesses(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Distributed.nworkers()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

### Threads ### 
function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUThreads,
    algorithm::Adam,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Flux.ADAM(),
        epochs = 1000,
        loss = loss_fct,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Threads.nthreads(),
        acceleration = MLJ.CPUThreads(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Threads.nthreads()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUThreads,
    algorithm::Radam,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Flux.RADAM(),
        epochs = 1000,
        loss = loss_fct,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Threads.nthreads(),
        acceleration = MLJ.CPUThreads(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Threads.nthreads()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUThreads,
    algorithm::Nadam,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Flux.NADAM(),
        epochs = 1000,
        loss = loss_fct,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Threads.nthreads(),
        acceleration = MLJ.CPUThreads(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Threads.nthreads()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUThreads,
    algorithm::Oadam,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Flux.OADAM(),
        epochs = 1000,
        loss = loss_fct,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Threads.nthreads(),
        acceleration = MLJ.CPUThreads(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Threads.nthreads()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

### Other solvers ###
function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUThreads,
    algorithm::Pso,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Optim.ParticleSwarm(),
        epochs = 1000,
        loss = loss_fct,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Threads.nthreads(),
        acceleration = MLJ.CPUThreads(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Threads.nthreads()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUThreads,
    algorithm::Oaccel,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Optim.OACCEL(),
        epochs = 1000,
        loss = loss_fct,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Threads.nthreads(),
        acceleration = MLJ.CPUThreads(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Threads.nthreads()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPUThreads,
    algorithm::Lbfgs,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Optim.LBFGS(),
        epochs = 1000,
        loss = loss_fct,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),# Threads.nthreads(),
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = Threads.nthreads(),# Threads.nthreads(),
        acceleration = MLJ.CPUThreads(),#MLJ.CPUThreads(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = Threads.nthreads()), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

### CPU1 ###
function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPU1,
    algorithm::Adam,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Flux.ADAM(),
        epochs = 1000,
        loss = loss_fct,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = 3,
        acceleration = MLJ.CPU1(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = 3), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPU1,
    algorithm::Radam,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Flux.RADAM(),
        epochs = 1000,
        loss = loss_fct,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = 3,
        acceleration = MLJ.CPU1(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = 3), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPU1,
    algorithm::Nadam,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Flux.NADAM(),
        epochs = 1000,
        loss = loss_fct,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = 3,
        acceleration = MLJ.CPU1(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = 3), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPU1,
    algorithm::Oadam,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Flux.OADAM(),
        epochs = 1000,
        loss = loss_fct,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = 3,
        acceleration = MLJ.CPU1(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = 3), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

### Other solvers ###
function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPU1,
    algorithm::Pso,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Optim.ParticleSwarm(),
        epochs = 1000,
        loss = loss_fct,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = 3,
        acceleration = MLJ.CPU1(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = 3), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPU1,
    algorithm::Oaccel,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Optim.OACCEL(),
        epochs = 1000,
        loss = loss_fct,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = 3,
        acceleration = MLJ.CPU1(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = 3), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end

function _neural_network_builder(
    nn::rnn,
    processor::MLJ.CPU1,
    algorithm::Lbfgs,
    max_time::Dates.TimePeriod;
    kws_...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs =
        get(kws, :neuralnet_minimum_epochs, NEURALNET_DEFAULT_PARAMETERS.minimum_epochs)
    maximum_epochs =
        get(kws, :neuralnet_maximum_epochs, NEURALNET_DEFAULT_PARAMETERS.maximum_epochs)
    minimum_layers =
        get(kws, :neuralnet_minimum_layers, NEURALNET_DEFAULT_PARAMETERS.minimum_layers)
    maximum_layers =
        get(kws, :neuralnet_maximum_layers, NEURALNET_DEFAULT_PARAMETERS.maximum_layers)
    minimum_neuron =
        get(kws, :neuralnet_minimum_neuron, NEURALNET_DEFAULT_PARAMETERS.minimum_neuron)
    maximum_neuron =
        get(kws, :neuralnet_maximum_neuron, NEURALNET_DEFAULT_PARAMETERS.maximum_neuron)
    batch_size = get(kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    if haskey(kws, :fraction_train) == true
        f_t = kws[:fraction_train]
    else
        @error "fraction_train is mandatory with recurrent neural networks"
    end

    loss_fct_multi = LOSS_FUNCTION_MULTI_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    loss_fct = LOSS_FUNCTION_LIST[Symbol(
        get(kws, :neuralnet_loss_function, NEURALNET_DEFAULT_PARAMETERS.loss_function),
    )]

    #Design the neural network Rnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = activation_function),
        batch_size = batch_size,
        optimiser = Optim.LBFGS(),
        epochs = 1000,
        loss = loss_fct,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,# 3,
        ),
        resampling = MLJ.Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = loss_fct_multi,
        n = 3,# 3,
        acceleration = MLJ.CPU1(),#MLJ.CPU1(),
    )

    iterated_nn = MLJ.IteratedModel(
        model = tuned_model,
        resampling = nothing,
        control = [MLJ.Step(n = 3), MLJ.TimeLimit(t = max_time)],
        iteration_parameter = :(n),
    )

    return iterated_nn

end
