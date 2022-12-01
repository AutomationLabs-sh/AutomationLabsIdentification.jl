# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

##### Multi-target neural network #####
"""
_neural_network_builder(
    nn::ExplorationModels,
    processor::MLJ.CPUProcesses,
    algorithm::Adam,
    max_time::Dates.TimePeriod;
    kws...
)
An artificial neural network builder with hyperparameters optimization, it returns a tuned model wich can be match with data and trained.
The function is multiple dispatched according to args type.

The following variables are mendatories:
* `nn`: a neural network architecture.
* `processor`: a processor selection for training.
* `algorithm`: an algorithm selection for neural network training. 
* `max_time` : a maximum time for training.

The following variables are optinals:
* `kws...`: optional variables.

"""
function _neural_network_builder(
    nn::ExplorationModels,
    processor::MLJ.CPUProcesses,
    algorithm::Adam,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Flux.ADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPUProcesses,
    algorithm::Radam,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Flux.RADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPUProcesses,
    algorithm::Nadam,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Flux.NADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPUProcesses,
    algorithm::Oadam,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Flux.OADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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

### others solvers ###
function _neural_network_builder(
    nn::ExplorationModels,
    processor::MLJ.CPUProcesses,
    algorithm::Lbfgs,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Optim.LBFGS(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPUProcesses,
    algorithm::Pso,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Optim.ParticleSwarm(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPUProcesses,
    algorithm::Oaccel,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Optim.OACCEL(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Distributed.nworkers(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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


### CPU Threads ###

function _neural_network_builder(
    nn::ExplorationModels,
    processor::MLJ.CPUThreads,
    algorithm::Adam,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Flux.ADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPUThreads,
    algorithm::Radam,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Flux.RADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPUThreads,
    algorithm::Nadam,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Flux.NADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPUThreads,
    algorithm::Oadam,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Flux.OADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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

### others solvers ###
function _neural_network_builder(
    nn::ExplorationModels,
    processor::MLJ.CPUThreads,
    algorithm::Lbfgs,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Optim.LBFGS(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPUThreads,
    algorithm::Pso,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Optim.ParticleSwarm(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPUThreads,
    algorithm::Oaccel,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Optim.OACCEL(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = Threads.nthreads(),
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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

### CPU1 ###
function _neural_network_builder(
    nn::ExplorationModels,
    processor::MLJ.CPU1,
    algorithm::Adam,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Flux.ADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPU1,
    algorithm::Radam,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Flux.RADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPU1,
    algorithm::Nadam,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Flux.NADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPU1,
    algorithm::Oadam,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Flux.OADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        acceleration = MLJ.CUDALibs(),
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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

### others solvers ###
function _neural_network_builder(
    nn::ExplorationModels,
    processor::MLJ.CPU1,
    algorithm::Lbfgs,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Optim.LBFGS(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPU1,
    algorithm::Pso,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Optim.ParticleSwarm(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
    nn::ExplorationModels,
    processor::MLJ.CPU1,
    algorithm::Oaccel,
    max_time::Dates.TimePeriod;
    kws...,
)

    # Get parameters from kwagrs for neural networks hyperparameters
    dict_kws = Dict{Symbol,Any}(kws)
    activation_function = ACTIVATION_FUNCTION_LIST[Symbol(
        get(
            dict_kws,
            :neuralnet_activation_function,
            NEURALNET_DEFAULT_PARAMETERS.activation_function,
        ),
    )]
    minimum_epochs = get(
        dict_kws,
        :neuralnet_minimum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.minimum_epochs,
    )
    maximum_epochs = get(
        dict_kws,
        :neuralnet_maximum_epochs,
        NEURALNET_DEFAULT_PARAMETERS.maximum_epochs,
    )
    minimum_layers = get(
        dict_kws,
        :neuralnet_minimum_layers,
        NEURALNET_DEFAULT_PARAMETERS.minimum_layers,
    )
    maximum_layers = get(
        dict_kws,
        :neuralnet_maximum_layers,
        NEURALNET_DEFAULT_PARAMETERS.maximum_layers,
    )
    minimum_neuron = get(
        dict_kws,
        :neuralnet_minimum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.minimum_neuron,
    )
    maximum_neuron = get(
        dict_kws,
        :neuralnet_maximum_neuron,
        NEURALNET_DEFAULT_PARAMETERS.maximum_neuron,
    )
    batch_size =
        get(dict_kws, :neuralnet_batch_size, NEURALNET_DEFAULT_PARAMETERS.batch_size)

    model_exploration = get(dict_kws, :model_exploration, DEFAULT_EXPLORATION_NETWORKS)

    #Design the neural network multitargetnn
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ExplorationOfNetworks(
            network = "Fnn",
            neuron = 10,
            layer = 2,
            σ = activation_function,
        ),
        batch_size = batch_size,
        optimiser = Optim.OACCEL(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    #Hyperparameters range
    r1 = range(model, :(builder.neuron), lower = minimum_neuron, upper = maximum_neuron)
    r2 = range(model, :(builder.layer), lower = minimum_layers, upper = maximum_layers)
    r3 = range(model, :epochs, lower = minimum_epochs, upper = maximum_epochs)
    r4 = range(model, :(builder.network), values = model_exploration)

    #tuned model with hyperparameters optimisation
    tuned_model = MLJ.TunedModel(
        model = model,
        tuning = MLJParticleSwarmOptimization.AdaptiveParticleSwarm(
            rng = StableRNGs.StableRNG(0),
            n_particles = 3,
        ),
        range = [r1, r2, r3, r4],
        measure = mae_multi,
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
