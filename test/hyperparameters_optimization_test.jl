# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
module HyparametersOptimizationTest


using Test
using MLJ
using MLJFlux
using Flux
using StableRNGs
using MLJParticleSwarmOptimization
using Distributed
using AutomationLabsIdentification
using CUDA

import AutomationLabsIdentification: ResNet

@testset "Test: neuron & layer & epoch & learning rate optimization" begin

    #data
    in = repeat(1:0.001:10, 1)[:, :]
    in1 = MLJ.table(in)
    out = vec(sin.(in))

    #resnet definition
    MLJ.default_resource(CPUThreads())
    model_resnet = MLJFlux.NeuralNetworkRegressor(
        builder = ResNet(neuron = 5, layer = 2, σ = Flux.relu),
        batch_size = 2048,
        optimiser = Flux.RADAM(),
        epochs = 100,
        loss = Flux.Losses.mae,
        acceleration = CUDALibs(),
    )

    r1 = range(model_resnet, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_resnet, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_resnet, :epochs, lower = 100, upper = 500)
    r4 = range(model_resnet, :(optimiser.eta), lower = 0.0001, upper = 0.001)

    #r5 = range(model_resnet, :(optimiser.beta), lower=0.9, upper =0.99)
    #r2 = range(model_resnet, :optimiser, values = [[Flux.RADAM()], [Flux.ADAM()], [Flux.NADAM()], [Flux.OADAM()]]);

    tuned_model = MLJ.TunedModel(
        model = model_resnet,
        tuning = AdaptiveParticleSwarm(n_particles = 6, rng = StableRNG(0)),
        #resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2, r3, r4],
        measure = mae,
        n = 20, #number of models to evaluate
        acceleration = MLJ.CPUThreads(),
    )

    mach = MLJ.machine(tuned_model, in1, out)
    MLJ.fit!(mach, verbosity = 5)

    best_model = MLJ.fitted_params(mach)

    @test size.(Flux.params(best_model.best_fitted_params.chain))[1][1] < 16
    @test size.(Flux.params(best_model.best_fitted_params.chain))[1][2] == 1
    @test size.(Flux.params(best_model.best_fitted_params.chain))[2][1] ==
          size.(Flux.params(best_model.best_fitted_params.chain))[1][1]

    param_best_model = fitted_params(mach).best_model

    @test param_best_model.builder.neuron < 16
    @test param_best_model.builder.neuron > 4

    @test param_best_model.builder.layer < 6
    @test param_best_model.builder.layer >= 1

    @test param_best_model.epochs <= 500
    @test param_best_model.epochs >= 100

    @test param_best_model.optimiser.eta <= 0.001
    @test param_best_model.optimiser.eta >= 0.0001

end

end
