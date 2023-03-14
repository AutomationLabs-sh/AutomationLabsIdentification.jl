# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
#module QTP_BackpropIdentification


using Distributed

using Test
using MLJ
using MLJFlux
using Flux
using StableRNGs
using MLJParticleSwarmOptimization
using DataFrames
using CSV
using Dates
using CUDA

using AutomationLabsIdentification

import AutomationLabsIdentification: data_formatting_identification

import AutomationLabsIdentification: Fnn
import AutomationLabsIdentification: Icnn
import AutomationLabsIdentification: DenseIcnn
import AutomationLabsIdentification: Rbf
import AutomationLabsIdentification: ResNet
import AutomationLabsIdentification: DenseNet
import AutomationLabsIdentification: PolyNet
import AutomationLabsIdentification: NeuralODE
import AutomationLabsIdentification: ExplorationOfNetworks
import AutomationLabsIdentification: Rnn
import AutomationLabsIdentification: Lstm
import AutomationLabsIdentification: Gru
import AutomationLabsIdentification: Rknn1
import AutomationLabsIdentification: Rknn2
import AutomationLabsIdentification: Rknn4



@testset "QTP identification Fnn" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:100000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:100000, :]

    n_delay = 1
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]


    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    #fnn definition
    model_fnn = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Fnn(neuron = 10, layer = 2, σ = NNlib.relu),
        batch_size = 2048,
        optimiser = Flux.RADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_fnn, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_fnn, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_fnn, :epochs, lower = 100, upper = 500)

    tuned_model_fnn = MLJ.TunedModel(
        model = model_fnn,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        #resampling=CV(nfolds=6, rng=StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 5,
        # acceleration = MLJ.CPUProcesses(),
    )

    iterated_model_fnn = IteratedModel(
        model = tuned_model_fnn,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
        iteration_parameter = :(n),
    )

    mach_fnn = MLJ.machine(iterated_model_fnn, in_data, out_data)

    MLJ.fit!(mach_fnn)

    #save the train model
    MLJ.save("./models_saved/fnn_train_result.jls", mach_fnn)

    @test fitted_params(fitted_params(mach_fnn).machine).best_model != 0
    @test report(fitted_params(mach_fnn).machine).best_history_entry != 0
    @test report(fitted_params(mach_fnn).machine).history != 0

    fnn_param_best_model = fitted_params(fitted_params(mach_fnn).machine).best_model

    @test fnn_param_best_model.builder.neuron < 16
    @test fnn_param_best_model.builder.neuron > 4

    @test fnn_param_best_model.builder.layer < 6
    @test fnn_param_best_model.builder.layer >= 1

    @test fnn_param_best_model.epochs <= 500
    @test fnn_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    fnn_chain_best_model_chain =
        fitted_params(fitted_params(mach_fnn).machine).best_fitted_params.chain

    mae_Train_fnn = Flux.mae(
        fnn_chain_best_model_chain(Matrix(Train_in_data)'),
        Matrix(Train_out_data)',
    )
    mae_Test_fnn =
        Flux.mae(fnn_chain_best_model_chain(Matrix(Test_in_data)'), Matrix(Test_out_data)')

    println("mae_Train_fnn $mae_Train_fnn")
    println("mae_Test_fnn $mae_Test_fnn")

    @test mae_Train_fnn <= 1
    @test mae_Test_fnn <= 1

end

@testset "QTP identification ResNet" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:100000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:100000, :]

    n_delay = 1
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]


    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    # resnet definition
    model_resnet = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ResNet(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 2048,
        optimiser = Flux.RADAM(),
        epochs = 100,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_resnet, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_resnet, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_resnet, :epochs, lower = 100, upper = 500)

    tuned_model_resnet = MLJ.TunedModel(
        model = model_resnet,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 5,
        #acceleration = MLJ.CPUProcesses(),
    )

    iterated_model_resenet = IteratedModel(
        model = tuned_model_resnet,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
    )

    mach_resnet = MLJ.machine(iterated_model_resenet, in_data, out_data)

    MLJ.fit!(mach_resnet)

    #save the model and optimisation results
    MLJ.save("./models_saved/resnet_train_result.jls", mach_resnet)

    #fitted_params(mach_resnet).chain
    @test fitted_params(fitted_params(mach_resnet).machine).best_model != 0
    @test report(fitted_params(mach_resnet).machine).best_history_entry != 0
    @test report(fitted_params(mach_resnet).machine).history != 0

    resnet_param_best_model = fitted_params(fitted_params(mach_resnet).machine).best_model

    @test resnet_param_best_model.builder.neuron < 16
    @test resnet_param_best_model.builder.neuron > 4

    @test resnet_param_best_model.builder.layer < 6
    @test resnet_param_best_model.builder.layer >= 1

    @test resnet_param_best_model.epochs <= 500
    @test resnet_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    resnet_chain_best_model_chain =
        fitted_params(fitted_params(mach_resnet).machine).best_fitted_params.chain

    mae_Train_resnet = Flux.mae(
        resnet_chain_best_model_chain(Matrix(Train_in_data)'),
        Matrix(Train_out_data)',
    )
    mae_Test_resnet = Flux.mae(
        resnet_chain_best_model_chain(Matrix(Test_in_data)'),
        Matrix(Test_out_data)',
    )

    println("mae_Train_resnet $mae_Train_resnet")
    println("mae_Test_resnet $mae_Test_resnet")

    @test mae_Train_resnet <= 1
    @test mae_Test_resnet <= 1

end

@testset "QTP identification polynet" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:100000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:100000, :]

    n_delay = 1
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]


    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    # polynet definition
    model_polynet = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = PolyNet(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 2048,
        optimiser = Flux.RADAM(),
        epochs = 100,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_polynet, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_polynet, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_polynet, :epochs, lower = 100, upper = 500)

    tuned_model_polynet = MLJ.TunedModel(
        model = model_polynet,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 5,
        #acceleration = MLJ.CPUProcesses(),
    )

    iterated_model_resenet = IteratedModel(
        model = tuned_model_polynet,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
    )

    mach_polynet = MLJ.machine(iterated_model_resenet, in_data, out_data)

    MLJ.fit!(mach_polynet)

    #save the model and optimisation results
    MLJ.save("./models_saved/polynet_train_result.jls", mach_polynet)

    #fitted_params(mach_polynet).chain
    @test fitted_params(fitted_params(mach_polynet).machine).best_model != 0
    @test report(fitted_params(mach_polynet).machine).best_history_entry != 0
    @test report(fitted_params(mach_polynet).machine).history != 0

    polynet_param_best_model = fitted_params(fitted_params(mach_polynet).machine).best_model

    @test polynet_param_best_model.builder.neuron < 16
    @test polynet_param_best_model.builder.neuron > 4

    @test polynet_param_best_model.builder.layer < 6
    @test polynet_param_best_model.builder.layer >= 1

    @test polynet_param_best_model.epochs <= 500
    @test polynet_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    polynet_chain_best_model_chain =
        fitted_params(fitted_params(mach_polynet).machine).best_fitted_params.chain

    mae_Train_polynet = Flux.mae(
        polynet_chain_best_model_chain(Matrix(Train_in_data)'),
        Matrix(Train_out_data)',
    )
    mae_Test_polynet = Flux.mae(
        polynet_chain_best_model_chain(Matrix(Test_in_data)'),
        Matrix(Test_out_data)',
    )

    println("mae_Train_polynet $mae_Train_polynet")
    println("mae_Test_polynet $mae_Test_polynet")

    @test mae_Train_polynet <= 1
    @test mae_Test_polynet <= 1

end

@testset "QTP identification DenseNet" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:100000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:100000, :]

    n_delay = 1
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]


    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    #densenet definition
    model_densenet = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = DenseNet(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 512,
        optimiser = Flux.RADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_densenet, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_densenet, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_densenet, :epochs, lower = 100, upper = 500)

    tuned_model_densenet = MLJ.TunedModel(
        model = model_densenet,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 5,
        #acceleration = MLJ.CPUProcesses(),
    )

    iterated_model_densenet = IteratedModel(
        model = tuned_model_densenet,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
    )

    mach_densenet = MLJ.machine(iterated_model_densenet, in_data, out_data)

    MLJ.fit!(mach_densenet)

    MLJ.save("./models_saved/densenet_train_result.jls", mach_densenet)

    @test fitted_params(fitted_params(mach_densenet).machine).best_model != 0
    @test report(fitted_params(mach_densenet).machine).best_history_entry != 0
    @test report(fitted_params(mach_densenet).machine).history != 0

    densenet_param_best_model =
        fitted_params(fitted_params(mach_densenet).machine).best_model

    @test densenet_param_best_model.builder.neuron < 16
    @test densenet_param_best_model.builder.neuron > 4

    @test densenet_param_best_model.builder.layer < 6
    @test densenet_param_best_model.builder.layer >= 1

    @test densenet_param_best_model.epochs <= 500
    @test densenet_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    densenet_chain_best_model_chain =
        fitted_params(fitted_params(mach_densenet).machine).best_fitted_params.chain

    mae_Train_densenet = Flux.mae(
        densenet_chain_best_model_chain(Matrix(Train_in_data)'),
        Matrix(Train_out_data)',
    )
    mae_Test_densenet = Flux.mae(
        densenet_chain_best_model_chain(Matrix(Test_in_data)'),
        Matrix(Test_out_data)',
    )

    println("mae_Train_densenet $mae_Train_densenet")
    println("mae_Test_densenet $mae_Test_densenet")

    @test mae_Train_densenet <= 1
    @test mae_Test_densenet <= 1

end

@testset "QTP identification Icnn" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:100000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:100000, :]

    n_delay = 1
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]


    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    #Icnn definition
    model_icnn = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Icnn(neuron = 5, layer = 1, σ = NNlib.relu),
        batch_size = 512,
        optimiser = Flux.RADAM(),
        epochs = 5000,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_icnn, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_icnn, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_icnn, :epochs, lower = 100, upper = 500)

    tuned_model_icnn = MLJ.TunedModel(
        model = model_icnn,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 5,
        #acceleration = MLJ.CPUProcesses(),
    )


    iterated_model_icnn = IteratedModel(
        model = tuned_model_icnn,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
    )

    mach_icnn = MLJ.machine(iterated_model_icnn, in_data, out_data)

    MLJ.fit!(mach_icnn)

    MLJ.save("./models_saved/icnn_train_result.jls", mach_icnn)

    @test fitted_params(fitted_params(mach_icnn).machine).best_model != 0
    @test report(fitted_params(mach_icnn).machine).best_history_entry != 0
    @test report(fitted_params(mach_icnn).machine).history != 0

    icnn_param_best_model = fitted_params(fitted_params(mach_icnn).machine).best_model

    @test icnn_param_best_model.builder.neuron < 16
    @test icnn_param_best_model.builder.neuron > 4

    @test icnn_param_best_model.builder.layer < 6
    @test icnn_param_best_model.builder.layer >= 1

    @test icnn_param_best_model.epochs <= 500
    @test icnn_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    icnn_chain_best_model_chain =
        fitted_params(fitted_params(mach_icnn).machine).best_fitted_params.chain

    mae_Train_icnn = Flux.mae(
        icnn_chain_best_model_chain(Matrix(Train_in_data)'),
        Matrix(Train_out_data)',
    )
    mae_Test_icnn =
        Flux.mae(icnn_chain_best_model_chain(Matrix(Test_in_data)'), Matrix(Test_out_data)')

    println("mae_Train_icnn $mae_Train_icnn")
    println("mae_Test_icnn $mae_Test_icnn")

    @test mae_Train_icnn <= 1
    @test mae_Test_icnn <= 1

end

@testset "QTP identification NeuralODE" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:50000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:50000, :]

    n_delay = 1
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    sample_time = 5.0

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    # Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    # NeuralNetODE_type2 definition
    model_neural_netODE = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = NeuralODE(
            neuron = 5,
            layer = 2,
            σ = NNlib.relu,
            sample_time = sample_time,
        ),
        batch_size = 4096,
        optimiser = Flux.RADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_neural_netODE, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_neural_netODE, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_neural_netODE, :epochs, lower = 100, upper = 500)

    tuned_model_neural_netODE = MLJ.TunedModel(
        model = model_neural_netODE,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        #resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 5,
        #acceleration = MLJ.CPUProcesses(),
    )

    iterated_model_neural_netODE = IteratedModel(
        model = tuned_model_neural_netODE,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
    )

    mach_neural_netODE = MLJ.machine(iterated_model_neural_netODE, in_data, out_data)

    MLJ.fit!(mach_neural_netODE)

    #save the model and optimisation results
    MLJ.save("./models_saved/NeuralODE_train_result.jls", mach_neural_netODE)

    @test fitted_params(fitted_params(mach_neural_netODE).machine).best_model != 0
    @test report(fitted_params(mach_neural_netODE).machine).best_history_entry != 0
    @test report(fitted_params(mach_neural_netODE).machine).history != 0

    NeuralNetODE_type2_param_best_model =
        fitted_params(fitted_params(mach_neural_netODE).machine).best_model

    @test NeuralNetODE_type2_param_best_model.builder.neuron < 16
    @test NeuralNetODE_type2_param_best_model.builder.neuron > 4

    @test NeuralNetODE_type2_param_best_model.builder.layer < 6
    @test NeuralNetODE_type2_param_best_model.builder.layer >= 1

    @test NeuralNetODE_type2_param_best_model.epochs <= 500
    @test NeuralNetODE_type2_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    neuralnet_chain_best_model_chain =
        fitted_params(fitted_params(mach_neural_netODE).machine).best_fitted_params.chain

    mae_Train_neuralnet = Flux.mae(
        neuralnet_chain_best_model_chain(Matrix(Train_in_data)'),
        Matrix(Train_out_data)',
    )
    mae_Test_neuralnet = Flux.mae(
        neuralnet_chain_best_model_chain(Matrix(Test_in_data)'),
        Matrix(Test_out_data)',
    )

    println("mae_Train_neuralnet $mae_Train_neuralnet")
    println("mae_Test_neuralnet $mae_Test_neuralnet")

    @test mae_Train_neuralnet <= 1
    @test mae_Test_neuralnet <= 1

end

@testset "QTP identification Rbf" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:100000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:100000, :]

    n_delay = 1
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]


    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    # Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    # Rbf definition
    model_rbf = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rbf(neuron = 5),
        batch_size = 2048,
        optimiser = Flux.RADAM(),
        epochs = 100,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_rbf, :(builder.neuron), lower = 5, upper = 25)
    r2 = range(model_rbf, :epochs, lower = 100, upper = 500)

    tuned_model_rbf = MLJ.TunedModel(
        model = model_rbf,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2],
        measure = my_loss,
        n = 5,
        #acceleration = MLJ.CPUProcesses(),
    )

    iterated_model_rbf = IteratedModel(
        model = tuned_model_rbf,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
    )

    mach_rbf = MLJ.machine(iterated_model_rbf, in_data, out_data)

    MLJ.fit!(mach_rbf)

    #save the model and optimisation results
    MLJ.save("./models_saved/rbf_train_result.jls", mach_rbf)

    @test fitted_params(fitted_params(mach_rbf).machine).best_model != 0
    @test report(fitted_params(mach_rbf).machine).best_history_entry != 0
    @test report(fitted_params(mach_rbf).machine).history != 0

    rbf_param_best_model = fitted_params(fitted_params(mach_rbf).machine).best_model

    @test rbf_param_best_model.builder.neuron < 26
    @test rbf_param_best_model.builder.neuron > 4

    @test rbf_param_best_model.epochs <= 500
    @test rbf_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    rbf_chain_best_model_chain =
        fitted_params(fitted_params(mach_rbf).machine).best_fitted_params.chain

    mae_Train_rbf = Flux.mae(
        rbf_chain_best_model_chain(Matrix(Train_in_data)'),
        Matrix(Train_out_data)',
    )
    mae_Test_rbf =
        Flux.mae(rbf_chain_best_model_chain(Matrix(Test_in_data)'), Matrix(Test_out_data)')

    println("mae_Train_rbf $mae_Train_rbf")
    println("mae_Test_rbf $mae_Test_rbf")

    @test mae_Train_rbf <= 1
    @test mae_Test_rbf <= 1

end

@testset "QTP identification rnn" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))

    n_sequence = 64
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_sequence = n_sequence,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    f_t = 1 - (1 / (size(in_data, 1) / n_sequence))
    # The test data is set to be equal to only one batch

    # Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    #rnn definition
    model_rnn = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rnn(neuron = 10, layer = 2, σ = NNlib.relu),
        batch_size = n_sequence,
        optimiser = Flux.RADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_rnn, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_rnn, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_rnn, :epochs, lower = 100, upper = 500)

    tuned_model_rnn = MLJ.TunedModel(
        model = model_rnn,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 5,
        # acceleration = MLJ.CPUProcesses(),
    )

    iterated_model_rnn = IteratedModel(
        model = tuned_model_rnn,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
        iteration_parameter = :(n),
    )

    mach_rnn = MLJ.machine(iterated_model_rnn, in_data, out_data)

    MLJ.fit!(mach_rnn)

    #save the train model
    MLJ.save("./models_saved/rnn_train_result.jls", mach_rnn)

    @test fitted_params(fitted_params(mach_rnn).machine).best_model != 0
    @test report(fitted_params(mach_rnn).machine).best_history_entry != 0
    @test report(fitted_params(mach_rnn).machine).history != 0

    rnn_param_best_model = fitted_params(fitted_params(mach_rnn).machine).best_model

    @test rnn_param_best_model.builder.neuron < 16
    @test rnn_param_best_model.builder.neuron > 4

    @test rnn_param_best_model.builder.layer < 6
    @test rnn_param_best_model.builder.layer >= 1

    @test rnn_param_best_model.epochs <= 500
    @test rnn_param_best_model.epochs >= 100

    # MAE with train data
    mae_Train_rnn = report(mach_rnn)[1].best_history_entry.measurement[1]

    println("mae_Train_rnn $mae_Train_rnn")

    @test mae_Train_rnn <= 1

    # Chain rnn
    rnn_chain_best_model_chain =
        fitted_params(fitted_params(mach_rnn).machine).best_fitted_params.chain

    rnn_chain_best_model_chain(rand(Float32, 6))

end

@testset "QTP identification lstm" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))

    n_sequence = 64
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_sequence = n_sequence,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    f_t = 1 - (1 / (size(in_data, 1) / n_sequence))

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    #lstm definition
    model_lstm = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Lstm(neuron = 10, layer = 2, σ = NNlib.relu),
        batch_size = n_sequence,
        optimiser = Flux.RADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_lstm, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_lstm, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_lstm, :epochs, lower = 100, upper = 500)

    tuned_model_lstm = MLJ.TunedModel(
        model = model_lstm,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 5,
        # acceleration = MLJ.CPUProcesses(),
    )

    iterated_model_lstm = IteratedModel(
        model = tuned_model_lstm,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
        iteration_parameter = :(n),
    )

    mach_lstm = MLJ.machine(iterated_model_lstm, in_data, out_data)

    MLJ.fit!(mach_lstm)

    #save the train model
    MLJ.save("./models_saved/lstm_train_result.jls", mach_lstm)

    @test fitted_params(fitted_params(mach_lstm).machine).best_model != 0
    @test report(fitted_params(mach_lstm).machine).best_history_entry != 0
    @test report(fitted_params(mach_lstm).machine).history != 0

    lstm_param_best_model = fitted_params(fitted_params(mach_lstm).machine).best_model

    @test lstm_param_best_model.builder.neuron < 16
    @test lstm_param_best_model.builder.neuron > 4

    @test lstm_param_best_model.builder.layer < 6
    @test lstm_param_best_model.builder.layer >= 1

    @test lstm_param_best_model.epochs <= 500
    @test lstm_param_best_model.epochs >= 100

    # MAE with train data
    mae_Train_lstm = report(mach_lstm)[1].best_history_entry.measurement[1]

    println("mae_Train_lstm $mae_Train_lstm")

    @test mae_Train_lstm <= 1

end

@testset "QTP identification gru" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))

    n_sequence = 64
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_sequence = n_sequence,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    f_t = 1 - (1 / (size(in_data, 1) / n_sequence))

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    #gru definition
    model_gru = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Gru(neuron = 10, layer = 2, σ = NNlib.relu),
        batch_size = n_sequence,
        optimiser = Flux.RADAM(),
        epochs = 1000,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_gru, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_gru, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_gru, :epochs, lower = 100, upper = 500)

    tuned_model_gru = MLJ.TunedModel(
        model = model_gru,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = Holdout(; fraction_train = f_t, shuffle = nothing, rng = nothing),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 5,
        # acceleration = MLJ.CPUProcesses(),
    )

    iterated_model_gru = IteratedModel(
        model = tuned_model_gru,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
        iteration_parameter = :(n),
    )

    mach_gru = MLJ.machine(iterated_model_gru, in_data, out_data)

    MLJ.fit!(mach_gru)

    #save the train model
    MLJ.save("./models_saved/gru_train_result.jls", mach_gru)

    @test fitted_params(fitted_params(mach_gru).machine).best_model != 0
    @test report(fitted_params(mach_gru).machine).best_history_entry != 0
    @test report(fitted_params(mach_gru).machine).history != 0

    gru_param_best_model = fitted_params(fitted_params(mach_gru).machine).best_model

    @test gru_param_best_model.builder.neuron < 16
    @test gru_param_best_model.builder.neuron > 4

    @test gru_param_best_model.builder.layer < 6
    @test gru_param_best_model.builder.layer >= 1

    @test gru_param_best_model.epochs <= 500
    @test gru_param_best_model.epochs >= 100

    # MAE with train data
    mae_Train_gru = report(mach_gru)[1].best_history_entry.measurement[1]

    println("mae_Train_gru $mae_Train_gru")

    @test mae_Train_gru <= 1

end


@testset "QTP identification Rknn1" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:10000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:10000, :]

    n_delay = 1
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]


    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    sample_time = 5.0
    nbr_states = 4
    nbr_inputs = 2

    #fnn definition
    model_rknn1 = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rknn1(neuron = 10, layer = 2, σ = NNlib.relu, sample_time = sample_time),
        batch_size = 2048,
        optimiser = Flux.RADAM(),
        epochs = 10000,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_rknn1, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_rknn1, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_rknn1, :epochs, lower = 100, upper = 500)

    tuned_model_rknn1 = MLJ.TunedModel(
        model = model_rknn1,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        #resampling=CV(nfolds=6, rng=StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 5,
        # acceleration = MLJ.CPUProcesses(),
    )

    iterated_model_rknn1 = IteratedModel(
        model = tuned_model_rknn1,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
        iteration_parameter = :(n),
    )

    mach_rknn1 = MLJ.machine(iterated_model_rknn1, in_data, out_data)

    MLJ.fit!(mach_rknn1)

    #save the train model
    MLJ.save("./models_saved/rknn1_train_result.jls", mach_rknn1)

    @test fitted_params(fitted_params(mach_rknn1).machine).best_model != 0
    @test report(fitted_params(mach_rknn1).machine).best_history_entry != 0
    @test report(fitted_params(mach_rknn1).machine).history != 0

    rknn_param_best_model = fitted_params(fitted_params(mach_rknn1).machine).best_model

    @test rknn_param_best_model.builder.neuron < 16
    @test rknn_param_best_model.builder.neuron > 4

    @test rknn_param_best_model.builder.layer < 6
    @test rknn_param_best_model.builder.layer >= 1

    @test rknn_param_best_model.epochs <= 500
    @test rknn_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    rknn_chain_best_model_chain =
        fitted_params(fitted_params(mach_rknn1).machine).best_fitted_params.chain

    mae_Train_rknn = Flux.mae(
        rknn_chain_best_model_chain(Matrix(Train_in_data)'),
        Matrix(Train_out_data)',
    )
    mae_Test_rknn =
        Flux.mae(rknn_chain_best_model_chain(Matrix(Test_in_data)'), Matrix(Test_out_data)')

    println("mae_Train_rknn 1 $mae_Train_rknn")
    println("mae_Test_rknn 1 $mae_Test_rknn")

    @test mae_Train_rknn <= 1
    @test mae_Test_rknn <= 1

end


@testset "QTP identification Rknn2" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:10000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:10000, :]

    n_delay = 1
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]


    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    sample_time = 5.0
    nbr_states = 4
    nbr_inputs = 2

    #fnn definition
    model_rknn2 = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rknn2(neuron = 10, layer = 2, σ = NNlib.relu, sample_time = sample_time),
        batch_size = 2048,
        optimiser = Flux.RADAM(),
        epochs = 10000,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_rknn2, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_rknn2, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_rknn2, :epochs, lower = 100, upper = 500)

    tuned_model_rknn2 = MLJ.TunedModel(
        model = model_rknn2,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        #resampling=CV(nfolds=6, rng=StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 5,
        # acceleration = MLJ.CPUProcesses(),
    )

    iterated_model_rknn2 = IteratedModel(
        model = tuned_model_rknn2,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
        iteration_parameter = :(n),
    )

    mach_rknn2 = MLJ.machine(iterated_model_rknn2, in_data, out_data)

    MLJ.fit!(mach_rknn2)

    #save the train model
    MLJ.save("./models_saved/rknn2_train_result.jls", mach_rknn2)

    @test fitted_params(fitted_params(mach_rknn2).machine).best_model != 0
    @test report(fitted_params(mach_rknn2).machine).best_history_entry != 0
    @test report(fitted_params(mach_rknn2).machine).history != 0

    rknn_param_best_model = fitted_params(fitted_params(mach_rknn2).machine).best_model

    @test rknn_param_best_model.builder.neuron < 16
    @test rknn_param_best_model.builder.neuron > 4

    @test rknn_param_best_model.builder.layer < 6
    @test rknn_param_best_model.builder.layer >= 1

    @test rknn_param_best_model.epochs <= 500
    @test rknn_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    rknn_chain_best_model_chain =
        fitted_params(fitted_params(mach_rknn2).machine).best_fitted_params.chain

    mae_Train_rknn = Flux.mae(
        rknn_chain_best_model_chain(Matrix(Train_in_data)'),
        Matrix(Train_out_data)',
    )
    mae_Test_rknn =
        Flux.mae(rknn_chain_best_model_chain(Matrix(Test_in_data)'), Matrix(Test_out_data)')

    println("mae_Train_rknn 2 $mae_Train_rknn")
    println("mae_Test_rknn 2 $mae_Test_rknn")

    @test mae_Train_rknn <= 1
    @test mae_Test_rknn <= 1

end


@testset "QTP identification Rknn4" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:10000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:10000, :]

    n_delay = 1
    normalisation = false

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]


    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    sample_time = 5.0
    nbr_states = 4
    nbr_inputs = 2

    #rknn4 definition
    model_rknn4 = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rknn4(neuron = 10, layer = 2, σ = NNlib.relu, sample_time = sample_time),
        batch_size = 2048,
        optimiser = Flux.RADAM(),
        epochs = 10000,
        loss = Flux.Losses.mae,
        #acceleration = CUDALibs(),
    )

    r1 = range(model_rknn4, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_rknn4, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_rknn4, :epochs, lower = 100, upper = 500)

    tuned_model_rknn4 = MLJ.TunedModel(
        model = model_rknn4,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        #resampling=CV(nfolds=6, rng=StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 5,
        # acceleration = MLJ.CPUProcesses(),
    )

    iterated_model_rknn4 = IteratedModel(
        model = tuned_model_rknn4,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(120))],
        iteration_parameter = :(n),
    )

    mach_rknn4 = MLJ.machine(iterated_model_rknn4, in_data, out_data)

    MLJ.fit!(mach_rknn4)


    #save the train model
    MLJ.save("./models_saved/rknn4_train_result.jls", mach_rknn4)

    @test fitted_params(fitted_params(mach_rknn4).machine).best_model != 0
    @test report(fitted_params(mach_rknn4).machine).best_history_entry != 0
    @test report(fitted_params(mach_rknn4).machine).history != 0

    rknn4_param_best_model = fitted_params(fitted_params(mach_rknn4).machine).best_model

    @test rknn4_param_best_model.builder.neuron < 16
    @test rknn4_param_best_model.builder.neuron > 4

    @test rknn4_param_best_model.builder.layer < 6
    @test rknn4_param_best_model.builder.layer >= 1

    @test rknn4_param_best_model.epochs <= 500
    @test rknn4_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    rknn_chain_best_model_chain =
        fitted_params(fitted_params(mach_rknn4).machine).best_fitted_params.chain

    mae_Train_rknn = Flux.mae(
        rknn_chain_best_model_chain(Matrix(Train_in_data)'),
        Matrix(Train_out_data)',
    )
    mae_Test_rknn =
        Flux.mae(rknn_chain_best_model_chain(Matrix(Test_in_data)'), Matrix(Test_out_data)')

    println("mae_Train_rknn 4 $mae_Train_rknn")
    println("mae_Test_rknn 4 $mae_Test_rknn")

    @test mae_Train_rknn <= 1
    @test mae_Test_rknn <= 1

end


#end
