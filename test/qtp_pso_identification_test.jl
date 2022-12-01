module QTP_ParticleSwarmIdentification

using Test
using CSV
using DataFrames
using MLJ
using MLJFlux
using Flux
using StableRNGs
using MLJParticleSwarmOptimization
using Optim
using Distributed

@everywhere import Pkg
@everywhere Pkg.activate("/home/pierre/CleverCloud/identification")
@everywhere using Identification

import Identification: Fnn
import Identification: FnnLinear
import Identification: Icnn
import Identification: DenseIcnn
import Identification: Rbf
import Identification: ResNet
import Identification: DenseNet
import Identification: NeuralNetODE
import Identification: ExplorationOfNetworks #meta-model 

@testset "QTP identification Linear Fnn" begin

    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:50000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:50000, :]

    n_delay = 1
    normalisation = false

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    ### Fnn ###
    model_fnn = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = FnnLinear(neuron = 5, layer = 2),
        batch_size = 2048,
        optimiser = Optim.ParticleSwarm(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    r1 = range(model_fnn, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_fnn, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_fnn, :epochs, lower = 100, upper = 1000)

    tuned_model_fnn = MLJ.TunedModel(
        model = model_fnn,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 1,
        #acceleration = MLJ.CPUProcesses(),
    )

    mach_fnn = MLJ.machine(tuned_model_fnn, in_data, out_data)

    MLJ.fit!(mach_fnn, verbosity = 1)

    @test fitted_params(mach_fnn).best_model != 0
    @test report(mach_fnn).best_history_entry != 0
    @test report(mach_fnn).history != 0

    fnn_param_best_model = fitted_params(mach_fnn).best_model

    @test fnn_param_best_model.builder.neuron < 16
    @test fnn_param_best_model.builder.neuron > 4

    @test fnn_param_best_model.builder.layer < 6
    @test fnn_param_best_model.builder.layer >= 1

    @test fnn_param_best_model.epochs <= 1000
    @test fnn_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    fnn_chain_best_model_chain = fitted_params(mach_fnn).best_fitted_params.chain

    mae_Train_fnn = Flux.mae(
        fnn_chain_best_model_chain(Matrix(Train_in_data)'),
        Matrix(Train_out_data)',
    )

    mae_Test_fnn =
        Flux.mae(fnn_chain_best_model_chain(Matrix(Test_in_data)'), Matrix(Test_out_data)')

    println("mae_Train_fnn linear $mae_Train_fnn")
    println("mae_Test_fnn linear $mae_Test_fnn")

    @test mae_Train_fnn <= 1
    @test mae_Test_fnn <= 1

end

@testset "QTP identification Fnn" begin

    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:50000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:50000, :]

    n_delay = 1
    normalisation = false

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    ### Fnn ###
    model_fnn = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Fnn(neuron = 5, layer = 2, σ = NNlib.swish),
        batch_size = 2048,
        optimiser = Optim.ParticleSwarm(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    r1 = range(model_fnn, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_fnn, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_fnn, :epochs, lower = 100, upper = 1000)

    tuned_model_fnn = MLJ.TunedModel(
        model = model_fnn,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 1,
        #acceleration = MLJ.CPUProcesses(),
    )

    mach_fnn = MLJ.machine(tuned_model_fnn, in_data, out_data)

    MLJ.fit!(mach_fnn, verbosity = 1)

    @test fitted_params(mach_fnn).best_model != 0
    @test report(mach_fnn).best_history_entry != 0
    @test report(mach_fnn).history != 0

    fnn_param_best_model = fitted_params(mach_fnn).best_model

    @test fnn_param_best_model.builder.neuron < 16
    @test fnn_param_best_model.builder.neuron > 4

    @test fnn_param_best_model.builder.layer < 6
    @test fnn_param_best_model.builder.layer >= 1

    @test fnn_param_best_model.epochs <= 1000
    @test fnn_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    fnn_chain_best_model_chain = fitted_params(mach_fnn).best_fitted_params.chain

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

@testset "QTP identification Rbf" begin

    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:50000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:50000, :]

    n_delay = 1
    normalisation = false

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    ### Rbf ### 
    model_rbf = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Rbf(neuron = 5),
        batch_size = 512,
        optimiser = Optim.ParticleSwarm(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    r1 = range(model_rbf, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_rbf, :epochs, lower = 100, upper = 1000)

    tuned_model_rbf = MLJ.TunedModel(
        model = model_rbf,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2],
        measure = my_loss,
        n = 1,
        #acceleration = MLJ.CPUProcesses(),
    )

    mach_rbf = MLJ.machine(tuned_model_rbf, in_data, out_data)

    MLJ.fit!(mach_rbf, verbosity = 1)

    @test fitted_params(mach_rbf).best_model != 0
    @test report(mach_rbf).best_history_entry != 0
    @test report(mach_rbf).history != 0

    rbf_param_best_model = fitted_params(mach_rbf).best_model

    @test rbf_param_best_model.builder.neuron < 16
    @test rbf_param_best_model.builder.neuron > 4

    @test rbf_param_best_model.epochs <= 1000
    @test rbf_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    rbf_chain_best_model_chain = fitted_params(mach_rbf).best_fitted_params.chain

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

@testset "QTP identification Icnn" begin

    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:50000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:50000, :]

    n_delay = 1
    normalisation = false

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    #### Icnn ###
    model_icnn = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Icnn(neuron = 10, layer = 2, σ = NNlib.relu),
        batch_size = 2048,
        optimiser = Optim.ParticleSwarm(),
        epochs = 500,
        loss = Flux.Losses.mae,
    )

    r1 = range(model_icnn, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_icnn, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_icnn, :epochs, lower = 100, upper = 1000)

    tuned_model_icnn = MLJ.TunedModel(
        model = model_icnn,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 1,
        #acceleration = MLJ.CPUProcesses(),
    )

    mach_icnn = MLJ.machine(tuned_model_icnn, in_data, out_data)

    MLJ.fit!(mach_icnn, verbosity = 5)

    @test fitted_params(mach_icnn).best_model != 0
    @test report(mach_icnn).best_history_entry != 0
    @test report(mach_icnn).history != 0

    icnn_param_best_model = fitted_params(mach_icnn).best_model

    @test icnn_param_best_model.builder.neuron < 16
    @test icnn_param_best_model.builder.neuron > 4

    @test icnn_param_best_model.builder.layer < 6
    @test icnn_param_best_model.builder.layer >= 1

    @test icnn_param_best_model.epochs <= 1000
    @test icnn_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    icnn_chain_best_model_chain = fitted_params(mach_icnn).best_fitted_params.chain

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

@testset "QTP identification ResNet" begin

    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:50000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:50000, :]

    n_delay = 1
    normalisation = false

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    #ResNet
    #MLJ.default_resource(MLJ.CPUProcesses())
    model_resnet = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = ResNet(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 2048,
        optimiser = Optim.ParticleSwarm(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    r1 = range(model_resnet, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_resnet, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_resnet, :epochs, lower = 100, upper = 1000)

    tuned_model_resnet = MLJ.TunedModel(
        model = model_resnet,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 1,
        #acceleration = MLJ.CPUProcesses(),
    )

    mach_resnet = MLJ.machine(tuned_model_resnet, in_data, out_data)

    MLJ.fit!(mach_resnet, verbosity = 1)

    @test fitted_params(mach_resnet).best_model != 0
    @test report(mach_resnet).best_history_entry != 0
    @test report(mach_resnet).history != 0

    resnet_param_best_model = fitted_params(mach_resnet).best_model

    @test resnet_param_best_model.builder.neuron < 16
    @test resnet_param_best_model.builder.neuron > 4

    @test resnet_param_best_model.builder.layer < 6
    @test resnet_param_best_model.builder.layer >= 1

    @test resnet_param_best_model.epochs <= 1000
    @test resnet_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    resnet_chain_best_model_chain = fitted_params(mach_resnet).best_fitted_params.chain

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

@testset "QTP identification DenseNet" begin

    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:50000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:50000, :]

    n_delay = 1
    normalisation = false

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    ### DenseNet ### 
    model_densenet = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = DenseNet(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 2048,
        optimiser = Optim.ParticleSwarm(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    r1 = range(model_densenet, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_densenet, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_densenet, :epochs, lower = 100, upper = 1000)

    tuned_model_densenet = MLJ.TunedModel(
        model = model_densenet,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 1,
        #acceleration = MLJ.CPUProcesses(),
    )

    mach_densenet = MLJ.machine(tuned_model_densenet, in_data, out_data)

    MLJ.fit!(mach_densenet, verbosity = 1)

    @test fitted_params(mach_densenet).best_model != 0
    @test report(mach_densenet).best_history_entry != 0
    @test report(mach_densenet).history != 0

    densenet_param_best_model = fitted_params(mach_densenet).best_model

    @test densenet_param_best_model.builder.neuron < 16
    @test densenet_param_best_model.builder.neuron > 4

    @test densenet_param_best_model.builder.layer < 6
    @test densenet_param_best_model.builder.layer >= 1

    @test densenet_param_best_model.epochs <= 1000
    @test densenet_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    densenet_chain_best_model_chain = fitted_params(mach_densenet).best_fitted_params.chain

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

@testset "QTP identification NeuralNetODE" begin

    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:50000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:50000, :]

    n_delay = 1
    normalisation = false

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin,
        dfout,
        n_delay = n_delay,
        normalisation = normalisation,
    )

    in_data = (DataTrainValidationTest.TrainDataIn)
    out_data = (DataTrainValidationTest.TrainDataOut)

    #Customs loss function for multiple neural outputs
    my_loss = function (yhat, y)
        loss = mean(abs.(Matrix(hcat(yhat[1], yhat[2], yhat[3], yhat[4])) .- Matrix(y)))
        return loss
    end

    ### NeuralNetODE ### 
    model_NeuralNetODE = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = NeuralNetODE(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 512,
        optimiser = Optim.ParticleSwarm(),
        epochs = 1000,
        loss = Flux.Losses.mae,
    )

    r1 = range(model_NeuralNetODE, :(builder.neuron), lower = 5, upper = 15)
    r2 = range(model_NeuralNetODE, :(builder.layer), lower = 1, upper = 5)
    r3 = range(model_NeuralNetODE, :epochs, lower = 100, upper = 1000)

    tuned_model_NeuralNetODE = MLJ.TunedModel(
        model = model_NeuralNetODE,
        tuning = AdaptiveParticleSwarm(rng = StableRNG(0)),
        resampling = CV(nfolds = 6, rng = StableRNG(1)),
        range = [r1, r2, r3],
        measure = my_loss,
        n = 1,
        #acceleration = MLJ.CPUProcesses(), #no acceleration too much memory usage
    )

    mach_NeuralNetODE = MLJ.machine(tuned_model_NeuralNetODE, in_data, out_data)

    MLJ.fit!(mach_NeuralNetODE, verbosity = 1)

    @test fitted_params(mach_NeuralNetODE).best_model != 0
    @test report(mach_NeuralNetODE).best_history_entry != 0
    @test report(mach_NeuralNetODE).history != 0

    NeuralNetODE_param_best_model = fitted_params(mach_NeuralNetODE).best_model

    @test NeuralNetODE_param_best_model.builder.neuron < 16
    @test NeuralNetODE_param_best_model.builder.neuron > 4

    @test NeuralNetODE_param_best_model.builder.layer < 6
    @test NeuralNetODE_param_best_model.builder.layer >= 1

    @test NeuralNetODE_param_best_model.epochs <= 1000
    @test NeuralNetODE_param_best_model.epochs >= 100

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    neuralnet_chain_best_model_chain =
        fitted_params(mach_NeuralNetODE).best_fitted_params.chain

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

end
