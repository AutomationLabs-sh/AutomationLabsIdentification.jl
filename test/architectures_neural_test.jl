# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
module ArchitecturesNeuralNetworks

using Test
using DataFrames
using Flux
using MLJ
using MLJFlux
using DifferentialEquations
using Distributed

using AutomationLabsIdentification

import AutomationLabsIdentification: Fnn
import AutomationLabsIdentification: Icnn
import AutomationLabsIdentification: DenseIcnn
import AutomationLabsIdentification: Rbf
import AutomationLabsIdentification: ResNet
import AutomationLabsIdentification: PolyNet
import AutomationLabsIdentification: DenseNet
import AutomationLabsIdentification: NeuralODE
import AutomationLabsIdentification: Rknn1
import AutomationLabsIdentification: Rknn2
import AutomationLabsIdentification: Rknn4
import AutomationLabsIdentification: Rnn
import AutomationLabsIdentification: Lstm
import AutomationLabsIdentification: Gru

import AutomationLabsIdentification: DenseSampleTime


@testset "Fnn" begin

    in_ = repeat(1:0.001:10, 1)[:, :]
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    #fnn definition
    model_fnn = MLJFlux.NeuralNetworkRegressor(
        builder = Fnn(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 1024,
        optimiser = Flux.RADAM(),
        epochs = 100,
    )

    mach_fnn = MLJ.machine(model_fnn, in1, out1)

    MLJ.fit!(mach_fnn)

    model_fnn_trained = fitted_params(mach_fnn).chain

    #test fnn
    @test size.(Flux.params(model_fnn_trained)) ==
          [(5, 1), (5, 5), (5,), (5, 5), (5,), (1, 5)]
    @test model_fnn_trained[1].σ == identity
    @test model_fnn_trained[2][1].σ == relu
    @test model_fnn_trained[2][2].σ == relu
    @test model_fnn_trained[3].σ == identity

end

@testset "DenseIcnn" begin

    nbr_neuron = 5
    # test neural convexity
    import AutomationLabsIdentification: DenseIcnn

    inner_icnn = DenseIcnn(nbr_neuron, nbr_neuron)
    @test (inner_icnn.weight .>= 0.0) == ones(Bool, nbr_neuron, nbr_neuron)
    @test inner_icnn.σ == relu

end

@testset "Icnn" begin

    in_ = repeat(1:0.001:10, 1)[:, :]
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    #icnn definition
    model_icnn = MLJFlux.NeuralNetworkRegressor(
        builder = Icnn(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 1024,
        optimiser = Flux.RADAM(),
        epochs = 100,
    )

    mach_icnn = MLJ.machine(model_icnn, in1, out1)

    MLJ.fit!(mach_icnn)

    model_icnn_trained = fitted_params(mach_icnn).chain

    #test icnn
    @test size.(Flux.params(model_icnn_trained)) ==
          [(5, 1), (5, 5), (5,), (5, 5), (5,), (1, 5)]
    @test model_icnn_trained[1].σ == identity
    @test model_icnn_trained[2][1].σ == relu
    @test model_icnn_trained[2][2].σ == relu
    @test model_icnn_trained[3].σ == identity

end

@testset "Rbf" begin

    #data
    in_ = repeat(1:0.001:10, 1)[:, :]
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    #icnn definition
    model_rbf = MLJFlux.NeuralNetworkRegressor(
        builder = Rbf(neuron = 5),
        batch_size = 1024,
        optimiser = Flux.RADAM(),
        epochs = 100,
    )

    mach_rbf = MLJ.machine(model_rbf, in1, out1)

    MLJ.fit!(mach_rbf)

    model_rbf_trained = fitted_params(mach_rbf).chain

    #test rbf
    @test size.(Flux.params(model_rbf_trained)) == [(5, 1), (5, 5), (1, 5)]
    @test model_rbf_trained[1].σ == identity
    @test model_rbf_trained[2].layers[1].σ == AutomationLabsIdentification.gaussian
    @test model_rbf_trained[3].σ == identity

end

@testset "ResNet" begin

    in_ = repeat(1:0.001:10, 1)[:, :]
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    #resnet definition
    model_resnet = MLJFlux.NeuralNetworkRegressor(
        builder = ResNet(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 1024,
        optimiser = Flux.RADAM(),
        epochs = 100,
    )

    mach_resnet = MLJ.machine(model_resnet, in1, out1)
    MLJ.fit!(mach_resnet)
    model_resnet_trained = fitted_params(mach_resnet).chain

    #test resnet
    @test size.(Flux.params(model_resnet_trained)) ==
          [(5, 1), (5, 5), (5,), (5, 5), (5,), (1, 5)]
    @test model_resnet_trained[1].σ == identity
    @test model_resnet_trained[2][1].layers.layers[1].σ == relu
    @test model_resnet_trained[2][2].layers.layers[1].σ == relu
    @test model_resnet_trained[3].σ == identity

end

@testset "PolyNet" begin

    in_ = repeat(1:0.001:10, 1)[:, :]
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    #polynet definition
    model_polynet = MLJFlux.NeuralNetworkRegressor(
        builder = PolyNet(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 1024,
        optimiser = Flux.RADAM(),
        epochs = 100,
    )

    mach_polynet = MLJ.machine(model_polynet, in1, out1)
    MLJ.fit!(mach_polynet)
    model_polynet_trained = fitted_params(mach_polynet).chain

    #test polynet
    @test size.(Flux.params(model_polynet_trained)) ==
          [(5, 1), (5, 5), (5,), (5, 5), (5,), (1, 5)]
    @test model_polynet_trained[1].σ == identity
    @test model_polynet_trained[2][1].layers.layers[1].σ == relu
    @test model_polynet_trained[2][2].layers.layers[1].σ == relu
    @test model_polynet_trained[3].σ == identity

end

@testset "DenseNet" begin

    #data
    in_ = repeat(1:0.001:10, 1)[:, :]
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    #densenet definition
    model_densenet = MLJFlux.NeuralNetworkRegressor(
        builder = DenseNet(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 1024,
        optimiser = Flux.RADAM(),
        epochs = 100,
    )

    mach_densenet = MLJ.machine(model_densenet, in1, out1)

    MLJ.fit!(mach_densenet)

    model_densenet_trained = fitted_params(mach_densenet).chain

    #test densenet
    @test size.(Flux.params(model_densenet_trained)) ==
          [(5, 1), (5, 5), (5,), (5, 10), (5,), (1, 15)]
    @test model_densenet_trained[1].σ == identity
    @test model_densenet_trained[2][1].layers.layers[1].σ == relu
    @test model_densenet_trained[2][2].layers.layers[1].σ == relu
    @test model_densenet_trained[3].σ == identity

end

@testset "NeuralODE" begin

    #data
    in_ = Float32.(repeat(1:0.001:10, 1)[:, :])
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    sample_time = 0.001

    #neural ode definition

    model_neural_ode = MLJFlux.NeuralNetworkRegressor(
        builder = NeuralODE(
            neuron = 5,
            layer = 2,
            σ = Flux.relu,
            sample_time = sample_time,
        ),
        batch_size = 1024,
        optimiser = Flux.ADAM(),
        epochs = 100,
    )

    mach_neural_ode = MLJ.machine(model_neural_ode, in1, out1)

    MLJ.fit!(mach_neural_ode)

    model_neural_ode_trained = fitted_params(mach_neural_ode).chain
    inner_chain = model_neural_ode_trained[1].model

    #test neural Ode
    @test size.(Flux.params(model_neural_ode_trained)) == [(70,)]
    @test size.(Flux.params(inner_chain)) == [(5, 1), (5, 5), (5,), (5, 5), (5,), (1, 5)]
    @test model_neural_ode_trained[1].tspan == (0.0f0, 0.001f0)
    @test model_neural_ode_trained[1].args == (DifferentialEquations.BS3(),)
    @test values(model_neural_ode_trained[1].kwargs) ==
          (save_everystep = false, reltol = 1e-6, abstol = 1e-6, save_start = false)

end

@testset "Rknn1" begin

    #data
    in_ = repeat(1:0.001:10, 1)[:, :]
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    sample_time = 0.001

    #neural ode definition

    model_rknn1 = MLJFlux.NeuralNetworkRegressor(
        builder = Rknn1(neuron = 5, layer = 2, σ = Flux.relu, sample_time = sample_time),
        batch_size = 1024,
        optimiser = Flux.ADAM(),
        epochs = 100,
    )

    mach_rknn1 = MLJ.machine(model_rknn1, in1, out1)

    MLJ.fit!(mach_rknn1)

    model_rknn1_trained = fitted_params(mach_rknn1).chain

    #test Rknn1
    @test size.(Flux.params(model_rknn1_trained)) ==
          [(5, 1), (5, 5), (5,), (5, 5), (5,), (1, 5)]

    @test typeof(model_rknn1_trained[1]) == AutomationLabsIdentification.DenseIdentityOut
    @test typeof(model_rknn1_trained[2][1]) == AutomationLabsIdentification.DenseSampleTime
    @test typeof(model_rknn1_trained[2][2]) == Chain{
        Tuple{
            Dense{typeof(identity),Matrix{Float32},Bool},
            Chain{
                Tuple{
                    Dense{typeof(relu),Matrix{Float32},Vector{Float32}},
                    Dense{typeof(relu),Matrix{Float32},Vector{Float32}},
                },
            },
            Dense{typeof(identity),Matrix{Float32},Bool},
        },
    }

end

@testset "Rknn2" begin

    #data
    in_ = repeat(1:0.001:10, 1)[:, :]
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    sample_time = 0.001

    #neural ode definition

    model_rknn2 = MLJFlux.NeuralNetworkRegressor(
        builder = Rknn2(neuron = 5, layer = 2, σ = Flux.relu, sample_time = sample_time),
        batch_size = 1024,
        optimiser = Flux.ADAM(),
        epochs = 100,
    )

    mach_rknn2 = MLJ.machine(model_rknn2, in1, out1)

    MLJ.fit!(mach_rknn2)

    model_rknn2_trained = fitted_params(mach_rknn2).chain

    #test Rknn1
    @test size.(Flux.params(model_rknn2_trained)) ==
          [(5, 1), (5, 5), (5,), (5, 5), (5,), (1, 5)]

    @test typeof(model_rknn2_trained[1]) == AutomationLabsIdentification.DenseIdentityOut
    @test typeof(model_rknn2_trained[2][1]) == AutomationLabsIdentification.Dense_over_z
    @test typeof(model_rknn2_trained[2][2]) == Chain{
        Tuple{
            DenseSampleTime,
            Chain{
                Tuple{
                    Dense{typeof(identity),Matrix{Float32},Bool},
                    Chain{
                        Tuple{
                            Dense{typeof(relu),Matrix{Float32},Vector{Float32}},
                            Dense{typeof(relu),Matrix{Float32},Vector{Float32}},
                        },
                    },
                    Dense{typeof(identity),Matrix{Float32},Bool},
                },
            },
        },
    }

end

@testset "Rknn4" begin

    #data
    in_ = repeat(1:0.001:10, 1)[:, :]
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    sample_time = 0.001

    #neural ode definition

    model_rknn4 = MLJFlux.NeuralNetworkRegressor(
        builder = Rknn4(neuron = 5, layer = 2, σ = Flux.relu, sample_time = sample_time),
        batch_size = 1024,
        optimiser = Flux.ADAM(),
        epochs = 100,
    )

    mach_rknn4 = MLJ.machine(model_rknn4, in1, out1)

    MLJ.fit!(mach_rknn4)

    model_rknn4_trained = fitted_params(mach_rknn4).chain

    #test Rknn4
    @test size.(Flux.params(model_rknn4_trained)) ==
          [(5, 1), (5, 5), (5,), (5, 5), (5,), (1, 5)]

    @test typeof(model_rknn4_trained[1]) == AutomationLabsIdentification.DenseIdentityOut
    @test typeof(model_rknn4_trained[2][2]) == Chain{
        Tuple{
            AutomationLabsIdentification.DenseSampleTime,
            Chain{
                Tuple{
                    Dense{typeof(identity),Matrix{Float32},Bool},
                    Chain{
                        Tuple{
                            Dense{typeof(relu),Matrix{Float32},Vector{Float32}},
                            Dense{typeof(relu),Matrix{Float32},Vector{Float32}},
                        },
                    },
                    Dense{typeof(identity),Matrix{Float32},Bool},
                },
            },
        },
    }

end



@testset "Rnn" begin

    #data
    in_ = Float32.(repeat(1:0.001:10, 1)[:, :])[1:140*64, :]
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    #neural rnn definition
    model_rnn = MLJFlux.NeuralNetworkRegressor(
        builder = Rnn(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 64,
        optimiser = Flux.RADAM(),
        epochs = 100,
    )

    mach_rnn = MLJ.machine(model_rnn, in1, out1)

    MLJ.fit!(mach_rnn)

    model_rnn_trained = fitted_params(mach_rnn).chain

    #test rnn
    @test size.(Flux.params(model_rnn_trained)) ==
          [(5, 1), (5, 5), (5, 5), (5,), (5, 1), (5, 5), (5, 5), (5,), (5, 1), (1, 5)]
    @test model_rnn_trained[1].σ == identity
    @test model_rnn_trained[2][1].cell.σ == relu
    @test model_rnn_trained[2][2].cell.σ == relu
    @test model_rnn_trained[3].σ == identity
end

@testset "Lstm" begin

    #data
    in_ = Float32.(repeat(1:0.001:10, 1)[:, :])[1:140*64, :]
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    #neural lstm definition
    model_lstm = MLJFlux.NeuralNetworkRegressor(
        builder = Lstm(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 64,
        optimiser = Flux.RADAM(),
        epochs = 100,
    )

    mach_lstm = MLJ.machine(model_lstm, in1, out1)

    MLJ.fit!(mach_lstm)

    model_lstm_trained = fitted_params(mach_lstm).chain

    #test lstm
    @test size.(Flux.params(model_lstm_trained)) == [
        (5, 1),
        (20, 5),
        (20, 5),
        (20,),
        (5, 1),
        (5, 1),
        (20, 5),
        (20, 5),
        (20,),
        (5, 1),
        (5, 1),
        (1, 5),
    ]
    @test model_lstm_trained[1].σ == identity
    @test model_lstm_trained[3].σ == identity
end


@testset "Gru" begin

    #data
    in_ = Float32.(repeat(1:0.001:10, 1)[:, :])[1:140*64, :]
    in1 = MLJ.table(in_)

    out1 = vec(sin.(in_))

    #neural gru definition
    model_gru = MLJFlux.NeuralNetworkRegressor(
        builder = Gru(neuron = 5, layer = 2, σ = NNlib.relu),
        batch_size = 64,
        optimiser = Flux.RADAM(),
        epochs = 100,
    )

    mach_gru = MLJ.machine(model_gru, in1, out1)

    MLJ.fit!(mach_gru)

    model_gru_trained = fitted_params(mach_gru).chain

    #test gru
    @test size.(Flux.params(model_gru_trained)) ==
          [(5, 1), (15, 5), (15, 5), (15,), (5, 1), (15, 5), (15, 5), (15,), (5, 1), (1, 5)]

    @test model_gru_trained[1].σ == identity
    @test model_gru_trained[3].σ == identity
end


end
