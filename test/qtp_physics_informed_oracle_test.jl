# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
module PhysicsInformedAndOracle

import Pkg
Pkg.activate("/home/pierre/CleverCloud/identification/test/")

using Test
using DataFrames
using CSV
using Flux
using MLJ
using MLJFlux
using Optim
using Zygote
using Dates
using Distributed

using Identification

import Identification: PhysicsInformed
import Identification: PhysicsInformedOracle
import Identification: Fnn

@testset "PhysicsInformed" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:5000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:5000, :]

    n_delay = 1
    normalisation = false

    # Separate data between test and train data with limits
    lower_in = [0.2 0.2 0.2 0.2 -Inf -Inf]
    upper_in = [1.32 1.32 1.32 1.32 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.32 1.32 1.32 1.32]

    DataTrainTest = Identification.data_formatting_identification(
        dfin,
        dfout;
        n_delay = n_delay,
        normalisation = normalisation,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    function QTP(du, u, p, t)
        #trainable parameters
        a1, a2, a3, a4 = p

        #constant parameters
        S = 0.06
        gamma_a = 0.3
        gamma_b = 0.4
        g = 9.81

        # a1 = 1.34e-4
        # a2 = 1.51e-4
        # a3 = 9.27e-5
        # a4 = 8.82e-5

        #states 
        x1 = u[1]
        x2 = u[2]
        x3 = u[3]
        x4 = u[4]
        qa = u[5]
        qb = u[6]

        du[1] =
            -a1 / S * sqrt(2 * g * x1) +
            a3 / S * sqrt(2 * g * x3) +
            gamma_a / (S * 3600) * qa
        du[2] =
            -a2 / S * sqrt(2 * g * x2) +
            a4 / S * sqrt(2 * g * x4) +
            gamma_b / (S * 3600) * qb
        du[3] = -a3 / S * sqrt(2 * g * x3) + (1 - gamma_b) / (S * 3600) * qb
        du[4] = -a4 / S * sqrt(2 * g * x4) + (1 - gamma_a) / (S * 3600) * qa
    end

    init_t_p = [1.1e-4, 1.2e-4, 9e-5, 9e-5]
    #init_t_p =#[1.34e-4, 1.51e-4, 9.27e-5, 8.82e-5]
    nbr_states = 4
    nbr_inputs = 2
    step_time = 5.0

    lower = [1e-5, 1e-5, 1e-5, 1e-5]
    upper = [10e-4, 10e-4, 10e-4, 10e-4]

    # QTP physical definition
    model_qtp = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Identification.physics_informed(
            QTP,
            init_t_p,
            nbr_states,
            nbr_inputs,
            step_time,
            lower_p = lower,
            upper_p = upper,
        ),
        batch_size = 2048,
        optimiser = Optim.LBFGS(),
        epochs = 50,
        loss = Flux.Losses.mae,
    )

    iterated_model_qtp = IteratedModel(
        model = model_qtp,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(5)), MLJ.NumberSinceBest(n=50), MLJ.Patience(n=5)],
    )

    mach_qtp = MLJ.machine(model_qtp, in_data, out_data)

    MLJ.fit!(mach_qtp, verbosity = 0)

    #save the model and optimisation results
    MLJ.save("./models_saved/qtp_physics_oracle_train_result.jls", mach_qtp)

    model_trained = fitted_params(mach_qtp).chain

    fitted_params(mach_qtp).chain.t_p

    ### test
    @test model_trained.t_p[1] != [1.1e-4]
    @test model_trained.t_p[2] != [1.2e-4]
    @test model_trained.t_p[3] != [9e-5]
    @test model_trained.t_p[4] != [9e-5]

    @test model_trained.t_p[1] <= upper[1]
    @test model_trained.t_p[2] <= upper[2]
    @test model_trained.t_p[3] <= upper[3]
    @test model_trained.t_p[4] <= upper[4]

    @test model_trained.t_p[1] >= lower[1]
    @test model_trained.t_p[2] >= lower[2]
    @test model_trained.t_p[3] >= lower[3]
    @test model_trained.t_p[4] >= lower[4]

end

@testset "PhysicsInformed + Oracle" begin

    # x+ = f(x, u) + O(x, u)

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:5000, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:5000, :]

    n_delay = 1
    normalisation = false

    # Separate data between test and train data with limits
    lower_in = [0.2 0.2 0.2 0.2 -Inf -Inf]
    upper_in = [1.32 1.32 1.32 1.32 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.32 1.32 1.32 1.32]

    DataTrainTest = data_formatting_identification(
        dfin,
        dfout;
        n_delay = n_delay,
        normalisation = normalisation,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    function QTP(du, u, p, t)
        #no trainable parameters
        #constant parameters
        S = 0.06
        gamma_a = 0.3
        gamma_b = 0.4
        g = 9.81

        a1 = 1.34e-4
        a2 = 1.51e-4
        a3 = 9.27e-5
        a4 = 8.82e-5

        #states 
        x1 = u[1]
        x2 = u[2]
        x3 = u[3]
        x4 = u[4]
        qa = u[5]
        qb = u[6]

        du[1] =
            -a1 / S * sqrt(2 * g * x1) +
            a3 / S * sqrt(2 * g * x3) +
            gamma_a / (S * 3600) * qa
        du[2] =
            -a2 / S * sqrt(2 * g * x2) +
            a4 / S * sqrt(2 * g * x4) +
            gamma_b / (S * 3600) * qb
        du[3] = -a3 / S * sqrt(2 * g * x3) + (1 - gamma_b) / (S * 3600) * qb
        du[4] = -a4 / S * sqrt(2 * g * x4) + (1 - gamma_a) / (S * 3600) * qa
    end

    init_t_p = [1.1e-4, 1.2e-4, 9e-5, 9e-5]
    #init_t_p =#[1.34e-4, 1.51e-4, 9.27e-5, 8.82e-5]
    nbr_states = 4
    nbr_inputs = 2
    step_time = 5.0

    # QTP physical definition
    model_qtp_oracle = MLJFlux.MultitargetNeuralNetworkRegressor(
        builder = Identification.physics_informed_oracle(
            QTP,
            nbr_states,
            nbr_inputs,
            step_time,
            Fnn(neuron = 5, layer = 2, σ = NNlib.swish),
        ),
        batch_size = 2048,
        optimiser = Optim.LBFGS(),
        epochs = 100,
        loss = Flux.Losses.mae,
    )

    iterated_model_qtp_oracle = IteratedModel(
        model = model_qtp_oracle,
        resampling = nothing,
        control = [Step(n = 1), TimeLimit(t = Minute(5)), MLJ.NumberSinceBest(n=50), MLJ.Patience(n=5)],
    )

    mach_qtp_oracle = MLJ.machine(model_qtp_oracle, in_data, out_data)

    MLJ.fit!(mach_qtp_oracle, verbosity = 0)

    #save the model and optimisation results
    MLJ.save("./models_saved/qtp_physics_oracle_train_result.jls", mach_qtp_oracle)

    model_trained_oracle = fitted_params(mach_qtp_oracle).chain
    model_trained_physical = model_trained_oracle[1]
    model_trained_fnn = model_trained_oracle[1]

    ### test
    mae_Train_QTP_oracle =
        Flux.mae(model_trained_oracle(Matrix(in_data)'), Matrix(out_data)')
    @test mae_Train_QTP_oracle <= 1

    mae_Train_QTP_physical =
        Flux.mae(model_trained_physical(Matrix(in_data)'), Matrix(out_data)')
    @test mae_Train_QTP_physical <= 1

end

end
