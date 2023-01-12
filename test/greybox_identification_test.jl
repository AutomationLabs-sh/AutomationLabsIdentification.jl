
# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
module GreyBoxIdentificationTest

using Test
using CSV
using DataFrames
using MLJ
using Flux
using Dates
using Distributed
using AutomationLabsIdentification


@testset "Physicsinformed grey box identification " begin

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

    function QTP(du, u, p, t)
        #constant parameters
        S = 0.06
        gamma_a = 0.3
        gamma_b = 0.4
        g = 9.81

        #trainable parameters
        a1, a2, a3, a4 = p

        #a1 = 1.34e-4
        #a2 = 1.51e-4
        #a3 = 9.27e-5
        #a4 = 8.82e-5

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


    #arguments and constraints on parameters
    lower_in = [0.2 0.2 0.2 0.2 -Inf -Inf]
    upper_in = [1.32 1.32 1.32 1.32 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.32 1.32 1.32 1.32]

    lower_params = [1e-5, 1e-5, 1e-5, 1e-5]
    upper_params = [10e-4, 10e-4, 10e-4, 10e-4]

    init_t_p = [1.1e-4, 1.2e-4, 9e-5, 9e-5]

    sample_time = 5.0

    grey_box_model_1 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "physics_informed",
        Minute(1);
        #option parameters
        f = QTP,
        trainable_parameters_init = init_t_p,
        sample_time = sample_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        lower_params = lower_params,
        upper_params = upper_params,
        computation_verbosity = 5,
        neuralnet_batch_size = n_sequence,
    )

    grey_box_model_3 = greybox_identification(
        dfin,
        dfout,
        "PSO",
        "PhysicsInformed",
        "CPU",
        Minute(5);
        #option parameters
        f = QTP,
        trainable_parameters_init = init_t_p,
        sample_time = sample_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        lower_params = lower_params,
        upper_params = upper_params,
        computation_verbosity = 0,
    )

    #Get best models
    greybox_best_model_chain_1 =
        fitted_params(fitted_params(grey_box_model_1).machine).chain

    greybox_best_model_chain_3 =
        fitted_params(fitted_params(grey_box_model_3[1]).machine).chain

    mae_Train_greybox_model_1 = Flux.mae(
        greybox_best_model_chain_1(Matrix(in_data)'),
        Matrix(out_data)',
    )

    mae_Train_greybox_model_1 = Flux.mae(
            greybox_best_model_chain_1(Matrix(grey_box_model_1[2].TrainDataIn)'),
            Matrix(grey_box_model_1[2].TrainDataOut)',
        )

    mae_Train_greybox_model_3 = Flux.mae(
            greybox_best_model_chain_3(Matrix(grey_box_model_3[2].TrainDataIn)'),
            Matrix(grey_box_model_3[2].TrainDataOut)',
        )

    mae_Test_greybox_model_1 = Flux.mae(
            greybox_best_model_chain_1(Matrix(grey_box_model_1[2].TestDataIn)'),
            Matrix(grey_box_model_1[2].TestDataOut)',
        )

    mae_Test_greybox_model_3 = Flux.mae(
            greybox_best_model_chain_3(Matrix(grey_box_model_3[2].TestDataIn)'),
            Matrix(grey_box_model_3[2].TestDataOut)',
        )

    println("mae_Train_greybox_model_1 $mae_Train_greybox_model_1")
    println("mae_Train_greybox_model_3 $mae_Train_greybox_model_3")

    println("mae_Test_greybox_model_1 $mae_Test_greybox_model_1")
    println("mae_Test_greybox_model_3 $mae_Test_greybox_model_3")

    @test mae_Train_greybox_model_1 <= 0.1
    @test mae_Test_greybox_model_1 <= 0.1
    @test mae_Train_greybox_model_3 <= 0.1
    @test mae_Test_greybox_model_3 <= 0.1
end

@testset "Physicsinformed + Oracle grey box identification " begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:2500, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:2500, :]

    function QTP(du, u, p, t)
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

    #kwargs
    lower_in = [0.2 0.2 0.2 0.2 -Inf -Inf]
    upper_in = [1.32 1.32 1.32 1.32 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.32 1.32 1.32 1.32]
    init_t_p = [1.1e-4, 1.2e-4, 9e-5, 9e-5]

    grey_box_model_1 = greybox_identification(
        dfin,
        dfout,
        "LBFGS",
        "PhysicsInformedOracle",
        "CPU",
        Minute(1);
        #option parameters
        f = QTP,
        trainable_parameters_init = init_t_p,
        sample_time = sample_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        lower_params = lower_params,
        upper_params = upper_params,
        computation_verbosity = 0,
    )

    grey_box_model_2 = greybox_identification(
        QTP,
        dfin,
        dfout,
        "OACCEL",
        "PhysicsInformedOracle",
        "CPU",
        Minute(1);
        #option parameters
        f = QTP,
        trainable_parameters_init = init_t_p,
        sample_time = sample_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        lower_params = lower_params,
        upper_params = upper_params,
        computation_verbosity = 0,
    )

    grey_box_model_3 = greybox_identification(
        QTP,
        dfin,
        dfout,
        "PSO",
        "PhysicsInformedOracle",
        "CPU",
        Minute(5);
        #option parameters
        f = QTP,
        trainable_parameters_init = init_t_p,
        sample_time = sample_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        lower_params = lower_params,
        upper_params = upper_params,
        computation_verbosity = 0,

    )

    # Get best models
    greybox_best_model_chain_1 =
        fitted_params(fitted_params(grey_box_model_1[1]).machine).chain
    greybox_best_model_chain_2 =
        fitted_params(fitted_params(grey_box_model_2[1]).machine).chain
    greybox_best_model_chain_3 =
        fitted_params(fitted_params(grey_box_model_3[1]).machine).chain

    mae_Train_greybox_model_1 = Flux.mae(
            greybox_best_model_chain_1(Matrix(grey_box_model_1[2].TrainDataIn)'),
            Matrix(grey_box_model_1[2].TrainDataOut)',
        )
    mae_Train_greybox_model_2 = Flux.mae(
            greybox_best_model_chain_2(Matrix(grey_box_model_2[2].TrainDataIn)'),
            Matrix(grey_box_model_2[2].TrainDataOut)',
        )
    mae_Train_greybox_model_3 = Flux.mae(
            greybox_best_model_chain_3(Matrix(grey_box_model_3[2].TrainDataIn)'),
            Matrix(grey_box_model_3[2].TrainDataOut)',
        )

    mae_Test_greybox_model_1 = Flux.mae(
            greybox_best_model_chain_1(Matrix(grey_box_model_1[2].TestDataIn)'),
            Matrix(grey_box_model_1[2].TestDataOut)',
        )
    mae_Test_greybox_model_2 = Flux.mae(
            greybox_best_model_chain_2(Matrix(grey_box_model_2[2].TestDataIn)'),
            Matrix(grey_box_model_2[2].TestDataOut)',
        )
    mae_Test_greybox_model_3 = Flux.mae(
            greybox_best_model_chain_3(Matrix(grey_box_model_3[2].TestDataIn)'),
            Matrix(grey_box_model_3[2].TestDataOut)',
        )

    println("mae_Train_greybox_model_1 $mae_Train_greybox_model_1")
    println("mae_Train_greybox_model_2 $mae_Train_greybox_model_2")
    println("mae_Train_greybox_model_3 $mae_Train_greybox_model_3")

    println("mae_Test_greybox_model_1 $mae_Test_greybox_model_1")
    println("mae_Test_greybox_model_2 $mae_Test_greybox_model_2")
    println("mae_Test_greybox_model_3 $mae_Test_greybox_model_3")

    @test mae_Train_greybox_model_1 <= 1
    @test mae_Test_greybox_model_1 <= 1
    @test mae_Train_greybox_model_2 <= 1
    @test mae_Test_greybox_model_2 <= 1
    @test mae_Train_greybox_model_3 != NaN
    @test mae_Test_greybox_model_3 != NaN

end
end