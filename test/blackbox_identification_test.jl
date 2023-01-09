# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
module BlackBoxIdentificationTest

using AutomationLabsIdentification

using Test
using CSV
using DataFrames
using MLJ
using Flux
using Dates
using Distributed

@testset "Linear architecture blackbox identification" begin

    #load data 
    dfout_raw = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))
    dfin_raw = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))

    #Train models
    max_time = Minute(5) #hours

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin_raw,
        dfout_raw,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )
    
    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    m_linear_1 = proceed_identification(
        in_data, 
        out_data, 
        "lls", 
        "linear", 
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    #get params
    A_t = fitted_params(m_linear_1)[1]
    A = A_t'

    Train_in_data = m_linear_1.data[1]
    Train_out_data = m_linear_1.data[2]

    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    mae_Train_fnn_1 = Flux.mae(A * (Matrix(Train_in_data)'), Matrix(Train_out_data)')

    mae_Test_fnn_1 = Flux.mae(A * (Matrix(Test_in_data)'), Matrix(Test_out_data)')

    println("mae_Train_fnn_1 $mae_Train_fnn_1")

    println("mae_Test_fnn_1 $mae_Test_fnn_1")

    @test mae_Train_fnn_1 <= 0.1

    @test mae_Test_fnn_1 <= 0.1

end

@testset "Fnn architecture blackbox identification" begin

    #load data 
    dfout_raw = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:2500, :]
    dfin_raw = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:2500, :]

    #Train models
    max_time = Minute(5) #hours

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin_raw,
        dfout_raw,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )
    
    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    m_fnn_0 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_fnn_1 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_fnn_2 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_fnn_3 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_fnn_4 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_fnn_5 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_fnn_6 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )


    m_fnn_7 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_fnn_8 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_fnn_9 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_fnn_10 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_fnn_11 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_fnn_12 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_fnn_13 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_fnn_14 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_fnn_15 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_fnn_16 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_fnn_17 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_fnn_18 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_fnn_19 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_fnn_20 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "fnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    #Get best models
    fnn_0_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_0[1]).machine).best_fitted_params.chain
    fnn_1_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_1[1]).machine).best_fitted_params.chain
    fnn_2_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_2[1]).machine).best_fitted_params.chain
    fnn_3_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_3[1]).machine).best_fitted_params.chain
    fnn_4_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_4[1]).machine).best_fitted_params.chain
    fnn_5_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_5[1]).machine).best_fitted_params.chain
    fnn_6_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_6[1]).machine).best_fitted_params.chain
    fnn_7_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_7[1]).machine).best_fitted_params.chain
    fnn_8_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_8[1]).machine).best_fitted_params.chain
    fnn_9_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_9[1]).machine).best_fitted_params.chain
    fnn_10_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_10[1]).machine).best_fitted_params.chain
    fnn_11_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_11[1]).machine).best_fitted_params.chain
    fnn_12_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_12[1]).machine).best_fitted_params.chain
    fnn_13_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_13[1]).machine).best_fitted_params.chain
    fnn_14_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_14[1]).machine).best_fitted_params.chain
    fnn_15_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_15[1]).machine).best_fitted_params.chain
    fnn_16_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_16[1]).machine).best_fitted_params.chain
    fnn_17_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_17[1]).machine).best_fitted_params.chain
    fnn_18_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_18[1]).machine).best_fitted_params.chain
    fnn_19_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_19[1]).machine).best_fitted_params.chain
    fnn_20_chain_best_model_chain =
        fitted_params(fitted_params(m_fnn_20[1]).machine).best_fitted_params.chain

    # MAE with train and test data
    mae_Train_fnn_0 = Flux.mae(
        fnn_0_chain_best_model_chain(Matrix(m_fnn_0[2].TrainDataIn)'),
        Matrix(m_fnn_0[2].TrainDataOut)',
    )
    mae_Train_fnn_1 = Flux.mae(
        fnn_1_chain_best_model_chain(Matrix(m_fnn_1[2].TrainDataIn)'),
        Matrix(m_fnn_1[2].TrainDataOut)',
    )
    mae_Train_fnn_2 = Flux.mae(
        fnn_2_chain_best_model_chain(Matrix(m_fnn_2[2].TrainDataIn)'),
        Matrix(m_fnn_2[2].TrainDataOut)',
    )
    mae_Train_fnn_3 = Flux.mae(
        fnn_3_chain_best_model_chain(Matrix(m_fnn_3[2].TrainDataIn)'),
        Matrix(m_fnn_3[2].TrainDataOut)',
    )
    mae_Train_fnn_4 = Flux.mae(
        fnn_4_chain_best_model_chain(Matrix(m_fnn_4[2].TrainDataIn)'),
        Matrix(m_fnn_4[2].TrainDataOut)',
    )
    mae_Train_fnn_5 = Flux.mae(
        fnn_5_chain_best_model_chain(Matrix(m_fnn_5[2].TrainDataIn)'),
        Matrix(m_fnn_5[2].TrainDataOut)',
    )
    mae_Train_fnn_6 = Flux.mae(
        fnn_6_chain_best_model_chain(Matrix(m_fnn_6[2].TrainDataIn)'),
        Matrix(m_fnn_6[2].TrainDataOut)',
    )
    mae_Train_fnn_7 = Flux.mae(
        fnn_7_chain_best_model_chain(Matrix(m_fnn_7[2].TrainDataIn)'),
        Matrix(m_fnn_7[2].TrainDataOut)',
    )
    mae_Train_fnn_8 = Flux.mae(
        fnn_8_chain_best_model_chain(Matrix(m_fnn_8[2].TrainDataIn)'),
        Matrix(m_fnn_8[2].TrainDataOut)',
    )
    mae_Train_fnn_9 = Flux.mae(
        fnn_9_chain_best_model_chain(Matrix(m_fnn_9[2].TrainDataIn)'),
        Matrix(m_fnn_9[2].TrainDataOut)',
    )
    mae_Train_fnn_10 = Flux.mae(
        fnn_10_chain_best_model_chain(Matrix(m_fnn_10[2].TrainDataIn)'),
        Matrix(m_fnn_10[2].TrainDataOut)',
    )
    mae_Train_fnn_11 = Flux.mae(
        fnn_11_chain_best_model_chain(Matrix(m_fnn_11[2].TrainDataIn)'),
        Matrix(m_fnn_11[2].TrainDataOut)',
    )
    mae_Train_fnn_12 = Flux.mae(
        fnn_12_chain_best_model_chain(Matrix(m_fnn_12[2].TrainDataIn)'),
        Matrix(m_fnn_12[2].TrainDataOut)',
    )
    mae_Train_fnn_13 = Flux.mae(
        fnn_13_chain_best_model_chain(Matrix(m_fnn_13[2].TrainDataIn)'),
        Matrix(m_fnn_13[2].TrainDataOut)',
    )
    mae_Train_fnn_14 = Flux.mae(
        fnn_14_chain_best_model_chain(Matrix(m_fnn_14[2].TrainDataIn)'),
        Matrix(m_fnn_14[2].TrainDataOut)',
    )
    mae_Train_fnn_15 = Flux.mae(
        fnn_15_chain_best_model_chain(Matrix(m_fnn_15[2].TrainDataIn)'),
        Matrix(m_fnn_15[2].TrainDataOut)',
    )
    mae_Train_fnn_16 = Flux.mae(
        fnn_16_chain_best_model_chain(Matrix(m_fnn_16[2].TrainDataIn)'),
        Matrix(m_fnn_16[2].TrainDataOut)',
    )
    mae_Train_fnn_17 = Flux.mae(
        fnn_17_chain_best_model_chain(Matrix(m_fnn_17[2].TrainDataIn)'),
        Matrix(m_fnn_17[2].TrainDataOut)',
    )
    mae_Train_fnn_18 = Flux.mae(
        fnn_18_chain_best_model_chain(Matrix(m_fnn_18[2].TrainDataIn)'),
        Matrix(m_fnn_18[2].TrainDataOut)',
    )
    mae_Train_fnn_19 = Flux.mae(
        fnn_19_chain_best_model_chain(Matrix(m_fnn_19[2].TrainDataIn)'),
        Matrix(m_fnn_19[2].TrainDataOut)',
    )
    mae_Train_fnn_20 = Flux.mae(
        fnn_20_chain_best_model_chain(Matrix(m_fnn_20[2].TrainDataIn)'),
        Matrix(m_fnn_20[2].TrainDataOut)',
    )

    mae_Test_fnn_0 = Flux.mae(
        fnn_0_chain_best_model_chain(Matrix(m_fnn_0[2].TestDataIn)'),
        Matrix(m_fnn_0[2].TestDataOut)',
    )
    mae_Test_fnn_1 = Flux.mae(
        fnn_1_chain_best_model_chain(Matrix(m_fnn_1[2].TestDataIn)'),
        Matrix(m_fnn_1[2].TestDataOut)',
    )
    mae_Test_fnn_2 = Flux.mae(
        fnn_2_chain_best_model_chain(Matrix(m_fnn_2[2].TestDataIn)'),
        Matrix(m_fnn_2[2].TestDataOut)',
    )
    mae_Test_fnn_3 = Flux.mae(
        fnn_3_chain_best_model_chain(Matrix(m_fnn_3[2].TestDataIn)'),
        Matrix(m_fnn_3[2].TestDataOut)',
    )
    mae_Test_fnn_4 = Flux.mae(
        fnn_4_chain_best_model_chain(Matrix(m_fnn_4[2].TestDataIn)'),
        Matrix(m_fnn_4[2].TestDataOut)',
    )
    mae_Test_fnn_5 = Flux.mae(
        fnn_5_chain_best_model_chain(Matrix(m_fnn_5[2].TestDataIn)'),
        Matrix(m_fnn_5[2].TestDataOut)',
    )
    mae_Test_fnn_6 = Flux.mae(
        fnn_6_chain_best_model_chain(Matrix(m_fnn_6[2].TestDataIn)'),
        Matrix(m_fnn_6[2].TestDataOut)',
    )
    mae_Test_fnn_7 = Flux.mae(
        fnn_7_chain_best_model_chain(Matrix(m_fnn_7[2].TestDataIn)'),
        Matrix(m_fnn_7[2].TestDataOut)',
    )
    mae_Test_fnn_8 = Flux.mae(
        fnn_8_chain_best_model_chain(Matrix(m_fnn_8[2].TestDataIn)'),
        Matrix(m_fnn_8[2].TestDataOut)',
    )
    mae_Test_fnn_9 = Flux.mae(
        fnn_9_chain_best_model_chain(Matrix(m_fnn_9[2].TestDataIn)'),
        Matrix(m_fnn_9[2].TestDataOut)',
    )
    mae_Test_fnn_10 = Flux.mae(
        fnn_10_chain_best_model_chain(Matrix(m_fnn_10[2].TestDataIn)'),
        Matrix(m_fnn_10[2].TestDataOut)',
    )
    mae_Test_fnn_11 = Flux.mae(
        fnn_11_chain_best_model_chain(Matrix(m_fnn_11[2].TestDataIn)'),
        Matrix(m_fnn_11[2].TestDataOut)',
    )
    mae_Test_fnn_12 = Flux.mae(
        fnn_12_chain_best_model_chain(Matrix(m_fnn_12[2].TestDataIn)'),
        Matrix(m_fnn_12[2].TestDataOut)',
    )
    mae_Test_fnn_13 = Flux.mae(
        fnn_13_chain_best_model_chain(Matrix(m_fnn_13[2].TestDataIn)'),
        Matrix(m_fnn_13[2].TestDataOut)',
    )
    mae_Test_fnn_14 = Flux.mae(
        fnn_14_chain_best_model_chain(Matrix(m_fnn_14[2].TestDataIn)'),
        Matrix(m_fnn_14[2].TestDataOut)',
    )
    mae_Test_fnn_15 = Flux.mae(
        fnn_15_chain_best_model_chain(Matrix(m_fnn_15[2].TestDataIn)'),
        Matrix(m_fnn_15[2].TestDataOut)',
    )
    mae_Test_fnn_16 = Flux.mae(
        fnn_16_chain_best_model_chain(Matrix(m_fnn_16[2].TestDataIn)'),
        Matrix(m_fnn_16[2].TestDataOut)',
    )
    mae_Test_fnn_17 = Flux.mae(
        fnn_17_chain_best_model_chain(Matrix(m_fnn_17[2].TestDataIn)'),
        Matrix(m_fnn_17[2].TestDataOut)',
    )
    mae_Test_fnn_18 = Flux.mae(
        fnn_18_chain_best_model_chain(Matrix(m_fnn_18[2].TestDataIn)'),
        Matrix(m_fnn_18[2].TestDataOut)',
    )
    mae_Test_fnn_19 = Flux.mae(
        fnn_19_chain_best_model_chain(Matrix(m_fnn_19[2].TestDataIn)'),
        Matrix(m_fnn_19[2].TestDataOut)',
    )
    mae_Test_fnn_20 = Flux.mae(
        fnn_20_chain_best_model_chain(Matrix(m_fnn_20[2].TestDataIn)'),
        Matrix(m_fnn_20[2].TestDataOut)',
    )

    println("mae_Train_fnn_0 $mae_Train_fnn_0")
    println("mae_Train_fnn_1 $mae_Train_fnn_1")
    println("mae_Train_fnn_2 $mae_Train_fnn_2")
    println("mae_Train_fnn_3 $mae_Train_fnn_3")
    println("mae_Train_fnn_4 $mae_Train_fnn_4")
    println("mae_Train_fnn_5 $mae_Train_fnn_5")
    println("mae_Train_fnn_6 $mae_Train_fnn_6")
    println("mae_Train_fnn_7 $mae_Train_fnn_7")
    println("mae_Train_fnn_8 $mae_Train_fnn_8")
    println("mae_Train_fnn_9 $mae_Train_fnn_9")
    println("mae_Train_fnn_10 $mae_Train_fnn_10")
    println("mae_Train_fnn_11 $mae_Train_fnn_11")
    println("mae_Train_fnn_12 $mae_Train_fnn_12")
    println("mae_Train_fnn_13 $mae_Train_fnn_13")
    println("mae_Train_fnn_14 $mae_Train_fnn_14")
    println("mae_Train_fnn_15 $mae_Train_fnn_15")
    println("mae_Train_fnn_16 $mae_Train_fnn_16")
    println("mae_Train_fnn_17 $mae_Train_fnn_17")
    println("mae_Train_fnn_18 $mae_Train_fnn_18")
    println("mae_Train_fnn_19 $mae_Train_fnn_19")
    println("mae_Train_fnn_20 $mae_Train_fnn_20")

    println("mae_Test_fnn_0 $mae_Test_fnn_0")
    println("mae_Test_fnn_1 $mae_Test_fnn_1")
    println("mae_Test_fnn_2 $mae_Test_fnn_2")
    println("mae_Test_fnn_3 $mae_Test_fnn_3")
    println("mae_Test_fnn_4 $mae_Test_fnn_4")
    println("mae_Test_fnn_5 $mae_Test_fnn_5")
    println("mae_Test_fnn_6 $mae_Test_fnn_6")
    println("mae_Test_fnn_7 $mae_Test_fnn_7")
    println("mae_Test_fnn_8 $mae_Test_fnn_8")
    println("mae_Test_fnn_9 $mae_Test_fnn_9")
    println("mae_Test_fnn_10 $mae_Test_fnn_10")
    println("mae_Test_fnn_11 $mae_Test_fnn_11")
    println("mae_Test_fnn_12 $mae_Test_fnn_12")
    println("mae_Test_fnn_13 $mae_Test_fnn_13")
    println("mae_Test_fnn_14 $mae_Test_fnn_14")
    println("mae_Test_fnn_15 $mae_Test_fnn_15")
    println("mae_Test_fnn_16 $mae_Test_fnn_16")
    println("mae_Test_fnn_17 $mae_Test_fnn_17")
    println("mae_Test_fnn_18 $mae_Test_fnn_18")
    println("mae_Test_fnn_19 $mae_Test_fnn_19")
    println("mae_Test_fnn_20 $mae_Test_fnn_20")

    @test mae_Train_fnn_0 <= 10
    @test mae_Train_fnn_1 <= 10
    @test mae_Train_fnn_2 <= 10
    @test mae_Train_fnn_3 <= 10
    @test mae_Train_fnn_4 <= 10
    @test mae_Train_fnn_5 <= 10
    @test mae_Train_fnn_6 != NaN
    @test mae_Train_fnn_7 <= 10
    @test mae_Train_fnn_8 <= 10
    @test mae_Train_fnn_9 <= 10
    @test mae_Train_fnn_10 <= 10
    @test mae_Train_fnn_11 <= 10
    @test mae_Train_fnn_12 <= 10
    @test mae_Train_fnn_13 != NaN
    @test mae_Train_fnn_14 <= 10
    @test mae_Train_fnn_15 <= 10
    @test mae_Train_fnn_16 <= 10
    @test mae_Train_fnn_17 <= 10
    @test mae_Train_fnn_18 <= 10
    @test mae_Train_fnn_19 <= 10
    @test mae_Train_fnn_20 != NaN

    @test mae_Test_fnn_0 <= 10
    @test mae_Test_fnn_1 <= 10
    @test mae_Test_fnn_2 <= 10
    @test mae_Test_fnn_3 <= 10
    @test mae_Test_fnn_4 <= 10
    @test mae_Test_fnn_5 <= 10
    @test mae_Test_fnn_6 != NaN
    @test mae_Test_fnn_7 <= 10
    @test mae_Test_fnn_8 <= 10
    @test mae_Test_fnn_9 <= 10
    @test mae_Test_fnn_10 <= 10
    @test mae_Test_fnn_11 <= 10
    @test mae_Test_fnn_12 <= 10
    @test mae_Test_fnn_13 != NaN
    @test mae_Test_fnn_14 <= 10
    @test mae_Test_fnn_15 <= 10
    @test mae_Test_fnn_16 <= 10
    @test mae_Test_fnn_17 <= 10
    @test mae_Test_fnn_18 <= 10
    @test mae_Test_fnn_19 <= 10
    @test mae_Test_fnn_20 != NaN

end

@testset "icnn architecture blackbox identification" begin

    #load data 
    dfout_raw = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:2500, :]
    dfin_raw = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:2500, :]

    #Train models
    max_time = Minute(5) #hours

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin_raw,
        dfout_raw,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )
    
    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    m_icnn_0 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_icnn_1 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_icnn_2 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_icnn_3 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_icnn_4 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_icnn_5 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_icnn_6 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )


    m_icnn_7 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_icnn_8 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_icnn_9 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_icnn_10 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_icnn_11 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_icnn_12 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_icnn_13 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_icnn_14 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_icnn_15 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_icnn_16 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_icnn_17 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_icnn_18 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_icnn_19 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_icnn_20 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "icnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    #Get best models
    icnn_0_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_0[1]).machine).best_fitted_params.chain
    icnn_1_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_1[1]).machine).best_fitted_params.chain
    icnn_2_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_2[1]).machine).best_fitted_params.chain
    icnn_3_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_3[1]).machine).best_fitted_params.chain
    icnn_4_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_4[1]).machine).best_fitted_params.chain
    icnn_5_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_5[1]).machine).best_fitted_params.chain
    icnn_6_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_6[1]).machine).best_fitted_params.chain
    icnn_7_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_7[1]).machine).best_fitted_params.chain
    icnn_8_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_8[1]).machine).best_fitted_params.chain
    icnn_9_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_9[1]).machine).best_fitted_params.chain
    icnn_10_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_10[1]).machine).best_fitted_params.chain
    icnn_11_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_11[1]).machine).best_fitted_params.chain
    icnn_12_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_12[1]).machine).best_fitted_params.chain
    icnn_13_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_13[1]).machine).best_fitted_params.chain
    icnn_14_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_14[1]).machine).best_fitted_params.chain
    icnn_15_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_15[1]).machine).best_fitted_params.chain
    icnn_16_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_16[1]).machine).best_fitted_params.chain
    icnn_17_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_17[1]).machine).best_fitted_params.chain
    icnn_18_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_18[1]).machine).best_fitted_params.chain
    icnn_19_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_19[1]).machine).best_fitted_params.chain
    icnn_20_chain_best_model_chain =
        fitted_params(fitted_params(m_icnn_20[1]).machine).best_fitted_params.chain

    # MAE with train and test data
    mae_Train_icnn_0 = Flux.mae(
        icnn_0_chain_best_model_chain(Matrix(m_icnn_0[2].TrainDataIn)'),
        Matrix(m_icnn_0[2].TrainDataOut)',
    )
    mae_Train_icnn_1 = Flux.mae(
        icnn_1_chain_best_model_chain(Matrix(m_icnn_1[2].TrainDataIn)'),
        Matrix(m_icnn_1[2].TrainDataOut)',
    )
    mae_Train_icnn_2 = Flux.mae(
        icnn_2_chain_best_model_chain(Matrix(m_icnn_2[2].TrainDataIn)'),
        Matrix(m_icnn_2[2].TrainDataOut)',
    )
    mae_Train_icnn_3 = Flux.mae(
        icnn_3_chain_best_model_chain(Matrix(m_icnn_3[2].TrainDataIn)'),
        Matrix(m_icnn_3[2].TrainDataOut)',
    )
    mae_Train_icnn_4 = Flux.mae(
        icnn_4_chain_best_model_chain(Matrix(m_icnn_4[2].TrainDataIn)'),
        Matrix(m_icnn_4[2].TrainDataOut)',
    )
    mae_Train_icnn_5 = Flux.mae(
        icnn_5_chain_best_model_chain(Matrix(m_icnn_5[2].TrainDataIn)'),
        Matrix(m_icnn_5[2].TrainDataOut)',
    )
    mae_Train_icnn_6 = Flux.mae(
        icnn_6_chain_best_model_chain(Matrix(m_icnn_6[2].TrainDataIn)'),
        Matrix(m_icnn_6[2].TrainDataOut)',
    )
    mae_Train_icnn_7 = Flux.mae(
        icnn_7_chain_best_model_chain(Matrix(m_icnn_7[2].TrainDataIn)'),
        Matrix(m_icnn_7[2].TrainDataOut)',
    )
    mae_Train_icnn_8 = Flux.mae(
        icnn_8_chain_best_model_chain(Matrix(m_icnn_8[2].TrainDataIn)'),
        Matrix(m_icnn_8[2].TrainDataOut)',
    )
    mae_Train_icnn_9 = Flux.mae(
        icnn_9_chain_best_model_chain(Matrix(m_icnn_9[2].TrainDataIn)'),
        Matrix(m_icnn_9[2].TrainDataOut)',
    )
    mae_Train_icnn_10 = Flux.mae(
        icnn_10_chain_best_model_chain(Matrix(m_icnn_10[2].TrainDataIn)'),
        Matrix(m_icnn_10[2].TrainDataOut)',
    )
    mae_Train_icnn_11 = Flux.mae(
        icnn_11_chain_best_model_chain(Matrix(m_icnn_11[2].TrainDataIn)'),
        Matrix(m_icnn_11[2].TrainDataOut)',
    )
    mae_Train_icnn_12 = Flux.mae(
        icnn_12_chain_best_model_chain(Matrix(m_icnn_12[2].TrainDataIn)'),
        Matrix(m_icnn_12[2].TrainDataOut)',
    )
    mae_Train_icnn_13 = Flux.mae(
        icnn_13_chain_best_model_chain(Matrix(m_icnn_13[2].TrainDataIn)'),
        Matrix(m_icnn_13[2].TrainDataOut)',
    )
    mae_Train_icnn_14 = Flux.mae(
        icnn_14_chain_best_model_chain(Matrix(m_icnn_14[2].TrainDataIn)'),
        Matrix(m_icnn_14[2].TrainDataOut)',
    )
    mae_Train_icnn_15 = Flux.mae(
        icnn_15_chain_best_model_chain(Matrix(m_icnn_15[2].TrainDataIn)'),
        Matrix(m_icnn_15[2].TrainDataOut)',
    )
    mae_Train_icnn_16 = Flux.mae(
        icnn_16_chain_best_model_chain(Matrix(m_icnn_16[2].TrainDataIn)'),
        Matrix(m_icnn_16[2].TrainDataOut)',
    )
    mae_Train_icnn_17 = Flux.mae(
        icnn_17_chain_best_model_chain(Matrix(m_icnn_17[2].TrainDataIn)'),
        Matrix(m_icnn_17[2].TrainDataOut)',
    )
    mae_Train_icnn_18 = Flux.mae(
        icnn_18_chain_best_model_chain(Matrix(m_icnn_18[2].TrainDataIn)'),
        Matrix(m_icnn_18[2].TrainDataOut)',
    )
    mae_Train_icnn_19 = Flux.mae(
        icnn_19_chain_best_model_chain(Matrix(m_icnn_19[2].TrainDataIn)'),
        Matrix(m_icnn_19[2].TrainDataOut)',
    )
    mae_Train_icnn_20 = Flux.mae(
        icnn_20_chain_best_model_chain(Matrix(m_icnn_20[2].TrainDataIn)'),
        Matrix(m_icnn_20[2].TrainDataOut)',
    )

    mae_Test_icnn_0 = Flux.mae(
        icnn_0_chain_best_model_chain(Matrix(m_icnn_0[2].TestDataIn)'),
        Matrix(m_icnn_0[2].TestDataOut)',
    )
    mae_Test_icnn_1 = Flux.mae(
        icnn_1_chain_best_model_chain(Matrix(m_icnn_1[2].TestDataIn)'),
        Matrix(m_icnn_1[2].TestDataOut)',
    )
    mae_Test_icnn_2 = Flux.mae(
        icnn_2_chain_best_model_chain(Matrix(m_icnn_2[2].TestDataIn)'),
        Matrix(m_icnn_2[2].TestDataOut)',
    )
    mae_Test_icnn_3 = Flux.mae(
        icnn_3_chain_best_model_chain(Matrix(m_icnn_3[2].TestDataIn)'),
        Matrix(m_icnn_3[2].TestDataOut)',
    )
    mae_Test_icnn_4 = Flux.mae(
        icnn_4_chain_best_model_chain(Matrix(m_icnn_4[2].TestDataIn)'),
        Matrix(m_icnn_4[2].TestDataOut)',
    )
    mae_Test_icnn_5 = Flux.mae(
        icnn_5_chain_best_model_chain(Matrix(m_icnn_5[2].TestDataIn)'),
        Matrix(m_icnn_5[2].TestDataOut)',
    )
    mae_Test_icnn_6 = Flux.mae(
        icnn_6_chain_best_model_chain(Matrix(m_icnn_6[2].TestDataIn)'),
        Matrix(m_icnn_6[2].TestDataOut)',
    )
    mae_Test_icnn_7 = Flux.mae(
        icnn_7_chain_best_model_chain(Matrix(m_icnn_7[2].TestDataIn)'),
        Matrix(m_icnn_7[2].TestDataOut)',
    )
    mae_Test_icnn_8 = Flux.mae(
        icnn_8_chain_best_model_chain(Matrix(m_icnn_8[2].TestDataIn)'),
        Matrix(m_icnn_8[2].TestDataOut)',
    )
    mae_Test_icnn_9 = Flux.mae(
        icnn_9_chain_best_model_chain(Matrix(m_icnn_9[2].TestDataIn)'),
        Matrix(m_icnn_9[2].TestDataOut)',
    )
    mae_Test_icnn_10 = Flux.mae(
        icnn_10_chain_best_model_chain(Matrix(m_icnn_10[2].TestDataIn)'),
        Matrix(m_icnn_10[2].TestDataOut)',
    )
    mae_Test_icnn_11 = Flux.mae(
        icnn_11_chain_best_model_chain(Matrix(m_icnn_11[2].TestDataIn)'),
        Matrix(m_icnn_11[2].TestDataOut)',
    )
    mae_Test_icnn_12 = Flux.mae(
        icnn_12_chain_best_model_chain(Matrix(m_icnn_12[2].TestDataIn)'),
        Matrix(m_icnn_12[2].TestDataOut)',
    )
    mae_Test_icnn_13 = Flux.mae(
        icnn_13_chain_best_model_chain(Matrix(m_icnn_13[2].TestDataIn)'),
        Matrix(m_icnn_13[2].TestDataOut)',
    )
    mae_Test_icnn_14 = Flux.mae(
        icnn_14_chain_best_model_chain(Matrix(m_icnn_14[2].TestDataIn)'),
        Matrix(m_icnn_14[2].TestDataOut)',
    )
    mae_Test_icnn_15 = Flux.mae(
        icnn_15_chain_best_model_chain(Matrix(m_icnn_15[2].TestDataIn)'),
        Matrix(m_icnn_15[2].TestDataOut)',
    )
    mae_Test_icnn_16 = Flux.mae(
        icnn_16_chain_best_model_chain(Matrix(m_icnn_16[2].TestDataIn)'),
        Matrix(m_icnn_16[2].TestDataOut)',
    )
    mae_Test_icnn_17 = Flux.mae(
        icnn_17_chain_best_model_chain(Matrix(m_icnn_17[2].TestDataIn)'),
        Matrix(m_icnn_17[2].TestDataOut)',
    )
    mae_Test_icnn_18 = Flux.mae(
        icnn_18_chain_best_model_chain(Matrix(m_icnn_18[2].TestDataIn)'),
        Matrix(m_icnn_18[2].TestDataOut)',
    )
    mae_Test_icnn_19 = Flux.mae(
        icnn_19_chain_best_model_chain(Matrix(m_icnn_19[2].TestDataIn)'),
        Matrix(m_icnn_19[2].TestDataOut)',
    )
    mae_Test_icnn_20 = Flux.mae(
        icnn_20_chain_best_model_chain(Matrix(m_icnn_20[2].TestDataIn)'),
        Matrix(m_icnn_20[2].TestDataOut)',
    )

    println("mae_Train_icnn_0 $mae_Train_icnn_0")
    println("mae_Train_icnn_1 $mae_Train_icnn_1")
    println("mae_Train_icnn_2 $mae_Train_icnn_2")
    println("mae_Train_icnn_3 $mae_Train_icnn_3")
    println("mae_Train_icnn_4 $mae_Train_icnn_4")
    println("mae_Train_icnn_5 $mae_Train_icnn_5")
    println("mae_Train_icnn_6 $mae_Train_icnn_6")
    println("mae_Train_icnn_7 $mae_Train_icnn_7")
    println("mae_Train_icnn_8 $mae_Train_icnn_8")
    println("mae_Train_icnn_9 $mae_Train_icnn_9")
    println("mae_Train_icnn_10 $mae_Train_icnn_10")
    println("mae_Train_icnn_11 $mae_Train_icnn_11")
    println("mae_Train_icnn_12 $mae_Train_icnn_12")
    println("mae_Train_icnn_13 $mae_Train_icnn_13")
    println("mae_Train_icnn_14 $mae_Train_icnn_14")
    println("mae_Train_icnn_15 $mae_Train_icnn_15")
    println("mae_Train_icnn_16 $mae_Train_icnn_16")
    println("mae_Train_icnn_17 $mae_Train_icnn_17")
    println("mae_Train_icnn_18 $mae_Train_icnn_18")
    println("mae_Train_icnn_19 $mae_Train_icnn_19")
    println("mae_Train_icnn_20 $mae_Train_icnn_20")

    println("mae_Test_icnn_0 $mae_Test_icnn_0")
    println("mae_Test_icnn_1 $mae_Test_icnn_1")
    println("mae_Test_icnn_2 $mae_Test_icnn_2")
    println("mae_Test_icnn_3 $mae_Test_icnn_3")
    println("mae_Test_icnn_4 $mae_Test_icnn_4")
    println("mae_Test_icnn_5 $mae_Test_icnn_5")
    println("mae_Test_icnn_6 $mae_Test_icnn_6")
    println("mae_Test_icnn_7 $mae_Test_icnn_7")
    println("mae_Test_icnn_8 $mae_Test_icnn_8")
    println("mae_Test_icnn_9 $mae_Test_icnn_9")
    println("mae_Test_icnn_10 $mae_Test_icnn_10")
    println("mae_Test_icnn_11 $mae_Test_icnn_11")
    println("mae_Test_icnn_12 $mae_Test_icnn_12")
    println("mae_Test_icnn_13 $mae_Test_icnn_13")
    println("mae_Test_icnn_14 $mae_Test_icnn_14")
    println("mae_Test_icnn_15 $mae_Test_icnn_15")
    println("mae_Test_icnn_16 $mae_Test_icnn_16")
    println("mae_Test_icnn_17 $mae_Test_icnn_17")
    println("mae_Test_icnn_18 $mae_Test_icnn_18")
    println("mae_Test_icnn_19 $mae_Test_icnn_19")
    println("mae_Test_icnn_20 $mae_Test_icnn_20")

    @test mae_Train_icnn_0 <= 10
    @test mae_Train_icnn_1 <= 10
    @test mae_Train_icnn_2 <= 10
    @test mae_Train_icnn_3 <= 10
    @test mae_Train_icnn_4 <= 10
    @test mae_Train_icnn_5 <= 10
    @test mae_Train_icnn_6 != NaN
    @test mae_Train_icnn_7 <= 10
    @test mae_Train_icnn_8 <= 10
    @test mae_Train_icnn_9 <= 10
    @test mae_Train_icnn_10 <= 10
    @test mae_Train_icnn_11 <= 10
    @test mae_Train_icnn_12 <= 10
    @test mae_Train_icnn_13 != NaN
    @test mae_Train_icnn_14 <= 10
    @test mae_Train_icnn_15 <= 10
    @test mae_Train_icnn_16 <= 10
    @test mae_Train_icnn_17 <= 10
    @test mae_Train_icnn_18 <= 10
    @test mae_Train_icnn_19 <= 10
    @test mae_Train_icnn_20 != NaN

    @test mae_Test_icnn_0 <= 10
    @test mae_Test_icnn_1 <= 10
    @test mae_Test_icnn_2 <= 10
    @test mae_Test_icnn_3 <= 10
    @test mae_Test_icnn_4 <= 10
    @test mae_Test_icnn_5 <= 10
    @test mae_Test_icnn_6 != NaN
    @test mae_Test_icnn_7 <= 10
    @test mae_Test_icnn_8 <= 10
    @test mae_Test_icnn_9 <= 10
    @test mae_Test_icnn_10 <= 10
    @test mae_Test_icnn_11 <= 10
    @test mae_Test_icnn_12 <= 10
    @test mae_Test_icnn_13 != NaN
    @test mae_Test_icnn_14 <= 10
    @test mae_Test_icnn_15 <= 10
    @test mae_Test_icnn_16 <= 10
    @test mae_Test_icnn_17 <= 10
    @test mae_Test_icnn_18 <= 10
    @test mae_Test_icnn_19 <= 10
    @test mae_Test_icnn_20 != NaN

end

@testset "rbf architecture blackbox identification" begin

    #load data 
    dfout_raw = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:2500, :]
    dfin_raw = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:2500, :]

    #Train models
    max_time = Minute(5) #hours

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin_raw,
        dfout_raw,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )
    
    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    m_rbf_0 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_rbf_1 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_rbf_2 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_rbf_3 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_rbf_4 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_rbf_5 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_rbf_6 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )


    m_rbf_7 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_rbf_8 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_rbf_9 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_rbf_10 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_rbf_11 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_rbf_12 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_rbf_13 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_rbf_14 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_rbf_15 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_rbf_16 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_rbf_17 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_rbf_18 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_rbf_19 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_rbf_20 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "rbf",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    #Get best models
    rbf_0_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_0[1]).machine).best_fitted_params.chain
    rbf_1_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_1[1]).machine).best_fitted_params.chain
    rbf_2_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_2[1]).machine).best_fitted_params.chain
    rbf_3_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_3[1]).machine).best_fitted_params.chain
    rbf_4_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_4[1]).machine).best_fitted_params.chain
    rbf_5_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_5[1]).machine).best_fitted_params.chain
    rbf_6_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_6[1]).machine).best_fitted_params.chain
    rbf_7_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_7[1]).machine).best_fitted_params.chain
    rbf_8_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_8[1]).machine).best_fitted_params.chain
    rbf_9_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_9[1]).machine).best_fitted_params.chain
    rbf_10_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_10[1]).machine).best_fitted_params.chain
    rbf_11_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_11[1]).machine).best_fitted_params.chain
    rbf_12_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_12[1]).machine).best_fitted_params.chain
    rbf_13_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_13[1]).machine).best_fitted_params.chain
    rbf_14_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_14[1]).machine).best_fitted_params.chain
    rbf_15_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_15[1]).machine).best_fitted_params.chain
    rbf_16_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_16[1]).machine).best_fitted_params.chain
    rbf_17_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_17[1]).machine).best_fitted_params.chain
    rbf_18_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_18[1]).machine).best_fitted_params.chain
    rbf_19_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_19[1]).machine).best_fitted_params.chain
    rbf_20_chain_best_model_chain =
        fitted_params(fitted_params(m_rbf_20[1]).machine).best_fitted_params.chain

    # MAE with train and test data
    mae_Train_rbf_0 = Flux.mae(
        rbf_0_chain_best_model_chain(Matrix(m_rbf_0[2].TrainDataIn)'),
        Matrix(m_rbf_0[2].TrainDataOut)',
    )
    mae_Train_rbf_1 = Flux.mae(
        rbf_1_chain_best_model_chain(Matrix(m_rbf_1[2].TrainDataIn)'),
        Matrix(m_rbf_1[2].TrainDataOut)',
    )
    mae_Train_rbf_2 = Flux.mae(
        rbf_2_chain_best_model_chain(Matrix(m_rbf_2[2].TrainDataIn)'),
        Matrix(m_rbf_2[2].TrainDataOut)',
    )
    mae_Train_rbf_3 = Flux.mae(
        rbf_3_chain_best_model_chain(Matrix(m_rbf_3[2].TrainDataIn)'),
        Matrix(m_rbf_3[2].TrainDataOut)',
    )
    mae_Train_rbf_4 = Flux.mae(
        rbf_4_chain_best_model_chain(Matrix(m_rbf_4[2].TrainDataIn)'),
        Matrix(m_rbf_4[2].TrainDataOut)',
    )
    mae_Train_rbf_5 = Flux.mae(
        rbf_5_chain_best_model_chain(Matrix(m_rbf_5[2].TrainDataIn)'),
        Matrix(m_rbf_5[2].TrainDataOut)',
    )
    mae_Train_rbf_6 = Flux.mae(
        rbf_6_chain_best_model_chain(Matrix(m_rbf_6[2].TrainDataIn)'),
        Matrix(m_rbf_6[2].TrainDataOut)',
    )
    mae_Train_rbf_7 = Flux.mae(
        rbf_7_chain_best_model_chain(Matrix(m_rbf_7[2].TrainDataIn)'),
        Matrix(m_rbf_7[2].TrainDataOut)',
    )
    mae_Train_rbf_8 = Flux.mae(
        rbf_8_chain_best_model_chain(Matrix(m_rbf_8[2].TrainDataIn)'),
        Matrix(m_rbf_8[2].TrainDataOut)',
    )
    mae_Train_rbf_9 = Flux.mae(
        rbf_9_chain_best_model_chain(Matrix(m_rbf_9[2].TrainDataIn)'),
        Matrix(m_rbf_9[2].TrainDataOut)',
    )
    mae_Train_rbf_10 = Flux.mae(
        rbf_10_chain_best_model_chain(Matrix(m_rbf_10[2].TrainDataIn)'),
        Matrix(m_rbf_10[2].TrainDataOut)',
    )
    mae_Train_rbf_11 = Flux.mae(
        rbf_11_chain_best_model_chain(Matrix(m_rbf_11[2].TrainDataIn)'),
        Matrix(m_rbf_11[2].TrainDataOut)',
    )
    mae_Train_rbf_12 = Flux.mae(
        rbf_12_chain_best_model_chain(Matrix(m_rbf_12[2].TrainDataIn)'),
        Matrix(m_rbf_12[2].TrainDataOut)',
    )
    mae_Train_rbf_13 = Flux.mae(
        rbf_13_chain_best_model_chain(Matrix(m_rbf_13[2].TrainDataIn)'),
        Matrix(m_rbf_13[2].TrainDataOut)',
    )
    mae_Train_rbf_14 = Flux.mae(
        rbf_14_chain_best_model_chain(Matrix(m_rbf_14[2].TrainDataIn)'),
        Matrix(m_rbf_14[2].TrainDataOut)',
    )
    mae_Train_rbf_15 = Flux.mae(
        rbf_15_chain_best_model_chain(Matrix(m_rbf_15[2].TrainDataIn)'),
        Matrix(m_rbf_15[2].TrainDataOut)',
    )
    mae_Train_rbf_16 = Flux.mae(
        rbf_16_chain_best_model_chain(Matrix(m_rbf_16[2].TrainDataIn)'),
        Matrix(m_rbf_16[2].TrainDataOut)',
    )
    mae_Train_rbf_17 = Flux.mae(
        rbf_17_chain_best_model_chain(Matrix(m_rbf_17[2].TrainDataIn)'),
        Matrix(m_rbf_17[2].TrainDataOut)',
    )
    mae_Train_rbf_18 = Flux.mae(
        rbf_18_chain_best_model_chain(Matrix(m_rbf_18[2].TrainDataIn)'),
        Matrix(m_rbf_18[2].TrainDataOut)',
    )
    mae_Train_rbf_19 = Flux.mae(
        rbf_19_chain_best_model_chain(Matrix(m_rbf_19[2].TrainDataIn)'),
        Matrix(m_rbf_19[2].TrainDataOut)',
    )
    mae_Train_rbf_20 = Flux.mae(
        rbf_20_chain_best_model_chain(Matrix(m_rbf_20[2].TrainDataIn)'),
        Matrix(m_rbf_20[2].TrainDataOut)',
    )

    mae_Test_rbf_0 = Flux.mae(
        rbf_0_chain_best_model_chain(Matrix(m_rbf_0[2].TestDataIn)'),
        Matrix(m_rbf_0[2].TestDataOut)',
    )
    mae_Test_rbf_1 = Flux.mae(
        rbf_1_chain_best_model_chain(Matrix(m_rbf_1[2].TestDataIn)'),
        Matrix(m_rbf_1[2].TestDataOut)',
    )
    mae_Test_rbf_2 = Flux.mae(
        rbf_2_chain_best_model_chain(Matrix(m_rbf_2[2].TestDataIn)'),
        Matrix(m_rbf_2[2].TestDataOut)',
    )
    mae_Test_rbf_3 = Flux.mae(
        rbf_3_chain_best_model_chain(Matrix(m_rbf_3[2].TestDataIn)'),
        Matrix(m_rbf_3[2].TestDataOut)',
    )
    mae_Test_rbf_4 = Flux.mae(
        rbf_4_chain_best_model_chain(Matrix(m_rbf_4[2].TestDataIn)'),
        Matrix(m_rbf_4[2].TestDataOut)',
    )
    mae_Test_rbf_5 = Flux.mae(
        rbf_5_chain_best_model_chain(Matrix(m_rbf_5[2].TestDataIn)'),
        Matrix(m_rbf_5[2].TestDataOut)',
    )
    mae_Test_rbf_6 = Flux.mae(
        rbf_6_chain_best_model_chain(Matrix(m_rbf_6[2].TestDataIn)'),
        Matrix(m_rbf_6[2].TestDataOut)',
    )
    mae_Test_rbf_7 = Flux.mae(
        rbf_7_chain_best_model_chain(Matrix(m_rbf_7[2].TestDataIn)'),
        Matrix(m_rbf_7[2].TestDataOut)',
    )
    mae_Test_rbf_8 = Flux.mae(
        rbf_8_chain_best_model_chain(Matrix(m_rbf_8[2].TestDataIn)'),
        Matrix(m_rbf_8[2].TestDataOut)',
    )
    mae_Test_rbf_9 = Flux.mae(
        rbf_9_chain_best_model_chain(Matrix(m_rbf_9[2].TestDataIn)'),
        Matrix(m_rbf_9[2].TestDataOut)',
    )
    mae_Test_rbf_10 = Flux.mae(
        rbf_10_chain_best_model_chain(Matrix(m_rbf_10[2].TestDataIn)'),
        Matrix(m_rbf_10[2].TestDataOut)',
    )
    mae_Test_rbf_11 = Flux.mae(
        rbf_11_chain_best_model_chain(Matrix(m_rbf_11[2].TestDataIn)'),
        Matrix(m_rbf_11[2].TestDataOut)',
    )
    mae_Test_rbf_12 = Flux.mae(
        rbf_12_chain_best_model_chain(Matrix(m_rbf_12[2].TestDataIn)'),
        Matrix(m_rbf_12[2].TestDataOut)',
    )
    mae_Test_rbf_13 = Flux.mae(
        rbf_13_chain_best_model_chain(Matrix(m_rbf_13[2].TestDataIn)'),
        Matrix(m_rbf_13[2].TestDataOut)',
    )
    mae_Test_rbf_14 = Flux.mae(
        rbf_14_chain_best_model_chain(Matrix(m_rbf_14[2].TestDataIn)'),
        Matrix(m_rbf_14[2].TestDataOut)',
    )
    mae_Test_rbf_15 = Flux.mae(
        rbf_15_chain_best_model_chain(Matrix(m_rbf_15[2].TestDataIn)'),
        Matrix(m_rbf_15[2].TestDataOut)',
    )
    mae_Test_rbf_16 = Flux.mae(
        rbf_16_chain_best_model_chain(Matrix(m_rbf_16[2].TestDataIn)'),
        Matrix(m_rbf_16[2].TestDataOut)',
    )
    mae_Test_rbf_17 = Flux.mae(
        rbf_17_chain_best_model_chain(Matrix(m_rbf_17[2].TestDataIn)'),
        Matrix(m_rbf_17[2].TestDataOut)',
    )
    mae_Test_rbf_18 = Flux.mae(
        rbf_18_chain_best_model_chain(Matrix(m_rbf_18[2].TestDataIn)'),
        Matrix(m_rbf_18[2].TestDataOut)',
    )
    mae_Test_rbf_19 = Flux.mae(
        rbf_19_chain_best_model_chain(Matrix(m_rbf_19[2].TestDataIn)'),
        Matrix(m_rbf_19[2].TestDataOut)',
    )
    mae_Test_rbf_20 = Flux.mae(
        rbf_20_chain_best_model_chain(Matrix(m_rbf_20[2].TestDataIn)'),
        Matrix(m_rbf_20[2].TestDataOut)',
    )

    println("mae_Train_rbf_0 $mae_Train_rbf_0")
    println("mae_Train_rbf_1 $mae_Train_rbf_1")
    println("mae_Train_rbf_2 $mae_Train_rbf_2")
    println("mae_Train_rbf_3 $mae_Train_rbf_3")
    println("mae_Train_rbf_4 $mae_Train_rbf_4")
    println("mae_Train_rbf_5 $mae_Train_rbf_5")
    println("mae_Train_rbf_6 $mae_Train_rbf_6")
    println("mae_Train_rbf_7 $mae_Train_rbf_7")
    println("mae_Train_rbf_8 $mae_Train_rbf_8")
    println("mae_Train_rbf_9 $mae_Train_rbf_9")
    println("mae_Train_rbf_10 $mae_Train_rbf_10")
    println("mae_Train_rbf_11 $mae_Train_rbf_11")
    println("mae_Train_rbf_12 $mae_Train_rbf_12")
    println("mae_Train_rbf_13 $mae_Train_rbf_13")
    println("mae_Train_rbf_14 $mae_Train_rbf_14")
    println("mae_Train_rbf_15 $mae_Train_rbf_15")
    println("mae_Train_rbf_16 $mae_Train_rbf_16")
    println("mae_Train_rbf_17 $mae_Train_rbf_17")
    println("mae_Train_rbf_18 $mae_Train_rbf_18")
    println("mae_Train_rbf_19 $mae_Train_rbf_19")
    println("mae_Train_rbf_20 $mae_Train_rbf_20")

    println("mae_Test_rbf_0 $mae_Test_rbf_0")
    println("mae_Test_rbf_1 $mae_Test_rbf_1")
    println("mae_Test_rbf_2 $mae_Test_rbf_2")
    println("mae_Test_rbf_3 $mae_Test_rbf_3")
    println("mae_Test_rbf_4 $mae_Test_rbf_4")
    println("mae_Test_rbf_5 $mae_Test_rbf_5")
    println("mae_Test_rbf_6 $mae_Test_rbf_6")
    println("mae_Test_rbf_7 $mae_Test_rbf_7")
    println("mae_Test_rbf_8 $mae_Test_rbf_8")
    println("mae_Test_rbf_9 $mae_Test_rbf_9")
    println("mae_Test_rbf_10 $mae_Test_rbf_10")
    println("mae_Test_rbf_11 $mae_Test_rbf_11")
    println("mae_Test_rbf_12 $mae_Test_rbf_12")
    println("mae_Test_rbf_13 $mae_Test_rbf_13")
    println("mae_Test_rbf_14 $mae_Test_rbf_14")
    println("mae_Test_rbf_15 $mae_Test_rbf_15")
    println("mae_Test_rbf_16 $mae_Test_rbf_16")
    println("mae_Test_rbf_17 $mae_Test_rbf_17")
    println("mae_Test_rbf_18 $mae_Test_rbf_18")
    println("mae_Test_rbf_19 $mae_Test_rbf_19")
    println("mae_Test_rbf_20 $mae_Test_rbf_20")

    @test mae_Train_rbf_0 <= 10
    @test mae_Train_rbf_1 <= 10
    @test mae_Train_rbf_2 <= 10
    @test mae_Train_rbf_3 <= 10
    @test mae_Train_rbf_4 <= 10
    @test mae_Train_rbf_5 <= 10
    @test mae_Train_rbf_6 != NaN
    @test mae_Train_rbf_7 <= 10
    @test mae_Train_rbf_8 <= 10
    @test mae_Train_rbf_9 <= 10
    @test mae_Train_rbf_10 <= 10
    @test mae_Train_rbf_11 <= 10
    @test mae_Train_rbf_12 <= 10
    @test mae_Train_rbf_13 != NaN
    @test mae_Train_rbf_14 <= 10
    @test mae_Train_rbf_15 <= 10
    @test mae_Train_rbf_16 <= 10
    @test mae_Train_rbf_17 <= 10
    @test mae_Train_rbf_18 <= 10
    @test mae_Train_rbf_19 <= 10
    @test mae_Train_rbf_20 != NaN

    @test mae_Test_rbf_0 <= 10
    @test mae_Test_rbf_1 <= 10
    @test mae_Test_rbf_2 <= 10
    @test mae_Test_rbf_3 <= 10
    @test mae_Test_rbf_4 <= 10
    @test mae_Test_rbf_5 <= 10
    @test mae_Test_rbf_6 != NaN
    @test mae_Test_rbf_7 <= 10
    @test mae_Test_rbf_8 <= 10
    @test mae_Test_rbf_9 <= 10
    @test mae_Test_rbf_10 <= 10
    @test mae_Test_rbf_11 <= 10
    @test mae_Test_rbf_12 <= 10
    @test mae_Test_rbf_13 != NaN
    @test mae_Test_rbf_14 <= 10
    @test mae_Test_rbf_15 <= 10
    @test mae_Test_rbf_16 <= 10
    @test mae_Test_rbf_17 <= 10
    @test mae_Test_rbf_18 <= 10
    @test mae_Test_rbf_19 <= 10
    @test mae_Test_rbf_20 != NaN

end

@testset "resnet architecture blackbox identification" begin

    #load data 
    dfout_raw = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:2500, :]
    dfin_raw = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:2500, :]

    #Train models
    max_time = Minute(5) #hours

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin_raw,
        dfout_raw,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )
    
    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    m_resnet_0 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_resnet_1 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_resnet_2 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_resnet_3 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_resnet_4 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_resnet_5 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_resnet_6 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )


    m_resnet_7 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_resnet_8 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_resnet_9 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_resnet_10 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_resnet_11 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_resnet_12 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_resnet_13 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_resnet_14 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_resnet_15 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_resnet_16 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_resnet_17 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_resnet_18 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_resnet_19 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_resnet_20 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "resnet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    #Get best models
    resnet_0_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_0[1]).machine).best_fitted_params.chain
    resnet_1_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_1[1]).machine).best_fitted_params.chain
    resnet_2_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_2[1]).machine).best_fitted_params.chain
    resnet_3_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_3[1]).machine).best_fitted_params.chain
    resnet_4_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_4[1]).machine).best_fitted_params.chain
    resnet_5_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_5[1]).machine).best_fitted_params.chain
    resnet_6_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_6[1]).machine).best_fitted_params.chain
    resnet_7_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_7[1]).machine).best_fitted_params.chain
    resnet_8_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_8[1]).machine).best_fitted_params.chain
    resnet_9_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_9[1]).machine).best_fitted_params.chain
    resnet_10_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_10[1]).machine).best_fitted_params.chain
    resnet_11_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_11[1]).machine).best_fitted_params.chain
    resnet_12_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_12[1]).machine).best_fitted_params.chain
    resnet_13_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_13[1]).machine).best_fitted_params.chain
    resnet_14_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_14[1]).machine).best_fitted_params.chain
    resnet_15_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_15[1]).machine).best_fitted_params.chain
    resnet_16_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_16[1]).machine).best_fitted_params.chain
    resnet_17_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_17[1]).machine).best_fitted_params.chain
    resnet_18_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_18[1]).machine).best_fitted_params.chain
    resnet_19_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_19[1]).machine).best_fitted_params.chain
    resnet_20_chain_best_model_chain =
        fitted_params(fitted_params(m_resnet_20[1]).machine).best_fitted_params.chain

    # MAE with train and test data
    mae_Train_resnet_0 = Flux.mae(
        resnet_0_chain_best_model_chain(Matrix(m_resnet_0[2].TrainDataIn)'),
        Matrix(m_resnet_0[2].TrainDataOut)',
    )
    mae_Train_resnet_1 = Flux.mae(
        resnet_1_chain_best_model_chain(Matrix(m_resnet_1[2].TrainDataIn)'),
        Matrix(m_resnet_1[2].TrainDataOut)',
    )
    mae_Train_resnet_2 = Flux.mae(
        resnet_2_chain_best_model_chain(Matrix(m_resnet_2[2].TrainDataIn)'),
        Matrix(m_resnet_2[2].TrainDataOut)',
    )
    mae_Train_resnet_3 = Flux.mae(
        resnet_3_chain_best_model_chain(Matrix(m_resnet_3[2].TrainDataIn)'),
        Matrix(m_resnet_3[2].TrainDataOut)',
    )
    mae_Train_resnet_4 = Flux.mae(
        resnet_4_chain_best_model_chain(Matrix(m_resnet_4[2].TrainDataIn)'),
        Matrix(m_resnet_4[2].TrainDataOut)',
    )
    mae_Train_resnet_5 = Flux.mae(
        resnet_5_chain_best_model_chain(Matrix(m_resnet_5[2].TrainDataIn)'),
        Matrix(m_resnet_5[2].TrainDataOut)',
    )
    mae_Train_resnet_6 = Flux.mae(
        resnet_6_chain_best_model_chain(Matrix(m_resnet_6[2].TrainDataIn)'),
        Matrix(m_resnet_6[2].TrainDataOut)',
    )
    mae_Train_resnet_7 = Flux.mae(
        resnet_7_chain_best_model_chain(Matrix(m_resnet_7[2].TrainDataIn)'),
        Matrix(m_resnet_7[2].TrainDataOut)',
    )
    mae_Train_resnet_8 = Flux.mae(
        resnet_8_chain_best_model_chain(Matrix(m_resnet_8[2].TrainDataIn)'),
        Matrix(m_resnet_8[2].TrainDataOut)',
    )
    mae_Train_resnet_9 = Flux.mae(
        resnet_9_chain_best_model_chain(Matrix(m_resnet_9[2].TrainDataIn)'),
        Matrix(m_resnet_9[2].TrainDataOut)',
    )
    mae_Train_resnet_10 = Flux.mae(
        resnet_10_chain_best_model_chain(Matrix(m_resnet_10[2].TrainDataIn)'),
        Matrix(m_resnet_10[2].TrainDataOut)',
    )
    mae_Train_resnet_11 = Flux.mae(
        resnet_11_chain_best_model_chain(Matrix(m_resnet_11[2].TrainDataIn)'),
        Matrix(m_resnet_11[2].TrainDataOut)',
    )
    mae_Train_resnet_12 = Flux.mae(
        resnet_12_chain_best_model_chain(Matrix(m_resnet_12[2].TrainDataIn)'),
        Matrix(m_resnet_12[2].TrainDataOut)',
    )
    mae_Train_resnet_13 = Flux.mae(
        resnet_13_chain_best_model_chain(Matrix(m_resnet_13[2].TrainDataIn)'),
        Matrix(m_resnet_13[2].TrainDataOut)',
    )
    mae_Train_resnet_14 = Flux.mae(
        resnet_14_chain_best_model_chain(Matrix(m_resnet_14[2].TrainDataIn)'),
        Matrix(m_resnet_14[2].TrainDataOut)',
    )
    mae_Train_resnet_15 = Flux.mae(
        resnet_15_chain_best_model_chain(Matrix(m_resnet_15[2].TrainDataIn)'),
        Matrix(m_resnet_15[2].TrainDataOut)',
    )
    mae_Train_resnet_16 = Flux.mae(
        resnet_16_chain_best_model_chain(Matrix(m_resnet_16[2].TrainDataIn)'),
        Matrix(m_resnet_16[2].TrainDataOut)',
    )
    mae_Train_resnet_17 = Flux.mae(
        resnet_17_chain_best_model_chain(Matrix(m_resnet_17[2].TrainDataIn)'),
        Matrix(m_resnet_17[2].TrainDataOut)',
    )
    mae_Train_resnet_18 = Flux.mae(
        resnet_18_chain_best_model_chain(Matrix(m_resnet_18[2].TrainDataIn)'),
        Matrix(m_resnet_18[2].TrainDataOut)',
    )
    mae_Train_resnet_19 = Flux.mae(
        resnet_19_chain_best_model_chain(Matrix(m_resnet_19[2].TrainDataIn)'),
        Matrix(m_resnet_19[2].TrainDataOut)',
    )
    mae_Train_resnet_20 = Flux.mae(
        resnet_20_chain_best_model_chain(Matrix(m_resnet_20[2].TrainDataIn)'),
        Matrix(m_resnet_20[2].TrainDataOut)',
    )

    mae_Test_resnet_0 = Flux.mae(
        resnet_0_chain_best_model_chain(Matrix(m_resnet_0[2].TestDataIn)'),
        Matrix(m_resnet_0[2].TestDataOut)',
    )
    mae_Test_resnet_1 = Flux.mae(
        resnet_1_chain_best_model_chain(Matrix(m_resnet_1[2].TestDataIn)'),
        Matrix(m_resnet_1[2].TestDataOut)',
    )
    mae_Test_resnet_2 = Flux.mae(
        resnet_2_chain_best_model_chain(Matrix(m_resnet_2[2].TestDataIn)'),
        Matrix(m_resnet_2[2].TestDataOut)',
    )
    mae_Test_resnet_3 = Flux.mae(
        resnet_3_chain_best_model_chain(Matrix(m_resnet_3[2].TestDataIn)'),
        Matrix(m_resnet_3[2].TestDataOut)',
    )
    mae_Test_resnet_4 = Flux.mae(
        resnet_4_chain_best_model_chain(Matrix(m_resnet_4[2].TestDataIn)'),
        Matrix(m_resnet_4[2].TestDataOut)',
    )
    mae_Test_resnet_5 = Flux.mae(
        resnet_5_chain_best_model_chain(Matrix(m_resnet_5[2].TestDataIn)'),
        Matrix(m_resnet_5[2].TestDataOut)',
    )
    mae_Test_resnet_6 = Flux.mae(
        resnet_6_chain_best_model_chain(Matrix(m_resnet_6[2].TestDataIn)'),
        Matrix(m_resnet_6[2].TestDataOut)',
    )
    mae_Test_resnet_7 = Flux.mae(
        resnet_7_chain_best_model_chain(Matrix(m_resnet_7[2].TestDataIn)'),
        Matrix(m_resnet_7[2].TestDataOut)',
    )
    mae_Test_resnet_8 = Flux.mae(
        resnet_8_chain_best_model_chain(Matrix(m_resnet_8[2].TestDataIn)'),
        Matrix(m_resnet_8[2].TestDataOut)',
    )
    mae_Test_resnet_9 = Flux.mae(
        resnet_9_chain_best_model_chain(Matrix(m_resnet_9[2].TestDataIn)'),
        Matrix(m_resnet_9[2].TestDataOut)',
    )
    mae_Test_resnet_10 = Flux.mae(
        resnet_10_chain_best_model_chain(Matrix(m_resnet_10[2].TestDataIn)'),
        Matrix(m_resnet_10[2].TestDataOut)',
    )
    mae_Test_resnet_11 = Flux.mae(
        resnet_11_chain_best_model_chain(Matrix(m_resnet_11[2].TestDataIn)'),
        Matrix(m_resnet_11[2].TestDataOut)',
    )
    mae_Test_resnet_12 = Flux.mae(
        resnet_12_chain_best_model_chain(Matrix(m_resnet_12[2].TestDataIn)'),
        Matrix(m_resnet_12[2].TestDataOut)',
    )
    mae_Test_resnet_13 = Flux.mae(
        resnet_13_chain_best_model_chain(Matrix(m_resnet_13[2].TestDataIn)'),
        Matrix(m_resnet_13[2].TestDataOut)',
    )
    mae_Test_resnet_14 = Flux.mae(
        resnet_14_chain_best_model_chain(Matrix(m_resnet_14[2].TestDataIn)'),
        Matrix(m_resnet_14[2].TestDataOut)',
    )
    mae_Test_resnet_15 = Flux.mae(
        resnet_15_chain_best_model_chain(Matrix(m_resnet_15[2].TestDataIn)'),
        Matrix(m_resnet_15[2].TestDataOut)',
    )
    mae_Test_resnet_16 = Flux.mae(
        resnet_16_chain_best_model_chain(Matrix(m_resnet_16[2].TestDataIn)'),
        Matrix(m_resnet_16[2].TestDataOut)',
    )
    mae_Test_resnet_17 = Flux.mae(
        resnet_17_chain_best_model_chain(Matrix(m_resnet_17[2].TestDataIn)'),
        Matrix(m_resnet_17[2].TestDataOut)',
    )
    mae_Test_resnet_18 = Flux.mae(
        resnet_18_chain_best_model_chain(Matrix(m_resnet_18[2].TestDataIn)'),
        Matrix(m_resnet_18[2].TestDataOut)',
    )
    mae_Test_resnet_19 = Flux.mae(
        resnet_19_chain_best_model_chain(Matrix(m_resnet_19[2].TestDataIn)'),
        Matrix(m_resnet_19[2].TestDataOut)',
    )
    mae_Test_resnet_20 = Flux.mae(
        resnet_20_chain_best_model_chain(Matrix(m_resnet_20[2].TestDataIn)'),
        Matrix(m_resnet_20[2].TestDataOut)',
    )

    println("mae_Train_resnet_0 $mae_Train_resnet_0")
    println("mae_Train_resnet_1 $mae_Train_resnet_1")
    println("mae_Train_resnet_2 $mae_Train_resnet_2")
    println("mae_Train_resnet_3 $mae_Train_resnet_3")
    println("mae_Train_resnet_4 $mae_Train_resnet_4")
    println("mae_Train_resnet_5 $mae_Train_resnet_5")
    println("mae_Train_resnet_6 $mae_Train_resnet_6")
    println("mae_Train_resnet_7 $mae_Train_resnet_7")
    println("mae_Train_resnet_8 $mae_Train_resnet_8")
    println("mae_Train_resnet_9 $mae_Train_resnet_9")
    println("mae_Train_resnet_10 $mae_Train_resnet_10")
    println("mae_Train_resnet_11 $mae_Train_resnet_11")
    println("mae_Train_resnet_12 $mae_Train_resnet_12")
    println("mae_Train_resnet_13 $mae_Train_resnet_13")
    println("mae_Train_resnet_14 $mae_Train_resnet_14")
    println("mae_Train_resnet_15 $mae_Train_resnet_15")
    println("mae_Train_resnet_16 $mae_Train_resnet_16")
    println("mae_Train_resnet_17 $mae_Train_resnet_17")
    println("mae_Train_resnet_18 $mae_Train_resnet_18")
    println("mae_Train_resnet_19 $mae_Train_resnet_19")
    println("mae_Train_resnet_20 $mae_Train_resnet_20")

    println("mae_Test_resnet_0 $mae_Test_resnet_0")
    println("mae_Test_resnet_1 $mae_Test_resnet_1")
    println("mae_Test_resnet_2 $mae_Test_resnet_2")
    println("mae_Test_resnet_3 $mae_Test_resnet_3")
    println("mae_Test_resnet_4 $mae_Test_resnet_4")
    println("mae_Test_resnet_5 $mae_Test_resnet_5")
    println("mae_Test_resnet_6 $mae_Test_resnet_6")
    println("mae_Test_resnet_7 $mae_Test_resnet_7")
    println("mae_Test_resnet_8 $mae_Test_resnet_8")
    println("mae_Test_resnet_9 $mae_Test_resnet_9")
    println("mae_Test_resnet_10 $mae_Test_resnet_10")
    println("mae_Test_resnet_11 $mae_Test_resnet_11")
    println("mae_Test_resnet_12 $mae_Test_resnet_12")
    println("mae_Test_resnet_13 $mae_Test_resnet_13")
    println("mae_Test_resnet_14 $mae_Test_resnet_14")
    println("mae_Test_resnet_15 $mae_Test_resnet_15")
    println("mae_Test_resnet_16 $mae_Test_resnet_16")
    println("mae_Test_resnet_17 $mae_Test_resnet_17")
    println("mae_Test_resnet_18 $mae_Test_resnet_18")
    println("mae_Test_resnet_19 $mae_Test_resnet_19")
    println("mae_Test_resnet_20 $mae_Test_resnet_20")

    @test mae_Train_resnet_0 <= 10
    @test mae_Train_resnet_1 <= 10
    @test mae_Train_resnet_2 <= 10
    @test mae_Train_resnet_3 <= 10
    @test mae_Train_resnet_4 <= 10
    @test mae_Train_resnet_5 <= 10
    @test mae_Train_resnet_6 != NaN
    @test mae_Train_resnet_7 <= 10
    @test mae_Train_resnet_8 <= 10
    @test mae_Train_resnet_9 <= 10
    @test mae_Train_resnet_10 <= 10
    @test mae_Train_resnet_11 <= 10
    @test mae_Train_resnet_12 <= 10
    @test mae_Train_resnet_13 != NaN
    @test mae_Train_resnet_14 <= 10
    @test mae_Train_resnet_15 <= 10
    @test mae_Train_resnet_16 <= 10
    @test mae_Train_resnet_17 <= 10
    @test mae_Train_resnet_18 <= 10
    @test mae_Train_resnet_19 <= 10
    @test mae_Train_resnet_20 != NaN

    @test mae_Test_resnet_0 <= 10
    @test mae_Test_resnet_1 <= 10
    @test mae_Test_resnet_2 <= 10
    @test mae_Test_resnet_3 <= 10
    @test mae_Test_resnet_4 <= 10
    @test mae_Test_resnet_5 <= 10
    @test mae_Test_resnet_6 != NaN
    @test mae_Test_resnet_7 <= 10
    @test mae_Test_resnet_8 <= 10
    @test mae_Test_resnet_9 <= 10
    @test mae_Test_resnet_10 <= 10
    @test mae_Test_resnet_11 <= 10
    @test mae_Test_resnet_12 <= 10
    @test mae_Test_resnet_13 != NaN
    @test mae_Test_resnet_14 <= 10
    @test mae_Test_resnet_15 <= 10
    @test mae_Test_resnet_16 <= 10
    @test mae_Test_resnet_17 <= 10
    @test mae_Test_resnet_18 <= 10
    @test mae_Test_resnet_19 <= 10
    @test mae_Test_resnet_20 != NaN

end

@testset "polynet architecture blackbox identification" begin

    #load data 
    dfout_raw = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:2500, :]
    dfin_raw = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:2500, :]

    #Train models
    max_time = Minute(5) #hours

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin_raw,
        dfout_raw,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )
    
    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    m_polynet_0 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_polynet_1 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_polynet_2 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_polynet_3 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_polynet_4 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_polynet_5 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_polynet_6 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )


    m_polynet_7 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_polynet_8 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_polynet_9 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_polynet_10 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_polynet_11 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_polynet_12 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_polynet_13 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_polynet_14 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_polynet_15 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_polynet_16 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_polynet_17 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_polynet_18 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_polynet_19 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_polynet_20 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "polynet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    #Get best models
    polynet_0_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_0[1]).machine).best_fitted_params.chain
    polynet_1_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_1[1]).machine).best_fitted_params.chain
    polynet_2_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_2[1]).machine).best_fitted_params.chain
    polynet_3_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_3[1]).machine).best_fitted_params.chain
    polynet_4_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_4[1]).machine).best_fitted_params.chain
    polynet_5_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_5[1]).machine).best_fitted_params.chain
    polynet_6_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_6[1]).machine).best_fitted_params.chain
    polynet_7_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_7[1]).machine).best_fitted_params.chain
    polynet_8_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_8[1]).machine).best_fitted_params.chain
    polynet_9_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_9[1]).machine).best_fitted_params.chain
    polynet_10_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_10[1]).machine).best_fitted_params.chain
    polynet_11_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_11[1]).machine).best_fitted_params.chain
    polynet_12_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_12[1]).machine).best_fitted_params.chain
    polynet_13_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_13[1]).machine).best_fitted_params.chain
    polynet_14_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_14[1]).machine).best_fitted_params.chain
    polynet_15_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_15[1]).machine).best_fitted_params.chain
    polynet_16_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_16[1]).machine).best_fitted_params.chain
    polynet_17_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_17[1]).machine).best_fitted_params.chain
    polynet_18_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_18[1]).machine).best_fitted_params.chain
    polynet_19_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_19[1]).machine).best_fitted_params.chain
    polynet_20_chain_best_model_chain =
        fitted_params(fitted_params(m_polynet_20[1]).machine).best_fitted_params.chain

    # MAE with train and test data
    mae_Train_polynet_0 = Flux.mae(
        polynet_0_chain_best_model_chain(Matrix(m_polynet_0[2].TrainDataIn)'),
        Matrix(m_polynet_0[2].TrainDataOut)',
    )
    mae_Train_polynet_1 = Flux.mae(
        polynet_1_chain_best_model_chain(Matrix(m_polynet_1[2].TrainDataIn)'),
        Matrix(m_polynet_1[2].TrainDataOut)',
    )
    mae_Train_polynet_2 = Flux.mae(
        polynet_2_chain_best_model_chain(Matrix(m_polynet_2[2].TrainDataIn)'),
        Matrix(m_polynet_2[2].TrainDataOut)',
    )
    mae_Train_polynet_3 = Flux.mae(
        polynet_3_chain_best_model_chain(Matrix(m_polynet_3[2].TrainDataIn)'),
        Matrix(m_polynet_3[2].TrainDataOut)',
    )
    mae_Train_polynet_4 = Flux.mae(
        polynet_4_chain_best_model_chain(Matrix(m_polynet_4[2].TrainDataIn)'),
        Matrix(m_polynet_4[2].TrainDataOut)',
    )
    mae_Train_polynet_5 = Flux.mae(
        polynet_5_chain_best_model_chain(Matrix(m_polynet_5[2].TrainDataIn)'),
        Matrix(m_polynet_5[2].TrainDataOut)',
    )
    mae_Train_polynet_6 = Flux.mae(
        polynet_6_chain_best_model_chain(Matrix(m_polynet_6[2].TrainDataIn)'),
        Matrix(m_polynet_6[2].TrainDataOut)',
    )
    mae_Train_polynet_7 = Flux.mae(
        polynet_7_chain_best_model_chain(Matrix(m_polynet_7[2].TrainDataIn)'),
        Matrix(m_polynet_7[2].TrainDataOut)',
    )
    mae_Train_polynet_8 = Flux.mae(
        polynet_8_chain_best_model_chain(Matrix(m_polynet_8[2].TrainDataIn)'),
        Matrix(m_polynet_8[2].TrainDataOut)',
    )
    mae_Train_polynet_9 = Flux.mae(
        polynet_9_chain_best_model_chain(Matrix(m_polynet_9[2].TrainDataIn)'),
        Matrix(m_polynet_9[2].TrainDataOut)',
    )
    mae_Train_polynet_10 = Flux.mae(
        polynet_10_chain_best_model_chain(Matrix(m_polynet_10[2].TrainDataIn)'),
        Matrix(m_polynet_10[2].TrainDataOut)',
    )
    mae_Train_polynet_11 = Flux.mae(
        polynet_11_chain_best_model_chain(Matrix(m_polynet_11[2].TrainDataIn)'),
        Matrix(m_polynet_11[2].TrainDataOut)',
    )
    mae_Train_polynet_12 = Flux.mae(
        polynet_12_chain_best_model_chain(Matrix(m_polynet_12[2].TrainDataIn)'),
        Matrix(m_polynet_12[2].TrainDataOut)',
    )
    mae_Train_polynet_13 = Flux.mae(
        polynet_13_chain_best_model_chain(Matrix(m_polynet_13[2].TrainDataIn)'),
        Matrix(m_polynet_13[2].TrainDataOut)',
    )
    mae_Train_polynet_14 = Flux.mae(
        polynet_14_chain_best_model_chain(Matrix(m_polynet_14[2].TrainDataIn)'),
        Matrix(m_polynet_14[2].TrainDataOut)',
    )
    mae_Train_polynet_15 = Flux.mae(
        polynet_15_chain_best_model_chain(Matrix(m_polynet_15[2].TrainDataIn)'),
        Matrix(m_polynet_15[2].TrainDataOut)',
    )
    mae_Train_polynet_16 = Flux.mae(
        polynet_16_chain_best_model_chain(Matrix(m_polynet_16[2].TrainDataIn)'),
        Matrix(m_polynet_16[2].TrainDataOut)',
    )
    mae_Train_polynet_17 = Flux.mae(
        polynet_17_chain_best_model_chain(Matrix(m_polynet_17[2].TrainDataIn)'),
        Matrix(m_polynet_17[2].TrainDataOut)',
    )
    mae_Train_polynet_18 = Flux.mae(
        polynet_18_chain_best_model_chain(Matrix(m_polynet_18[2].TrainDataIn)'),
        Matrix(m_polynet_18[2].TrainDataOut)',
    )
    mae_Train_polynet_19 = Flux.mae(
        polynet_19_chain_best_model_chain(Matrix(m_polynet_19[2].TrainDataIn)'),
        Matrix(m_polynet_19[2].TrainDataOut)',
    )
    mae_Train_polynet_20 = Flux.mae(
        polynet_20_chain_best_model_chain(Matrix(m_polynet_20[2].TrainDataIn)'),
        Matrix(m_polynet_20[2].TrainDataOut)',
    )

    mae_Test_polynet_0 = Flux.mae(
        polynet_0_chain_best_model_chain(Matrix(m_polynet_0[2].TestDataIn)'),
        Matrix(m_polynet_0[2].TestDataOut)',
    )
    mae_Test_polynet_1 = Flux.mae(
        polynet_1_chain_best_model_chain(Matrix(m_polynet_1[2].TestDataIn)'),
        Matrix(m_polynet_1[2].TestDataOut)',
    )
    mae_Test_polynet_2 = Flux.mae(
        polynet_2_chain_best_model_chain(Matrix(m_polynet_2[2].TestDataIn)'),
        Matrix(m_polynet_2[2].TestDataOut)',
    )
    mae_Test_polynet_3 = Flux.mae(
        polynet_3_chain_best_model_chain(Matrix(m_polynet_3[2].TestDataIn)'),
        Matrix(m_polynet_3[2].TestDataOut)',
    )
    mae_Test_polynet_4 = Flux.mae(
        polynet_4_chain_best_model_chain(Matrix(m_polynet_4[2].TestDataIn)'),
        Matrix(m_polynet_4[2].TestDataOut)',
    )
    mae_Test_polynet_5 = Flux.mae(
        polynet_5_chain_best_model_chain(Matrix(m_polynet_5[2].TestDataIn)'),
        Matrix(m_polynet_5[2].TestDataOut)',
    )
    mae_Test_polynet_6 = Flux.mae(
        polynet_6_chain_best_model_chain(Matrix(m_polynet_6[2].TestDataIn)'),
        Matrix(m_polynet_6[2].TestDataOut)',
    )
    mae_Test_polynet_7 = Flux.mae(
        polynet_7_chain_best_model_chain(Matrix(m_polynet_7[2].TestDataIn)'),
        Matrix(m_polynet_7[2].TestDataOut)',
    )
    mae_Test_polynet_8 = Flux.mae(
        polynet_8_chain_best_model_chain(Matrix(m_polynet_8[2].TestDataIn)'),
        Matrix(m_polynet_8[2].TestDataOut)',
    )
    mae_Test_polynet_9 = Flux.mae(
        polynet_9_chain_best_model_chain(Matrix(m_polynet_9[2].TestDataIn)'),
        Matrix(m_polynet_9[2].TestDataOut)',
    )
    mae_Test_polynet_10 = Flux.mae(
        polynet_10_chain_best_model_chain(Matrix(m_polynet_10[2].TestDataIn)'),
        Matrix(m_polynet_10[2].TestDataOut)',
    )
    mae_Test_polynet_11 = Flux.mae(
        polynet_11_chain_best_model_chain(Matrix(m_polynet_11[2].TestDataIn)'),
        Matrix(m_polynet_11[2].TestDataOut)',
    )
    mae_Test_polynet_12 = Flux.mae(
        polynet_12_chain_best_model_chain(Matrix(m_polynet_12[2].TestDataIn)'),
        Matrix(m_polynet_12[2].TestDataOut)',
    )
    mae_Test_polynet_13 = Flux.mae(
        polynet_13_chain_best_model_chain(Matrix(m_polynet_13[2].TestDataIn)'),
        Matrix(m_polynet_13[2].TestDataOut)',
    )
    mae_Test_polynet_14 = Flux.mae(
        polynet_14_chain_best_model_chain(Matrix(m_polynet_14[2].TestDataIn)'),
        Matrix(m_polynet_14[2].TestDataOut)',
    )
    mae_Test_polynet_15 = Flux.mae(
        polynet_15_chain_best_model_chain(Matrix(m_polynet_15[2].TestDataIn)'),
        Matrix(m_polynet_15[2].TestDataOut)',
    )
    mae_Test_polynet_16 = Flux.mae(
        polynet_16_chain_best_model_chain(Matrix(m_polynet_16[2].TestDataIn)'),
        Matrix(m_polynet_16[2].TestDataOut)',
    )
    mae_Test_polynet_17 = Flux.mae(
        polynet_17_chain_best_model_chain(Matrix(m_polynet_17[2].TestDataIn)'),
        Matrix(m_polynet_17[2].TestDataOut)',
    )
    mae_Test_polynet_18 = Flux.mae(
        polynet_18_chain_best_model_chain(Matrix(m_polynet_18[2].TestDataIn)'),
        Matrix(m_polynet_18[2].TestDataOut)',
    )
    mae_Test_polynet_19 = Flux.mae(
        polynet_19_chain_best_model_chain(Matrix(m_polynet_19[2].TestDataIn)'),
        Matrix(m_polynet_19[2].TestDataOut)',
    )
    mae_Test_polynet_20 = Flux.mae(
        polynet_20_chain_best_model_chain(Matrix(m_polynet_20[2].TestDataIn)'),
        Matrix(m_polynet_20[2].TestDataOut)',
    )

    println("mae_Train_polynet_0 $mae_Train_polynet_0")
    println("mae_Train_polynet_1 $mae_Train_polynet_1")
    println("mae_Train_polynet_2 $mae_Train_polynet_2")
    println("mae_Train_polynet_3 $mae_Train_polynet_3")
    println("mae_Train_polynet_4 $mae_Train_polynet_4")
    println("mae_Train_polynet_5 $mae_Train_polynet_5")
    println("mae_Train_polynet_6 $mae_Train_polynet_6")
    println("mae_Train_polynet_7 $mae_Train_polynet_7")
    println("mae_Train_polynet_8 $mae_Train_polynet_8")
    println("mae_Train_polynet_9 $mae_Train_polynet_9")
    println("mae_Train_polynet_10 $mae_Train_polynet_10")
    println("mae_Train_polynet_11 $mae_Train_polynet_11")
    println("mae_Train_polynet_12 $mae_Train_polynet_12")
    println("mae_Train_polynet_13 $mae_Train_polynet_13")
    println("mae_Train_polynet_14 $mae_Train_polynet_14")
    println("mae_Train_polynet_15 $mae_Train_polynet_15")
    println("mae_Train_polynet_16 $mae_Train_polynet_16")
    println("mae_Train_polynet_17 $mae_Train_polynet_17")
    println("mae_Train_polynet_18 $mae_Train_polynet_18")
    println("mae_Train_polynet_19 $mae_Train_polynet_19")
    println("mae_Train_polynet_20 $mae_Train_polynet_20")

    println("mae_Test_polynet_0 $mae_Test_polynet_0")
    println("mae_Test_polynet_1 $mae_Test_polynet_1")
    println("mae_Test_polynet_2 $mae_Test_polynet_2")
    println("mae_Test_polynet_3 $mae_Test_polynet_3")
    println("mae_Test_polynet_4 $mae_Test_polynet_4")
    println("mae_Test_polynet_5 $mae_Test_polynet_5")
    println("mae_Test_polynet_6 $mae_Test_polynet_6")
    println("mae_Test_polynet_7 $mae_Test_polynet_7")
    println("mae_Test_polynet_8 $mae_Test_polynet_8")
    println("mae_Test_polynet_9 $mae_Test_polynet_9")
    println("mae_Test_polynet_10 $mae_Test_polynet_10")
    println("mae_Test_polynet_11 $mae_Test_polynet_11")
    println("mae_Test_polynet_12 $mae_Test_polynet_12")
    println("mae_Test_polynet_13 $mae_Test_polynet_13")
    println("mae_Test_polynet_14 $mae_Test_polynet_14")
    println("mae_Test_polynet_15 $mae_Test_polynet_15")
    println("mae_Test_polynet_16 $mae_Test_polynet_16")
    println("mae_Test_polynet_17 $mae_Test_polynet_17")
    println("mae_Test_polynet_18 $mae_Test_polynet_18")
    println("mae_Test_polynet_19 $mae_Test_polynet_19")
    println("mae_Test_polynet_20 $mae_Test_polynet_20")

    @test mae_Train_polynet_0 <= 10
    @test mae_Train_polynet_1 <= 10
    @test mae_Train_polynet_2 <= 10
    @test mae_Train_polynet_3 <= 10
    @test mae_Train_polynet_4 <= 10
    @test mae_Train_polynet_5 <= 10
    @test mae_Train_polynet_6 != NaN
    @test mae_Train_polynet_7 <= 10
    @test mae_Train_polynet_8 <= 10
    @test mae_Train_polynet_9 <= 10
    @test mae_Train_polynet_10 <= 10
    @test mae_Train_polynet_11 <= 10
    @test mae_Train_polynet_12 <= 10
    @test mae_Train_polynet_13 != NaN
    @test mae_Train_polynet_14 <= 10
    @test mae_Train_polynet_15 <= 10
    @test mae_Train_polynet_16 <= 10
    @test mae_Train_polynet_17 <= 10
    @test mae_Train_polynet_18 <= 10
    @test mae_Train_polynet_19 <= 10
    @test mae_Train_polynet_20 != NaN

    @test mae_Test_polynet_0 <= 10
    @test mae_Test_polynet_1 <= 10
    @test mae_Test_polynet_2 <= 10
    @test mae_Test_polynet_3 <= 10
    @test mae_Test_polynet_4 <= 10
    @test mae_Test_polynet_5 <= 10
    @test mae_Test_polynet_6 != NaN
    @test mae_Test_polynet_7 <= 10
    @test mae_Test_polynet_8 <= 10
    @test mae_Test_polynet_9 <= 10
    @test mae_Test_polynet_10 <= 10
    @test mae_Test_polynet_11 <= 10
    @test mae_Test_polynet_12 <= 10
    @test mae_Test_polynet_13 != NaN
    @test mae_Test_polynet_14 <= 10
    @test mae_Test_polynet_15 <= 10
    @test mae_Test_polynet_16 <= 10
    @test mae_Test_polynet_17 <= 10
    @test mae_Test_polynet_18 <= 10
    @test mae_Test_polynet_19 <= 10
    @test mae_Test_polynet_20 != NaN

end

@testset "densenet architecture blackbox identification" begin

    #load data 
    dfout_raw = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:2500, :]
    dfin_raw = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:2500, :]

    #Train models
    max_time = Minute(5) #hours

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin_raw,
        dfout_raw,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )
    
    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    m_densenet_0 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_densenet_1 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_densenet_2 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_densenet_3 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_densenet_4 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_densenet_5 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_densenet_6 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )


    m_densenet_7 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_densenet_8 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_densenet_9 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_densenet_10 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_densenet_11 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_densenet_12 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_densenet_13 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_densenet_14 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_densenet_15 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_densenet_16 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_densenet_17 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_densenet_18 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_densenet_19 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_densenet_20 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "densenet",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    #Get best models
    densenet_0_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_0[1]).machine).best_fitted_params.chain
    densenet_1_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_1[1]).machine).best_fitted_params.chain
    densenet_2_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_2[1]).machine).best_fitted_params.chain
    densenet_3_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_3[1]).machine).best_fitted_params.chain
    densenet_4_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_4[1]).machine).best_fitted_params.chain
    densenet_5_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_5[1]).machine).best_fitted_params.chain
    densenet_6_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_6[1]).machine).best_fitted_params.chain
    densenet_7_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_7[1]).machine).best_fitted_params.chain
    densenet_8_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_8[1]).machine).best_fitted_params.chain
    densenet_9_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_9[1]).machine).best_fitted_params.chain
    densenet_10_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_10[1]).machine).best_fitted_params.chain
    densenet_11_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_11[1]).machine).best_fitted_params.chain
    densenet_12_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_12[1]).machine).best_fitted_params.chain
    densenet_13_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_13[1]).machine).best_fitted_params.chain
    densenet_14_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_14[1]).machine).best_fitted_params.chain
    densenet_15_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_15[1]).machine).best_fitted_params.chain
    densenet_16_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_16[1]).machine).best_fitted_params.chain
    densenet_17_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_17[1]).machine).best_fitted_params.chain
    densenet_18_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_18[1]).machine).best_fitted_params.chain
    densenet_19_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_19[1]).machine).best_fitted_params.chain
    densenet_20_chain_best_model_chain =
        fitted_params(fitted_params(m_densenet_20[1]).machine).best_fitted_params.chain

    # MAE with train and test data
    mae_Train_densenet_0 = Flux.mae(
        densenet_0_chain_best_model_chain(Matrix(m_densenet_0[2].TrainDataIn)'),
        Matrix(m_densenet_0[2].TrainDataOut)',
    )
    mae_Train_densenet_1 = Flux.mae(
        densenet_1_chain_best_model_chain(Matrix(m_densenet_1[2].TrainDataIn)'),
        Matrix(m_densenet_1[2].TrainDataOut)',
    )
    mae_Train_densenet_2 = Flux.mae(
        densenet_2_chain_best_model_chain(Matrix(m_densenet_2[2].TrainDataIn)'),
        Matrix(m_densenet_2[2].TrainDataOut)',
    )
    mae_Train_densenet_3 = Flux.mae(
        densenet_3_chain_best_model_chain(Matrix(m_densenet_3[2].TrainDataIn)'),
        Matrix(m_densenet_3[2].TrainDataOut)',
    )
    mae_Train_densenet_4 = Flux.mae(
        densenet_4_chain_best_model_chain(Matrix(m_densenet_4[2].TrainDataIn)'),
        Matrix(m_densenet_4[2].TrainDataOut)',
    )
    mae_Train_densenet_5 = Flux.mae(
        densenet_5_chain_best_model_chain(Matrix(m_densenet_5[2].TrainDataIn)'),
        Matrix(m_densenet_5[2].TrainDataOut)',
    )
    mae_Train_densenet_6 = Flux.mae(
        densenet_6_chain_best_model_chain(Matrix(m_densenet_6[2].TrainDataIn)'),
        Matrix(m_densenet_6[2].TrainDataOut)',
    )
    mae_Train_densenet_7 = Flux.mae(
        densenet_7_chain_best_model_chain(Matrix(m_densenet_7[2].TrainDataIn)'),
        Matrix(m_densenet_7[2].TrainDataOut)',
    )
    mae_Train_densenet_8 = Flux.mae(
        densenet_8_chain_best_model_chain(Matrix(m_densenet_8[2].TrainDataIn)'),
        Matrix(m_densenet_8[2].TrainDataOut)',
    )
    mae_Train_densenet_9 = Flux.mae(
        densenet_9_chain_best_model_chain(Matrix(m_densenet_9[2].TrainDataIn)'),
        Matrix(m_densenet_9[2].TrainDataOut)',
    )
    mae_Train_densenet_10 = Flux.mae(
        densenet_10_chain_best_model_chain(Matrix(m_densenet_10[2].TrainDataIn)'),
        Matrix(m_densenet_10[2].TrainDataOut)',
    )
    mae_Train_densenet_11 = Flux.mae(
        densenet_11_chain_best_model_chain(Matrix(m_densenet_11[2].TrainDataIn)'),
        Matrix(m_densenet_11[2].TrainDataOut)',
    )
    mae_Train_densenet_12 = Flux.mae(
        densenet_12_chain_best_model_chain(Matrix(m_densenet_12[2].TrainDataIn)'),
        Matrix(m_densenet_12[2].TrainDataOut)',
    )
    mae_Train_densenet_13 = Flux.mae(
        densenet_13_chain_best_model_chain(Matrix(m_densenet_13[2].TrainDataIn)'),
        Matrix(m_densenet_13[2].TrainDataOut)',
    )
    mae_Train_densenet_14 = Flux.mae(
        densenet_14_chain_best_model_chain(Matrix(m_densenet_14[2].TrainDataIn)'),
        Matrix(m_densenet_14[2].TrainDataOut)',
    )
    mae_Train_densenet_15 = Flux.mae(
        densenet_15_chain_best_model_chain(Matrix(m_densenet_15[2].TrainDataIn)'),
        Matrix(m_densenet_15[2].TrainDataOut)',
    )
    mae_Train_densenet_16 = Flux.mae(
        densenet_16_chain_best_model_chain(Matrix(m_densenet_16[2].TrainDataIn)'),
        Matrix(m_densenet_16[2].TrainDataOut)',
    )
    mae_Train_densenet_17 = Flux.mae(
        densenet_17_chain_best_model_chain(Matrix(m_densenet_17[2].TrainDataIn)'),
        Matrix(m_densenet_17[2].TrainDataOut)',
    )
    mae_Train_densenet_18 = Flux.mae(
        densenet_18_chain_best_model_chain(Matrix(m_densenet_18[2].TrainDataIn)'),
        Matrix(m_densenet_18[2].TrainDataOut)',
    )
    mae_Train_densenet_19 = Flux.mae(
        densenet_19_chain_best_model_chain(Matrix(m_densenet_19[2].TrainDataIn)'),
        Matrix(m_densenet_19[2].TrainDataOut)',
    )
    mae_Train_densenet_20 = Flux.mae(
        densenet_20_chain_best_model_chain(Matrix(m_densenet_20[2].TrainDataIn)'),
        Matrix(m_densenet_20[2].TrainDataOut)',
    )

    mae_Test_densenet_0 = Flux.mae(
        densenet_0_chain_best_model_chain(Matrix(m_densenet_0[2].TestDataIn)'),
        Matrix(m_densenet_0[2].TestDataOut)',
    )
    mae_Test_densenet_1 = Flux.mae(
        densenet_1_chain_best_model_chain(Matrix(m_densenet_1[2].TestDataIn)'),
        Matrix(m_densenet_1[2].TestDataOut)',
    )
    mae_Test_densenet_2 = Flux.mae(
        densenet_2_chain_best_model_chain(Matrix(m_densenet_2[2].TestDataIn)'),
        Matrix(m_densenet_2[2].TestDataOut)',
    )
    mae_Test_densenet_3 = Flux.mae(
        densenet_3_chain_best_model_chain(Matrix(m_densenet_3[2].TestDataIn)'),
        Matrix(m_densenet_3[2].TestDataOut)',
    )
    mae_Test_densenet_4 = Flux.mae(
        densenet_4_chain_best_model_chain(Matrix(m_densenet_4[2].TestDataIn)'),
        Matrix(m_densenet_4[2].TestDataOut)',
    )
    mae_Test_densenet_5 = Flux.mae(
        densenet_5_chain_best_model_chain(Matrix(m_densenet_5[2].TestDataIn)'),
        Matrix(m_densenet_5[2].TestDataOut)',
    )
    mae_Test_densenet_6 = Flux.mae(
        densenet_6_chain_best_model_chain(Matrix(m_densenet_6[2].TestDataIn)'),
        Matrix(m_densenet_6[2].TestDataOut)',
    )
    mae_Test_densenet_7 = Flux.mae(
        densenet_7_chain_best_model_chain(Matrix(m_densenet_7[2].TestDataIn)'),
        Matrix(m_densenet_7[2].TestDataOut)',
    )
    mae_Test_densenet_8 = Flux.mae(
        densenet_8_chain_best_model_chain(Matrix(m_densenet_8[2].TestDataIn)'),
        Matrix(m_densenet_8[2].TestDataOut)',
    )
    mae_Test_densenet_9 = Flux.mae(
        densenet_9_chain_best_model_chain(Matrix(m_densenet_9[2].TestDataIn)'),
        Matrix(m_densenet_9[2].TestDataOut)',
    )
    mae_Test_densenet_10 = Flux.mae(
        densenet_10_chain_best_model_chain(Matrix(m_densenet_10[2].TestDataIn)'),
        Matrix(m_densenet_10[2].TestDataOut)',
    )
    mae_Test_densenet_11 = Flux.mae(
        densenet_11_chain_best_model_chain(Matrix(m_densenet_11[2].TestDataIn)'),
        Matrix(m_densenet_11[2].TestDataOut)',
    )
    mae_Test_densenet_12 = Flux.mae(
        densenet_12_chain_best_model_chain(Matrix(m_densenet_12[2].TestDataIn)'),
        Matrix(m_densenet_12[2].TestDataOut)',
    )
    mae_Test_densenet_13 = Flux.mae(
        densenet_13_chain_best_model_chain(Matrix(m_densenet_13[2].TestDataIn)'),
        Matrix(m_densenet_13[2].TestDataOut)',
    )
    mae_Test_densenet_14 = Flux.mae(
        densenet_14_chain_best_model_chain(Matrix(m_densenet_14[2].TestDataIn)'),
        Matrix(m_densenet_14[2].TestDataOut)',
    )
    mae_Test_densenet_15 = Flux.mae(
        densenet_15_chain_best_model_chain(Matrix(m_densenet_15[2].TestDataIn)'),
        Matrix(m_densenet_15[2].TestDataOut)',
    )
    mae_Test_densenet_16 = Flux.mae(
        densenet_16_chain_best_model_chain(Matrix(m_densenet_16[2].TestDataIn)'),
        Matrix(m_densenet_16[2].TestDataOut)',
    )
    mae_Test_densenet_17 = Flux.mae(
        densenet_17_chain_best_model_chain(Matrix(m_densenet_17[2].TestDataIn)'),
        Matrix(m_densenet_17[2].TestDataOut)',
    )
    mae_Test_densenet_18 = Flux.mae(
        densenet_18_chain_best_model_chain(Matrix(m_densenet_18[2].TestDataIn)'),
        Matrix(m_densenet_18[2].TestDataOut)',
    )
    mae_Test_densenet_19 = Flux.mae(
        densenet_19_chain_best_model_chain(Matrix(m_densenet_19[2].TestDataIn)'),
        Matrix(m_densenet_19[2].TestDataOut)',
    )
    mae_Test_densenet_20 = Flux.mae(
        densenet_20_chain_best_model_chain(Matrix(m_densenet_20[2].TestDataIn)'),
        Matrix(m_densenet_20[2].TestDataOut)',
    )

    println("mae_Train_densenet_0 $mae_Train_densenet_0")
    println("mae_Train_densenet_1 $mae_Train_densenet_1")
    println("mae_Train_densenet_2 $mae_Train_densenet_2")
    println("mae_Train_densenet_3 $mae_Train_densenet_3")
    println("mae_Train_densenet_4 $mae_Train_densenet_4")
    println("mae_Train_densenet_5 $mae_Train_densenet_5")
    println("mae_Train_densenet_6 $mae_Train_densenet_6")
    println("mae_Train_densenet_7 $mae_Train_densenet_7")
    println("mae_Train_densenet_8 $mae_Train_densenet_8")
    println("mae_Train_densenet_9 $mae_Train_densenet_9")
    println("mae_Train_densenet_10 $mae_Train_densenet_10")
    println("mae_Train_densenet_11 $mae_Train_densenet_11")
    println("mae_Train_densenet_12 $mae_Train_densenet_12")
    println("mae_Train_densenet_13 $mae_Train_densenet_13")
    println("mae_Train_densenet_14 $mae_Train_densenet_14")
    println("mae_Train_densenet_15 $mae_Train_densenet_15")
    println("mae_Train_densenet_16 $mae_Train_densenet_16")
    println("mae_Train_densenet_17 $mae_Train_densenet_17")
    println("mae_Train_densenet_18 $mae_Train_densenet_18")
    println("mae_Train_densenet_19 $mae_Train_densenet_19")
    println("mae_Train_densenet_20 $mae_Train_densenet_20")

    println("mae_Test_densenet_0 $mae_Test_densenet_0")
    println("mae_Test_densenet_1 $mae_Test_densenet_1")
    println("mae_Test_densenet_2 $mae_Test_densenet_2")
    println("mae_Test_densenet_3 $mae_Test_densenet_3")
    println("mae_Test_densenet_4 $mae_Test_densenet_4")
    println("mae_Test_densenet_5 $mae_Test_densenet_5")
    println("mae_Test_densenet_6 $mae_Test_densenet_6")
    println("mae_Test_densenet_7 $mae_Test_densenet_7")
    println("mae_Test_densenet_8 $mae_Test_densenet_8")
    println("mae_Test_densenet_9 $mae_Test_densenet_9")
    println("mae_Test_densenet_10 $mae_Test_densenet_10")
    println("mae_Test_densenet_11 $mae_Test_densenet_11")
    println("mae_Test_densenet_12 $mae_Test_densenet_12")
    println("mae_Test_densenet_13 $mae_Test_densenet_13")
    println("mae_Test_densenet_14 $mae_Test_densenet_14")
    println("mae_Test_densenet_15 $mae_Test_densenet_15")
    println("mae_Test_densenet_16 $mae_Test_densenet_16")
    println("mae_Test_densenet_17 $mae_Test_densenet_17")
    println("mae_Test_densenet_18 $mae_Test_densenet_18")
    println("mae_Test_densenet_19 $mae_Test_densenet_19")
    println("mae_Test_densenet_20 $mae_Test_densenet_20")

    @test mae_Train_densenet_0 <= 10
    @test mae_Train_densenet_1 <= 10
    @test mae_Train_densenet_2 <= 10
    @test mae_Train_densenet_3 <= 10
    @test mae_Train_densenet_4 <= 10
    @test mae_Train_densenet_5 <= 10
    @test mae_Train_densenet_6 != NaN
    @test mae_Train_densenet_7 <= 10
    @test mae_Train_densenet_8 <= 10
    @test mae_Train_densenet_9 <= 10
    @test mae_Train_densenet_10 <= 10
    @test mae_Train_densenet_11 <= 10
    @test mae_Train_densenet_12 <= 10
    @test mae_Train_densenet_13 != NaN
    @test mae_Train_densenet_14 <= 10
    @test mae_Train_densenet_15 <= 10
    @test mae_Train_densenet_16 <= 10
    @test mae_Train_densenet_17 <= 10
    @test mae_Train_densenet_18 <= 10
    @test mae_Train_densenet_19 <= 10
    @test mae_Train_densenet_20 != NaN

    @test mae_Test_densenet_0 <= 10
    @test mae_Test_densenet_1 <= 10
    @test mae_Test_densenet_2 <= 10
    @test mae_Test_densenet_3 <= 10
    @test mae_Test_densenet_4 <= 10
    @test mae_Test_densenet_5 <= 10
    @test mae_Test_densenet_6 != NaN
    @test mae_Test_densenet_7 <= 10
    @test mae_Test_densenet_8 <= 10
    @test mae_Test_densenet_9 <= 10
    @test mae_Test_densenet_10 <= 10
    @test mae_Test_densenet_11 <= 10
    @test mae_Test_densenet_12 <= 10
    @test mae_Test_densenet_13 != NaN
    @test mae_Test_densenet_14 <= 10
    @test mae_Test_densenet_15 <= 10
    @test mae_Test_densenet_16 <= 10
    @test mae_Test_densenet_17 <= 10
    @test mae_Test_densenet_18 <= 10
    @test mae_Test_densenet_19 <= 10
    @test mae_Test_densenet_20 != NaN

end

@testset "neuralnet ode type 1 architecture blackbox identification" begin

    #load data 
    dfout_raw = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:2500, :]
    dfin_raw = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:2500, :]

    #Train models
    max_time = Minute(5) #hours

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin_raw,
        dfout_raw,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )
    
    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    m_neuralnet_0 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_neuralnet_1 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_neuralnet_2 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_neuralnet_3 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_neuralnet_4 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_neuralnet_5 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )

    m_neuralnet_6 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
    )


    m_neuralnet_7 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_neuralnet_8 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_neuralnet_9 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_neuralnet_10 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_neuralnet_11 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_neuralnet_12 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_neuralnet_13 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
    )

    m_neuralnet_14 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_neuralnet_15 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_neuralnet_16 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_neuralnet_17 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_neuralnet_18 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_neuralnet_19 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    m_neuralnet_20 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "neuralnet_ode_type1",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
    )

    #Get best models
    neuralnet_0_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_0[1]).machine).best_fitted_params.chain
    neuralnet_1_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_1[1]).machine).best_fitted_params.chain
    neuralnet_2_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_2[1]).machine).best_fitted_params.chain
    neuralnet_3_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_3[1]).machine).best_fitted_params.chain
    neuralnet_4_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_4[1]).machine).best_fitted_params.chain
    neuralnet_5_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_5[1]).machine).best_fitted_params.chain
    neuralnet_6_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_6[1]).machine).best_fitted_params.chain
    neuralnet_7_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_7[1]).machine).best_fitted_params.chain
    neuralnet_8_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_8[1]).machine).best_fitted_params.chain
    neuralnet_9_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_9[1]).machine).best_fitted_params.chain
    neuralnet_10_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_10[1]).machine).best_fitted_params.chain
    neuralnet_11_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_11[1]).machine).best_fitted_params.chain
    neuralnet_12_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_12[1]).machine).best_fitted_params.chain
    neuralnet_13_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_13[1]).machine).best_fitted_params.chain
    neuralnet_14_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_14[1]).machine).best_fitted_params.chain
    neuralnet_15_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_15[1]).machine).best_fitted_params.chain
    neuralnet_16_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_16[1]).machine).best_fitted_params.chain
    neuralnet_17_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_17[1]).machine).best_fitted_params.chain
    neuralnet_18_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_18[1]).machine).best_fitted_params.chain
    neuralnet_19_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_19[1]).machine).best_fitted_params.chain
    neuralnet_20_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_20[1]).machine).best_fitted_params.chain

    # MAE with train and test data
    mae_Train_neuralnet_0 = Flux.mae(
        neuralnet_0_chain_best_model_chain(Matrix(m_neuralnet_0[2].TrainDataIn)'),
        Matrix(m_neuralnet_0[2].TrainDataOut)',
    )
    mae_Train_neuralnet_1 = Flux.mae(
        neuralnet_1_chain_best_model_chain(Matrix(m_neuralnet_1[2].TrainDataIn)'),
        Matrix(m_neuralnet_1[2].TrainDataOut)',
    )
    mae_Train_neuralnet_2 = Flux.mae(
        neuralnet_2_chain_best_model_chain(Matrix(m_neuralnet_2[2].TrainDataIn)'),
        Matrix(m_neuralnet_2[2].TrainDataOut)',
    )
    mae_Train_neuralnet_3 = Flux.mae(
        neuralnet_3_chain_best_model_chain(Matrix(m_neuralnet_3[2].TrainDataIn)'),
        Matrix(m_neuralnet_3[2].TrainDataOut)',
    )
    mae_Train_neuralnet_4 = Flux.mae(
        neuralnet_4_chain_best_model_chain(Matrix(m_neuralnet_4[2].TrainDataIn)'),
        Matrix(m_neuralnet_4[2].TrainDataOut)',
    )
    mae_Train_neuralnet_5 = Flux.mae(
        neuralnet_5_chain_best_model_chain(Matrix(m_neuralnet_5[2].TrainDataIn)'),
        Matrix(m_neuralnet_5[2].TrainDataOut)',
    )
    mae_Train_neuralnet_6 = Flux.mae(
        neuralnet_6_chain_best_model_chain(Matrix(m_neuralnet_6[2].TrainDataIn)'),
        Matrix(m_neuralnet_6[2].TrainDataOut)',
    )
    mae_Train_neuralnet_7 = Flux.mae(
        neuralnet_7_chain_best_model_chain(Matrix(m_neuralnet_7[2].TrainDataIn)'),
        Matrix(m_neuralnet_7[2].TrainDataOut)',
    )
    mae_Train_neuralnet_8 = Flux.mae(
        neuralnet_8_chain_best_model_chain(Matrix(m_neuralnet_8[2].TrainDataIn)'),
        Matrix(m_neuralnet_8[2].TrainDataOut)',
    )
    mae_Train_neuralnet_9 = Flux.mae(
        neuralnet_9_chain_best_model_chain(Matrix(m_neuralnet_9[2].TrainDataIn)'),
        Matrix(m_neuralnet_9[2].TrainDataOut)',
    )
    mae_Train_neuralnet_10 = Flux.mae(
        neuralnet_10_chain_best_model_chain(Matrix(m_neuralnet_10[2].TrainDataIn)'),
        Matrix(m_neuralnet_10[2].TrainDataOut)',
    )
    mae_Train_neuralnet_11 = Flux.mae(
        neuralnet_11_chain_best_model_chain(Matrix(m_neuralnet_11[2].TrainDataIn)'),
        Matrix(m_neuralnet_11[2].TrainDataOut)',
    )
    mae_Train_neuralnet_12 = Flux.mae(
        neuralnet_12_chain_best_model_chain(Matrix(m_neuralnet_12[2].TrainDataIn)'),
        Matrix(m_neuralnet_12[2].TrainDataOut)',
    )
    mae_Train_neuralnet_13 = Flux.mae(
        neuralnet_13_chain_best_model_chain(Matrix(m_neuralnet_13[2].TrainDataIn)'),
        Matrix(m_neuralnet_13[2].TrainDataOut)',
    )
    mae_Train_neuralnet_14 = Flux.mae(
        neuralnet_14_chain_best_model_chain(Matrix(m_neuralnet_14[2].TrainDataIn)'),
        Matrix(m_neuralnet_14[2].TrainDataOut)',
    )
    mae_Train_neuralnet_15 = Flux.mae(
        neuralnet_15_chain_best_model_chain(Matrix(m_neuralnet_15[2].TrainDataIn)'),
        Matrix(m_neuralnet_15[2].TrainDataOut)',
    )
    mae_Train_neuralnet_16 = Flux.mae(
        neuralnet_16_chain_best_model_chain(Matrix(m_neuralnet_16[2].TrainDataIn)'),
        Matrix(m_neuralnet_16[2].TrainDataOut)',
    )
    mae_Train_neuralnet_17 = Flux.mae(
        neuralnet_17_chain_best_model_chain(Matrix(m_neuralnet_17[2].TrainDataIn)'),
        Matrix(m_neuralnet_17[2].TrainDataOut)',
    )
    mae_Train_neuralnet_18 = Flux.mae(
        neuralnet_18_chain_best_model_chain(Matrix(m_neuralnet_18[2].TrainDataIn)'),
        Matrix(m_neuralnet_18[2].TrainDataOut)',
    )
    mae_Train_neuralnet_19 = Flux.mae(
        neuralnet_19_chain_best_model_chain(Matrix(m_neuralnet_19[2].TrainDataIn)'),
        Matrix(m_neuralnet_19[2].TrainDataOut)',
    )
    mae_Train_neuralnet_20 = Flux.mae(
        neuralnet_20_chain_best_model_chain(Matrix(m_neuralnet_20[2].TrainDataIn)'),
        Matrix(m_neuralnet_20[2].TrainDataOut)',
    )

    mae_Test_neuralnet_0 = Flux.mae(
        neuralnet_0_chain_best_model_chain(Matrix(m_neuralnet_0[2].TestDataIn)'),
        Matrix(m_neuralnet_0[2].TestDataOut)',
    )
    mae_Test_neuralnet_1 = Flux.mae(
        neuralnet_1_chain_best_model_chain(Matrix(m_neuralnet_1[2].TestDataIn)'),
        Matrix(m_neuralnet_1[2].TestDataOut)',
    )
    mae_Test_neuralnet_2 = Flux.mae(
        neuralnet_2_chain_best_model_chain(Matrix(m_neuralnet_2[2].TestDataIn)'),
        Matrix(m_neuralnet_2[2].TestDataOut)',
    )
    mae_Test_neuralnet_3 = Flux.mae(
        neuralnet_3_chain_best_model_chain(Matrix(m_neuralnet_3[2].TestDataIn)'),
        Matrix(m_neuralnet_3[2].TestDataOut)',
    )
    mae_Test_neuralnet_4 = Flux.mae(
        neuralnet_4_chain_best_model_chain(Matrix(m_neuralnet_4[2].TestDataIn)'),
        Matrix(m_neuralnet_4[2].TestDataOut)',
    )
    mae_Test_neuralnet_5 = Flux.mae(
        neuralnet_5_chain_best_model_chain(Matrix(m_neuralnet_5[2].TestDataIn)'),
        Matrix(m_neuralnet_5[2].TestDataOut)',
    )
    mae_Test_neuralnet_6 = Flux.mae(
        neuralnet_6_chain_best_model_chain(Matrix(m_neuralnet_6[2].TestDataIn)'),
        Matrix(m_neuralnet_6[2].TestDataOut)',
    )
    mae_Test_neuralnet_7 = Flux.mae(
        neuralnet_7_chain_best_model_chain(Matrix(m_neuralnet_7[2].TestDataIn)'),
        Matrix(m_neuralnet_7[2].TestDataOut)',
    )
    mae_Test_neuralnet_8 = Flux.mae(
        neuralnet_8_chain_best_model_chain(Matrix(m_neuralnet_8[2].TestDataIn)'),
        Matrix(m_neuralnet_8[2].TestDataOut)',
    )
    mae_Test_neuralnet_9 = Flux.mae(
        neuralnet_9_chain_best_model_chain(Matrix(m_neuralnet_9[2].TestDataIn)'),
        Matrix(m_neuralnet_9[2].TestDataOut)',
    )
    mae_Test_neuralnet_10 = Flux.mae(
        neuralnet_10_chain_best_model_chain(Matrix(m_neuralnet_10[2].TestDataIn)'),
        Matrix(m_neuralnet_10[2].TestDataOut)',
    )
    mae_Test_neuralnet_11 = Flux.mae(
        neuralnet_11_chain_best_model_chain(Matrix(m_neuralnet_11[2].TestDataIn)'),
        Matrix(m_neuralnet_11[2].TestDataOut)',
    )
    mae_Test_neuralnet_12 = Flux.mae(
        neuralnet_12_chain_best_model_chain(Matrix(m_neuralnet_12[2].TestDataIn)'),
        Matrix(m_neuralnet_12[2].TestDataOut)',
    )
    mae_Test_neuralnet_13 = Flux.mae(
        neuralnet_13_chain_best_model_chain(Matrix(m_neuralnet_13[2].TestDataIn)'),
        Matrix(m_neuralnet_13[2].TestDataOut)',
    )
    mae_Test_neuralnet_14 = Flux.mae(
        neuralnet_14_chain_best_model_chain(Matrix(m_neuralnet_14[2].TestDataIn)'),
        Matrix(m_neuralnet_14[2].TestDataOut)',
    )
    mae_Test_neuralnet_15 = Flux.mae(
        neuralnet_15_chain_best_model_chain(Matrix(m_neuralnet_15[2].TestDataIn)'),
        Matrix(m_neuralnet_15[2].TestDataOut)',
    )
    mae_Test_neuralnet_16 = Flux.mae(
        neuralnet_16_chain_best_model_chain(Matrix(m_neuralnet_16[2].TestDataIn)'),
        Matrix(m_neuralnet_16[2].TestDataOut)',
    )
    mae_Test_neuralnet_17 = Flux.mae(
        neuralnet_17_chain_best_model_chain(Matrix(m_neuralnet_17[2].TestDataIn)'),
        Matrix(m_neuralnet_17[2].TestDataOut)',
    )
    mae_Test_neuralnet_18 = Flux.mae(
        neuralnet_18_chain_best_model_chain(Matrix(m_neuralnet_18[2].TestDataIn)'),
        Matrix(m_neuralnet_18[2].TestDataOut)',
    )
    mae_Test_neuralnet_19 = Flux.mae(
        neuralnet_19_chain_best_model_chain(Matrix(m_neuralnet_19[2].TestDataIn)'),
        Matrix(m_neuralnet_19[2].TestDataOut)',
    )
    mae_Test_neuralnet_20 = Flux.mae(
        neuralnet_20_chain_best_model_chain(Matrix(m_neuralnet_20[2].TestDataIn)'),
        Matrix(m_neuralnet_20[2].TestDataOut)',
    )

    println("mae_Train_neuralnet_0 $mae_Train_neuralnet_0")
    println("mae_Train_neuralnet_1 $mae_Train_neuralnet_1")
    println("mae_Train_neuralnet_2 $mae_Train_neuralnet_2")
    println("mae_Train_neuralnet_3 $mae_Train_neuralnet_3")
    println("mae_Train_neuralnet_4 $mae_Train_neuralnet_4")
    println("mae_Train_neuralnet_5 $mae_Train_neuralnet_5")
    println("mae_Train_neuralnet_6 $mae_Train_neuralnet_6")
    println("mae_Train_neuralnet_7 $mae_Train_neuralnet_7")
    println("mae_Train_neuralnet_8 $mae_Train_neuralnet_8")
    println("mae_Train_neuralnet_9 $mae_Train_neuralnet_9")
    println("mae_Train_neuralnet_10 $mae_Train_neuralnet_10")
    println("mae_Train_neuralnet_11 $mae_Train_neuralnet_11")
    println("mae_Train_neuralnet_12 $mae_Train_neuralnet_12")
    println("mae_Train_neuralnet_13 $mae_Train_neuralnet_13")
    println("mae_Train_neuralnet_14 $mae_Train_neuralnet_14")
    println("mae_Train_neuralnet_15 $mae_Train_neuralnet_15")
    println("mae_Train_neuralnet_16 $mae_Train_neuralnet_16")
    println("mae_Train_neuralnet_17 $mae_Train_neuralnet_17")
    println("mae_Train_neuralnet_18 $mae_Train_neuralnet_18")
    println("mae_Train_neuralnet_19 $mae_Train_neuralnet_19")
    println("mae_Train_neuralnet_20 $mae_Train_neuralnet_20")

    println("mae_Test_neuralnet_0 $mae_Test_neuralnet_0")
    println("mae_Test_neuralnet_1 $mae_Test_neuralnet_1")
    println("mae_Test_neuralnet_2 $mae_Test_neuralnet_2")
    println("mae_Test_neuralnet_3 $mae_Test_neuralnet_3")
    println("mae_Test_neuralnet_4 $mae_Test_neuralnet_4")
    println("mae_Test_neuralnet_5 $mae_Test_neuralnet_5")
    println("mae_Test_neuralnet_6 $mae_Test_neuralnet_6")
    println("mae_Test_neuralnet_7 $mae_Test_neuralnet_7")
    println("mae_Test_neuralnet_8 $mae_Test_neuralnet_8")
    println("mae_Test_neuralnet_9 $mae_Test_neuralnet_9")
    println("mae_Test_neuralnet_10 $mae_Test_neuralnet_10")
    println("mae_Test_neuralnet_11 $mae_Test_neuralnet_11")
    println("mae_Test_neuralnet_12 $mae_Test_neuralnet_12")
    println("mae_Test_neuralnet_13 $mae_Test_neuralnet_13")
    println("mae_Test_neuralnet_14 $mae_Test_neuralnet_14")
    println("mae_Test_neuralnet_15 $mae_Test_neuralnet_15")
    println("mae_Test_neuralnet_16 $mae_Test_neuralnet_16")
    println("mae_Test_neuralnet_17 $mae_Test_neuralnet_17")
    println("mae_Test_neuralnet_18 $mae_Test_neuralnet_18")
    println("mae_Test_neuralnet_19 $mae_Test_neuralnet_19")
    println("mae_Test_neuralnet_20 $mae_Test_neuralnet_20")

    @test mae_Train_neuralnet_0 <= 10
    @test mae_Train_neuralnet_1 <= 10
    @test mae_Train_neuralnet_2 <= 10
    @test mae_Train_neuralnet_3 <= 10
    @test mae_Train_neuralnet_4 <= 10
    @test mae_Train_neuralnet_5 <= 10
    @test mae_Train_neuralnet_6 != NaN
    @test mae_Train_neuralnet_7 <= 10
    @test mae_Train_neuralnet_8 <= 10
    @test mae_Train_neuralnet_9 <= 10
    @test mae_Train_neuralnet_10 <= 10
    @test mae_Train_neuralnet_11 <= 10
    @test mae_Train_neuralnet_12 <= 10
    @test mae_Train_neuralnet_13 != NaN
    @test mae_Train_neuralnet_14 <= 10
    @test mae_Train_neuralnet_15 <= 10
    @test mae_Train_neuralnet_16 <= 10
    @test mae_Train_neuralnet_17 <= 10
    @test mae_Train_neuralnet_18 <= 10
    @test mae_Train_neuralnet_19 <= 10
    @test mae_Train_neuralnet_20 != NaN

    @test mae_Test_neuralnet_0 <= 10
    @test mae_Test_neuralnet_1 <= 10
    @test mae_Test_neuralnet_2 <= 10
    @test mae_Test_neuralnet_3 <= 10
    @test mae_Test_neuralnet_4 <= 10
    @test mae_Test_neuralnet_5 <= 10
    @test mae_Test_neuralnet_6 != NaN
    @test mae_Test_neuralnet_7 <= 10
    @test mae_Test_neuralnet_8 <= 10
    @test mae_Test_neuralnet_9 <= 10
    @test mae_Test_neuralnet_10 <= 10
    @test mae_Test_neuralnet_11 <= 10
    @test mae_Test_neuralnet_12 <= 10
    @test mae_Test_neuralnet_13 != NaN
    @test mae_Test_neuralnet_14 <= 10
    @test mae_Test_neuralnet_15 <= 10
    @test mae_Test_neuralnet_16 <= 10
    @test mae_Test_neuralnet_17 <= 10
    @test mae_Test_neuralnet_18 <= 10
    @test mae_Test_neuralnet_19 <= 10
    @test mae_Test_neuralnet_20 != NaN

end

@testset "neuralnet ode type 1 architecture blackbox identification" begin

    #load data 
    dfout_raw = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:2500, :]
    dfin_raw = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:2500, :]

    #Train models
    max_time = Minute(5) #hours

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin_raw,
        dfout_raw,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )
    
    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)
    m_neuralnet_0 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        sample_time = 5.0,
    )

    m_neuralnet_1 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        sample_time = 5.0,
    )

    m_neuralnet_2 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        sample_time = 5.0,
    )

    m_neuralnet_3 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        sample_time = 5.0,
    )

    m_neuralnet_4 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        sample_time = 5.0,
    )

    m_neuralnet_5 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        sample_time = 5.0,
    )

    m_neuralnet_6 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        sample_time = 5.0,
    )


    m_neuralnet_7 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
        sample_time = 5.0,
    )

    m_neuralnet_8 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
        sample_time = 5.0,
    )

    m_neuralnet_9 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
        sample_time = 5.0,
    )

    m_neuralnet_10 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
        sample_time = 5.0,
    )

    m_neuralnet_11 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
        sample_time = 5.0,
    )

    m_neuralnet_12 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
        sample_time = 5.0,
    )

    m_neuralnet_13 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_threads",
        sample_time = 5.0,
    )

    m_neuralnet_14 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
        sample_time = 5.0,
    )

    m_neuralnet_15 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
        sample_time = 5.0,
    )

    m_neuralnet_16 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
        sample_time = 5.0,
    )

    m_neuralnet_17 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
        sample_time = 5.0,
    )

    m_neuralnet_18 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
        sample_time = 5.0,
    )

    m_neuralnet_19 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
        sample_time = 5.0,
    )

    m_neuralnet_20 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "neuralnet_ode_type2",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        computation_processor = "cpu_processes",
        sample_time = 5.0,
    )

    #Get best models
    neuralnet_0_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_0[1]).machine).best_fitted_params.chain
    neuralnet_1_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_1[1]).machine).best_fitted_params.chain
    neuralnet_2_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_2[1]).machine).best_fitted_params.chain
    neuralnet_3_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_3[1]).machine).best_fitted_params.chain
    neuralnet_4_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_4[1]).machine).best_fitted_params.chain
    neuralnet_5_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_5[1]).machine).best_fitted_params.chain
    neuralnet_6_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_6[1]).machine).best_fitted_params.chain
    neuralnet_7_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_7[1]).machine).best_fitted_params.chain
    neuralnet_8_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_8[1]).machine).best_fitted_params.chain
    neuralnet_9_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_9[1]).machine).best_fitted_params.chain
    neuralnet_10_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_10[1]).machine).best_fitted_params.chain
    neuralnet_11_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_11[1]).machine).best_fitted_params.chain
    neuralnet_12_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_12[1]).machine).best_fitted_params.chain
    neuralnet_13_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_13[1]).machine).best_fitted_params.chain
    neuralnet_14_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_14[1]).machine).best_fitted_params.chain
    neuralnet_15_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_15[1]).machine).best_fitted_params.chain
    neuralnet_16_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_16[1]).machine).best_fitted_params.chain
    neuralnet_17_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_17[1]).machine).best_fitted_params.chain
    neuralnet_18_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_18[1]).machine).best_fitted_params.chain
    neuralnet_19_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_19[1]).machine).best_fitted_params.chain
    neuralnet_20_chain_best_model_chain =
        fitted_params(fitted_params(m_neuralnet_20[1]).machine).best_fitted_params.chain

    # MAE with train and test data
    mae_Train_neuralnet_0 = Flux.mae(
        neuralnet_0_chain_best_model_chain(Matrix(m_neuralnet_0[2].TrainDataIn)'),
        Matrix(m_neuralnet_0[2].TrainDataOut)',
    )
    mae_Train_neuralnet_1 = Flux.mae(
        neuralnet_1_chain_best_model_chain(Matrix(m_neuralnet_1[2].TrainDataIn)'),
        Matrix(m_neuralnet_1[2].TrainDataOut)',
    )
    mae_Train_neuralnet_2 = Flux.mae(
        neuralnet_2_chain_best_model_chain(Matrix(m_neuralnet_2[2].TrainDataIn)'),
        Matrix(m_neuralnet_2[2].TrainDataOut)',
    )
    mae_Train_neuralnet_3 = Flux.mae(
        neuralnet_3_chain_best_model_chain(Matrix(m_neuralnet_3[2].TrainDataIn)'),
        Matrix(m_neuralnet_3[2].TrainDataOut)',
    )
    mae_Train_neuralnet_4 = Flux.mae(
        neuralnet_4_chain_best_model_chain(Matrix(m_neuralnet_4[2].TrainDataIn)'),
        Matrix(m_neuralnet_4[2].TrainDataOut)',
    )
    mae_Train_neuralnet_5 = Flux.mae(
        neuralnet_5_chain_best_model_chain(Matrix(m_neuralnet_5[2].TrainDataIn)'),
        Matrix(m_neuralnet_5[2].TrainDataOut)',
    )
    mae_Train_neuralnet_6 = Flux.mae(
        neuralnet_6_chain_best_model_chain(Matrix(m_neuralnet_6[2].TrainDataIn)'),
        Matrix(m_neuralnet_6[2].TrainDataOut)',
    )
    mae_Train_neuralnet_7 = Flux.mae(
        neuralnet_7_chain_best_model_chain(Matrix(m_neuralnet_7[2].TrainDataIn)'),
        Matrix(m_neuralnet_7[2].TrainDataOut)',
    )
    mae_Train_neuralnet_8 = Flux.mae(
        neuralnet_8_chain_best_model_chain(Matrix(m_neuralnet_8[2].TrainDataIn)'),
        Matrix(m_neuralnet_8[2].TrainDataOut)',
    )
    mae_Train_neuralnet_9 = Flux.mae(
        neuralnet_9_chain_best_model_chain(Matrix(m_neuralnet_9[2].TrainDataIn)'),
        Matrix(m_neuralnet_9[2].TrainDataOut)',
    )
    mae_Train_neuralnet_10 = Flux.mae(
        neuralnet_10_chain_best_model_chain(Matrix(m_neuralnet_10[2].TrainDataIn)'),
        Matrix(m_neuralnet_10[2].TrainDataOut)',
    )
    mae_Train_neuralnet_11 = Flux.mae(
        neuralnet_11_chain_best_model_chain(Matrix(m_neuralnet_11[2].TrainDataIn)'),
        Matrix(m_neuralnet_11[2].TrainDataOut)',
    )
    mae_Train_neuralnet_12 = Flux.mae(
        neuralnet_12_chain_best_model_chain(Matrix(m_neuralnet_12[2].TrainDataIn)'),
        Matrix(m_neuralnet_12[2].TrainDataOut)',
    )
    mae_Train_neuralnet_13 = Flux.mae(
        neuralnet_13_chain_best_model_chain(Matrix(m_neuralnet_13[2].TrainDataIn)'),
        Matrix(m_neuralnet_13[2].TrainDataOut)',
    )
    mae_Train_neuralnet_14 = Flux.mae(
        neuralnet_14_chain_best_model_chain(Matrix(m_neuralnet_14[2].TrainDataIn)'),
        Matrix(m_neuralnet_14[2].TrainDataOut)',
    )
    mae_Train_neuralnet_15 = Flux.mae(
        neuralnet_15_chain_best_model_chain(Matrix(m_neuralnet_15[2].TrainDataIn)'),
        Matrix(m_neuralnet_15[2].TrainDataOut)',
    )
    mae_Train_neuralnet_16 = Flux.mae(
        neuralnet_16_chain_best_model_chain(Matrix(m_neuralnet_16[2].TrainDataIn)'),
        Matrix(m_neuralnet_16[2].TrainDataOut)',
    )
    mae_Train_neuralnet_17 = Flux.mae(
        neuralnet_17_chain_best_model_chain(Matrix(m_neuralnet_17[2].TrainDataIn)'),
        Matrix(m_neuralnet_17[2].TrainDataOut)',
    )
    mae_Train_neuralnet_18 = Flux.mae(
        neuralnet_18_chain_best_model_chain(Matrix(m_neuralnet_18[2].TrainDataIn)'),
        Matrix(m_neuralnet_18[2].TrainDataOut)',
    )
    mae_Train_neuralnet_19 = Flux.mae(
        neuralnet_19_chain_best_model_chain(Matrix(m_neuralnet_19[2].TrainDataIn)'),
        Matrix(m_neuralnet_19[2].TrainDataOut)',
    )
    mae_Train_neuralnet_20 = Flux.mae(
        neuralnet_20_chain_best_model_chain(Matrix(m_neuralnet_20[2].TrainDataIn)'),
        Matrix(m_neuralnet_20[2].TrainDataOut)',
    )

    mae_Test_neuralnet_0 = Flux.mae(
        neuralnet_0_chain_best_model_chain(Matrix(m_neuralnet_0[2].TestDataIn)'),
        Matrix(m_neuralnet_0[2].TestDataOut)',
    )
    mae_Test_neuralnet_1 = Flux.mae(
        neuralnet_1_chain_best_model_chain(Matrix(m_neuralnet_1[2].TestDataIn)'),
        Matrix(m_neuralnet_1[2].TestDataOut)',
    )
    mae_Test_neuralnet_2 = Flux.mae(
        neuralnet_2_chain_best_model_chain(Matrix(m_neuralnet_2[2].TestDataIn)'),
        Matrix(m_neuralnet_2[2].TestDataOut)',
    )
    mae_Test_neuralnet_3 = Flux.mae(
        neuralnet_3_chain_best_model_chain(Matrix(m_neuralnet_3[2].TestDataIn)'),
        Matrix(m_neuralnet_3[2].TestDataOut)',
    )
    mae_Test_neuralnet_4 = Flux.mae(
        neuralnet_4_chain_best_model_chain(Matrix(m_neuralnet_4[2].TestDataIn)'),
        Matrix(m_neuralnet_4[2].TestDataOut)',
    )
    mae_Test_neuralnet_5 = Flux.mae(
        neuralnet_5_chain_best_model_chain(Matrix(m_neuralnet_5[2].TestDataIn)'),
        Matrix(m_neuralnet_5[2].TestDataOut)',
    )
    mae_Test_neuralnet_6 = Flux.mae(
        neuralnet_6_chain_best_model_chain(Matrix(m_neuralnet_6[2].TestDataIn)'),
        Matrix(m_neuralnet_6[2].TestDataOut)',
    )
    mae_Test_neuralnet_7 = Flux.mae(
        neuralnet_7_chain_best_model_chain(Matrix(m_neuralnet_7[2].TestDataIn)'),
        Matrix(m_neuralnet_7[2].TestDataOut)',
    )
    mae_Test_neuralnet_8 = Flux.mae(
        neuralnet_8_chain_best_model_chain(Matrix(m_neuralnet_8[2].TestDataIn)'),
        Matrix(m_neuralnet_8[2].TestDataOut)',
    )
    mae_Test_neuralnet_9 = Flux.mae(
        neuralnet_9_chain_best_model_chain(Matrix(m_neuralnet_9[2].TestDataIn)'),
        Matrix(m_neuralnet_9[2].TestDataOut)',
    )
    mae_Test_neuralnet_10 = Flux.mae(
        neuralnet_10_chain_best_model_chain(Matrix(m_neuralnet_10[2].TestDataIn)'),
        Matrix(m_neuralnet_10[2].TestDataOut)',
    )
    mae_Test_neuralnet_11 = Flux.mae(
        neuralnet_11_chain_best_model_chain(Matrix(m_neuralnet_11[2].TestDataIn)'),
        Matrix(m_neuralnet_11[2].TestDataOut)',
    )
    mae_Test_neuralnet_12 = Flux.mae(
        neuralnet_12_chain_best_model_chain(Matrix(m_neuralnet_12[2].TestDataIn)'),
        Matrix(m_neuralnet_12[2].TestDataOut)',
    )
    mae_Test_neuralnet_13 = Flux.mae(
        neuralnet_13_chain_best_model_chain(Matrix(m_neuralnet_13[2].TestDataIn)'),
        Matrix(m_neuralnet_13[2].TestDataOut)',
    )
    mae_Test_neuralnet_14 = Flux.mae(
        neuralnet_14_chain_best_model_chain(Matrix(m_neuralnet_14[2].TestDataIn)'),
        Matrix(m_neuralnet_14[2].TestDataOut)',
    )
    mae_Test_neuralnet_15 = Flux.mae(
        neuralnet_15_chain_best_model_chain(Matrix(m_neuralnet_15[2].TestDataIn)'),
        Matrix(m_neuralnet_15[2].TestDataOut)',
    )
    mae_Test_neuralnet_16 = Flux.mae(
        neuralnet_16_chain_best_model_chain(Matrix(m_neuralnet_16[2].TestDataIn)'),
        Matrix(m_neuralnet_16[2].TestDataOut)',
    )
    mae_Test_neuralnet_17 = Flux.mae(
        neuralnet_17_chain_best_model_chain(Matrix(m_neuralnet_17[2].TestDataIn)'),
        Matrix(m_neuralnet_17[2].TestDataOut)',
    )
    mae_Test_neuralnet_18 = Flux.mae(
        neuralnet_18_chain_best_model_chain(Matrix(m_neuralnet_18[2].TestDataIn)'),
        Matrix(m_neuralnet_18[2].TestDataOut)',
    )
    mae_Test_neuralnet_19 = Flux.mae(
        neuralnet_19_chain_best_model_chain(Matrix(m_neuralnet_19[2].TestDataIn)'),
        Matrix(m_neuralnet_19[2].TestDataOut)',
    )
    mae_Test_neuralnet_20 = Flux.mae(
        neuralnet_20_chain_best_model_chain(Matrix(m_neuralnet_20[2].TestDataIn)'),
        Matrix(m_neuralnet_20[2].TestDataOut)',
    )

    println("mae_Train_neuralnet_0 $mae_Train_neuralnet_0")
    println("mae_Train_neuralnet_1 $mae_Train_neuralnet_1")
    println("mae_Train_neuralnet_2 $mae_Train_neuralnet_2")
    println("mae_Train_neuralnet_3 $mae_Train_neuralnet_3")
    println("mae_Train_neuralnet_4 $mae_Train_neuralnet_4")
    println("mae_Train_neuralnet_5 $mae_Train_neuralnet_5")
    println("mae_Train_neuralnet_6 $mae_Train_neuralnet_6")
    println("mae_Train_neuralnet_7 $mae_Train_neuralnet_7")
    println("mae_Train_neuralnet_8 $mae_Train_neuralnet_8")
    println("mae_Train_neuralnet_9 $mae_Train_neuralnet_9")
    println("mae_Train_neuralnet_10 $mae_Train_neuralnet_10")
    println("mae_Train_neuralnet_11 $mae_Train_neuralnet_11")
    println("mae_Train_neuralnet_12 $mae_Train_neuralnet_12")
    println("mae_Train_neuralnet_13 $mae_Train_neuralnet_13")
    println("mae_Train_neuralnet_14 $mae_Train_neuralnet_14")
    println("mae_Train_neuralnet_15 $mae_Train_neuralnet_15")
    println("mae_Train_neuralnet_16 $mae_Train_neuralnet_16")
    println("mae_Train_neuralnet_17 $mae_Train_neuralnet_17")
    println("mae_Train_neuralnet_18 $mae_Train_neuralnet_18")
    println("mae_Train_neuralnet_19 $mae_Train_neuralnet_19")
    println("mae_Train_neuralnet_20 $mae_Train_neuralnet_20")

    println("mae_Test_neuralnet_0 $mae_Test_neuralnet_0")
    println("mae_Test_neuralnet_1 $mae_Test_neuralnet_1")
    println("mae_Test_neuralnet_2 $mae_Test_neuralnet_2")
    println("mae_Test_neuralnet_3 $mae_Test_neuralnet_3")
    println("mae_Test_neuralnet_4 $mae_Test_neuralnet_4")
    println("mae_Test_neuralnet_5 $mae_Test_neuralnet_5")
    println("mae_Test_neuralnet_6 $mae_Test_neuralnet_6")
    println("mae_Test_neuralnet_7 $mae_Test_neuralnet_7")
    println("mae_Test_neuralnet_8 $mae_Test_neuralnet_8")
    println("mae_Test_neuralnet_9 $mae_Test_neuralnet_9")
    println("mae_Test_neuralnet_10 $mae_Test_neuralnet_10")
    println("mae_Test_neuralnet_11 $mae_Test_neuralnet_11")
    println("mae_Test_neuralnet_12 $mae_Test_neuralnet_12")
    println("mae_Test_neuralnet_13 $mae_Test_neuralnet_13")
    println("mae_Test_neuralnet_14 $mae_Test_neuralnet_14")
    println("mae_Test_neuralnet_15 $mae_Test_neuralnet_15")
    println("mae_Test_neuralnet_16 $mae_Test_neuralnet_16")
    println("mae_Test_neuralnet_17 $mae_Test_neuralnet_17")
    println("mae_Test_neuralnet_18 $mae_Test_neuralnet_18")
    println("mae_Test_neuralnet_19 $mae_Test_neuralnet_19")
    println("mae_Test_neuralnet_20 $mae_Test_neuralnet_20")

    @test mae_Train_neuralnet_0 <= 10
    @test mae_Train_neuralnet_1 <= 10
    @test mae_Train_neuralnet_2 <= 10
    @test mae_Train_neuralnet_3 <= 10
    @test mae_Train_neuralnet_4 <= 10
    @test mae_Train_neuralnet_5 <= 10
    @test mae_Train_neuralnet_6 != NaN
    @test mae_Train_neuralnet_7 <= 10
    @test mae_Train_neuralnet_8 <= 10
    @test mae_Train_neuralnet_9 <= 10
    @test mae_Train_neuralnet_10 <= 10
    @test mae_Train_neuralnet_11 <= 10
    @test mae_Train_neuralnet_12 <= 10
    @test mae_Train_neuralnet_13 != NaN
    @test mae_Train_neuralnet_14 <= 10
    @test mae_Train_neuralnet_15 <= 10
    @test mae_Train_neuralnet_16 <= 10
    @test mae_Train_neuralnet_17 <= 10
    @test mae_Train_neuralnet_18 <= 10
    @test mae_Train_neuralnet_19 <= 10
    @test mae_Train_neuralnet_20 != NaN

    @test mae_Test_neuralnet_0 <= 10
    @test mae_Test_neuralnet_1 <= 10
    @test mae_Test_neuralnet_2 <= 10
    @test mae_Test_neuralnet_3 <= 10
    @test mae_Test_neuralnet_4 <= 10
    @test mae_Test_neuralnet_5 <= 10
    @test mae_Test_neuralnet_6 != NaN
    @test mae_Test_neuralnet_7 <= 10
    @test mae_Test_neuralnet_8 <= 10
    @test mae_Test_neuralnet_9 <= 10
    @test mae_Test_neuralnet_10 <= 10
    @test mae_Test_neuralnet_11 <= 10
    @test mae_Test_neuralnet_12 <= 10
    @test mae_Test_neuralnet_13 != NaN
    @test mae_Test_neuralnet_14 <= 10
    @test mae_Test_neuralnet_15 <= 10
    @test mae_Test_neuralnet_16 <= 10
    @test mae_Test_neuralnet_17 <= 10
    @test mae_Test_neuralnet_18 <= 10
    @test mae_Test_neuralnet_19 <= 10
    @test mae_Test_neuralnet_20 != NaN

end


@testset "Exploration of networks architecture blackbox identification" begin

    #load data 
    dfout_raw = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:2500, :]
    dfin_raw = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:2500, :]

    #Train models
    max_time = Minute(5) #hours

    lower_in = [0.2 0.2 0.2 0.2 0 0]
    upper_in = [1.2 1.2 1.2 1.2 Inf Inf]

    lower_out = [0.2 0.2 0.2 0.2]
    upper_out = [1.2 1.2 1.2 1.2]

    # Separate data between test and train data
    DataTrainTest = data_formatting_identification(
        dfin_raw,
        dfout_raw,
        data_type = Float32,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )
    
    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    m_exploration_0 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
    )

    m_exploration_1 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
    )

    m_exploration_2 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
    )

    m_exploration_3 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
    )

    m_exploration_4 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
    )

    m_exploration_5 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
    )

    m_exploration_6 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
    )

    ### CPU threads ###

    m_exploration_7 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_threads",
    )

    m_exploration_8 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_threads",
    )

    m_exploration_9 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_threads",
    )

    m_exploration_10 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_threads",
    )

    m_exploration_11 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_threads",
    )

    m_exploration_12 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_threads",
    )

    m_exploration_13 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_threads",
    )

    ### CPU processes ###

    m_exploration_14 = proceed_identification(
        in_data,
        out_data,
        "adam",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_processes",
    )
    
    m_exploration_15 = proceed_identification(
        in_data,
        out_data,
        "radam",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_processes",
    )
    
    m_exploration_16 = proceed_identification(
        in_data,
        out_data,
        "nadam",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_processes",
    )
    
    m_exploration_17 = proceed_identification(
        in_data,
        out_data,
        "oadam",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_processes",
    )
    
    m_exploration_18 = proceed_identification(
        in_data,
        out_data,
        "lbfgs",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_processes",
    )
    
    m_exploration_19 = proceed_identification(
        in_data,
        out_data,
        "oaccel",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_processes",
    )
    
    m_exploration_20 = proceed_identification(
        in_data,
        out_data,
        "pso",
        "exploration_models",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        model_exploration = ["fnn", "resnet", "densenet"],
        computation_processor = "cpu_processes",
    )

   #Get best models
exploration_0_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_0[1]).machine).best_fitted_params.chain
exploration_1_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_1[1]).machine).best_fitted_params.chain
exploration_2_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_2[1]).machine).best_fitted_params.chain
exploration_3_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_3[1]).machine).best_fitted_params.chain
exploration_4_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_4[1]).machine).best_fitted_params.chain
exploration_5_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_5[1]).machine).best_fitted_params.chain
exploration_6_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_6[1]).machine).best_fitted_params.chain
exploration_7_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_7[1]).machine).best_fitted_params.chain
exploration_8_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_8[1]).machine).best_fitted_params.chain
exploration_9_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_9[1]).machine).best_fitted_params.chain
exploration_10_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_10[1]).machine).best_fitted_params.chain
exploration_11_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_11[1]).machine).best_fitted_params.chain
exploration_12_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_12[1]).machine).best_fitted_params.chain
exploration_13_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_13[1]).machine).best_fitted_params.chain
exploration_14_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_14[1]).machine).best_fitted_params.chain
exploration_15_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_15[1]).machine).best_fitted_params.chain
exploration_16_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_16[1]).machine).best_fitted_params.chain
exploration_17_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_17[1]).machine).best_fitted_params.chain
exploration_18_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_18[1]).machine).best_fitted_params.chain
exploration_19_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_19[1]).machine).best_fitted_params.chain
exploration_20_chain_best_model_chain =
fitted_params(fitted_params(m_exploration_20[1]).machine).best_fitted_params.chain

# MAE with train and test data
mae_Train_exploration_0 = Flux.mae(
exploration_0_chain_best_model_chain(Matrix(m_exploration_0[2].TrainDataIn)'),
Matrix(m_exploration_0[2].TrainDataOut)',
)
mae_Train_exploration_1 = Flux.mae(
exploration_1_chain_best_model_chain(Matrix(m_exploration_1[2].TrainDataIn)'),
Matrix(m_exploration_1[2].TrainDataOut)',
)
mae_Train_exploration_2 = Flux.mae(
exploration_2_chain_best_model_chain(Matrix(m_exploration_2[2].TrainDataIn)'),
Matrix(m_exploration_2[2].TrainDataOut)',
)
mae_Train_exploration_3 = Flux.mae(
exploration_3_chain_best_model_chain(Matrix(m_exploration_3[2].TrainDataIn)'),
Matrix(m_exploration_3[2].TrainDataOut)',
)
mae_Train_exploration_4 = Flux.mae(
exploration_4_chain_best_model_chain(Matrix(m_exploration_4[2].TrainDataIn)'),
Matrix(m_exploration_4[2].TrainDataOut)',
)
mae_Train_exploration_5 = Flux.mae(
exploration_5_chain_best_model_chain(Matrix(m_exploration_5[2].TrainDataIn)'),
Matrix(m_exploration_5[2].TrainDataOut)',
)
mae_Train_exploration_6 = Flux.mae(
exploration_6_chain_best_model_chain(Matrix(m_exploration_6[2].TrainDataIn)'),
Matrix(m_exploration_6[2].TrainDataOut)',
)
mae_Train_exploration_7 = Flux.mae(
exploration_7_chain_best_model_chain(Matrix(m_exploration_7[2].TrainDataIn)'),
Matrix(m_exploration_7[2].TrainDataOut)',
)
mae_Train_exploration_8 = Flux.mae(
exploration_8_chain_best_model_chain(Matrix(m_exploration_8[2].TrainDataIn)'),
Matrix(m_exploration_8[2].TrainDataOut)',
)
mae_Train_exploration_9 = Flux.mae(
exploration_9_chain_best_model_chain(Matrix(m_exploration_9[2].TrainDataIn)'),
Matrix(m_exploration_9[2].TrainDataOut)',
)
mae_Train_exploration_10 = Flux.mae(
exploration_10_chain_best_model_chain(Matrix(m_exploration_10[2].TrainDataIn)'),
Matrix(m_exploration_10[2].TrainDataOut)',
)
mae_Train_exploration_11 = Flux.mae(
exploration_11_chain_best_model_chain(Matrix(m_exploration_11[2].TrainDataIn)'),
Matrix(m_exploration_11[2].TrainDataOut)',
)
mae_Train_exploration_12 = Flux.mae(
exploration_12_chain_best_model_chain(Matrix(m_exploration_12[2].TrainDataIn)'),
Matrix(m_exploration_12[2].TrainDataOut)',
)
mae_Train_exploration_13 = Flux.mae(
exploration_13_chain_best_model_chain(Matrix(m_exploration_13[2].TrainDataIn)'),
Matrix(m_exploration_13[2].TrainDataOut)',
)
mae_Train_exploration_14 = Flux.mae(
exploration_14_chain_best_model_chain(Matrix(m_exploration_14[2].TrainDataIn)'),
Matrix(m_exploration_14[2].TrainDataOut)',
)
mae_Train_exploration_15 = Flux.mae(
exploration_15_chain_best_model_chain(Matrix(m_exploration_15[2].TrainDataIn)'),
Matrix(m_exploration_15[2].TrainDataOut)',
)
mae_Train_exploration_16 = Flux.mae(
exploration_16_chain_best_model_chain(Matrix(m_exploration_16[2].TrainDataIn)'),
Matrix(m_exploration_16[2].TrainDataOut)',
)
mae_Train_exploration_17 = Flux.mae(
exploration_17_chain_best_model_chain(Matrix(m_exploration_17[2].TrainDataIn)'),
Matrix(m_exploration_17[2].TrainDataOut)',
)
mae_Train_exploration_18 = Flux.mae(
exploration_18_chain_best_model_chain(Matrix(m_exploration_18[2].TrainDataIn)'),
Matrix(m_exploration_18[2].TrainDataOut)',
)
mae_Train_exploration_19 = Flux.mae(
exploration_19_chain_best_model_chain(Matrix(m_exploration_19[2].TrainDataIn)'),
Matrix(m_exploration_19[2].TrainDataOut)',
)
mae_Train_exploration_20 = Flux.mae(
exploration_20_chain_best_model_chain(Matrix(m_exploration_20[2].TrainDataIn)'),
Matrix(m_exploration_20[2].TrainDataOut)',
)

mae_Test_exploration_0 = Flux.mae(
exploration_0_chain_best_model_chain(Matrix(m_exploration_0[2].TestDataIn)'),
Matrix(m_exploration_0[2].TestDataOut)',
)
mae_Test_exploration_1 = Flux.mae(
exploration_1_chain_best_model_chain(Matrix(m_exploration_1[2].TestDataIn)'),
Matrix(m_exploration_1[2].TestDataOut)',
)
mae_Test_exploration_2 = Flux.mae(
exploration_2_chain_best_model_chain(Matrix(m_exploration_2[2].TestDataIn)'),
Matrix(m_exploration_2[2].TestDataOut)',
)
mae_Test_exploration_3 = Flux.mae(
exploration_3_chain_best_model_chain(Matrix(m_exploration_3[2].TestDataIn)'),
Matrix(m_exploration_3[2].TestDataOut)',
)
mae_Test_exploration_4 = Flux.mae(
exploration_4_chain_best_model_chain(Matrix(m_exploration_4[2].TestDataIn)'),
Matrix(m_exploration_4[2].TestDataOut)',
)
mae_Test_exploration_5 = Flux.mae(
exploration_5_chain_best_model_chain(Matrix(m_exploration_5[2].TestDataIn)'),
Matrix(m_exploration_5[2].TestDataOut)',
)
mae_Test_exploration_6 = Flux.mae(
exploration_6_chain_best_model_chain(Matrix(m_exploration_6[2].TestDataIn)'),
Matrix(m_exploration_6[2].TestDataOut)',
)
mae_Test_exploration_7 = Flux.mae(
exploration_7_chain_best_model_chain(Matrix(m_exploration_7[2].TestDataIn)'),
Matrix(m_exploration_7[2].TestDataOut)',
)
mae_Test_exploration_8 = Flux.mae(
exploration_8_chain_best_model_chain(Matrix(m_exploration_8[2].TestDataIn)'),
Matrix(m_exploration_8[2].TestDataOut)',
)
mae_Test_exploration_9 = Flux.mae(
exploration_9_chain_best_model_chain(Matrix(m_exploration_9[2].TestDataIn)'),
Matrix(m_exploration_9[2].TestDataOut)',
)
mae_Test_exploration_10 = Flux.mae(
exploration_10_chain_best_model_chain(Matrix(m_exploration_10[2].TestDataIn)'),
Matrix(m_exploration_10[2].TestDataOut)',
)
mae_Test_exploration_11 = Flux.mae(
exploration_11_chain_best_model_chain(Matrix(m_exploration_11[2].TestDataIn)'),
Matrix(m_exploration_11[2].TestDataOut)',
)
mae_Test_exploration_12 = Flux.mae(
exploration_12_chain_best_model_chain(Matrix(m_exploration_12[2].TestDataIn)'),
Matrix(m_exploration_12[2].TestDataOut)',
)
mae_Test_exploration_13 = Flux.mae(
exploration_13_chain_best_model_chain(Matrix(m_exploration_13[2].TestDataIn)'),
Matrix(m_exploration_13[2].TestDataOut)',
)
mae_Test_exploration_14 = Flux.mae(
exploration_14_chain_best_model_chain(Matrix(m_exploration_14[2].TestDataIn)'),
Matrix(m_exploration_14[2].TestDataOut)',
)
mae_Test_exploration_15 = Flux.mae(
exploration_15_chain_best_model_chain(Matrix(m_exploration_15[2].TestDataIn)'),
Matrix(m_exploration_15[2].TestDataOut)',
)
mae_Test_exploration_16 = Flux.mae(
exploration_16_chain_best_model_chain(Matrix(m_exploration_16[2].TestDataIn)'),
Matrix(m_exploration_16[2].TestDataOut)',
)
mae_Test_exploration_17 = Flux.mae(
exploration_17_chain_best_model_chain(Matrix(m_exploration_17[2].TestDataIn)'),
Matrix(m_exploration_17[2].TestDataOut)',
)
mae_Test_exploration_18 = Flux.mae(
exploration_18_chain_best_model_chain(Matrix(m_exploration_18[2].TestDataIn)'),
Matrix(m_exploration_18[2].TestDataOut)',
)
mae_Test_exploration_19 = Flux.mae(
exploration_19_chain_best_model_chain(Matrix(m_exploration_19[2].TestDataIn)'),
Matrix(m_exploration_19[2].TestDataOut)',
)
mae_Test_exploration_20 = Flux.mae(
exploration_20_chain_best_model_chain(Matrix(m_exploration_20[2].TestDataIn)'),
Matrix(m_exploration_20[2].TestDataOut)',
)

println("mae_Train_exploration_0 $mae_Train_exploration_0")
println("mae_Train_exploration_1 $mae_Train_exploration_1")
println("mae_Train_exploration_2 $mae_Train_exploration_2")
println("mae_Train_exploration_3 $mae_Train_exploration_3")
println("mae_Train_exploration_4 $mae_Train_exploration_4")
println("mae_Train_exploration_5 $mae_Train_exploration_5")
println("mae_Train_exploration_6 $mae_Train_exploration_6")
println("mae_Train_exploration_7 $mae_Train_exploration_7")
println("mae_Train_exploration_8 $mae_Train_exploration_8")
println("mae_Train_exploration_9 $mae_Train_exploration_9")
println("mae_Train_exploration_10 $mae_Train_exploration_10")
println("mae_Train_exploration_11 $mae_Train_exploration_11")
println("mae_Train_exploration_12 $mae_Train_exploration_12")
println("mae_Train_exploration_13 $mae_Train_exploration_13")
println("mae_Train_exploration_14 $mae_Train_exploration_14")
println("mae_Train_exploration_15 $mae_Train_exploration_15")
println("mae_Train_exploration_16 $mae_Train_exploration_16")
println("mae_Train_exploration_17 $mae_Train_exploration_17")
println("mae_Train_exploration_18 $mae_Train_exploration_18")
println("mae_Train_exploration_19 $mae_Train_exploration_19")
println("mae_Train_exploration_20 $mae_Train_exploration_20")

println("mae_Test_exploration_0 $mae_Test_exploration_0")
println("mae_Test_exploration_1 $mae_Test_exploration_1")
println("mae_Test_exploration_2 $mae_Test_exploration_2")
println("mae_Test_exploration_3 $mae_Test_exploration_3")
println("mae_Test_exploration_4 $mae_Test_exploration_4")
println("mae_Test_exploration_5 $mae_Test_exploration_5")
println("mae_Test_exploration_6 $mae_Test_exploration_6")
println("mae_Test_exploration_7 $mae_Test_exploration_7")
println("mae_Test_exploration_8 $mae_Test_exploration_8")
println("mae_Test_exploration_9 $mae_Test_exploration_9")
println("mae_Test_exploration_10 $mae_Test_exploration_10")
println("mae_Test_exploration_11 $mae_Test_exploration_11")
println("mae_Test_exploration_12 $mae_Test_exploration_12")
println("mae_Test_exploration_13 $mae_Test_exploration_13")
println("mae_Test_exploration_14 $mae_Test_exploration_14")
println("mae_Test_exploration_15 $mae_Test_exploration_15")
println("mae_Test_exploration_16 $mae_Test_exploration_16")
println("mae_Test_exploration_17 $mae_Test_exploration_17")
println("mae_Test_exploration_18 $mae_Test_exploration_18")
println("mae_Test_exploration_19 $mae_Test_exploration_19")
println("mae_Test_exploration_20 $mae_Test_exploration_20")

@test mae_Train_exploration_0 <= 10
@test mae_Train_exploration_1 <= 10
@test mae_Train_exploration_2 <= 10
@test mae_Train_exploration_3 <= 10
@test mae_Train_exploration_4 <= 10
@test mae_Train_exploration_5 <= 10
@test mae_Train_exploration_6 != NaN
@test mae_Train_exploration_7 <= 10
@test mae_Train_exploration_8 <= 10
@test mae_Train_exploration_9 <= 10
@test mae_Train_exploration_10 <= 10
@test mae_Train_exploration_11 <= 10
@test mae_Train_exploration_12 <= 10
@test mae_Train_exploration_13 != NaN
@test mae_Train_exploration_14 <= 10
@test mae_Train_exploration_15 <= 10
@test mae_Train_exploration_16 <= 10
@test mae_Train_exploration_17 <= 10
@test mae_Train_exploration_18 <= 10
@test mae_Train_exploration_19 <= 10
@test mae_Train_exploration_20 != NaN

@test mae_Test_exploration_0 <= 10
@test mae_Test_exploration_1 <= 10
@test mae_Test_exploration_2 <= 10
@test mae_Test_exploration_3 <= 10
@test mae_Test_exploration_4 <= 10
@test mae_Test_exploration_5 <= 10
@test mae_Test_exploration_6 != NaN
@test mae_Test_exploration_7 <= 10
@test mae_Test_exploration_8 <= 10
@test mae_Test_exploration_9 <= 10
@test mae_Test_exploration_10 <= 10
@test mae_Test_exploration_11 <= 10
@test mae_Test_exploration_12 <= 10
@test mae_Test_exploration_13 != NaN
@test mae_Test_exploration_14 <= 10
@test mae_Test_exploration_15 <= 10
@test mae_Test_exploration_16 <= 10
@test mae_Test_exploration_17 <= 10
@test mae_Test_exploration_18 <= 10
@test mae_Test_exploration_19 <= 10
@test mae_Test_exploration_20 != NaN

end

end