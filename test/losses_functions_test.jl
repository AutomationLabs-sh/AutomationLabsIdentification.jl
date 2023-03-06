# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
module BlackBoxIdentificationTestLosses

using AutomationLabsIdentification

using Test
using CSV
using DataFrames
using MLJ
using Flux
using Dates
using Statistics
using Distributed

#import AutomationLabsIdentification: mse_losses
#import AutomationLabsIdentification: rmse_losses
#import AutomationLabsIdentification: mae_losses
#import AutomationLabsIdentification: mape_losses

@testset "Fnn evaluation mse rmse mae mape" begin

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

    mae_losses = function (x, y)
        return Flux.Losses.mae(x, y)
    end

    mse_losses = function (x, y)
        return Flux.Losses.mse(x, y)
    end

    rmse_losses = function (x, y)
        return sqrt(Flux.Losses.mse(x, y))
    end

    mape_losses = function (x, y)
        return Statistics.mean(abs.((x .- y) ./ y))
    end

    m_fnn_mse = proceed_identification(
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
        neuralnet_loss_function = "mse",
    )

    m_fnn_rmse = proceed_identification(
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
        neuralnet_loss_function = "rmse",
    )

    m_fnn_mae = proceed_identification(
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
        neuralnet_loss_function = "mae",
    )

    m_fnn_mape = proceed_identification(
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
        neuralnet_loss_function = "mape",
    )

    report_fnn_mse = report(fitted_params(m_fnn_mse).machine).best_history_entry
    report_fnn_rmse = report(fitted_params(m_fnn_rmse).machine).best_history_entry
    report_fnn_mae = report(fitted_params(m_fnn_mae).machine).best_history_entry
    report_fnn_mape = report(fitted_params(m_fnn_mape).machine).best_history_entry

    @test report_fnn_mse.measurement[1] < 1
    @test report_fnn_rmse.measurement[1] < 1
    @test report_fnn_mae.measurement[1] < 1
    @test report_fnn_mape.measurement[1] < 1

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    fnn_mse_chain = fitted_params(fitted_params(m_fnn_mse).machine).best_fitted_params.chain
    fnn_rmse_chain =
        fitted_params(fitted_params(m_fnn_rmse).machine).best_fitted_params.chain
    fnn_mae_chain = fitted_params(fitted_params(m_fnn_mae).machine).best_fitted_params.chain
    fnn_mape_chain =
        fitted_params(fitted_params(m_fnn_mape).machine).best_fitted_params.chain

    mse_fnn_from_fnn_mse =
        mse_losses(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_mse =
        rmse_losses(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mae_fnn_from_fnn_mse =
        mae_losses(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_mse =
        mape_losses(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')

    mse_fnn_from_fnn_mse_2 =
        Flux.Losses.mse(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_mse_2 = sqrt(
        Flux.Losses.mse(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)'),
    )
    mae_fnn_from_fnn_mse_2 =
        Flux.Losses.mae(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_mse_2 = Statistics.mean(
        abs.(
            (fnn_mse_chain(Matrix(Train_in_data)') .- Matrix(Train_out_data)') ./
            Matrix(Train_out_data)'
        ),
    )

    @test mse_fnn_from_fnn_mse == mse_fnn_from_fnn_mse_2
    @test rmse_fnn_from_fnn_mse == rmse_fnn_from_fnn_mse_2
    @test mae_fnn_from_fnn_mse == mae_fnn_from_fnn_mse_2
    @test mape_fnn_from_fnn_mse == mape_fnn_from_fnn_mse_2

    mse_fnn_from_fnn_rmse =
        mse_losses(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_rmse =
        rmse_losses(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mae_fnn_from_fnn_rmse =
        mae_losses(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_rmse =
        mape_losses(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')

    mse_fnn_from_fnn_rmse_2 =
        Flux.Losses.mse(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_rmse_2 = sqrt(
        Flux.Losses.mse(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)'),
    )
    mae_fnn_from_fnn_rmse_2 =
        Flux.Losses.mae(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_rmse_2 = Statistics.mean(
        abs.(
            (fnn_rmse_chain(Matrix(Train_in_data)') .- Matrix(Train_out_data)') ./
            Matrix(Train_out_data)'
        ),
    )

    @test mse_fnn_from_fnn_rmse == mse_fnn_from_fnn_rmse_2
    @test rmse_fnn_from_fnn_rmse == rmse_fnn_from_fnn_rmse_2
    @test mae_fnn_from_fnn_rmse == mae_fnn_from_fnn_rmse_2
    @test mape_fnn_from_fnn_rmse == mape_fnn_from_fnn_rmse_2

    mse_fnn_from_fnn_mae =
        mse_losses(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_mae =
        rmse_losses(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mae_fnn_from_fnn_mae =
        mae_losses(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_mae =
        mape_losses(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')

    mse_fnn_from_fnn_mae_2 =
        Flux.Losses.mse(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_mae_2 = sqrt(
        Flux.Losses.mse(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)'),
    )
    mae_fnn_from_fnn_mae_2 =
        Flux.Losses.mae(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_mae_2 = Statistics.mean(
        abs.(
            (fnn_mae_chain(Matrix(Train_in_data)') .- Matrix(Train_out_data)') ./
            Matrix(Train_out_data)'
        ),
    )

    @test mse_fnn_from_fnn_mae == mse_fnn_from_fnn_mae_2
    @test rmse_fnn_from_fnn_mae == rmse_fnn_from_fnn_mae_2
    @test mae_fnn_from_fnn_mae == mae_fnn_from_fnn_mae_2
    @test mape_fnn_from_fnn_mae == mape_fnn_from_fnn_mae_2

    mse_fnn_from_fnn_mape =
        mse_losses(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_mape =
        rmse_losses(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mae_fnn_from_fnn_mape =
        mae_losses(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_mape =
        mape_losses(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')

    mse_fnn_from_fnn_mape_2 =
        Flux.Losses.mse(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_mape_2 = sqrt(
        Flux.Losses.mse(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)'),
    )
    mae_fnn_from_fnn_mape_2 =
        Flux.Losses.mae(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_mape_2 = Statistics.mean(
        abs.(
            (fnn_mape_chain(Matrix(Train_in_data)') .- Matrix(Train_out_data)') ./
            Matrix(Train_out_data)'
        ),
    )

    @test mse_fnn_from_fnn_mape == mse_fnn_from_fnn_mape_2
    @test rmse_fnn_from_fnn_mape == rmse_fnn_from_fnn_mape_2
    @test mae_fnn_from_fnn_mape == mae_fnn_from_fnn_mape_2
    @test mape_fnn_from_fnn_mape == mape_fnn_from_fnn_mape_2

end

@testset "Rnn evaluation mse rmse mae mape" begin

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

    #Train models
    max_time = Minute(5) #hours

    in_data = (DataTrainTest.TrainDataIn)
    out_data = (DataTrainTest.TrainDataOut)

    mae_losses = function (x, y)
        return Flux.Losses.mae(x, y)
    end

    mse_losses = function (x, y)
        return Flux.Losses.mse(x, y)
    end

    rmse_losses = function (x, y)
        return sqrt(Flux.Losses.mse(x, y))
    end

    mape_losses = function (x, y)
        return Statistics.mean(abs.((x .- y) ./ y))
    end

    m_rnn_mse = proceed_identification(
        in_data,
        out_data,
        "adam",
        "rnn",
        max_time;
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        neuralnet_loss_function = "mse",
        fraction_train = f_t,
        neuralnet_batch_size = n_sequence,
    )

    m_rnn_rmse = proceed_identification(
        in_data,
        out_data,
        "adam",
        "rnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        neuralnet_loss_function = "rmse",
        fraction_train = f_t,
        neuralnet_batch_size = n_sequence,
    )

    m_rnn_mae = proceed_identification(
        in_data,
        out_data,
        "adam",
        "rnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        neuralnet_loss_function = "mae",
        fraction_train = f_t,
        neuralnet_batch_size = n_sequence,
    )

    m_rnn_mape = proceed_identification(
        in_data,
        out_data,
        "adam",
        "rnn",
        max_time,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
        computation_verbosity = 0,
        neuralnet_loss_function = "mape",
        fraction_train = f_t,
        neuralnet_batch_size = n_sequence,
    )

    report_rnn_mse = report(fitted_params(m_rnn_mse).machine).best_history_entry
    report_rnn_rmse = report(fitted_params(m_rnn_rmse).machine).best_history_entry
    report_rnn_mae = report(fitted_params(m_rnn_mae).machine).best_history_entry
    report_rnn_mape = report(fitted_params(m_rnn_mape).machine).best_history_entry

    @test report_rnn_mse.measurement[1] < 1
    @test report_rnn_rmse.measurement[1] < 1
    @test report_rnn_mae.measurement[1] < 1
    @test report_rnn_mape.measurement[1] < 1

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)

    rnn_mse_chain = fitted_params(fitted_params(m_rnn_mse).machine).best_fitted_params.chain
    rnn_rmse_chain =
        fitted_params(fitted_params(m_rnn_rmse).machine).best_fitted_params.chain
    rnn_mae_chain = fitted_params(fitted_params(m_rnn_mae).machine).best_fitted_params.chain
    rnn_mape_chain =
        fitted_params(fitted_params(m_rnn_mape).machine).best_fitted_params.chain

    Flux.reset!(rnn_mse_chain)
    mse_rnn_from_rnn_mse =
        mse_losses(rnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mse_chain)
    rmse_rnn_from_rnn_mse =
        rmse_losses(rnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mse_chain)
    mae_rnn_from_rnn_mse =
        mae_losses(rnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mse_chain)
    mape_rnn_from_rnn_mse =
        mape_losses(rnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')

    Flux.reset!(rnn_mse_chain)
    mse_rnn_from_rnn_mse_2 =
        Flux.Losses.mse(rnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mse_chain)
    rmse_rnn_from_rnn_mse_2 = sqrt(
        Flux.Losses.mse(rnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)'),
    )
    Flux.reset!(rnn_mse_chain)
    mae_rnn_from_rnn_mse_2 =
        Flux.Losses.mae(rnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mse_chain)
    mape_rnn_from_rnn_mse_2 = Statistics.mean(
        abs.(
            (rnn_mse_chain(Matrix(Train_in_data)') .- Matrix(Train_out_data)') ./
            Matrix(Train_out_data)'
        ),
    )

    @test mse_rnn_from_rnn_mse == mse_rnn_from_rnn_mse_2
    @test rmse_rnn_from_rnn_mse == rmse_rnn_from_rnn_mse_2
    @test mae_rnn_from_rnn_mse == mae_rnn_from_rnn_mse_2
    @test mape_rnn_from_rnn_mse == mape_rnn_from_rnn_mse_2

    Flux.reset!(rnn_rmse_chain)
    mse_rnn_from_rnn_rmse =
        mse_losses(rnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_rmse_chain)
    rmse_rnn_from_rnn_rmse =
        rmse_losses(rnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_rmse_chain)
    mae_rnn_from_rnn_rmse =
        mae_losses(rnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_rmse_chain)
    mape_rnn_from_rnn_rmse =
        mape_losses(rnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')

    Flux.reset!(rnn_rmse_chain)
    mse_rnn_from_rnn_rmse_2 =
        Flux.Losses.mse(rnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_rmse_chain)
    rmse_rnn_from_rnn_rmse_2 = sqrt(
        Flux.Losses.mse(rnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)'),
    )
    Flux.reset!(rnn_rmse_chain)
    mae_rnn_from_rnn_rmse_2 =
        Flux.Losses.mae(rnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_rmse_chain)
    mape_rnn_from_rnn_rmse_2 = Statistics.mean(
        abs.(
            (rnn_rmse_chain(Matrix(Train_in_data)') .- Matrix(Train_out_data)') ./
            Matrix(Train_out_data)'
        ),
    )

    @test mse_rnn_from_rnn_rmse == mse_rnn_from_rnn_rmse_2
    @test rmse_rnn_from_rnn_rmse == rmse_rnn_from_rnn_rmse_2
    @test mae_rnn_from_rnn_rmse == mae_rnn_from_rnn_rmse_2
    @test mape_rnn_from_rnn_rmse == mape_rnn_from_rnn_rmse_2

    Flux.reset!(rnn_mae_chain)
    mse_rnn_from_rnn_mae =
        mse_losses(rnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mae_chain)
    rmse_rnn_from_rnn_mae =
        rmse_losses(rnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mae_chain)
    mae_rnn_from_rnn_mae =
        mae_losses(rnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mae_chain)
    mape_rnn_from_rnn_mae =
        mape_losses(rnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')

    Flux.reset!(rnn_mae_chain)
    mse_rnn_from_rnn_mae_2 =
        Flux.Losses.mse(rnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mae_chain)
    rmse_rnn_from_rnn_mae_2 = sqrt(
        Flux.Losses.mse(rnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)'),
    )
    Flux.reset!(rnn_mae_chain)
    mae_rnn_from_rnn_mae_2 =
        Flux.Losses.mae(rnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mae_chain)
    mape_rnn_from_rnn_mae_2 = Statistics.mean(
        abs.(
            (rnn_mae_chain(Matrix(Train_in_data)') .- Matrix(Train_out_data)') ./
            Matrix(Train_out_data)'
        ),
    )

    @test mse_rnn_from_rnn_mae == mse_rnn_from_rnn_mae_2
    @test rmse_rnn_from_rnn_mae == rmse_rnn_from_rnn_mae_2
    @test mae_rnn_from_rnn_mae == mae_rnn_from_rnn_mae_2
    @test mape_rnn_from_rnn_mae == mape_rnn_from_rnn_mae_2

    Flux.reset!(rnn_mape_chain)
    mse_rnn_from_rnn_mape =
        mse_losses(rnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mape_chain)
    rmse_rnn_from_rnn_mape =
        rmse_losses(rnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mape_chain)
    mae_rnn_from_rnn_mape =
        mae_losses(rnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mape_chain)
    mape_rnn_from_rnn_mape =
        mape_losses(rnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')

    Flux.reset!(rnn_mape_chain)
    mse_rnn_from_rnn_mape_2 =
        Flux.Losses.mse(rnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mape_chain)
    rmse_rnn_from_rnn_mape_2 = sqrt(
        Flux.Losses.mse(rnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)'),
    )
    Flux.reset!(rnn_mape_chain)
    mae_rnn_from_rnn_mape_2 =
        Flux.Losses.mae(rnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    Flux.reset!(rnn_mape_chain)
    mape_rnn_from_rnn_mape_2 = Statistics.mean(
        abs.(
            (rnn_mape_chain(Matrix(Train_in_data)') .- Matrix(Train_out_data)') ./
            Matrix(Train_out_data)'
        ),
    )

    @test mse_rnn_from_rnn_mape == mse_rnn_from_rnn_mape_2
    @test rmse_rnn_from_rnn_mape == rmse_rnn_from_rnn_mape_2
    @test mae_rnn_from_rnn_mape == mae_rnn_from_rnn_mape_2
    @test mape_rnn_from_rnn_mape == mape_rnn_from_rnn_mape_2

end

end
