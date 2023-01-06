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
using Statistics
using Distributed

import AutomationLabsIdentification: mse_losses
import AutomationLabsIdentification: rmse_losses
import AutomationLabsIdentification: mae_losses
import AutomationLabsIdentification: mape_losses

@testset "Fnn architecture blackbox identification MSE" begin

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
        neuralnet_loss_function = "mse"
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
        neuralnet_loss_function = "rmse"
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
        neuralnet_loss_function = "mae"
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
        neuralnet_loss_function = "mape"
    )

    report(fitted_params(m_fnn_mse).machine).best_history_entry 
    report(fitted_params(m_fnn_rmse).machine).best_history_entry 
    report(fitted_params(m_fnn_mae).machine).best_history_entry 
    report(fitted_params(m_fnn_mape).machine).best_history_entry 

    # MAE with train and test data
    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)
    Test_in_data = (DataTrainTest.TestDataIn)
    Test_out_data = (DataTrainTest.TestDataOut)
    
    fnn_mse_chain = fitted_params(fitted_params(m_fnn_mse).machine).best_fitted_params.chain
    fnn_rmse_chain = fitted_params(fitted_params(m_fnn_rmse).machine).best_fitted_params.chain
    fnn_mae_chain = fitted_params(fitted_params(m_fnn_mae).machine).best_fitted_params.chain
    fnn_mape_chain = fitted_params(fitted_params(m_fnn_mape).machine).best_fitted_params.chain

    mse_fnn_from_fnn_mse = mse_losses(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_mse = rmse_losses(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mae_fnn_from_fnn_mse = mae_losses(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_mse = mape_losses(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')

    mse_fnn_from_fnn_mse_2 = Flux.Losses.mse(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_mse_2 = sqrt(Flux.Losses.mse(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)'))
    mae_fnn_from_fnn_mse_2 = Flux.Losses.mae(fnn_mse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_mse_2 = Statistics.mean(abs.( (fnn_mse_chain(Matrix(Train_in_data)') .- Matrix(Train_out_data)') ./ Matrix(Train_out_data)'))
    
    @test mse_fnn_from_fnn_mse == mse_fnn_from_fnn_mse_2
    @test rmse_fnn_from_fnn_mse == rmse_fnn_from_fnn_mse_2
    @test mae_fnn_from_fnn_mse == mae_fnn_from_fnn_mse_2
    @test mape_fnn_from_fnn_mse == mape_fnn_from_fnn_mse_2

    mse_fnn_from_fnn_rmse = mse_losses(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_rmse = rmse_losses(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mae_fnn_from_fnn_rmse = mae_losses(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_rmse = mape_losses(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')

    mse_fnn_from_fnn_rmse_2 = Flux.Losses.mse(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_rmse_2 = sqrt(Flux.Losses.mse(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)'))
    mae_fnn_from_fnn_rmse_2 = Flux.Losses.mae(fnn_rmse_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_rmse_2 = Statistics.mean(abs.( (fnn_rmse_chain(Matrix(Train_in_data)') .- Matrix(Train_out_data)') ./ Matrix(Train_out_data)'))

    @test mse_fnn_from_fnn_rmse == mse_fnn_from_fnn_rmse_2
    @test rmse_fnn_from_fnn_rmse == rmse_fnn_from_fnn_rmse_2
    @test mae_fnn_from_fnn_rmse == mae_fnn_from_fnn_rmse_2
    @test mape_fnn_from_fnn_rmse == mape_fnn_from_fnn_rmse_2
    
    mse_fnn_from_fnn_mae = mse_losses(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_mae = rmse_losses(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mae_fnn_from_fnn_mae = mae_losses(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_mae = mape_losses(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')

    mse_fnn_from_fnn_mae_2 = Flux.Losses.mse(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_mae_2 = sqrt(Flux.Losses.mse(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)'))
    mae_fnn_from_fnn_mae_2 = Flux.Losses.mae(fnn_mae_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_mae_2 = Statistics.mean(abs.( (fnn_mae_chain(Matrix(Train_in_data)') .- Matrix(Train_out_data)') ./ Matrix(Train_out_data)'))

    @test mse_fnn_from_fnn_mae == mse_fnn_from_fnn_mae_2
    @test rmse_fnn_from_fnn_mae == rmse_fnn_from_fnn_mae_2
    @test mae_fnn_from_fnn_mae == mae_fnn_from_fnn_mae_2
    @test mape_fnn_from_fnn_mae == mape_fnn_from_fnn_mae_2

    mse_fnn_from_fnn_mape = mse_losses(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_mape = rmse_losses(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mae_fnn_from_fnn_mape = mae_losses(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_mape = mape_losses(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')

    mse_fnn_from_fnn_mape_2 = Flux.Losses.mse(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    rmse_fnn_from_fnn_mape_2 = sqrt(Flux.Losses.mse(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)'))
    mae_fnn_from_fnn_mape_2 = Flux.Losses.mae(fnn_mape_chain(Matrix(Train_in_data)'), Matrix(Train_out_data)')
    mape_fnn_from_fnn_mape_2 = Statistics.mean(abs.( (fnn_mape_chain(Matrix(Train_in_data)') .- Matrix(Train_out_data)') ./ Matrix(Train_out_data)'))

    @test mse_fnn_from_fnn_mape == mse_fnn_from_fnn_mape_2
    @test rmse_fnn_from_fnn_mape == rmse_fnn_from_fnn_mape_2
    @test mae_fnn_from_fnn_mape == mae_fnn_from_fnn_mape_2
    @test mape_fnn_from_fnn_mape == mape_fnn_from_fnn_mape_2

end

end