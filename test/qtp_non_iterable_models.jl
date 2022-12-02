module QtpNonIterableModel

using Test
using MLJ
using MLJMultivariateStatsInterface
using MultivariateStats
using CSV
using DataFrames
using Flux
using AutomationLabsIdentification

@testset "Linear model Ax + Bu" begin

    # load the inputs and outputs data
    dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))#[1:2500, :]
    dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))#[1:2500, :]
    
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

    # Linear regression Ax + Bu
    linear_regressor =
        MLJMultivariateStatsInterface.MultitargetLinearRegressor(bias = false)

    mach_linear_regressor = MLJ.machine(linear_regressor, in_data, out_data)

    MLJ.fit!(mach_linear_regressor)

    #save the model and optimisation results
    MLJ.save("./models_saved/linear_regressor_train_result.jls", mach_linear_regressor)

    #get params
    A_t = fitted_params(mach_linear_regressor).coefficients
    A = A_t'

    Train_in_data = (DataTrainTest.TrainDataIn)
    Train_out_data = (DataTrainTest.TrainDataOut)

    mae_Train_linear_regression =
        Flux.mae(A * (Matrix(Train_in_data)'), Matrix(Train_out_data)')

    println("mae_Train_linear_regression $mae_Train_linear_regression")

    @test mae_Train_linear_regression <= 0.01

end

end