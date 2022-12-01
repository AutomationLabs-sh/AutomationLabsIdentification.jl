module DataSeprationTests

using Test
using DataFrames
using Distributed

Distributed.@everywhere import Pkg
Distributed.@everywhere Pkg.activate("/home/pierre/CleverCloud/identification")
Distributed.@everywhere using Identification

@testset "Train & Test & Delay" begin

    data_in = DataFrames.DataFrame(A = rand(500000))
    data_out = DataFrames.DataFrame(B = rand(500000), C = rand(500000))
    n_delay = 5
    normalisation = false

    DataTrainTest = data_formatting_identification(
        data_in,
        data_out;
        n_delay = n_delay,
        normalisation = normalisation,
    )

    @test size(DataTrainTest.TrainDataIn) == (399996, 15)
    @test size(DataTrainTest.TrainDataOut) == (399996, 2)
    @test size(DataTrainTest.TestDataIn) == (99999, 15)
    @test size(DataTrainTest.TestDataOut) == (99999, 2)

end

@testset "Normalisation 0-1" begin

    data_in = DataFrames.DataFrame(A = 10 * rand(500000))
    data_out = DataFrames.DataFrame(B = 10 * rand(500000), C = 10 * rand(500000))
    n_delay = 5
    normalisation = true

    DataTrainTest = data_formatting_identification(
        data_in,
        data_out;
        n_delay = n_delay,
        normalisation = normalisation,
    )

    @test maximum(Matrix(DataTrainTest.TrainDataIn)) <= 1.0
    @test minimum(Matrix(DataTrainTest.TrainDataIn)) >= 0.0
    @test maximum(Matrix(DataTrainTest.TestDataIn)) <= 1.0
    @test minimum(Matrix(DataTrainTest.TestDataIn)) >= 0.0

end

@testset "DataSeparationFloat32" begin

    data_in = DataFrames.DataFrame(A = 10 * rand(500000))
    data_out = DataFrames.DataFrame(B = 10 * rand(500000), C = 10 * rand(500000))
    n_delay = 5
    normalisation = true

    DataTrainTest = data_formatting_identification(
        data_in,
        data_out;
        n_delay = n_delay,
        normalisation = normalisation,
        data_type = Float32,
    )

    @test typeof(DataTrainTest.TrainDataIn[!, 1]) == Vector{Float32}
    @test typeof(DataTrainTest.TrainDataIn[!, 2]) == Vector{Float32}
    @test typeof(DataTrainTest.TrainDataIn[!, 3]) == Vector{Float32}

    @test typeof(DataTrainTest.TestDataIn[!, 1]) == Vector{Float32}
    @test typeof(DataTrainTest.TestDataIn[!, 2]) == Vector{Float32}
    @test typeof(DataTrainTest.TestDataIn[!, 3]) == Vector{Float32}

end

@testset "Limits-data" begin

    data_in = DataFrames.DataFrame(A = 10 * rand(500000))
    data_out = DataFrames.DataFrame(B = 10 * rand(500000), C = 10 * rand(500000))
    n_delay = 1
    normalisation = false

    lower_in = [1 1 2]
    upper_in = [9 Inf 9]

    lower_out = [0.25 0.5]
    upper_out = [9 Inf]


    DataTrainTest = data_formatting_identification(
        data_in,
        data_out;
        n_delay = n_delay,
        normalisation = normalisation,
        data_lower_input = lower_in,
        data_upper_input = upper_in,
        data_lower_output = lower_out,
        data_upper_output = upper_out,
    )

    @test maximum(Matrix(DataTrainTest.TrainDataIn)[:, 1]) <= 9.0
    @test minimum(Matrix(DataTrainTest.TrainDataIn)[:, 1]) >= 1.0

    @test maximum(Matrix(DataTrainTest.TrainDataIn)[:, 2]) <= Inf
    @test minimum(Matrix(DataTrainTest.TrainDataIn)[:, 2]) >= 1.0

    @test maximum(Matrix(DataTrainTest.TrainDataIn)[:, 3]) <= 9.0
    @test minimum(Matrix(DataTrainTest.TrainDataIn)[:, 3]) >= 2.0

    @test maximum(Matrix(DataTrainTest.TrainDataOut)[:, 1]) <= 9.0
    @test minimum(Matrix(DataTrainTest.TrainDataOut)[:, 1]) >= 0.25

    @test maximum(Matrix(DataTrainTest.TrainDataOut)[:, 2]) <= Inf
    @test minimum(Matrix(DataTrainTest.TrainDataOut)[:, 2]) >= 0.5

    @test maximum(Matrix(DataTrainTest.TestDataIn)[:, 1]) <= 9.0
    @test minimum(Matrix(DataTrainTest.TestDataIn)[:, 1]) >= 1.0

    @test maximum(Matrix(DataTrainTest.TestDataIn)[:, 2]) <= Inf
    @test minimum(Matrix(DataTrainTest.TestDataIn)[:, 2]) >= 1.0

    @test maximum(Matrix(DataTrainTest.TestDataIn)[:, 3]) <= 9.0
    @test minimum(Matrix(DataTrainTest.TestDataIn)[:, 3]) >= 2.0

    @test maximum(Matrix(DataTrainTest.TestDataOut)[:, 1]) <= 9.0
    @test minimum(Matrix(DataTrainTest.TestDataOut)[:, 1]) >= 0.25

    @test maximum(Matrix(DataTrainTest.TestDataOut)[:, 2]) <= Inf
    @test minimum(Matrix(DataTrainTest.TestDataOut)[:, 2]) >= 0.5

end


end


data_in = DataFrames.DataFrame(A = rand(500000))
data_out = DataFrames.DataFrame(B = rand(500000), C = rand(500000))
n_delay = 5
normalisation = false
n_input = 1
n_output = 2
data_type = Float64

data_table_df = hcat(data_in, data_out)

in, out = input_output_formatting(
    Matrix(data_table_df),
    n_input,
    n_output,
    n_delay,
    data_type,
)


function input_output_formatting_recurrrent(
    data_table_in::Matrix,
    n_input::Int,
    n_output::Int,
    n_seq::Int,
    data_type::DataType,
)

    # Float64 or Float32 convertion
    data_table = data_type.(data_table_in)

    # Data inputs and outputs and predict separation
    DataInputs = data_table[:, begin:n_input]
    DataOutputs = data_table[:, n_input+1:end]
    DataPredict = data_table[2:end, n_input+1:end]

    # Memory allocation
    nbr_sequence = trunc(Int, size(DataInputs, 1) / n_seq)
    data_neural_input =
        zeros(data_type, n_seq, n_input + n_output, nbr_sequence - 1)
    data_neural_output = zeros(data_type, n_seq, n_output, nbr_sequence - 1)

    # Data separation, neural input and neural output
    j_seq = nbr_sequence - 1
    Threads.@threads for i = size(DataPredict, 1): -n_seq :n_seq 
        data_neural_input[:, :, j_seq] =
            hcat(DataOutputs[i-n_seq+1:i, :], DataInputs[i-n_seq+1:i, :])
        data_neural_output[:, :, j_seq] = DataPredict[i-n_seq+1:i, :]  #[i:i, :]
        j_seq = j_seq - 1
    end

    return data_neural_input, data_neural_output

end




