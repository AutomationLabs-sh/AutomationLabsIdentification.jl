# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

### Struct necessary for the package ###
"""
	AbstractData
An abstract type that should be subtyped for data extensions.
"""
abstract type AbstractData end

"""
	DataTrainTest
An data type from dynamical system identification problem, with train, validation and test data. They are all scale from 0 to 1.
"""
struct DataTrainTest <: AbstractData
    dt_input::Any
    dt_output::Any
    TrainDataIn::Any
    TrainDataOut::Any
    TestDataIn::Any
    TestDataOut::Any
end

"""
    data_formatting_identification
An artificial neural network for dynamical system identification problem needs data.
This function allows a data formatting with a recurrent equation of the form : x(k+1) = f(x(k), u(k)), as such, 
the output is formed with x(k+1) vector and input with (x(k), u(k)). The function return a DataTrainTest with train data inputs/outputs and
test data data inputs/ouputs.

The following variables are mendatories:
* `dfin` the dynamical system measure inputs DataFrame u(k).
* `dfout` the dynamical system measure outputs DataFrame x(k).

The following variables are optionals inside kws...:
* `n_delay`: The number of inputs delay for time delay neural network, default is 1.
* `normalisation`: A normalisation of the data from 0-1 with `normalisation = true`.
* `data_type`: type of the output for training. For GPU neural networks training `data_type = Float32` should be selected.
* `data_lower_input` : The data input lower limit.
* `data_upper_input` : The data input upper limit.
* `data_lower_output` : The data output lower limit.
* `data_upper_output` : The data output upper limit.

## Example
```
in1 = repeat(1:0.001: 1000, inner=1)[1:10000]
out1 = sin.(in1) 
data_in = DataFrames.DataFrame(A=in1)
data_out = DataFrames.DataFrame(B=out1)
n_delay = 3
normalisation = false

DataTrainTest = data_formatting_identification(
    data_in,
    data_out;
    n_delay = n_delay,
    normalisation = normalisation)

```
"""
function data_formatting_identification(
    data_table_dfin::DataFrames.DataFrame,
    data_table_dfout::DataFrames.DataFrame;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    normalisation = get(kws, :normalisation, false)
    data_lower_input = get(kws, :data_lower_input, [-Inf])
    data_upper_input = get(kws, :data_upper_input, [Inf])
    data_lower_output = get(kws, :data_lower_output, [-Inf])
    data_upper_output = get(kws, :data_upper_output, [Inf])
    data_type = get(kws, :data_type, Float64)

    # Evaluate if there is a n_delay or n_sequence or nothing 
    if haskey(kws, :n_delay) == true
        #forward is selected
        n_delay = kws[:n_delay]
    elseif haskey(kws, :n_sequence) == true
        #recurrent is selected
        n_sequence = kws[:n_sequence]
    elseif haskey(kws, :n_delay) == true && haskey(kws, :n_sequence) == true
        #error cannot be forward and recurrent networks 
        @warn "n_delay and n_sequence cannot be selected together"
        @info "n_sequence is forgotten"
        n_delay = kws[:n_delay]
    else
        #there is no n_sequence and n_delay by default forward is selected
        n_delay = get(kws, :n_delay, 1)
    end


    # Data informations
    data_table_df = hcat(data_table_dfin, data_table_dfout)
    n_input = size(data_table_dfin, 2)
    n_output = size(data_table_dfout, 2)

    if @isdefined(n_delay) == true

        # First level of modification, set the data inputs and outputs with delay : x+ = f(x, u)
        in_tmp, out_tmp = input_output_formatting(
            Matrix(data_table_df),
            n_input,
            n_output,
            n_delay,
            data_type,
        )

        return data_separation_train_test(
            in_tmp,
            out_tmp,
            normalisation,
            data_lower_input,
            data_upper_input,
            data_lower_output,
            data_upper_output,
            n_input,
            n_output,
            n_delay,
            data_type,
        )
    end

    if @isdefined(n_sequence) == true
        # First level of modification, set the data inputs and outputs with delay : x+ = f(x, u)
        in_tmp, out_tmp = input_output_formatting_recurrent(
            Matrix(data_table_df),
            n_input,
            n_output,
            n_sequence,
            data_type,
        )

        return data_separation_train_test_recurrent(
            in_tmp,
            out_tmp,
            normalisation,
            data_lower_input,
            data_upper_input,
            data_lower_output,
            data_upper_output,
            n_input,
            n_output,
            n_sequence,
            data_type,
        )
    end

end

"""
    input_output_formatting
This function allows a data formatting with a recurrent equation of the form : x(k+1) = f(x(k), u(k)), as such, 
the output is formed with x(k+1) vector and input with (x(k), u(k)). Also, the program separate data from inputs, (x(k), u(k)), 
predict, (x(k+1)). 
"""
function input_output_formatting(
    data_table_in::Matrix,
    n_input::Int,
    n_output::Int,
    n_delay::Int,
    data_type::DataType,
)

    # Float64 or Float32 convertion
    data_table = data_type.(data_table_in)

    # Data inputs and outputs and predict separation
    DataInputs = data_table[:, begin:n_input]
    DataOutputs = data_table[:, n_input+1:end]
    DataPredict = data_table[2:end, n_input+1:end]

    # Memory allocation
    data_neural_input =
        zeros(data_type, n_delay, n_input + n_output, size(DataInputs, 1) - n_delay)
    data_neural_output = zeros(data_type, 1, n_output, size(DataInputs, 1) - n_delay)

    # Data separation, neural input and neural output
    Threads.@threads for i = size(DataPredict, 1):-1:n_delay
        data_neural_input[:, :, i-n_delay+1] =
            hcat(DataOutputs[i-n_delay+1:i, :], DataInputs[i-n_delay+1:i, :])
        data_neural_output[:, :, i-n_delay+1] = DataPredict[i:i, :]
    end

    return data_neural_input, data_neural_output

end

"""
    data_separation_train_test
This function allows a data formatting. It allows to limit the data, to normalised the data (from 0 to 1), 
and to separate the data from train and test sets.
"""
function data_separation_train_test(
    DataIn_inputs,
    DataOut_inputs,
    normalisation,
    lower_in,
    upper_in,
    lower_out,
    upper_out,
    n_input,
    n_output,
    n_delay,
    data_type,
)

    # Keep only data inside lower and upper limits
    sol_lower_in = DataIn_inputs[:, :, :] .>= lower_in
    sol_upper_in = DataIn_inputs[:, :, :] .<= upper_in

    sol_lower_out = DataOut_inputs[:, :, :] .>= lower_out
    sol_upper_out = DataOut_inputs[:, :, :] .<= upper_out


    DataIn_limited = []
    DataOut_limited = []

    for i = 1:1:size(DataIn_inputs, 3)

        if (
            (0 ⊈ sol_lower_in[:, :, i]) == true &&
            (0 ⊈ sol_upper_in[:, :, i]) == true &&
            (0 ⊈ sol_lower_out[:, :, i]) == true &&
            (0 ⊈ sol_upper_out[:, :, i]) == true
        )

            push!(DataIn_limited, DataIn_inputs[:, :, i])
            push!(DataOut_limited, DataOut_inputs[:, :, i])
        end
    end

    DataIn = zeros(data_type, n_delay, n_input + n_output, size(DataIn_limited, 1))
    DataOut = zeros(data_type, 1, n_output, size(DataOut_limited, 1))

    Threads.@threads for i = 1:1:size(DataIn_limited, 1)
        DataIn[:, :, i] = DataIn_limited[i]
        DataOut[:, :, i] = DataOut_limited[i]
    end

    # Data normalization
    if normalisation == true
        # Normalise and scale data into 0 and 1
        dt_input = StatsBase.fit(
            StatsBase.UnitRangeTransform,
            reshape(DataIn, (:, size(DataIn, 2))),
            dims = 1,
        )
        DataIn_0_1 = reshape(
            StatsBase.transform(dt_input, reshape(DataIn, (:, size(DataIn, 2)))),
            (size(DataIn)),
        )

        dt_output = StatsBase.fit(
            StatsBase.UnitRangeTransform,
            reshape(DataOut, (:, size(DataOut, 2))),
            dims = 1,
        )
        DataOut_0_1 = reshape(
            StatsBase.transform(dt_output, reshape(DataOut, (:, size(DataOut, 2)))),
            (size(DataOut)),
        )

    else
        DataIn_0_1 = DataIn
        DataOut_0_1 = DataOut
        dt_input = 0.0
        dt_output = 0.0
    end

    # DataPair
    DataPair_0_1 = []
    for i = 1:1:size(DataIn_0_1, 3)
        push!(DataPair_0_1, (DataIn_0_1[:, :, i], DataOut_0_1[:, :, i]))
    end

    # Data separation between train & validation & test data
    TrainDataPair, TestDataPair =
        MLUtils.splitobs(MLUtils.shuffleobs(DataPair_0_1), at = 0.8)

    # Input and output separation
    TrainDataIn, TrainDataOut = pair_data_separation(TrainDataPair, data_type)
    TestDataIn, TestDataOut = pair_data_separation(TestDataPair, data_type)

    # DataFrame x1(k), x1(k-1), x1(k-2), x2(k), x2(k-1),...

    TrainDataInDf = array_to_dataframe(TrainDataIn, data_type)
    TrainDataOutDf = array_to_dataframe(TrainDataOut, data_type)

    TestDataInDf = array_to_dataframe(TestDataIn, data_type)
    TestDataOutDf = array_to_dataframe(TestDataOut, data_type)

    # Struct design with data separation
    AllData = DataTrainTest(
        dt_input,
        dt_output,
        TrainDataInDf,
        TrainDataOutDf,
        TestDataInDf,
        TestDataOutDf,
    )

    return AllData
end

"""
    pair_data_separation
This function allows a data separation from Pairs. The pairs are formed from neural data inputs and outputs.
The output of the function are formed of neural data inputs and neural data outputs.
"""
function pair_data_separation(DataPair, data_type)

    # Memory allocation
    DataIn = zeros(
        data_type,
        size(DataPair[1][1], 1),
        size(DataPair[1][1], 2),
        size(DataPair, 1),
    )
    DataOut = zeros(
        data_type,
        size(DataPair[1][2], 1),
        size(DataPair[1][2], 2),
        size(DataPair, 1),
    )

    # Separation input data and output data
    Threads.@threads for i = 1:1:size(DataPair, 1)
        DataIn[:, :, i] = DataPair[i][1]
        DataOut[:, :, i] = DataPair[i][2]
    end

    return DataIn, DataOut

end


"""
    array_to_dataframe
This function allows a data from Julia Array to DataFrames.DataFrame.
"""
function array_to_dataframe(data, data_type)

    # Memory allocation
    data_horizontal = zeros(data_type, size(data, 3), size(data, 1) * size(data, 2))

    Threads.@threads for i = 1:1:size(data, 3)
        data_horizontal[i, :] = hcat(Matrix(data[:, :, i])...)
    end

    return DataFrames.DataFrame(data_horizontal, :auto)

end


function array_to_dataframe_recurrent(data, data_type)

    # Memory allocation
    data_horizontal = zeros(data_type, size(data, 3) * size(data, 1), size(data, 2))
    data_tmp = []
    for i = 1:1:size(data, 3)
        push!(data_tmp, data[:, :, i])
    end
    data_horizontal = vcat(data_tmp...)

    return DataFrames.DataFrame(data_horizontal, :auto)

end

### Recurrent with sequence separation

function input_output_formatting_recurrent(
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
    data_neural_input = zeros(data_type, n_seq, n_input + n_output, nbr_sequence)
    data_neural_output = zeros(data_type, n_seq, n_output, nbr_sequence)

    # Data separation, neural input and neural output
    j_seq = nbr_sequence
    for i = size(DataPredict, 1):-n_seq:n_seq
        data_neural_input[:, :, j_seq] .=
            hcat(DataOutputs[i-n_seq+1:i, :], DataInputs[i-n_seq+1:i, :])
        data_neural_output[:, :, j_seq] .= DataPredict[i-n_seq+1:i, :]  #[i:i, :]
        j_seq = j_seq - 1
    end

    return data_neural_input, data_neural_output
end


"""
    data_separation_train_test
This function allows a data formatting. It allows to limit the data, to normalised the data (from 0 to 1), 
and to separate the data from train and test sets.
"""
function data_separation_train_test_recurrent(
    DataIn_inputs,
    DataOut_inputs,
    normalisation,
    lower_in,
    upper_in,
    lower_out,
    upper_out,
    n_input,
    n_output,
    n_seq,
    data_type,
)

    # Keep only data inside lower and upper limits
    sol_lower_in = DataIn_inputs[:, :, :] .>= lower_in
    sol_upper_in = DataIn_inputs[:, :, :] .<= upper_in

    sol_lower_out = DataOut_inputs[:, :, :] .>= lower_out
    sol_upper_out = DataOut_inputs[:, :, :] .<= upper_out


    DataIn_limited = []
    DataOut_limited = []

    for i = 1:1:size(DataIn_inputs, 3)

        if (
            (0 ⊈ sol_lower_in[:, :, i]) == true &&
            (0 ⊈ sol_upper_in[:, :, i]) == true &&
            (0 ⊈ sol_lower_out[:, :, i]) == true &&
            (0 ⊈ sol_upper_out[:, :, i]) == true
        )

            push!(DataIn_limited, DataIn_inputs[:, :, i])
            push!(DataOut_limited, DataOut_inputs[:, :, i])
        end
    end

    DataIn = zeros(data_type, n_seq, n_input + n_output, size(DataIn_limited, 1))
    DataOut = zeros(data_type, n_seq, n_output, size(DataOut_limited, 1))

    Threads.@threads for i = 1:1:size(DataIn_limited, 1)
        DataIn[:, :, i] = DataIn_limited[i]
        DataOut[:, :, i] = DataOut_limited[i]
    end

    # Data normalization
    if normalisation == true
        # Normalise and scale data into 0 and 1
        dt_input = StatsBase.fit(
            StatsBase.UnitRangeTransform,
            reshape(DataIn, (:, size(DataIn, 2))),
            dims = 1,
        )
        DataIn_0_1 = reshape(
            StatsBase.transform(dt_input, reshape(DataIn, (:, size(DataIn, 2)))),
            (size(DataIn)),
        )

        dt_output = StatsBase.fit(
            StatsBase.UnitRangeTransform,
            reshape(DataOut, (:, size(DataOut, 2))),
            dims = 1,
        )
        DataOut_0_1 = reshape(
            StatsBase.transform(dt_output, reshape(DataOut, (:, size(DataOut, 2)))),
            (size(DataOut)),
        )

    else
        DataIn_0_1 = DataIn
        DataOut_0_1 = DataOut
        dt_input = 0.0
        dt_output = 0.0
    end

    # DataPair
    DataPair_0_1 = []
    for i = 1:1:size(DataIn_0_1, 3)
        push!(DataPair_0_1, (DataIn_0_1[:, :, i], DataOut_0_1[:, :, i]))
    end

    # Data separation between train & validation & test data
    TrainDataPair, TestDataPair =
        MLUtils.splitobs(MLUtils.shuffleobs(DataPair_0_1), at = 0.8)

    # Input and output separation
    TrainDataIn, TrainDataOut = pair_data_separation(TrainDataPair, data_type)
    TestDataIn, TestDataOut = pair_data_separation(TestDataPair, data_type)

    # DataFrame x1(k), x1(k-1), x1(k-2), x2(k), x2(k-1),...

    TrainDataInDf = array_to_dataframe_recurrent(TrainDataIn, data_type)
    TrainDataOutDf = array_to_dataframe_recurrent(TrainDataOut, data_type)

    TestDataInDf = array_to_dataframe_recurrent(TestDataIn, data_type)
    TestDataOutDf = array_to_dataframe_recurrent(TestDataOut, data_type)

    # Struct design with data separation
    AllData = DataTrainTest(
        dt_input,
        dt_output,
        TrainDataInDf,
        TrainDataOutDf,
        TestDataInDf,
        TestDataOutDf,
    )

    return AllData
end
