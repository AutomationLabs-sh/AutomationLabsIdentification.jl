# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

##### Linear model #####
"""
    _neural_network_builder(
    nn::linear,
    processor::Union{MLJ.CPU1, MLJ.CPUThreads, MLJ.CPUProcesses},
    algorithm::adam,
    max_time::Union{Float64, Dates.TimePeriod};
    kwargs...
)
An artificial neural network builder with hyperparameters optimization, it returns a tuned model wich can be match with data and trained.
The function is multiple dispatched according to args type.

The following variables are mendatories:
* `nn`: a neural network architecture.
* `processor`: a processor selection for training.
* `algorithm`: an algorithm selection for neural network training. 
* `max_time` : a maximum time for training.

The following variables are optinals:
* `kwargs...`: optional variables.

"""
function _neural_network_builder(
    nn::linear,
    processor::Union{MLJ.CPU1,MLJ.CPUThreads,MLJ.CPUProcesses},
    algorithm::Lls,
    max_time::Dates.TimePeriod;
    kwargs...,
)
    #CPU only and algorithm Linear Least Square

    # Linear regression: Wx --> Ax + Bu
    linear_regressor =
        MLJMultivariateStatsInterface.MultitargetLinearRegressor(bias = false)

    return linear_regressor

end
