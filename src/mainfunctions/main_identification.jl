# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
    proceed_identification
A function for blackbox model or greybox model for dynamical system identification problem.

The following variables are mendatories:
* `train_dfin`: a DataFrame data table from measure data from dynamical system inputs.
* `train_dfout`: a DataFrame data table from measure data from dynamical system outputs.
* `solver`: an algorithm selection for model tuning. 
* `architecture`: an architecture selection for model. 
* `maximum_time` : a time limit for the hyperparameters optimisation.

It is possible to define optional variables from blackbox or greybox identification.
"""
function proceed_identification(
    train_dfin::DataFrames.DataFrame,
    train_dfout::DataFrames.DataFrame,
    solver::String,
    architecture::String,
    maximum_time::Dates.TimePeriod;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    # Evaluate if architecture is blackbox identification
    if haskey(ARCHITECTURE_LIST, Symbol(architecture))
        # it is a blackbox architecture

        mach_model = blackbox_identification(
            train_dfin,
            train_dfout,
            solver,
            architecture,
            maximum_time;
            kws,
        )

        return mach_model

    end

    # Evaluation if architecture is greybox identification
    if haskey(PHYSICS_INFORMED_LIST, Symbol(architecture))
        # It is a greybox identification

        # get f 
        f = kws[:f]

        # Get init_t_p 
        trainable_parameters_init = kws[:trainable_parameters_init]

        # get sample_time 
        sample_time = kws[:sample_time]

        mach_model = greybox_identification(
            f,
            train_dfin,
            train_dfout,
            solver,
            architecture,
            trainable_parameters_init,
            sample_time,
            maximum_time;
            kws,
        )

        return mach_model

    end

    return nothing

end
