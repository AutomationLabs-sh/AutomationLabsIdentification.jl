# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
"""
    greybox_identification
A greybox model for dynamical system identification problem with recurrent equation of the form : x(k+1) = f(x(k), u(k)).
It needs a continuous system of the dynamical system in the form of julia `function` and the algorithm discretises the function.
Also, this function needs some trainable parameters. For instance for the quadruple tank process, the function is defined as:
```
function f(du, u, p, t)
    # Constant parameters
    S = 0.06
    gamma_a = 0.3
    gamma_b = 0.4
    g = 9.81

    # Trainable parameters
    a1, a2, a3, a4 = p

    # State initialization 
    x1 = u[1]
    x2 = u[2]
    x3 = u[3]
    x4 = u[4]

    # Input initialization
    qa = u[5]
    qb = u[6]

    # Derivatives
    du[1] =
        -a1 / S * sqrt(2 * g * x1) +
        a3 / S * sqrt(2 * g * x3) +
        gamma_a / (S * 3600) * qa
    du[2] =
        -a2 / S * sqrt(2 * g * x2) +
        a4 / S * sqrt(2 * g * x4) +
        gamma_b / (S * 3600) * qb
    du[3] = 
        -a3 / S * sqrt(2 * g * x3) + 
        (1 - gamma_b) / (S * 3600) * qb
    du[4] = 
    -a4 / S * sqrt(2 * g * x4) + 
    (1 - gamma_a) / (S * 3600) * qa
end
```

The following variables are mendatories:
* `f`: The continuous state of the dynamical system.
* `dfin`: a DataFrame data table from measure data inputs.
* `dfout`: a DataFrame data table from measure data outputs.
* `solver`: an solver algorithm selection for neural network training, such as, "LBFGS".
* `architecture`: a selection for greybox identification, such as "PhysicsInformed" or "PhysicsInformedOracle".
* `init_t_p`: a vector for trainable parameters initialization.
* `sample_time`: a sample time for discretation, the sample time related to the measure data steps.
* `maximum_time`: a maximum time for the optimization, if FLoat64 the time is in hours, or a Dates Period is also possible.

The selection of the architecture depends on the grey box model. It is possible to add an oracle to the grey box model.
In this case, the trainable parameters are present in the oracle, and by default, the oracle is a neural network.
In case of greybox identification with oracle, the continuous dynamical system `f` does not have trainable parameters. 

It is possible to define optional variables:
* `data_lower_input`: Set lower limits to data formatting with inputs vector of the dynamical systems. The inputs vector represents the state x(k) and u(k).
* `data_upper_input`: Set upper limits to data formatting with inputs vector of the dynamical systems. The inputs vector represents the state x(k) and u(k).
* `data_lower_output`: Set lower limits to data formatting with outputs vector of the dynamical systems. The outputs vector represents the predicted state x(k+1).
* `data_upper_output`: Set upper limits to data formatting with outputs vector of the dynamical systems. The outputs vector represents the predicted state x(k+1).
* `lower_params`: Set lower limits to the trainable parameters.
* `upper_params`: Set lower limits to the trainable parameters.
* `computation_verbosity`: Set verbosity REPL informations.

The constraints/limits are mendatories when mathematical function with limited domain of a function are employed. 
For instance, the domain of sqrt(x) is [0, +Inf[. The containts sets limits to the optimizer and they are set with `lower_params` and `upper_params`.

## Example
The following example comes from the init testing file of the greebox_identification.
```
# load the inputs and outputs data
dfout = DataFrame(CSV.File("./data_QTP/data_outputs.csv"))[1:5000, :]
dfin = DataFrame(CSV.File("./data_QTP/data_inputs_m3h.csv"))[1:5000, :]

function f(du, u, p, t)
    # Constant parameters
    S = 0.06
    gamma_a = 0.3
    gamma_b = 0.4
    g = 9.81

    # Trainable parameters
    a1, a2, a3, a4 = p

    # State initialization 
    x1 = u[1]
    x2 = u[2]
    x3 = u[3]
    x4 = u[4]

    # Input initialization
    qa = u[5]
    qb = u[6]

    # Derivatives
    du[1] =
        -a1 / S * sqrt(2 * g * x1) +
        a3 / S * sqrt(2 * g * x3) +
        gamma_a / (S * 3600) * qa
    du[2] =
        -a2 / S * sqrt(2 * g * x2) +
        a4 / S * sqrt(2 * g * x4) +
        gamma_b / (S * 3600) * qb
    du[3] = 
        -a3 / S * sqrt(2 * g * x3) + 
        (1 - gamma_b) / (S * 3600) * qb
    du[4] = 
    -a4 / S * sqrt(2 * g * x4) + 
    (1 - gamma_a) / (S * 3600) * qa
end

# Optional variable
lower_in = [0.2 0.2 0.2 0.2 -Inf -Inf]
upper_in = [1.32 1.32 1.32 1.32 Inf Inf]

lower_out = [0.2 0.2 0.2 0.2]
upper_out = [1.32 1.32 1.32 1.32]

lower_params = [1e-5, 1e-5, 1e-5, 1e-5]
upper_params = [10e-4, 10e-4, 10e-4, 10e-4]

# Parameters intiialization
init_t_p = [1.1e-4, 1.2e-4, 9e-5, 9e-5]

train_model = greybox_identification(
    f,
    dfin,
    dfout,
    "LBFGS",
    "PhysicsInformed",
    "CPU",
    init_t_p,
    5.0,
    Minute(1);
    #option parameters
    data_lower_input = lower_in,
    data_upper_input = upper_in,
    data_lower_output = lower_out,
    data_upper_output = upper_out,
    lower_params = lower_params,
    upper_params = upper_params
)
```
"""
function greybox_identification(
    f::Function,
    train_dfin::DataFrames.DataFrame,
    train_dfout::DataFrames.DataFrame,
    solver::String,
    architecture::String,
    init_t_p::Vector,
    sample_time::Float64,
    maximum_time::Union{Float64,Dates.TimePeriod};
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    # Get parameters from kws 
    lower_params = get(kws, :lower_params, [-Inf])
    upper_params = get(kws, :upper_params, [Inf])

    # Get parameters from kws for computation informations
    verbosity = get(kws, :computation_verbosity, 0)

    # df all size
    n_inputs = size(train_dfin, 2)
    n_outputs = size(train_dfout, 2)

    # Architecture selection, fnn, physics_informed_builder, densenet, icnn
    tuned_model = physics_informed_builder(
        PHYSICS_INFORMED_LIST[Symbol(architecture)],
        f,
        ALGORITHM_LIST[Symbol(solver)],
        init_t_p,
        n_inputs,
        n_outputs,
        sample_time,
        maximum_time;
        lower_params = lower_params,
        upper_params = upper_params,
    ) #dispatched function

    #mach the model with the data
    mach_nn = MLJ.machine(tuned_model, train_dfin, train_dfout)

    #train the neural network
    MLJ.fit!(mach_nn, verbosity = verbosity)

    return mach_nn

end

#struct for physics-informed
"""
    AbstractPhysicsInformedArchitecture
An abstract type that should be subtyped for physics informed.
"""
abstract type AbstractPhysicsInformedArchitecture end

"""
    physicsinformed
An physics informed architecture type for dynamical system identification problem [ref].
"""
struct physicsinformed <: AbstractNeuralArchitecture end

"""
    physics_informed_oracle
An physics informed oracle architecture type for dynamical system identification problem [ref].
"""
struct physicsinformedoracle <: AbstractNeuralArchitecture end

#NamedTuple definition
PHYSICS_INFORMED_LIST =
    (physics_informed = physicsinformed(), physics_informed_oracle = physicsinformedoracle())
