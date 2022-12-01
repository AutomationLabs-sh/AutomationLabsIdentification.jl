## Roadmap

I/ Improve the documentation

II/ Add dynamical system identification models:
* CliqueNet
* CNN+LSTM 
* CNN+GRU
* ARX
* ARMAX 
* FIR 
* OE 
* Hammerstein–Wiener
* Sandi method
* Steady state identification
* Extrem learning machine (ELM)

III/ A reflexion on a quadratic cost function to add instead of MAE.

IV/ Add algorithms to tune the identification models (need JuMP):
* MILP to tune neural networks with relu activation function
* Non linear solver such as Ipopt
* Commercial solvers

V/ Report the issue with physics informed model and Zygote, add a modification to compute the gradient with ForwardDiff.

VI/ Improve the tests: speed, function to test, links with depot.