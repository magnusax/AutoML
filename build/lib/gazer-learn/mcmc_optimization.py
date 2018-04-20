"""

MCMC utility module 

Formulas based on "Monte Carlo Statistical Methods"  (Robert & Casella: 2nd Ed. 2004)

The underlying idea is as follows:

The mcmc framework describes simulation techniques which can be adapted to e.g. trace out
a hyperdimensional surface of any adequately smooth function parameterized by a set of tunable parameters.
The fashion in which the surface is explored has in some scenarios significant advantages over e.g. a 
brute-force type of search which places unnecessary weight on "uninteresting" parts of the parameter space due
to the nature of its sampling scheme. This is especially true when the function to be evaluated is expensive to 
compute wrt CPU-time. 

The mcmc scheme is Bayesian in nature. The mode of the function depends on the "best-estimates" of parameters. But
each parameter has its own distribution (which allows us to inspect the uncertainty in our estimate) which we can 
observe in 1D by marginalizing over all other parameters: in a frequentist paradigm one thinks of each parameter 
estimate as a point-estimate which does not have any certainty by itself (the uncertainty is captured in the target 
function only). 

While interesting in its own right, we will not consider parameter distributions here: we are purely interested in
optimizing the target function (our machine learning model) with respect to the performance metric. 



Notes:

* Implement a stopping criterion: 
    1) stop after N iterations
    2) stop when function cannot be further optimized withing a set tolerance
    3)    
  
* As the algo will run on a single machine we cannot use the "intra- and between chain variance" stopping criterion.
  (This variable is normally referred to as 'R' in the literature. See e.g. Rubin.)

* 

"""


if __name__ == '__main__':
    import sys
    sys.exit(-1)

