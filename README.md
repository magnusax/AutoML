# ml-meta-wrapper (to be renamed to something more appropriate)

The project aims to develop a customized ML framework written for python versions => 3.4.x. on top of existing libraries. Currently, the project is still in its infancy, and only the bare minimum of functionality is present.  No original features 
have thus been added.

However, some (perhaps) original ideas drawing upon the my background from statistical cosmology to be implemented are:

* Markov Chain Monte Carlo scheme for optimization of (very-) large ensemble models for large data sets where training
  times are relatively long.
  
* Model perturbation and greedy adaptation.


Libraries which will be heavily relied upon are:

### Preprocessing
* scikit-learn (for e.g. outlier detection and standardizing)
* scipy
* ...
* More to follow

### Visualization
* Seaborn
* Hypertools (convenient for visualizing e.g. low-dimensional representations of clusters)

### Learning and evaluation
* scikit-learn
* tensorflow/theano wrapped with keras
* xgboost
* ...
* More to follow

### Intepretation 
* LIME (perhaps, if deemed appropriate. In any case, it will probably be in the form of a wrapper module to ensure easy API-use)

TODO: add direct links to each github project mentioned on this page, 
to ensure that credit and recognition goes to developers


## Comments
Wrapper modules will be added with time. Albeit most machine learning tasks fit on a single laptop, I will consider writing some modules in Spark (perhaps best suited for ETL jobs) or DASK/BLAZE.

Project maintainer can be reached at: johanmagnusaxelsson [at] gmail [dot] com