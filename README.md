
# Gazer

> "Standing on the shoulders of giants, gazing at the stars"

The project aims to develop a customized "automl" framework, leveraging the power of existing popular libraries. 
As time passes the intention of the author is to add non-standard capabilities such as advanced forms of ensemble
building and Markov Chain Monte Carlo (MCMC) like optimization algorithms (see further below for a brief overview of ongoing efforts).
The project is still in its infancy, and fairly basic functionality has been implemented; but it is currently 
under active development (conditional on time and energy).

Install will (shortly) be available through the python package index `pip install gazer` (dev version). Currently 
tested for python versions `>=3.4` only. 

### Using the library
The entrance to functionality is the **GazerMetaLearner** object;
    
```python
from gazer import GazerMetaLearner
learner = GazerMetaLearner()
```
	
Set parameter `method='all'` to initialize all available algorithms. Default is to initialize 3 algorithms randomly.

Post initialization, (meta) algorithms are available in the `.clf` variable of the initialized object. To be more specific, the variable 
returns a dictionary of (name, MetaClassifier) tuples, where the **MetaClassifier** object is a wrapper around a "sklearn-like"
classifier. The current version of the library also implements *xgboost* and *keras* classifiers.

```python    	
learner.clf = {name1: MetaClassifier1, name2: MetaClassifier2, ... , nameN: MetaClassifierN}
```
   	
To inspect loaded algorithms, simply call `learner.names`.

Assuming a standard supervised learning scenario with data available in the form of `X, y`, you can easily fit 
available algorithms using default (i.e. reasonble) hyperparameter settings:

```python	
learner.fit(X, y)
```

This method trains all initialized algorithms. Moreover, Random, Grid search, and Bayesian optimization methods
are implemented and may be directly called from the `learner` instance. To see parameters, simply call:
```python
learner.clf['name'].cv_params.

```
This returns a list containing one or more parameter dictionaries that may be edited in place.


### Roadmap
Some ideas are currently being implemented, e.g.
* Markov Chain Monte Carlo scheme for optimization of (very-) large ensemble models 
for large data sets where training times are relatively long.
* Model perturbation and greedy adaptation
* Advanced ensembling techniques
  - See [these two papers from Caruana et al.](http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf)


ML libraries which will be heavily relied upon are first and foremost:
* (Done) Scikit-learn
* (Done) Xgboost
* (Done) Keras 
* (Coming) Pytorch (>= 0.4)


TODO: add direct links to each github project mentioned on this page, 
to ensure that credit and recognition goes to developers


### Comments
As this project is in its initial development phase, the code has not been tested properly yet. It thus follows
that it has not yet been uploaded to the PyPi repository yet. Project developer's contact info is available in the setup.py file. 
