# Gazer-learn

> "Standing on the shoulders of giants, gazing at the stars"

The project aims to develop a customized AUTO-ML framework, leveraging the power of existing popular libraries. 
As time passes the intention of the author is to add non-standard capabilities such as advanced forms of ensemble
building and MCMC like optimization algorithms (see further below for a brief overview of ongoing efforts).
The project is still in its infancy, and fairly basic functionality has been implemented; but it is currently 
under active development (conditional on time and energy).

Install will (shortly) be available through the python package index `pip install gazer` (dev version). Currently 
tested for python versions `>=3.4` only. 

#### Using the library
The entrance to functionality is the **GazerMetaLearner** object;
    
```python
from gazer import GazerMetaLearner
gz = GazerMetaLearner()
```
	
Set parameter `method='complete'` to initialize all available algorithms. Default is to initialize 3 algorithms randomly.

Post initialization, (meta) algorithms are available in the `.clf` variable of the initialized object. To be more specific, the variable 
returns a list of (name, MetaClassifier) tuples, where the **MetaClassifier** object is a wrapper around a scikit-learn
classifier.

```python    	
gz.clf = [(name1, MetaClassifier1), (name2, MetaClassifier2), ... , (nameN, MetaClassifierN)]
```
   	
Assuming a standard supervised learning scenario with data available in the form of `X, y`, you can easily fit 
available algorithms using default (i.e. reasonble) hyperparameter settings:

```python	
gz.meta_fit(X, y)
```

This method trains all available algorithms. Bayesian optimization is available.



Some ideas are currently being implemented, e.g.
* Markov Chain Monte Carlo scheme for optimization of (very-) large ensemble models 
for large data sets where training times are relatively long.
* Model perturbation and greedy adaptation
* Advanced ensembling techniques
  - See [these two papers from Caruana et al.](http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf)

Libraries which will be heavily relied upon are:

### Preprocessing
* Scikit-learn 
* Scipy

### Visualization
* Matplotlib
* Seaborn 

### Learning and evaluation
* Scikit-learn
* (Tentative) Keras (Tensorflow/Theano)
* (Tentative) Xgboost

TODO: add direct links to each github project mentioned on this page, 
to ensure that credit and recognition goes to developers


## Comments
As this project is in its initial development phase, the code has not been tested properly yet. It thus follows
that it has not yet been uploaded to the PyPi repository yet. Project developer's contact info is available in the setup.py file. 
