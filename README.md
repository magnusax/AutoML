# Gazer-learn

As in: "standing on the shoulders of giants, gazing at the stars"

The project aims to develop a customized ML framework, leveraging the power of existing popular libraries. Currently tested for python versions>=3.4. 

The project is still in its infancy, and only the bare minimum of functionality is present.

Install will (shortly) be available through the python package index `pip install gazer`.

The entrance to functionality is the MetaLearner object. It is initialized like so:
    
```python
from gazer import GazerMetaLearner
gz = GazerMetaLearner(method='complete, verbose=1)
```
	
Algorithms are available in the `.clf` variable
    	
    classifiers = gz.clf
    	
Assuming you have data available in the form of `X,y` data you can easily fit available algorithms 
using default hyperparameter settings:
	
    gz.meta_fit(X, y)



Some ideas are currently being implemented, e.g.
* Markov Chain Monte Carlo scheme for optimization of (very-) large ensemble models 
for large data sets where training times are relatively long.
* Model perturbation and greedy adaptation
* Ensembling techniques

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
