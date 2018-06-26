from __future__ import print_function

import os
import sys
import copy
import warnings
import numpy as np
from operator import itemgetter

from tqdm import tqdm
from scipy.stats import uniform
from sklearn.externals import joblib
from sklearn.exceptions import NotFittedError

from .metrics import get_scorer
from .sampling import Loguniform
from .core import GazerMetaLearner
from .library import library_config


def build(learner, X):
    """Build ensemble from base learners contained
    in the `learner` object.

    Parameters:
    -----------
        learner : object
            instance of GazerMetaLearner class

        X : matrix-like
            input 2D matrix of shape (n_samples, n_features)
            We need some meta data to be able to make sensible choices
            on parameters.

    Returns:
    ---------
        dictionary : (algorithm[str]: classifiers[list])
            Dictionary containing name keys with corresponding values being
            a list of possible learners with varying settings of hyperparameters.
    """
    lib = library_config(learner.names, *X.shape)
    return {name: _generate(name, grid) for name, grid in lib}


def _generate(name, params):    
    """Here we generate estimators to later fit."""
    learner = GazerMetaLearner(
        method='selected', 
        estimators=[name])
    clf = learner._get_algorithm(name)
    del learner
            
    estimators = []
    for param in params:
        par = param['param']
        premise = param['config']
        values = _generate_grid(param['grid'])        
        for value in values:
            estimator = copy.deepcopy(clf.estimator)
            pars = {par:value}
            pars.update(premise)
            try:
                estimator.set_params(**pars)
            except:
                warnings.warn("Failed to set: %s" % par)
                continue
            estimators.append(estimator)
            del estimator                    
    return estimators


def _generate_grid(grid):
    """ Generate a config grid. """
  
    method = grid.get('method', None)
    assert method in ('take', 'sample')  
    category = grid.get('category', None)        
    assert category is not None
        
    if method=='take':
        return grid['values']
    
    elif method=='sample':                
        low, high, points, prior = (
            grid['low'], grid['high'], grid['numval'], grid['prior'])
        
        if category=='discrete':
            raise ValueError('Discrete sampling not implemented.')                   
        
        elif category=='continuous':                                  
            if prior=='loguniform':
                return Loguniform(low=low, high=high, size=points).range()
            else:
                return np.linspace(low, high, points, endpoint=True)

            
def fit(ensemble, X, y, save_dir, scoring='accuracy', **kwargs):
    """
    Fit an ensemble of algorithms. If `save_dir` is set then models
    are pickled to that directory 
    If directory does not exist, we attempt to create it. 
    Method returns a (flat) list of fitted learners.
    
    Parameters:
    ------------
        ensemble: dict(str:list)
            Dictionary of name, list-of-estimators key-value pairs
                   
        X : matrix-like
            2D matrix of shape (n_samples, n_columns)
        
        y : array-like
            Label vector of shape (n_samples,)
        
        save_dir: str             
            Directory to pickle fitted algorithms
        
        scoring : str or callable
            Used when obtaining training data score
            Fetches get_scorer() from local metrics.py module
            
        **kwargs: 
            Variables related to scikit-learn estimator.
            Used to alter estimator parameters if needed (such as e.g. n_jobs)
            
            Example: 
                - Use {'random_forest': {'n_jobs': 4}} to use parallel
                  processing when fitting the random forest algorithm. 
                - Note that the key needs to match the a key in the `ensemble` dict
                  to take effect. 
                - The change takes place through estimator.set_params()
    
    Returns:
    ---------
    Dictionary with paths to fitted and pickled learners, as well as scores on 
    training data. Note that joblib is used to pickle the data.
    
    """ 
    if (save_dir is None or len(save_dir)==0):
        raise Exception(
            "Please specify a valid directory.")
    
    if os.path.exists(save_dir):
        raise Exception(
            "{} already exits. Please choose a different directory."
            .format(save_dir))
    try:
        os.makedirs(save_dir)
    except:
        raise Exception(
            "Could not create folder {}.".format(save_dir))
    
    scorer = get_scorer(scoring)
    
    # Keep track of model and scores in `models`
    # All relevant data is available in `history`
    history = {}
    
    for name, clfs in ensemble.items():        
        os.makedirs(os.path.join(save_dir, name))        
        fkwargs = kwargs.get(name, {})
        print()   
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            models = []
            for idx, estimator in enumerate(tqdm(clfs, desc="{}".format(name), ncols=120)):
                modelname = "{}_{:04d}train.pkl".format(name,(idx+1))
                try:
                    estimator.set_params(**fkwargs)
                    estimator.fit(X, y)
                except:
                    _, desc, _ = sys.exc_info()
                    raise NotFittedError("Could not fit: {}".format(desc))
                try:
                    joblib.dump(estimator, os.path.join(save_dir, name, modelname))
                except:
                    raise Exception("Could not pickle: {}".format(modelname))
                models.append(
                    (modelname, scorer(estimator.predict(X), y)))
        # Return sorted history dict  
        history[name] = sorted(models, key=lambda x: -x[1])
    return history