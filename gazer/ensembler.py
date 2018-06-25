from __future__ import print_function

import os
import sys
import copy
import warnings
import numpy as np

from tqdm import trange
from scipy.stats import uniform
from sklearn.externals import joblib
from sklearn.exceptions import NotFittedError

from .metrics import get_scorer
from .sampling import Loguniform
from .library import library_config
from .core import GazerMetaLearner


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
            Dictionary of (name, estimator) tuples
                   
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
            Variables related to scikit-learn estimator's 
            `fit` method (such as e.g. n_jobs)
    
    Returns:
    ---------
        List of fitted learners. 
            If save_dir is a valid directory it will contain the 
            pickled versions of all fitted classifiers.
    """ 
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # Get scorer from metrics
    scorer = get_scorer(scoring)
    
    # Keep track of model score on train data
    model_scores = []
    
    for name, clfs in ensemble.items():
        total_fits = len(clfs)
        
        for i, estimator in trange(enumerate(clfs)):
            
            # Fit model
            if hasattr(estimator, 'fit'):
                estimator.fit(X, y)
            else:
                warnings.warn("`fit` method required") 
                break
                
            # Save model
            if save_dir is not None:
                model_name = 'model_%s%s.pkl' % (name, (i+1))
                try:
                    joblib.dump(estimator, os.path.join(save_dir, model_name))
                except:
                    raise Exception("Could not pickle %s" % model_name)
            
            # Save score
            model_scores.append(
                (model_name, scorer(estimator.predict(X), y)))

    
    return model_scores