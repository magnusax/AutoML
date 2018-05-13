from __future__ import print_function

import os
import sys
import warnings
from copy import deepcopy

import numpy as np
from scipy.stats import uniform
from sklearn.externals import joblib
from sklearn.exceptions import NotFittedError
from sklearn.metrics import get_scorer

from .sampling import Loguniform
from .progress import show_progress
from .library import library_config
from .gazer import GazerMetaLearner



def build(X, names=None):
    """ 
    Build ensemble from base learners. 
    If 'names' is not set then method returns an empty dictionary.
    
    Parameters:
    -----------

        X : matrix-like
            input 2D matrix of shape (n_samples, n_columns)
            
        names : array-like
            iterable of names of algorithms to fetch from repository.
            If not set then names are read from a GazerMetaLearner object.
    
    Returns:
    ---------

        dict : (name_of_algorithm[str]: list(sklearn classifiers))
            Dictionary containing name keys with corresponding values being 
            a list of possible learners with varying settings of hyperparameters.
    """    
    
    if names is None:
        names = GazerMetaLearner(method='complete').get_names()
        print("Possible names: %s" % ",".join(names))
        return
    
    lib = library_config(names, *X.shape)

    return { name: _generate(name, grid) for name, grid in lib }


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
            Fetches get_scorer() from sklearn.metrics
            
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
        
        for i, estimator in enumerate(clfs):
            
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
                (model_name, scorer(estimator.predict(X), y))
            
            # Show progress bar
            show_progress(i, total_fits)
    
    return model_scores


def _generate(estimator_name, estimator_params):    
    """ Here we generate estimators to later fit. """
    
    learner = GazerMetaLearner(method='chosen', estimators=[estimator_name])
    clfs = learner.clf
    if len(clfs)==1:
        clf = clfs[0][1]
    else:
        raise ValueError("Should find 1 algorithm only.")       
    del learner
    
    estimators = []
    for estimator_param in estimator_params:
        param = estimator_param['param']
        premise = estimator_param['premise']
        values = _generate_grid(estimator_param['grid'])        
        for value in values:
            estimator = deepcopy(clf.estimator)
            pars = { param:value }
            pars.update(premise)
            try:
                estimator.set_params(**pars)
            except:
                warnings.warn("Failed to set param: %s" % param)
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
        low = grid['low']
        high = grid['high']
        grid_points = grid['numval']        
        prior = grid['prior']
        
        if category=='discrete':
            raise ValueError('Discrete sampling not implemented.')                   
        elif category=='continuous':                                  
            if prior=='loguniform':
                return loguniform(low=low, high=high, size=grid_points).range()
            else:
                return np.linspace(low, high, grid_points, endpoint=True)
