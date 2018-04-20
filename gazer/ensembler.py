from __future__ import print_function

import os
import sys
import warnings
from copy import deepcopy

import numpy as np
from scipy.stats import uniform
from sklearn.externals import joblib
from sklearn.exceptions import NotFittedError

from sampling import Loguniform
from library import library_config
from gazer import GazerMetaLearner
from progress import show_progress


def build(X, names=None):
    """ 
    Build ensemble from base learners. 
    If 'names' is not set then method returns an empty dictionary.
    
    Input:
    -----------
        X : matrix-like
            input 2D matrix of shape (n_samples, n_columns)
            
        names : array-like
            iterable of names of algorithms to fetch from repository.
            If not set then names are read from a GazerMetaLearner object.
    
    Returns:
    -----------
        dict : (name_of_algorithm[str]: list( sklearn classifiers ))
            Dictionary containing name keys with corresponding values being 
            a list of possible learners with varying settings of hyperparameters.
    """    
    if names is None:
        names = GazerMetaLearner(method='complete').get_names()
        print("Possible names: %s" % ",".join(names))
        return {}
    
    name_grid = library_config(names, *X.shape)
    return {name:_generate(name, grid) for name, grid in name_grid}

def fit(ens, X, y, save_dir=None, **kwargs):
    """ Fit an ensemble of algorithms. If 'save_dir' is set then models
    are pickled to that directory (if directory does not exist, we attempt
    to create it). Method returns a (flat) list of fitted learners.
    
    Input:
    ----------------
        ens: dict([key=str]:[value=list])
            dictionary of (name, learner) tuples. The learner 
            must have a fit method.
        X (2D array-like or matrix-like): 
            2D matrix of shape (n_samples, n_columns)
        y (iterable, array-like): 
            Label vector of shape (n_samples,)
        save_dir (str): 
            directory to pickle fitted algorithms
        **kwargs: 
            variables related to scikit-learn estimators 
            (such as e.g. n_jobs)
    
    Returns:
    ----------------
        List(fitted scikit-learn classifiers):
            List of fitted learners. If save_dir is a valid directory 
            it will contain the pickled versions of all fitted classifiers.
    """
    
    if not isinstance(ens, dict):
        raise TypeError(__name__+".fit_ensemble: expect 'ens' to be a dictionary.")
    
    if save_dir is not None:
        if save_dir[-1]!="/":
            save_dir += "/"        
        if not os.path.isdir(save_dir):
            try:
                os.makedirs(save_dir)
            except:
                raise ValueError("Could not create '%s'." % save_dir)
    
    for name, clfs in ens.items():
        total_fits = len(clfs)
        checked = False
        for i, clf in enumerate(clfs):
            if not checked:
                if hasattr(clf.estimator, 'fit'):
                    clf.estimator.fit(X, y)
                else:
                    raise Exception("'fit' method required.") 
                checked = True
            train_score = get_score(clf.estimator.predict(X), y)
            if save_dir is not None:
                model_name = 'model_%s_%s.pkl' % (name, (i+1))
                try:
                    joblib.dump(clf.estimator, save_dir+model_name)
                except:
                    warnings.warn("Could not pickle '%s'." % model_name)
            show_progress(i, total_fits)
        print("Fitted all versions of '%s' algorithm." % name)
    return 

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