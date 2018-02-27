import numpy as np
import warnings
from copy import deepcopy

from scipy.stats import uniform
from sampling import Loguniform
from library import library_config
from gazer import GazerMetaLearner


def ensemble_builder(X, names=None):
    """ Build ensemble from base learners
    Input:
        X: input 2D matrix of shape (n_samples, n_columns)
        names: names of algorithms to fetch from repository (optional).
            If not set then names are read from a GazerMetaLearner object.
    Output:
        Dictionary containing name-keys with corresponding values being 
        a list of possible learners with varying settings of hyperparameters.
    """
    if names is None:
        names = GazerMetaLearner(method='complete').get_names()
    _lib = library_config(names, X.shape[0], X.shape[1])
    
    return {name:_generate(name, grid) for name, grid in _lib}

def fit_ensemble(ens, X, y):
    """ Fit an ensemble of algorithms 
    Input:
        ens: dictionary of (name, learner) tuples. The learner must have a fit method.
        X: 2D matrix of shape (n_samples, n_columns)
        y: Label vector of shape (n_samples,)
    Output:
        List of fitted learners
    """
    if not isinstance(ens, dict):
        raise TypeError(__name__+".fit_ensemble: expects 'ens' to be of type dictionary")
        
    return [clf.fit(X,y) for _, clfs in ens.items() for clf in clfs]
           
def _generate(estimator_name, estimator_params):    
    """ Here we generate estimators to later fit. """
    
    if isinstance(estimator_name, str):
        item = GazerMetaLearner(method='chosen', estimators=[estimator_name])
        clfs = item.clf
        if len(clfs)==1:
            _, clf = clfs[0]
        else:
            raise ValueError(__name__+"._generate: should only find 1 algorithm.")
    else:
        raise TypeError(__name__+"._generate: expected string input. \nFound: %s" 
                        % str(estimator_name))        
    
    estimators_to_fit = []
    for estimator_param in estimator_params:
        param = estimator_param['param']
        premise = estimator_param['premise']
        values = _generate_grid(estimator_param['grid'])        
        for value in values:
            estimator = deepcopy(clf.estimator)
            pars = {param:value}
            pars.update(premise)
            try:
                estimator.set_params(**pars)
            except:
                warnings.warn(__name__+"._generate: failed to set param '%s'" % param)
                continue
            estimators_to_fit.append(estimator)
            del estimator                    
    return estimators_to_fit

def _generate_grid(grid):
    """ Generate a config grid. """
    
    method = grid.get('method', None)
    category = grid.get('category', None)        
    
    if method == 'take':
        return grid.get('values', None)    
    
    elif method == 'sample':        
        low = grid.get('low')
        high = grid.get('high')
        num = grid.get('numval')
        prior = grid.get('prior')                
        
        if category == 'discrete':
            raise ValueError('Discrete sampling not allowed yet...check if you need it.')           
        
        elif category == 'continuous':                       
            if prior == 'uniform':
                return np.linspace(low, high, num, endpoint=True)            
            elif prior == 'loguniform':
                logs = loguniform(low=low, high=high, size=num)
                return logs.range()
    else:
        raise ValueError(__name__+"._generate_grid: 'method' should be (take, sample)")
