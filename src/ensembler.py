import sys
import numpy as np
import warnings
from copy import deepcopy
from library import expose_library
from ml_meta_wrapper import MetaWrapperClassifier


def _gen_numeric_grid(grid):
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
                from scipy.stats import uniform
                return np.linspace(low, high, num, endpoint=True)            
            elif prior == 'loguniform':
                from sampling import loguniform
                logs = loguniform(low=low, high=high, size=num)
                return logs.range()
    else:
        raise ValueError("Method should be ('take','sample')")

        
def _generate(estimator_name, estimator_params):    
    if isinstance(estimator_name, str):
        item = MetaWrapperClassifier(method='chosen', estimators=[estimator_name])
        clfs = item.clf
        if len(clfs)==1:
            _, clf = clfs[0]
        else:
            raise ValueError("Should only have 1 algorithm")
    else:
        raise TypeError("Expecting string input. Found: %s" % str(estimator_name))        
    estimators_to_fit = []
    for estimator_param in estimator_params:
        param = estimator_param['param']
        premise = estimator_param['premise']
        values = _gen_numeric_grid(estimator_param['grid'])        
        for value in values:
            estimator = deepcopy(clf.estimator)
            pars = {param:value}
            pars.update(premise)
            try:
                estimator.set_params(**pars)
            except:
                warnings.warn("Failed to set param '%s'" % param)
                continue
            estimators_to_fit.append(estimator)
            del estimator                    
    return estimators_to_fit


def fit_ensemble(ensemble, X, y):
    if not isinstance(ensemble, dict):
        raise TypeError("Expecting dict input")
    return [clf.fit(X,y) for _, clfs in ensemble.items() for clf in clfs]


def construct_ensemble(X, names=None):
    if names is None:
        names = MetaWrapperClassifier(method='complete').get_names()
    library = expose_library(names, X.shape[0], X.shape[1])
    return {name:_generate(name, grid) for name, grid in library}


if __name__ == '__main__':
    sys.exit(-1)