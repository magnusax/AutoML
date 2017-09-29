import sys
import numpy as np
import warnings
from copy import deepcopy
from ml_meta_wrapper import MetaWrapperClassifier

def _gen_numeric_values(low, high, num_values):
    return np.linspace(low, high, num_values, endpoint=True)

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
    values = _gen_numeric_values(estimator_params['low'], 
                                 estimator_params['high'], 
                                 estimator_params['numval']) 
    param = estimator_params['param']
    for value in values:
        estimator = deepcopy(clf.estimator)
        try:
            estimator.set_params(**{param:value})
        except:
            warnings.warn("Failed to set param '%s'" % param)
            continue
        estimators_to_fit.append(estimator)
        del estimator        
    return estimators_to_fit

def generate_fit(estimator_name, estimator_params, X, y):
    return [estimator.fit(X,y) for estimator in _generate(estimator_name, estimator_params)]



def ensemble(X, y, mode='default', strategy='passive'):
    # 1. Get names of classifier algorithms
    names = MetaWrapperClassifier().get_names()
    # 2. From 'names' construct a standard library (which we should be able to modify and customize)
    library = load_classifier_library(names, mode=mode, strategy=strategy)
    # 3. For all classifiers in the library: fit to training data
    ensemble = [(lib.name, lib.params) for lib in library]
    # 4. Fit ensemble to data (or just return ensemble?)
    fitted_ensemble = [(name, generate_fit(name, params, X, y)) for name, params in ensemble]
    # 5. Return fitted ensemble
    return fitted_ensemble

if __name__ == '__main__':
    sys.exit(-1)