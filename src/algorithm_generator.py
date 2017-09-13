import sys
import numpy as np
from copy import deepcopy
from ml_meta_wrapper import MetaWrapperClassifier

def _gen_numeric_values(low, high, num_values):
    return np.linspace(low, high, num_values, endpoint=True)

def _generate(estimator_name, estimator_params):    
    if isinstance(estimator_name, str):
        item = MetaWrapperClassifier(method = 'chosen', estimators = [estimator_name])
        clfs = item.clf
        if len(clfs)==1:
            _, clf = clfs[0]
        else:
            raise ValueError("Should only have 1 algorithm")    
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
            sys.exit("Failed to set param '%s'" % param)
        estimators_to_fit.append(estimator)
        del estimator        
    return estimators_to_fit

def generate_fit(estimator_name, estimator_params, X, y):
    return [estimator.fit(X,y) for estimator in _generate(estimator_name, estimator_params)]

if __name__ == '__main__':
    sys.exit(-1)