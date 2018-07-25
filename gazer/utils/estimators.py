import os
import sys
from sklearn.externals import joblib

__estimator_types__ = ('keras', 'sklearn')


def save_model(estimator, file, estimator_type=None):
    """ Save an estimator """   
    
    if ((estimator_type is not None) and not
    (estimator_type in __estimator_types__)):
        raise ValueError(
            "Expected 'estimator_type' in: {}"
            .format(", ".join(__estimator_types__)))
    
    if estimator_type is None:
        if hasattr(estimator, 'save'):
            estimator_type = 'keras'
        else:
            estimator_type = 'sklearn'
            
    folder = os.path.dirname(file)
    if not os.path.isdir(folder):
        os.makedirs(folder)    
        
    if estimator_type == 'keras':   
        estimator.save(file, overwrite=True)
        
    elif estimator_type == 'sklearn':
        joblib.dump(estimator, file)        
    
    return