import sys
import numpy as np
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression

from ..base import BaseClassifier
from ..sampling import Loguniform

class MetaLogisticRegressionClassifier(BaseClassifier):
    """
    Meta classifier object that sits on top of scikit-learn's
    'LogisticRegression' algorithm. We provide extra utilities and functionality
    that aims to automate several steps in e.g. a cross-validation context.
    """
    # Use the defaults 
    def __init__(self, penalty='l2', C=1.0, fit_intercept=True, random_state=None, solver='liblinear', max_iter=100, warm_start=False):
        
        self.name = "logreg"
        self.max_n_iter = 1000
        
        self.init_params = {}      
        self.init_params['penalty'] = penalty
        self.init_params['C'] = C
        self.init_params['fit_intercept'] = fit_intercept
        self.init_params['solver'] = solver
        self.init_params['max_iter'] = max_iter
        self.init_params['warm_start'] = warm_start
        self.init_params['random_state'] = random_state
        self.name = "logreg"
        self.max_n_iter = 1000
        
        self.init_params = {}      
        self.init_params['penalty'] = penalty
        self.init_params['C'] = C
        self.init_params['fit_intercept'] = fit_intercept
        self.init_params['solver'] = solver
        self.init_params['max_iter'] = max_iter
        self.init_params['warm_start'] = warm_start
        self.init_params['random_state'] = random_state
        
        # Initialize algorithm and make it available
        self.estimator = self._get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []

        
    def _get_clf(self):
        return LogisticRegression(**self.init_params)    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
    
    def adjust_param(self, d):
         return super().adjust_params(d)
    
    def set_tune_params(self, params, num_params=1, mode='random', keys=list()):
        return super().set_tune_params(params, num_params, mode, keys) 
    
    def _set_cv_params(self):
        """
        Dictionary containing all trainable parameters      
        """
        # Trainable params available in self.cv_params().keys()
        return [{ 'penalty': ['l1','l2'],
                  'C': Loguniform(low=1e-7, high=1e+7),
                  'fit_intercept': [True, False],
                  'class_weight': ['balanced', None],
                  'max_iter': [50, 100, 200] }]
        # Initialize algorithm and make it available
        self.estimator = self._get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []

        
    def _get_clf(self):
        return LogisticRegression(**self.init_params)    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
    
    def adjust_param(self, d):
         return super().adjust_params(d)
    
    def set_tune_params(self, params, num_params=1, mode='random', keys=list()):
        return super().set_tune_params(params, num_params, mode, keys) 
    
    def _set_cv_params(self):
        """
        Dictionary containing all trainable parameters      
        """
        # Trainable params available in self.cv_params().keys()
        return [{ 'penalty': ['l1','l2'],
                  'C': Loguniform(low=1e-7, high=1e+7),
                  'fit_intercept': [True, False],
                  'class_weight': ['balanced', None],
                  'max_iter': [50, 100, 200] }]