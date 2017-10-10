import sys
import numpy as np
from scipy.stats import randint, uniform
from sampling import loguniform
from sklearn.linear_model import LogisticRegression
from base import BaseClassifier


class MetaLogisticRegressionClassifierAlgorithm(BaseClassifier):
    """
    Meta classifier object that sits on top of scikit-learn's
    'LogisticRegression' algorithm. We provide extra utilities and functionality
    that aims to automate several steps in e.g. a cross-validation context.
    """
    # Use the defaults 
    def __init__(self, penalty='l2', C=1.0, fit_intercept=True, random_state=None, solver='liblinear', max_iter=100, warm_start=False):
        
        self.name = "logreg"
        self.max_n_iter = 1000
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.random_state = random_state
        
        # Initialize algorithm and make it available
        self.estimator = self.get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []

        
    def get_clf(self):
        return LogisticRegression(penalty = self.penalty, 
                                  C = self.C,
                                  fit_intercept = self.fit_intercept, 
                                  solver = self.solver,
                                  max_iter = self.max_iter,
                                  warm_start = self.warm_start,
                                  random_state = self.random_state)    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
    
    def adjust_param(self, d):
         return super().adjust_params(d)
    
    def sample_hyperparams(self, params, num_params, mode, keys):
        # We let the child class inherit a general method from its super class
        return super().trainable_hyperparams(params, num_params, mode, keys)  
    
    def _set_cv_params(self):
        """
        Dictionary containing all trainable parameters      
        """
        # Trainable params available in self.cv_params().keys()
        return [{ 'penalty': ['l1','l2'],
                  'C': loguniform(low=1e-7, high=1e+7),
                  'fit_intercept': [True, False],
                  'class_weight': ['balanced', None],
                  'max_iter': [50, 100, 200] }]
                                  
if __name__ == '__main__':
    sys.exit(-1)