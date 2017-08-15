import numpy as np
from scipy.stats import randint, uniform

from sklearn.linear_model import SGDClassifier

from base import BaseClassifier


class MetaSGDClassifierAlgorithm(BaseClassifier):

    # Use the defaults from scikit-learn package
    def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, learning_rate='optimal', random_state=None):
        
        self.name = "sgd_clf"
        self.max_n_iter = 1000
        
        self.init_params = {}
        self.init_params['loss'] = loss
        self.init_params['penalty'] = penalty
        self.init_params['alpha'] = alpha
        self.init_params['l1_ratio'] = l1_ratio
        self.init_params['fit_intercept'] = fit_intercept
        self.init_params['max_iter'] = max_iter
        self.init_params['learning_rate'] = learning_rate
        self.init_params['random_state'] = random_state
        
        # Initialize algorithm and make it available
        self.estimator = self.get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []

        
    def get_clf(self):
        return SGDClassifier(**self.init_params)
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
    
    def adjust_param(self, d):
         return super().adjust_params(d)
    
    def sample_hyperparams(self, params, num_params, mode, keys):
        return super().trainable_hyperparams(params, num_params, mode, keys)  
    
    def _set_cv_params(self):
        """
        Dictionary containing all trainable parameters
        
        """
        # Trainable params available in self.cv_params().keys()
        return { 'penalty': ['l1', 'l2', 'elasticnet'],
                 'alpha':  
                 'fit_intercept': [True, False],
                 'class_weight': ['balanced', None],
                 'max_iter': [50, 100, 200] }
                                  
if __name__ == '__main__':
    import sys
    sys.exit(-1)