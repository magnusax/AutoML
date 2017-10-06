import sys
from scipy.stats import uniform
from sampling import loguniform
from base import BaseClassifier
from sklearn.linear_model import SGDClassifier


class MetaSGDClassifierAlgorithm(BaseClassifier):

    # Use the defaults from scikit-learn package
    def __init__(self, loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, learning_rate='optimal', random_state=None):
        
        self.name = "sgd_%s_loss" % str(loss)
        self.max_n_iter = 1000
        
        self.init_params = {}
        self.init_params['loss'] = loss
        self.init_params['penalty'] = penalty
        self.init_params['alpha'] = alpha
        self.init_params['l1_ratio'] = l1_ratio
        self.init_params['fit_intercept'] = fit_intercept
        self.init_params['n_iter'] = n_iter
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
    
    def adjust_params(self, d):
         return super().adjust_params(d)
    
    def sample_hyperparams(self, params, num_params, mode, keys):
        return super().trainable_hyperparams(params, num_params, mode, keys)  
    
    def _set_cv_params(self):
        """
        Dictionary containing all trainable parameters
        
        """
        # Trainable params available in self.cv_params[i].keys() for i in len(self.cv_params)
        return [
             {'penalty': ['l1', 'l2'],
              'alpha': loguniform(low=1e-7, high=1e+7), 
              'fit_intercept': [True, False],
              'class_weight': ['balanced', None],
              'n_iter': [5, 10, 25, 50, 100],
              'learning_rate': ['optimal', 1e-1, 1e-2] },
             
             {'penalty': ['elasticnet'],
              'l1_ratio': uniform(0, 1),
              'alpha': loguniform(low=1e-7, high=1e+7), 
              'fit_intercept': [True, False],
              'class_weight': ['balanced', None],
              'n_iter': [5, 10, 25, 50, 100],
              'learning_rate': ['optimal', 1e-1, 1e-2] }  
             ]

    
if __name__ == '__main__':
    sys.exit(-1)