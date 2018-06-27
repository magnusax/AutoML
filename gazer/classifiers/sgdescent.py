from scipy.stats import uniform
from sklearn.linear_model import SGDClassifier
from ..sampling import Loguniform
from ..base import BaseClassifier


class MetaSGDClassifier(BaseClassifier):

    def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, 
                 fit_intercept=True, max_iter=5, learning_rate='optimal', random_state=None):
        
        self.name = "sgd_%s" % str(loss)
        self.max_n_iter = 1000
        
        self.init_params = {}
        self.init_params['loss'] = loss
        self.init_params['penalty'] = penalty
        self.init_params['alpha'] = alpha
        self.init_params['l1_ratio'] = l1_ratio
        self.init_params['fit_intercept'] = fit_intercept
        self.init_params['n_iter'] = max_iter
        self.init_params['learning_rate'] = learning_rate
        self.init_params['random_state'] = random_state
        
        # Initialize algorithm and make it available
        self.estimator = self._get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []

        
    def _get_clf(self):
        return SGDClassifier(**self.init_params)
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': 
                    hasattr(self.estimator, 'predict_proba')}
    
    def adjust_params(self, par):
         return super().adjust_params(par)
    
    def set_tune_params(self, params, num_params=1, mode='random', keys=list()):
        return super().set_tune_params(params, num_params, mode, keys)    
    
    def _set_cv_params(self):
        """ Dictionary containing all trainable parameters """
        
        # Trainable params available in: 
        # self.cv_params[i].keys() for i in len(self.cv_params)
        return [
             {'penalty': ['l1', 'l2'],
              'alpha': Loguniform(low=1e-8, high=1e+8), 
              'fit_intercept': [True, False],
              'class_weight': ['balanced', None],
              'max_iter': [5, 10, 25, 50, 100],
              'learning_rate': ['optimal'] },
             
             {'penalty': ['elasticnet'],
              'l1_ratio': uniform(0, 1),
              'alpha': Loguniform(low=1e-8, high=1e+8), 
              'fit_intercept': [True, False],
              'class_weight': ['balanced', None],
              'max_iter': [5, 10, 25, 50, 100],
              'learning_rate': ['optimal'] } ]