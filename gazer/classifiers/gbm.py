import numpy as np
from scipy.stats import randint, uniform 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from ..base import BaseClassifier
from ..utils.stats import _uniform

  
class MetaGradBoostingClassifier(BaseClassifier):
    """ 
    Meta classifier wrapping the gradient boosting classifier in sklearn.
    
    Api reference: 
        ```http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier
        .html#sklearn.ensemble.GradientBoostingClassifier.__init__```
           
    """
    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                 criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
                 min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, 
                 min_impurity_split=None, init=None, random_state=None, max_features=None, 
                 max_leaf_nodes=None):
        
        self.name = "gbm"
        self.max_n_iter = 1000
        
        self.init_params = {
            'loss': loss, 
            'learning_rate': learning_rate,
            'n_estimators': n_estimators, 
            'subsample': subsample,
            'criterion': criterion, 
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf, 
            'min_weight_fraction_leaf': min_weight_fraction_leaf,
            'max_depth': max_depth, 
            'min_impurity_decrease': min_impurity_decrease, 
            'min_impurity_split': min_impurity_split, 
            'init': init, 
            'random_state': random_state, 
            'max_features': max_features, 
            'max_leaf_nodes': max_leaf_nodes}        

        self.estimator = self._get_clf()        
        self.cv_params = self._set_cv_params()       
        self.cv_params_to_tune = []

        
    def _get_clf(self):
        return GradientBoostingClassifier(**self.init_params)    
    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'external': False,
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
     
    def set_params(self, params):
         return super().set_params(params)
        
        
    def set_tune_params(self, params, n_params, mode, keys):
        """ Used by random search procedure to set which parameters to tune 
        by cross validation.
        """
        return super().set_tune_params(params, n_params, mode, keys)
    
    
    def update_cv_params(self, params):
        """ Update parameter dictionary. """
        assert self.cv_params
        return super().update_cv_params(params)       
    
    
    def _set_cv_params(self):
        """ Parameter dictionary used in random- and bayesian searches. """  
        mxft = tuple(v for v in np.linspace(0.1, 0.9, 9))       
        params = {            
            'learning_rate': _uniform(0.01, 0.1),
            'n_estimators': randint(10, 1000),
            'max_depth': randint(1, 10),
            'criterion': ('friedman_mse', 'mae'),
            'subsample': _uniform(0.3, 0.9),
            'max_features': ('sqrt', 'log2', None) + mxft,
            'min_samples_leaf': randint(1, 10), 
        }
        return [params]
    