from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint, uniform
from ..base import BaseClassifier


class MetaDecisionTreeClassifier(BaseClassifier):
    
    def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                 min_impurity_split=None, class_weight=None, presort=False):
    
        self.name = "tree"
        self.max_n_iter = 500

        self.init_params = {}      
        self.init_params['criterion'] = criterion
        self.init_params['max_depth'] = max_depth
        self.init_params['min_samples_split'] = min_samples_split
        self.init_params['min_samples_leaf'] = min_samples_leaf
        self.init_params['max_features'] = max_features
        self.init_params['class_weight'] = class_weight
        self.init_params['random_state'] = random_state

        # Initialize algorithm and make it available
        self.estimator = self._get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []

        
    def _get_clf(self):
        return DecisionTreeClassifier(**self.init_params)    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': 
                    hasattr(self.estimator, 'predict_proba')}
    
    def adjust_param(self, d):
         return super().adjust_params(d)
    
    def set_tune_params(self, params, num_params=1, mode='random', keys=list()):
        return super().set_tune_params(params, num_params, mode, keys) 
    
    def _set_cv_params(self):
        """ Dictionary containing all trainable parameters.      
        """
        return [{ 'criterion': ['entropy', 'gini'],
                  'max_depth': randint(2, 51),
                  'min_samples_split': randint(2, 201),
                  'min_samples_leaf': randint(2, 201),
                  'max_features': uniform(0.25, 0.95),
                  'class_weight': ['balanced', None] }]
                