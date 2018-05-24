from ..base import BaseClassifier
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
    
    
class MetaRandomForestClassifier(BaseClassifier):
    """
    Implementation of random forest classifier:
    http://scikit-learn.org/0.17/modules/generated/sklearn.ensemble.\
                RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    """
    
    def __init__(self, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, 
                 oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None):
        
        self.name = "random_forest"
        self.max_n_iter = 1000
        
        self.init_params = {}
        self.init_params['n_estimators'] = n_estimators
        self.init_params['criterion'] = criterion
        self.init_params['max_depth'] = max_depth
        self.init_params['min_samples_split'] = min_samples_split
        self.init_params['min_samples_leaf'] = min_samples_leaf
        self.init_params['min_weight_fraction_leaf'] = min_weight_fraction_leaf
        self.init_params['max_features'] = max_features
        self.init_params['max_leaf_nodes'] = max_leaf_nodes
        self.init_params['bootstrap'] = bootstrap
        self.init_params['oob_score'] = oob_score
        self.init_params['random_state'] = random_state
        self.init_params['warm_start'] = warm_start
        self.init_params['class_weight'] = class_weight
        
        # Initialize algorithm and make it available
        self.estimator = self._get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()        
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []
        
        
    def _get_clf(self):
        return RandomForestClassifier(**self.init_params)

    def get_info(self):
        return {'does_classification': True, 'does_multiclass': True,
                'does_regression': False, 'predict_probas': hasattr(self.estimator, 'predict_proba')}
        
    def adjust_params(self, d):
        return super().adjust_params(d)
    
    def set_tune_params(self, params, num_params=1, mode='random', keys=list()):
        return super().set_tune_params(params, num_params, mode, keys)  
    
    def _set_cv_params(self):
        """
        Trainable params available in self.cv_params[i].keys() for i in len(self.cv_params)
        """
        return [{
            "max_depth": [None, 3, 5, 7],
            "max_features": randint(1, 21),
            "min_samples_split": randint(2, 21),
            "min_samples_leaf": randint(1, 21),
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"],
            "n_estimators": [10, 64, 128, 512]},
        ]