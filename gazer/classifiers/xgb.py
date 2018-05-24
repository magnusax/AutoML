import numpy as np
from scipy.stats import randint, uniform
from xgboost import XGBClassifier
from ..sampling import Loguniform
from ..base import BaseClassifier


class MetaXGBoostClassifier(BaseClassifier):
    """
    Classifier from the XGBOOST project adapted to sklearn api
    """
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic', 
                 booster='gbtree', gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, 
                 colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, 
                 missing=None):
        
        self.name = 'xgboost'
        self.max_n_iter = 1000
        
        self.init_params = {}
        self.init_params['max_depth'] = max_depth
        self.init_params['learning_rate'] = learning_rate
        self.init_params['n_estimators'] = n_estimators
        self.init_params['silent'] = silent
        self.init_params['objective'] = objective
        self.init_params['booster'] = booster
        self.init_params['gamma'] = gamma
        self.init_params['min_child_weight'] = min_child_weight
        self.init_params['max_delta_step'] = max_delta_step
        self.init_params['subsample'] = subsample
        self.init_params['colsample_bytree'] = colsample_bytree
        self.init_params['colsample_bylevel'] = colsample_bylevel
        self.init_params['reg_alpha'] = reg_alpha
        self.init_params['reg_lambda'] = reg_lambda
        self.init_params['scale_pos_weight'] = scale_pos_weight
        self.init_params['base_score'] = base_score
        self.init_params['random_state'] = random_state
        self.init_params['missing'] = missing 
        
        self.estimator = self._get_clf()        
        self.cv_params = self._set_cv_params()       
        self.cv_params_to_tune = list()
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
    
    def _get_clf(self):
        return XGBClassifier(**self.init_params)
        
    def _set_cv_params(self):
        """ 
        These default parameter settings are borrowed from 
        HyperOpt :: https://github.com/hyperopt/
        """ 
        return [{
            'n_estimators': list(range(50, 550, 50)),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(2, 10, 1),
            'min_child_weight': randint(1, 10, 1),
            'subsample': uniform(0.5, 1.0),
            'colsample_bytree': 
                list(np.linspace(0.5, 1.0, 6, endpoint=True)),
            'colsample_bylevel': 
                list(np.linspace(0.5, 1.0, 6, endpoint=True)),
            'gamma': uniform(0, 1),
            'reg_alpha': Loguniform(1e-10, 1),
            'reg_lambda': uniform(0.1, 10),
            'base_score': uniform(0.1, 0.9),
            'scale_pos_weight': uniform(0.1, 10),
        }]