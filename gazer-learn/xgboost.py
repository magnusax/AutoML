try:
    from xgboost import XGBClassifier
except ImportError:
    raise 
from base import BaseClassifier


class MetaXGBoostClassifier(BaseClassifier):
    """
    scikit-learn api compatible classifier from the XGBOOST project
    """
    def __init__(self, ):
        
        self.name = 'xgboost'
        self.max_iter = 1000
        
        self.init_params = {}
        
        self.estimator = self._get_clf()        
        self.cv_params = self._set_cv_params()       
        self.cv_params_to_tune = list()
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
    
    def _get_clf(self):
        return XGBClassifier(**init_params)
        
    def _set_cv_params(self):
        pass