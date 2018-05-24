from scipy.stats import randint
from sklearn.neighbors import KNeighborsClassifier
from ..base import BaseClassifier


class MetaKNearestNeighborClassifier(BaseClassifier):
    """
    Meta classifier object that sits on top of scikit-learn's
    'KNeighborsClassifier' algorithm. We provide extra utilities and functionality
    that aims to automate several steps in e.g. a cross-validation context.
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski'):
        
        self.name = "knn"
        self.max_n_iter = 1000
        
        self.init_params = {}
        self.init_params['n_neighbors'] = n_neighbors
        self.init_params['weights'] = weights
        self.init_params['algorithm'] = algorithm
        self.init_params['leaf_size'] = leaf_size
        self.init_params['p'] = p
        self.init_params['metric'] = metric
        
        # Initialize algorithm and make it available
        self.estimator = self._get_clf()       
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []

        
    def _get_clf(self):
        return KNeighborsClassifier(**self.init_params)   
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
    
    def adjust_params(self, params):
        """ Update parameter values in algorithm """
        return super().adjust_params(params)
    
    def set_tune_params(self, params, num_params=1, mode='random', keys=list()):
        return super().set_tune_params(params, num_params, mode, keys)
        
    def _set_cv_params(self):
        """
        Dictionary containing all trainable parameters
       (Consider making it public)        
        """
        return [{ 
            'n_neighbors': randint(2, 100),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': randint(15, 45), # see: http://scikit-learn.org/stable/modules/neighbors.html#neighbors (1.6.4.5. Effect of leaf_size)
            'p': [1,2,3],
            'metric': ['minkowski'] }]