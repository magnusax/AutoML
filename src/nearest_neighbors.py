from scipy.stats import randint, uniform
from sklearn.neighbors import KNeighborsClassifier
from base import BaseClassifier


class MetaKNearestNeighborClassifierAlgorithm(BaseClassifier):
    """
    Meta classifier object that sits on top of scikit-learn's
    'KNeighborsClassifier' algorithm. We provide extra utilities and functionality
    that aims to automate several steps in e.g. a cross-validation context.
    """
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski'):
        
        self.name = "nearestneigbors"
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        
        # Initialize algorithm and make it available
        self.estimator = self.get_clf()
        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        
    def get_clf(self):
        return KNeighborsClassifier(n_neighbors = self.n_neighbors, 
                                    weights = self.weights,
                                    algorithm = self.algorithm, 
                                    leaf_size = self.leaf_size,
                                    p = self.p,
                                    metric = self.metric)    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
    
    def adjust_param(self, d):
        """
        Update parameter values in algorithm
        """
        import warnings
        
        if not isinstance(d, dict):
            raise ValueError("Expect 'dict'. Got '%s'" % type(d))
        
        for param, value in d.items():
            try: 
                self.estimator.set_params(**{param:value})
            except: 
                warnings.warn("warning: '%s' not set (%s)" % (param, sys.exc_info()[1]))
        return 
    
    def set_hyparms(self, params, num_params, mode, keys):
        # We let the child class inherit a general method from its super class
        return super().trainable_hyperparams(params, num_params, mode, keys)
        
    def _set_cv_params(self):
        """
        Dictionary containing all trainable parameters
       (Consider making it public)        
        """
        return { 'n_neighbors': randint(2, 20),
                 'weights': ['uniform', 'distance'],
                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                 'leaf_size': [15, 30, 45], # see: http://scikit-learn.org/stable/modules/neighbors.html#neighbors (1.6.4.5. Effect of leaf_size)
                 'p': [1,2,3],
                 'metric': ['minkowski']}
                           
if __name__ == '__main__':
    import sys
    sys.exit(-1)