from scipy.stats import randint, uniform
from sklearn.neighbors import KNeighborsClassifier

class MetaKNearestNeighborClassifierAlgorithm(object):
    """
    Meta classifier object that sits on top of scikit-learn's
    'KNeighborsClassifier' algorithm. We provide extra utilities and functionality
    that aims to automate several steps in e.g. a cross-validation context.
    """
    def __init__(self, 
                 n_neighbors=5, 
                 weights='uniform', 
                 algorithm='auto', 
                 leaf_size=30, 
                 p=2, 
                 metric='minkowski'):
        self.name_ = "KNearestNeighbors"
        self.n_neighbors=n_neighbors
        self.weights=weights
        self.algorithm=algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        # Here we keep track of which parameters to train
        # in a cross validation setting
        self.trainable = None       
        # Initialize empty dictionary which eventually
        # becomes populated with trainable parameters
        self.cv_param_dist = self._param_dist()
        # Initialize algorithm and make it available
        self.estimator = self.get_clf()
        
    def get_clf(self):
        return KNeighborsClassifier(n_neighbors=self.n_neighbors, 
                                    weights=self.weights,
                                    algorithm=self.algorithm, 
                                    leaf_size=self.leaf_size,
                                    p=self.p,
                                    metric=self.metric)    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
    
    def set_cv_params(self, list_of_tuples):
        """
        Results will be available in "cv_param_dist" 
        dictionary and "trainable" list of tuples
        """
        params = list()
        for param, is_trainable in list_of_tuples:
            if is_trainable: params.append(param)
                
        for k, v in self._param_dist().items():
            if k in params: 
                self.cv_param_dist[k] = v        
        return   
        
    def _param_dist(self):
        """
        Dictionary containing all trainable parameters
       (Consider making it public)        
        """
        dict = { 'n_neighbors': randint(2, 20),
                 'weights': ['uniform', 'distance'],
                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                 'leaf_size': [15, 30, 45], # see: http://scikit-learn.org/stable/modules/neighbors.html#neighbors (1.6.4.5. Effect of leaf_size)
                 'p': [1,2,3],
                 'metric': ['minkowski']}
        return dict
        
                           
if __name__ == '__main__':
    sys.exit(-1)