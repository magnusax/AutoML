from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from base import BaseClassifier

class MetadSVMClassifier(BaseClassifier):
    
    def __init__(
        
                kernel='rbf', gamma=None, coef0=1, degree=3, kernel_params=None, n_components=100, random_state=None)
        """
        In some cases we are cheating: we emulate e.g. an RBF kernel by using a kernel approximation on the input data.
        
        """
        self.name = 'svm'
        self.max_n_iter = 1000
        
        self.init_params = {}
        self.init_params['random_state'] = random_state
        
        self.init_params_kernel = {}
        self.init_params_kernel['kernel'] = kernel
        self.init_params_kernel['gamma'] = gamma
        self.init_params_kernel['coef0'] = coef0
        self.init_params_kernel['degree'] = degree
        self.init_params_kernel['kernel_params'] = kernel_params
        self.init_params_kernel['n_components'] = n_components
        self.init_params_kernel['random_state'] = random_state
        
        self.estimator = self.get_clf()    
        
    def get_clf(self):
        return Pipeline([
            ('kernelize', Nystroem(**init_params_kernel)), 
            ('model', SGDClassifier(**init_params))
        ])
        pipeline