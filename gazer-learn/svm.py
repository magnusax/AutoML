from scipy.stats import uniform
from sampling import loguniform

from base import BaseClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem



class MetaSVMClassifier(BaseClassifier):
    
    def __init__(self, alpha=0.0001, n_iter=5, learning_rate='optimal', kernel='rbf', 
                 gamma=None, coef0=1, degree=3, kernel_params=None, n_components=100, 
                 random_state=None)
        """
        In some cases we are cheating: we emulate e.g. an RBF kernel by using a kernel approximation on the input data.
        
        """
        self.name = 'svm_%' % kernel
        self.max_n_iter = 1000
        
        self.init_params = {}
        self.init_params['loss'] = 'hinge'
        self.init_params['penalty'] = 'l2'
        self.init_params['alpha'] = alpha
        self.init_params['fit_intercept'] = False
        self.init_params['n_iter'] = n_iter
        self.init_params['learning_rate'] = learning_rate
        self.init_params['random_state'] = random_state
        
        self.init_params_kernel = {}
        self.init_params_kernel['kernel'] = kernel
        self.init_params_kernel['gamma'] = gamma
        self.init_params_kernel['coef0'] = coef0
        self.init_params_kernel['degree'] = degree
        self.init_params_kernel['kernel_params'] = kernel_params
        self.init_params_kernel['n_components'] = n_components
        self.init_params_kernel['random_state'] = random_state
        
        self.estimator = self._get_clf()    
        self.cv_params = self._set_cv_params() 
        self.cv_params_to_tune = []
        
    def _get_clf(self):
        return Pipeline([
            ('kernel', Nystroem(**self.init_params_kernel)), 
            ('model', SGDClassifier(**self.init_params))
        ])
        
    def _set_cv_params(self):
        kernel = {
            'kernel__gamma': loguniform(low=1e-3, high=1e+3),
            'kernel__n_components': [100, ]
        }
        
        model = [{
            'model__alpha': loguniform(low=1e-7, high=1e+7), 
            'model__n_iter': [3, 6, 12, 25, 50],
        }]
        
        
        