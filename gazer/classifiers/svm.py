from scipy.stats import uniform, randint
from ..sampling import Loguniform
from ..base import BaseClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem


class MetaSVMClassifier(BaseClassifier):

    def __init__(self, alpha=0.0001, fit_intercept=True, penalty='l2', max_iter=5, 
                 learning_rate='optimal', kernel='rbf', gamma=None, coef0=1, degree=3, 
                 kernel_params=None, n_components=100, random_state=None):
        """
        In a way we are cheating: we emulate e.g. an RBF kernel by using 
        a kernel approximation on the input data. This is done in order to
        avoid training too long.
        
        """
        
        # Meta data
        self.name = 'svm'
        self.max_n_iter = 1000
                
        # Init params
        self.init_params_kernel = {}
        self.init_params_kernel['kernel'] = kernel
        self.init_params_kernel['gamma'] = gamma
        self.init_params_kernel['coef0'] = coef0
        self.init_params_kernel['degree'] = degree
        self.init_params_kernel['kernel_params'] = kernel_params
        self.init_params_kernel['n_components'] = n_components
        self.init_params_kernel['random_state'] = random_state
        
        self.init_params = {}
        self.init_params['loss'] = 'hinge'
        self.init_params['penalty'] = penalty
        self.init_params['alpha'] = alpha
        self.init_params['fit_intercept'] = fit_intercept
        self.init_params['max_iter'] = max_iter
        self.init_params['learning_rate'] = learning_rate
        self.init_params['random_state'] = random_state

        # Methods
        self.estimator = self._get_clf()    
        self.cv_params = self._set_cv_params() 
        self.cv_params_to_tune = []
        
    def _get_clf(self):
        return Pipeline([
            ('kernel', Nystroem(**self.init_params_kernel)), 
            ('model', SGDClassifier(**self.init_params))
        ])
        
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
        
    def adjust_params(self, par):
        return super().adjust_params(par)    
    
    def _set_cv_params(self):
        
        kernel1 = {
            'kernel__kernel': ['rbf'],
            'kernel__gamma': Loguniform(low=1e-4, high=1e+4),
            'kernel__n_components': [100, 250, 500, 1000, 1200]}
        kernel2 = {
            'kernel__kernel': ['poly'],
            'kernel__degree': randint(1, 5),
            'kernel__coef0': uniform(0., 1.0),
            'kernel__n_components': [100, 250, 500, 1000, 1200]}               
        model = {
            'model__alpha': Loguniform(low=1e-8, high=1e+8), 
            'model__max_iter': [6, 8, 12, 24, 48, 96],
            'model__fit_intercept': [True, False],
            'model__penalty': ['l1', 'l2']}
        
        kernels = [kernel1, kernel2]
        cv_params = list()
        for kernel in kernels:
            temp = model.copy()
            temp.update(kernel)
            cv_params.append(temp)                           
        return cv_params
        
        