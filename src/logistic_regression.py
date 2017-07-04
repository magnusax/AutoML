import numpy as np
from math import pow
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression

class MetaLogisticRegressionClassifierAlgorithm(object):
    """
    Meta classifier object that sits on top of scikit-learn's
    'LogisticRegression' algorithm. We provide extra utilities and functionality
    that aims to automate several steps in e.g. a cross-validation context.
    """
    def __init__(self, 
                 penalty='l2', 
                 C=1.0, 
                 fit_intercept=True, 
                 random_state=None, 
                 solver='liblinear', 
                 max_iter=100, 
                 warm_start=False):
        self.name_ = "LogReg"
        self.penalty=penalty
        self.C=C
        self.fit_intercept=fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.random_state = random_state
        # Here we keep track of which parameters to train
        # in a cross validation setting
        self.trainable = None       
        # Initialize empty dictionary which eventually
        # becomes populated with trainable parameters
        self.cv_param_dist = self._param_dist()
        # Initialize algorithm and make it available
        self.estimator = self.get_clf()
        
    def get_clf(self):
        return LogisticRegression(penalty=self.penalty, 
                                  C=self.C,
                                  fit_intercept=self.fit_intercept, 
                                  solver=self.solver,
                                  max_iter=self.max_iter,
                                  warm_start=self.warm_start,
                                  random_state=self.random_state)    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
    
    def adjust_params(self, d):
        """
        Update parameter values in algorithm
        """
        if not isinstance(d, dict):
            raise ValueError("Expecting a dictionary. Got %s" % type(d))
        import warnings
        for param, value in d.items():
            try: self.estimator.set_params(**{param:value})
            # Catch any errors and warn
            except: warnings.warn("warning: '%s' not set (%s)" % (param, sys.exc_info()[1]))
        return 
    
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
        # Prefer to keep method private
        def stochastic_C(num_samples): 
            """ 
            Return randomly sampled floats in [10^-7, 10^3]
            Check if there are methods in scipy.stats which can replace this method
            If you want deterministic output, then set np.random.seed 
            """
            return np.random.choice([pow(10, float(f)) for f in np.linspace(-7, 3, 300)], num_samples)
        
        # Some reasonable values
        dict = { 'penalty': ['l1','l2'],
                 'C': stochastic_C(100),
                 'fit_intercept': [True, False],
                 'class_weight': ['balanced', None],
                 'max_iter': [100, 200]}
        
        # Trainable params available in self._param_dist().keys()
        return dict
        
                           
if __name__ == '__main__':
    sys.exit(-1)