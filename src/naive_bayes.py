, , BernoulliNB
from base import BaseClassifier


class MetaGNBayesClassifierAlgorithm(BaseClassifier):
    """
    Docstring:    
    Gaussian naive bayes classifier
    """
    from sklearn.naive_bayes import GaussianNB
    
    def __init__(priors=None):
        self.name = 'gaussian_nb'
        self.priors = priors                    
        # Initialize algorithm and make it available
        self.estimator = self.get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []
        
    def get_clf(self):
        return GaussianNB(priors = self.priors)            
    
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
    
    def sample_hyperparams(self, params, num_params, mode, keys):
        # We let the child class inherit a general method from its super class
        return super().trainable_hyperparams(params, num_params, mode, keys)  
        
    def _set_cv_params(self):
        """
        Dictionary containing all trainable parameters
        
        """
        return {}

    
class MetaMultinomNBayesClassifierAlgorithm(BaseClassifier):
    """
    Docstring:    
    Multinomial naive bayes classifier
    
    From sklearn documentation (http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes):
    The smoothing priors alpha \ge 0 accounts for features not present in the learning samples 
    and prevents zero probabilities in further computations. Setting alpha = 1 is called 
    Laplace smoothing, while alpha < 1 is called Lidstone smoothing.
    
    """
    from sklearn.naive_bayes import MultinomialNB
    
    def __init__(alpha=1.0, fit_prior=True, class_prior=None):
        self.name = 'multinomial_nb'
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        
        # Initialize algorithm and make it available
        self.estimator = self.get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []
        
    def get_clf(self):
        return MultinomialNB(alpha = self.alpha,
                             fit_prior = self.fit_prior,
                             class_prior = self.class_prior)
    
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
    
    def sample_hyperparams(self, params, num_params, mode, keys):
        # We let the child class inherit a general method from its super class
        return super().trainable_hyperparams(params, num_params, mode, keys)  
        
    def _set_cv_params(self):
        """
        Dictionary containing all trainable parameters
        
        """
        from scipy.stats import uniform as sp_uniform
        return {'alpha': sp_uniform(0, 1)}    

    
  #  elif self.bayes_type == 'BernoulliNB':
  #      return BernoulliNB(alpha = self.alpha, 
  #                           binarize = self.binarize, 
  #                           fit_prior = self.fit_prior, 
  #                           class_prior = self.class_prior)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    import sys
    sys.exit(-1)