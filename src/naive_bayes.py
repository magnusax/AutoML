from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from base import BaseClassifier


class MetaNaiveBayesClassifierAlgorithm(BaseClassifier):
    """
    Meta classifier object that sits on top of scikit-learn's
    'LogisticRegression' algorithm. We provide extra utilities and functionality
    that aims to automate several steps in e.g. a cross-validation context.
    """
    # Use the defaults 
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, binarize=None, priors=None, bayes_type=None):
        
        self.alpha = alpha
        self.fit_prior = fit_prior,
        self.class_prior = class_prior
        self.binarize = binarize
        self.priors = priors
        self.bayes_type = bayes_type
                    
        # Initialize algorithm and make it available
        self.estimator = self.get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []

        
    def get_clf(self):
        
        if self.bayes_type == 'GaussianNB' or self.bayes_type is None:
            return GaussianNB(priors = self.priors)
        elif self.bayes_type == 'MultinomialNB':
            return MultinomialNB(alpha = self.alpha, 
                                 fit_prior = self.fit_prior, 
                                 class_prior = self.class_prior)
        elif self.bayes_type == 'BernoulliNB':
            return BernoulliNB(alpha = self.alpha, 
                               binarize = self.binarize, 
                               fit_prior = self.fit_prior, 
                               class_prior = self.class_prior)
        else:
            raise ValueError("Wrong 'bayes_type' value")
            
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
        return { }
                                  
if __name__ == '__main__':
    import sys
    sys.exit(-1)