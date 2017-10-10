from base import BaseClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


class MetaGaussianNBayesClassifierAlgorithm(BaseClassifier):
    """
    Docstring:    
    Gaussian naive bayes classifier
    """
    
    def __init__(self, priors=None):
        self.name = 'gaussian_nb'
        self.max_n_iter = 0
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
    
    def adjust_params(self, params):
        """ Update parameter values in algorithm """
        return super().adjust_params(params)

    def sample_hyperparams(self, params, num_params=1, mode='random', keys=[]):
        """ Sample a subset of hyperparameters to optimize """
        return super().trainable_hyperparams(params, num_params, mode, keys)  
        
    def _set_cv_params(self):
        """ Dictionary containing all trainable parameters """
        return [{}]

        
class MetaMultinomialNBayesClassifierAlgorithm(BaseClassifier):
    """
    Docstring:    
    Multinomial naive bayes classifier
    
    From sklearn documentation (http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes):
    The smoothing priors alpha \ge 0 accounts for features not present in the learning samples 
    and prevents zero probabilities in further computations. Setting alpha = 1 is called 
    Laplace smoothing, while alpha < 1 is called Lidstone smoothing.
    
    """
    
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.name = 'multinomial_nb'
        self.max_n_iter = 50
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
    
    def adjust_params(self, params):
        """ Update parameter values in algorithm """
        return super().adjust_params(params)

    def sample_hyperparams(self, params, num_params=1, mode='random', keys=[]):
        """ Sample a subset of hyperparameters to optimize """
        return super().trainable_hyperparams(params, num_params, mode, keys)   
        
    def _set_cv_params(self):
        """ Dictionary containing all trainable parameters """
        from scipy.stats import uniform as sp_uniform        
        return [{'alpha': sp_uniform(0, 1),
                 'fit_prior': [True, False] }]    


class MetaBernoulliNBayesClassifierAlgorithm(BaseClassifier):
    """
    Docstring:    
    Bernoulli naive bayes classifier
    """
    
    def __init__(self, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):
        self.name = 'bernoulli_nb'
        self.max_n_iter = 50
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        
        # Initialize algorithm and make it available
        self.estimator = self.get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []
        
    def get_clf(self):
        return BernoulliNB(alpha = self.alpha,
                           binarize = self.binarize, 
                           fit_prior = self.fit_prior, 
                           class_prior = self.class_prior)
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}    
    
    def adjust_params(self, params):
        """ Update parameter values in algorithm """
        return super().adjust_params(params)

    def sample_hyperparams(self, params, num_params, mode, keys):
        """ Sample a subset of hyperparameters to optimize """
        return super().trainable_hyperparams(params, num_params, mode, keys)   
        
    def _set_cv_params(self):
        """ Dictionary containing all trainable parameters """
        from scipy.stats import uniform as sp_uniform        
        return [{'alpha': sp_uniform(0, 1),  
                 'fit_prior': [True, False] }]


if __name__ == '__main__':
    import sys
    sys.exit(-1)