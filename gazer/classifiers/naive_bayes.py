from scipy.stats import uniform
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from ..base import BaseClassifier


class MetaGaussianNBayesClassifier(BaseClassifier):
    """
    Docstring:    
    Gaussian naive bayes classifier
    """
    
    def __init__(self, priors=None):
        
        self.name = 'gaussian_nb'
        self.max_n_iter = 0
        
        self.init_params = {}
        self.init_params['priors'] = priors                    
        
        # Initialize algorithm and make it available
        self.estimator = self._get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []
        
    def _get_clf(self):
        return GaussianNB(**self.init_params)            
    
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
        """ Dictionary containing all trainable parameters """
        return [{}]

        
class MetaMultinomialNBayesClassifier(BaseClassifier):
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
        
        self.init_params = {}
        self.init_params['alpha'] = alpha
        self.init_params['fit_prior'] = fit_prior
        self.init_params['class_prior'] = class_prior
        
        # Initialize algorithm and make it available
        self.estimator = self._get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []
        
    def _get_clf(self):
        return MultinomialNB(**self.init_params)
    
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
        """ Dictionary containing all trainable parameters """      
        return [{
            'alpha': uniform(0, 1),
            'fit_prior': [True, False] 
        }]    


class MetaBernoulliNBayesClassifier(BaseClassifier):
    """
    Docstring:    
    Bernoulli naive bayes classifier
    """
    
    def __init__(self, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):
        
        self.name = 'bernoulli_nb'
        self.max_n_iter = 50
        
        self.init_params = {}
        self.init_params['alpha'] = alpha
        self.init_params['binarize'] = binarize
        self.init_params['fit_prior'] = fit_prior
        self.init_params['class_prior'] = class_prior
        
        # Initialize algorithm and make it available
        self.estimator = self._get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = []
        
    def _get_clf(self):
        return BernoulliNB(**self.init_params)
    
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
        """ Dictionary containing all trainable parameters """
        return [{
            'alpha': uniform(0, 1),  
            'fit_prior': [True, False] 
        }]