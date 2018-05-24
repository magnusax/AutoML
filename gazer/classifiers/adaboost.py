from scipy.stats import randint, uniform 

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from ..base import EnsembleBaseClassifier, EnsembleBaseRegressor


  
class MetaAdaBoostClassifier(EnsembleBaseClassifier):
    """
    Meta classifier object that sits on top of scikit-learn's
    'adaboost' algorithm. We provide extra utilities and functionality
    that aims to automate several steps in e.g. a cross-validation context.
    """
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=0.1, algorithm='SAMME.R', random_state=None):
        
        self.name = "adaboost"
        self.max_n_iter = 1000
        
        self.init_params = {}        
        if base_estimator is None:
            self.init_params['base_estimator'] = DecisionTreeClassifier()
        else:
            self.init_params['base_estimator'] = base_estimator
        self.init_params['n_estimators'] = n_estimators
        self.init_params['learning_rate'] = learning_rate
        self.init_params['algorithm'] = algorithm
        self.init_params['random_state'] = random_state
        
        # Initialize algorithm and make it available
        self.estimator = self._get_clf()        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()       
        # Initialize list which can be populated with params to tune 
        self.cv_params_to_tune = list()

        
    def _get_clf(self):
        return AdaBoostClassifier(**self.init_params)    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
        
    def set_tune_params(self, params, num_params=1, mode='random', keys=list()):
        return super().set_tune_params(params, num_params, mode, keys)
         
    def _set_cv_params(self):
        """
        Dictionary containing all trainable parameters. This method assumes that we are using 
        the DecisionTreeClassifier as base estimator. Consider including more base estimators later.
        """
        _base_estimator = self.init_params['base_estimator']
        
        ad = {'n_estimators': [50, 100, 200],
              'learning_rate': [0.1, 0.05, 0.01]}
        
        if isinstance(_base_estimator, type(DecisionTreeClassifier())):
            be = {'base_estimator__criterion': ['gini', 'entropy'],
                  'base_estimator__max_depth': randint(1, 8), # Do not let it go too deep
                  'base_estimator__min_samples_leaf': randint(2, 20),
                  'base_estimator__max_features': [0.1, 'auto', 'log2'],
                  'base_estimator__class_weight': ['balanced', None]}
        
        # Tends to overfit: maybe use weak learners only, or perhaps skip 
        # parameter tuning entirely
        elif isinstance(_base_estimator, type(LogisticRegression())): 
            be = {'base_estimator__C': uniform(0, 1000),
                  'base_estimator__fit_intercept': [True, False],
                  'base_estimator__penalty': ['l1', 'l2']} 
        else:
            be = {} # base estimator specific options not implemented for other classifiers
        
        # This procedure is consistent and likely "version proof".
        params = ad.copy()
        params.update(be) # Mutates dict (returns None)
        return [params]