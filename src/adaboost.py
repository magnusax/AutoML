from scipy.stats import randint, uniform 
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# Import base classifier and regressor objects
from base import BaseClassifier, BaseRegressor


class MetaAdaBoostRegressorAlgorithm(BaseRegressor):
    """ 
    Regression object
    """
    def __init__(self, *args):
        return AdaBoostRegressor()

    
class MetaAdaBoostClassifierAlgorithm(BaseClassifier):
    """
    Meta classifier object that sits on top of scikit-learn's
    'adaboost' algorithm. We provide extra utilities and functionality
    that aims to automate several steps in e.g. a cross-validation context.
    """
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=0.1, algorithm='SAMME.R', random_state=None):
        self.name = "adaboost"
        
        if base_estimator == 'logreg':
            self.base_estimator = LogisticRegression()
        else:
            self.base_estimator = DecisionTreeClassifier()
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        
        # Initialize algorithm and make it available
        self.estimator = self.get_clf()
        
        # Initialize dictionary with trainable parameters
        self.cv_params = self._set_cv_params()
        
    def get_clf(self):
        return AdaBoostClassifier(base_estimator = self.base_estimator, 
                                  n_estimators = self.n_estimators,
                                  learning_rate = self.learning_rate, 
                                  algorithm = self.algorithm,
                                  random_state = self.random_state)    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
        
    def set_hyparms(self, params, num_params, mode, keys):
        # We let the child class inherit a general method from its super class
        return super().trainable_hyperparams(params, num_params, mode, keys)
         
    def _set_cv_params(self):
        """
        Dictionary containing all trainable parameters. This method assumes that we are using the DecisionTreeClassifier
        as base estimator. Consider including more base estimators later.
        """
        
        ad = {'n_estimators': [50, 100, 200],
              'learning_rate': [0.1, 0.05, 0.01]}
        
        if isinstance(self.base_estimator, DecisionTreeClassifier()):
            be = {'base_estimator__criterion': ['gini', 'entropy'],
                  'base_estimator__max_depth': randint(3, 8), # Do not let it go too deep
                  'base_estimator__min_samples_leaf': randint(2, 20),
                  'base_estimator__max_features': [0.1, 'auto', 'log2'],
                  'base_estimator__class_weight': ['balanced', None]}
        
        # Tends to overfit: maybe use weak learners only, or perhaps skip parameter tuning entirely
        elif isinstance(self.base_estimator, LogisticRegression()): 
            be = {'base_estimator__C': uniform(0, 150),
                  'base_estimator__fit_intercept': [True, False],
                  'base_estimator__penalty': ['l1', 'l2']} 
        else:
            be = {} # base estimator specific options not implemented for other classifiers
        
        # This procedure is consistent and likely "version proof".
        d = ad.copy()
        d.update(be) # Mutates 'd' so it returns None
        return d
        
        
# Do not allow calls to this module from the command line
if __name__ == '__main__':
    import sys
    sys.exit(-1)