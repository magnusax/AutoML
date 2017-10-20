import numpy as np
import warnings
import time
import sys

from skopt import gp_minimize
from utils import skopt_space_mapping 
from base import EnsembleBaseClassifier, BaseClassifier
from algorithms import all_algorithms

from importlib import import_module  
from operator import itemgetter

import seaborn as sns; sns.set()
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, \
recall_score, precision_score, log_loss, matthews_corrcoef as mcc
from sklearn.model_selection import cross_val_score, RandomizedSearchCV


class GazerMetaLearner(object):
    """
    Docstring:        
    Class that keeps track of available classifiers and implements functionality around them    
    
    Import:
    ------------------
    from gazer import GazerMetaLearner
    gzer = GazerMetaLearner()    
    
    Input:
    ------------------
       method = str: ('random' : Choose num_sample random classifiers (default),
                      'complete' : Choose all available classifiers,
                      'chosen' : Send in a list (iterable will work) of classifier names and initialize)                      
       num_sample: (int: 3) choose number of classifiers to sample. Used when method = 'random'
       estimators: (None or list/list-like: None): Send a list of classifiers to initialize. Used when method = 'chosen'
       base_estimator: (None or sklearn-type estimator: None) decide which classifier to use as a weak learner for ensemble methods
       verbose: (int, 0) if > 0 then be 'chatty'       
    
    Output:
    ------------------
    GazerMetaLearner object: see GazerMetaLearner().clf for a list of initialized learning algorithms: (name, clf) tuples
    
    """
    def __init__(self, method='random', num_sample=3, estimators=None, base_estimator=None, exclude=list(), verbose=0):                    
        _options = ('random', 'complete', 'chosen')
        if method not in _options:
            raise ValueError("'method' should be one of (%s)" % ", ".join(_options))                
        
        if method == 'chosen' and (estimators is None or len(estimators) == 0):
            suggestions = [name for name, _ in self.__build_classifier_repository()]
            raise ValueError("Specify name of at least one algorithm in 'estimators'. "\
                             "\nValid options: (%s)" % ", ".join(suggestions))
        
        # Does this improve reproducibility???
        self.random_state = np.random.RandomState(12345678)    
        
        self.verbose = verbose   
        self.method = method
        self.num_sample = num_sample
        
        # This estimator is used as 'base_estimator' in ensembling algorithms
        # It will override defaults in the individual __init__ scripts
        self.base_estimator = base_estimator
        self.exclude = exclude
        
        if method == 'chosen':
            self.clf = [(name, c) for name, c in self.__build_classifier_repository() if name in estimators]
        else:
            self.clf = self.__build_classifier_list()    
        self.clf = [(n, c) for n, c in self.clf if not (n in self.exclude)]               
        
        if self.verbose > 0:    
            print("Initialized classifiers: %s" % ", ".join(self.get_names()))         
    
    def get_names(self):
        return [name for name, _ in self.clf]
    
    def __build_classifier_list(self):        
        clfs = self.__build_classifier_repository()        
        
        if self.method == 'random':
            print("Sampling %i algorithms" % self.num_sample)
            clfs = [clfs[i] for i in np.random.choice(len(clfs), self.num_sample, replace=False)]        
        
        return clfs
        
    def __build_classifier_repository(self):
        """
        Return list of (classifier_name, classifier) tuples        
        """        
        _algorithms = list()       
        _added = all_algorithms()
        for m, c in _added:
            name, algo = self._add_algorithm(m, c)
            if name is not None and algo is not None:
                _algorithms.append((name, algo))
        return _algorithms

    def _add_algorithm(self, module_name, algorithm_name):                     
        try:
            module = import_module(module_name)
        # Should prevent crashing if some library (e.g. xgboost) is missing
        except ImportError: 
            warnings.warn("Could not import %s. Traceback: %s" % (module_name, sys.exc_info()[1]))
            return None, None
            
        algorithm = getattr(module, algorithm_name)
        
        if issubclass(algorithm, EnsembleBaseClassifier):
            instance = algorithm(base_estimator=self.base_estimator, 
                                 random_state=self.random_state)
        elif issubclass(algorithm, BaseClassifier):
            if hasattr(algorithm(), 'random_state'):
                instance = algorithm(random_state=self.random_state)
            else:
                instance = algorithm()
                
        return instance.name, instance
    
    def meta_fit(self, X, y, n_jobs=1):
        for name, clf in self.clf:
            try:
                clf.estimator.set_params(**{'n_jobs': n_jobs})
            except:
                pass
            st = time.time()
            clf.estimator.fit(X, y)
            if self.verbose > 0:
                print("Classifier %s trained (time: %.2f min)" % (name, (time.time()-st)/60.))
        return
    
    def meta_set_params(self, name, params):
        clf = self._get_algorithm(name)
        try:
            clf.adjust_params(params)
        except AttributeError:
            raise        
        return self
    
    def _get_algorithm(self, name):
        if not name in self.get_names():
            raise ValueError("%s not found. Available: %s" % (name, ", ".join(self.get_names())))
        _clf = [clf for n, clf in self.clf if n == name]
        if len(_clf) == 1:
            return _clf[0]
        else:
            raise ValueError("No unique algorithm found: %s" % str(_clf))
    
    def meta_predict(self, X):
        return [(name, clf.estimator.predict(X)) for name, clf in self.clf]
    
    def meta_predict_proba(self, X):
        probas = []
        for name, clf in self.clf:
            # Check if classifier implements 'predict_proba'
            info = clf.get_info()
            if info['predict_probas']:
                probas.append((name, clf.estimator.predict_proba(X)))
        return probas
    
    def meta_evaluate(self, preds, y_true, metric=None, multiclass=None, **kwargs):
        """
        Evalute predictions (preds) against ground truth (y_true) using scikit-learn metrics
        Input:
        -------------
        preds:       a list of (name (str), array-of-predictions (iterable)) tuples (default output from 'meta_predict' method)
                     e.g. [('my_classifier1', np.array(...)), ('my_classifier2', np.array(...)), ...]  
        y_true:      an array (iterable) of ground truth labels
        metric:      a label (str) that indicates type of metric to use (accuracy, auc, f1, recall, precision, log_loss)  
                     Must be set.
        multiclass:  True or False. Indicate if this is a multiclass problem or not. Must be set.
        
        Output:
        -------------
        A list of 3-tuples (one for each algorithm) in the following form: 
        (classifier: name, classifier: metric performance, classifier: log loss)
        """
        
        if metric is None:
            raise ValueError("Please specify a metric")
        if multiclass is None:
            raise ValueError("Please specify if multiclass or not")
            
        # List of supported metric names and the corresponding method (tuples)
        metrics = [('accuracy', accuracy_score),
                   ('auc', roc_auc_score),
                   ('f1', f1_score),
                   ('recall', recall_score),
                   ('precision', precision_score),
                   ('log_loss', log_loss),]
        
        metrics_names = [f for f, _ in metrics]
        
        if not metric in metrics_names:
            raise ValueError("'metric' should be one of the following: '%s'" % ", ".join(metrics_names))
        
        scorer = [sc for name, sc in metrics if name == metric][0]
        scores = []
        
        for name, y_pred in preds:
            val = scorer(y_true, y_pred)
            if multiclass and len(np.array(y_pred).shape) == 1:
                lb = LabelBinarizer()
                yhat = lb.fit_transform(y_pred)
            else:
                yhat = np.array(y_pred)
            loss = log_loss(y_true, yhat)            
            scores.append((name, val, loss))                        
        scores.sort(key=itemgetter(1), reverse=True)
        
        if self.verbose > 0:
            for name, met, loss in scores: print("log_loss: %.4f \t %s: %.4f \t %s" % (loss, metric, met, name))
        return scores
    
    def meta_crossval_optimize(self, X, y, 
                               n_iter=12, scoring='accuracy', cv=10, n_jobs=1, 
                               sample_hyperparams=False, min_hyperparams=None, # min_hyperparams = 2 
                               get_hyperparams=False, random_state=None):
        """
        Docstring:        
        This method is a wrapper to cross validation using RandomizedSearchCV from scikit-learn, wherein we optimize each defined algorithm
        Default behavior is to optimize all parameters available to each algorithm, but it is possible to sample (randomly) a subset of them
        to optimize (sample_hyperparams=True), or to choose a set of parameters (get_hyperparams=True).
        
        Input parameters:
        ------------------
        X: data matrix (n_samples, n_features)
        y: labels/ground truth (n_samples,)
        
        n_iter: (int: 1) number of iterations to use in RandomizedSearchCV method, i.e. number of independent draws from parameter dictionary
        scoring: (str or callable: 'accuracy') type of scorer to use in optimization
        cv: (int or callable: 10) number of cross validation folds, or callable of correct type
        n_jobs: (int: 1) specify number of parallel processes to use
        
        sample_hyperparams: (bool: False) randomly sample a subset of algorithm parameters and tune these  
        min_hyperparams: (int: 2) when sample_hyperparams=True, choose number of parameters to sample
        get_hyperparams: (bool: False) instead of random sampling, use previously chosen set of parameters to optimize (must be preceeded by ...)
        random_state: (None or int: None) used for reproducible results
        
        Ouput:
        ------------------
        List containing (classifier name, most optimized classifier) tuples       
        """
        
        def _random_grid_search(p_index, clf, clf_name, param_dict):
            clf_name = clf_name + "_pd%i" % (p_index+1) if p_index > 0 else clf_name

            param_dist = param_dict.copy()           
            if len(param_dist) == 0:
                warnings.warn("%s has an empty parameter dictionary (returning basic 'fit')" % clf_name)
                return (clf_name, clf.estimator.fit(X,y))
            
            if sample_hyperparams and not get_hyperparams:
                num_params = np.random.randint(min_hyperparams, len(param_dist))
                param_dist = clf.set_tune_params(param_dist, num_params=num_params, mode='random')

            if get_hyperparams and not sample_hyperparams:
                if len(clf.cv_params_to_tune)>0:                    
                    param_dist = clf.set_tune_params(param_dist, keys=clf.cv_params_to_tune, mode='select')            
            
            n_iter_ = min(n_iter, clf.max_n_iter)        
                      
            random_search = RandomizedSearchCV(clf.estimator, param_distributions=param_dist, n_iter=n_iter_, scoring=scoring, cv=cv, n_jobs=n_jobs, 
                                               verbose=0, error_score=0, return_train_score=True, random_state=random_state)            
            fitted = False
            start_time = time.time()
            try:
                random_search.fit(X, y)
                fitted = True
            except:
                warnings.warn("Failed to search through %s (returning basic 'fit'). \nInfo: %s" % (clf_name, sys.exc_info()[1]))
                return (clf_name, clf.estimator.fit(X,y))
            else:
                return (clf_name, random_search.best_estimator_)            
            finally:
                if self.verbose>0 and fitted:
                    print("=" * 75)
                    print("Algorithm: %s \tSearch time: %.2f min. \tBest score: %.5f" 
                          % (clf_name, (time.time()-start_time)/60., random_search.best_score_))
                    print("=" * 75)
        return [
            _random_grid_search(idx, clf, name, param_dict) 
            for name, clf in self.clf 
            for idx, param_dict in enumerate(clf.cv_params)
        ]

    def meta_bayes_opt(self, X, y, n_calls=50, scoring='accuracy', greater_is_better=True, cv=10, n_jobs=1, random_state=None):
        """        
        Use package 'scikit-optimize' >=0.3 in order to do Bayesian optimization instead of random grid search.
        Package URL: https://github.com/scikit-optimize/scikit-optimize/        
        """           

        # cv_params is a list with >= 1 dicts, so we have to iterate over them
        # and add each to the repository of space
        skopt_spaces = skopt_space_mapping([
            (name, params) for name, clf in self.clf for params in clf.cv_params])  
        
        results = []        
        for name, classifier in self.clf:       
            
            # Define search space (could get more than a single hit here), so len(space) >= 1 (potentially)
            spaces = [(name_, skopt_space_) 
                      for name_, skopt_space_ in skopt_spaces if name_ == name]                        
            
            if len(spaces) == 0:
                raise ValueError("Space is undefined [name = %s] (%s)" % (name, space))
            
            for space in spaces:
                
                if not (isinstance(space, tuple) and isinstance(space[1], dict)):
                    raise ValueError("'space' should contain (str, dict)-tuples, got: %s" % str(spaces))    
                if len(space[1]) == 0:
                    warnings.warn("%s:\tempty parameter dictionary (continue)" % name)
                    continue
                          
                param_names = [ n for n, _ in space[1].items() ]
                dim_space = [ dim for _, dim in space[1].items() ]
            
                # Define objective function (it will have access to externally defined variables in 
                # the calling method namespace
                def _objective(params):
                    param_ = {param_name:param for param_name, param in zip(param_names, params)}
                    classifier.estimator.set_params(**param_) 
                    score = np.mean(cross_val_score(classifier.estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs))
                    if greater_is_better:
                        return -score
                    else:
                        return score
                
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        res_gp = gp_minimize(_objective, dim_space, n_calls=n_calls, random_state=random_state)  
                    except:
                        warnings.warn("Optimization failed. Info: %s" % sys.exc_info()[1])
                        continue
                        
                # Classifier with optimized parameters
                best_params = {k:v for k,v in zip(param_names, res_gp.x)}                
                results.append({name:(best_params, res_gp.fun)})                        
                if self.verbose > 0:
                    print("Name: %s \tBest score: %.4f" % (name, -res_gp.fun if greater_is_better else res_gp.fun))        
        
        return results