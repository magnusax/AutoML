from __future__ import print_function

import sys
import time
import inspect
import warnings

from operator import itemgetter
from importlib import import_module  
import numpy as np

from skopt import gp_minimize
from sklearn.model_selection import (cross_val_score, 
                                     RandomizedSearchCV)

from .metrics import get_scorer
from .algorithms import implemented        
from .utils.mappings import skopt_space_mapping
from .base import EnsembleBaseClassifier, BaseClassifier


from gazer import __importflags__
if __importflags__[0]: 
    import keras
if __importflags__[1]: 
    import xgboost
if not __package__:
    __package__ = __name__

    
    
class GazerMetaLearner():
    """        
    Meta class that keeps track of available classifiers 
    and implements functionality around them.        
    
    Importing:
    -----------
    from gazer import GazerMetaLearner
    learner = GazerMetaLearner()     
      
      
    Parameters:
    ------------
    
        method : str,  default, 'random'
            random; choose 'n_samples' random classifiers
            all;  choose all available classifiers,
            select; send in an iterable of classifier names (arg: 'estimators')
                      
        n_samples : integer, default: 3
            Choose number of classifiers to sample. Used when method='random' else ignored.
       
        estimators : None or list-like, default: None
            Send a list of classifiers to initialize. Used when method='selected' else ignored.

        base_estimator: None or sklearn estimator, default: None 
            Decide which classifier to use as a weak learner in ensemble estimators.
       
        verbose : integer, default: 0 
            If verbose>0 then output feedback messages.       
    
    Returns:
    ---------
    GazerMetaLearner : object class
    
        - See 'self.clf' for a dictionary of initialized learning algorithms.

        - Each algorithm is a wrapper, or "meta estimator" that implements extra functionality
          around scikit-learn like algorithms.

        - Algorithms are accessible through self.clf[name] where name is of str type.

        - Names are available in 'self.names' and you may consult this property whenever you need
          a hint on how to inspect an algorithm (or change it).
    
    """
    def __init__(self, 
                 method='random', 
                 n_samples=3, 
                 estimators=None, 
                 base_estimator=None, 
                 exclude=None, 
                 verbose=0, 
                 random_state=None):                    
        
        options = ('random', 'all', 'select')
        
        options_string = ", ".join(options)        
        if method not in options:
            raise ValueError("Allowed `method` values: {}".format(options_string))                 
        self.method = method
        
        self.verbose = verbose   
        
        self.n_samples = n_samples if method=='random' else None

        # For reproducibility
        self.random_state = (np.random.RandomState(random_state) 
                             if random_state is not None else None)   

        # This estimator is used as 'base_estimator' in ensembling algorithms
        # It will override defaults in the individual init scripts
        self.base_estimator = base_estimator

        # Algorithms to exclude (to avoid sampling them if asking for random set) 
        self.exclude = [] if exclude is None else exclude
        
        # Estimators (used when method == 'select')
        self.estimators = [] if estimators is None else list(estimators)
        
        # For internal use only
        self._mapper = {}

        # Build repository of classifiers
        try:
            self.clf = self._build_repository() 
        except Exception:
            # If we are here then we need advice on available algorithms:
            setattr(self, 'method', 'all'); setattr(self, 'exclude', [])
            self.clf = self._build_repository()
            raise ValueError("""Specify at least 1 algorithm in 'estimators'. 
                \nRecommended: {}""".format(", ".join(self.names)))
        
        if self.verbose > 0: 
            print("Available algorithms (use '.clf' attribute for access):\n{}"
                  .format(", ".join(self.names)))         
    
    @property
    def names(self):
        return list(self.clf.keys())
    
    @property
    def n_algorithms(self):
        return len(self.names)
    
    
    def update(self, name, params):
        """ 
        
        Update any meta estimator's parameters. For now,
        we first delete the old version of the meta estimator in
        the 'self.clf' dictionary, then attempt to replace it with a new
        version. If update fails, we fall back to the old version again, 
        and throw a warning.
        
        - Note: if self.verbose > 0 we print the meta estimator's init signature
          if for some reason the update procedure fails.
          
        Parameters:
        ------------
            name : str
                Name of an initialized meta estimator.
                Must match an entry in 'self.names' property.
            
            params : dict
                A dictionary containing parameters to beupdated. 
                To see available parameters check each meta estimator's
                __init__ signature.
                  
        Returns:
        ---------
        Nothing. The 'self.clf' dictionary is edited inplace.
        
        """
        if not len(params)>0:
            raise ValueError("'params' cannot be empty.")
            
        if not name in self.names:
            raise ValueError("'name' not a valid name: see 'self.names'.")
                             
        mod_name, meta_estimator = self._mapper[name].split("__")
        
        # Find correct module
        module = import_module(
            ".".join((__package__, "classifiers", mod_name)), package=True)
        
        new = getattr(module, meta_estimator)        
        
        # If we fail to update; keep old version
        old = self.clf.pop(name)
        try:
            self.clf[name] = new(**params)
        except:
            _, desc, _ = sys.exc_info()
            warnings.warn("Failed to update {}. Msg: {}"
                          .format(name, desc))
            if self.verbose > 0:
                signature = inspect.getfullargspec(old.__init__)
                print("{0:18}  {1}".format("Variable:", 'Default value:'))
                for arg, default in zip(signature.args[1:], 
                                        signature.defaults):
                    print("{0:18}  {1}".format("<{}>".format(arg), default))
            self.clf[name] = old
        return
        
        
    def _build_repository(self):
                
        # Read implemented algorithms
        global __importflags__
        to_add = implemented(*__importflags__)
        
        # Build from available
        repo = []        
        for module, cls in to_add:
            instance = self._add_algorithm(module, cls)            
            if (instance is not None) and (instance.name not in self.exclude):
                repo.append((instance.name, instance))  
                self._mapper[instance.name] = "__".join((module, cls))
                
        # All
        if self.method=='all':
            return {name:clf for name, clf in repo}
        
        
        # Random
        if self.method=='random':
            num = min(len(repo), self.n_samples)
            if self.verbose>0: 
                print("Sampling {} algorithms".format(num))
            repo = [repo[i] for i in np.random.choice(len(repo), num, replace=False)]           
            return {name:clf for name, clf in repo}
        
        
        # Select 
        if self.method=='select':
            if len(self.estimators)>0:
                return {name:clf for name, clf in repo if name in self.estimators}
            elif self.estimators is None or (len(self.estimators)==0):
                raise Exception()

                
    def _add_algorithm(self, module_name, algorithm_name):                     
        """ Import classifier algorithms """
                    
        path_to_module = ".".join((__package__, "classifiers", module_name))                           
        try:
            module = import_module(path_to_module, package=True)
        except ImportError: 
            warnings.warn("Could not import {}\n{}"
                          .format(module_name, sys.exc_info()[1]), 
                          RuntimeWarning)
            return None
        
        algorithm = getattr(module, algorithm_name)        
        
        if issubclass(algorithm, EnsembleBaseClassifier):
            instance = algorithm(random_state = self.random_state,
                                 base_estimator = self.base_estimator)                    
        elif issubclass(algorithm, BaseClassifier):            
            if hasattr(algorithm(), 'random_state'):
                instance = algorithm(random_state = self.random_state)
            else:
                instance = algorithm()                
        return instance
    
    
    def fit(self, X, y, n_jobs=1):
        """ 
        Fit available algorithms.
        
        Note: the neural network module needs to be treated separately
        and so has its own `fit` method implemented.
        
        Parameters:
        ------------
        
            X : numpy matrix, pandas DataFrame, 2D numpy array
                Training data.
                
            y : numpy array, iterable
                Training labels.
                
            n_jobs : integer, optional, default: 1
                Perform parallel computation if implemented in algorithm.
        
        """
        for name, clf in self.clf.items():            
            start = time.time()            
            
            if hasattr(clf, 'fit'):
                clf.fit(X, y, verbose=self.verbose)
            else:
                if n_jobs != 1 and hasattr(clf.estimator, 'n_jobs'):
                    clf.estimator.set_params(**{'n_jobs': n_jobs})
                clf.estimator.fit(X, y)
            
            if self.verbose>0:
                delta = (time.time()-start)/float(60)
                print("{}: training time = {:.1f} min.".format(name, delta))
        return
    
    
    def set_params(self, name, params):
        clf = self._get_algorithm(name)
        try:
            clf.adjust_params(params)
        except AttributeError:
            raise Exception(
                "Could not adjust params: %s" 
                % sys.exc_info()[1])
        return self
    
    
    def _get_algorithm(self, name):
        if not name in self.names:
            raise ValueError("{} not found.".format(name))
        alg = self.clf.get(name, None)
        assert alg is not None
        return alg
    

    def predict(self, X):
        """
        Compute class labels from data matrix 'X'.
        
        Parameters:
        -----------
            X : array-like, matrix-like
                Data with shape (n_samples, n_features) 
                to predict on.
                
        Returns:
        ---------
            labels : list of (name, class_label) tuples where
            'name' is name of algorithm, and class_label is a numpy
            array of shape (n_samples,).
            - Length of labels: len('GazerMetaLearner().names')

        """
        # Wrap 'predict' using a Lambda expression
        predict = (lambda clf, x: clf.predict(x) if hasattr(clf, 'predict') 
                                               else clf.estimator.predict(x))        
        return [(name, predict(clf, X)) for name, clf in self.clf.items()]
    

    def predict_proba(self, X):
        """
        Compute class probabilities from data matrix 'X'.
        
        Parameters:
        -----------
            X : array-like, matrix-like
                Data with shape (n_samples, n_features) 
                to predict on.
        
        Returns:
        ---------
            probas : list of (name, class_proba) tuples where
            'name' is name of algorithm, and class_proba is a numpy
            array of shape (n_samples, n_classes).
            - Length of probas: len('GazerMetaLearner().names')
         
        """
        # Wrap 'predict_proba' using Lambda expression
        predict_proba = (lambda clf, x: clf.estimator.predict_proba(x) 
                         if clf.get_info()['predict_probas'] else np.zeros(x.shape[0]))
        return [(name, predict_proba(clf, X)) for name, clf in self.clf.items()] 
    

    def evaluate(self, X, y, metric='accuracy', get_loss=True, **kwargs):
        """
        Evalute predictions computed from X against ground truth given by y 
        using native scikit-learn metrics.
 
        Parameters:
        ------------
            X : matrix-like, 2D-array      
                A matrix-like object of shape (n_samples, n_features)
                - The data we wish to predict and evaluate score on

            y : array-like, list-like, iterable     
                An array (iterable) of ground truth labels. 
                Shape: (n_samples,)

            metric : str, default: 'accuracy'   
                Label that indicates type of metric to use 
                (accuracy, auc, f1, recall, precision, log_loss). 

            get_loss : boolean, default: True
                Compute the log loss score whenever possible.
        
        
        Returns:
        ---------
            scores : dictionary
                Specifies loss and score for each algorithm.
                - Format: {name: {'score': score, 'loss': loss}}
        
        """
        if metric is None:
            raise ValueError("Please specify a metric")        
        scorer = get_scorer(metric)
        
        scores = {}
        for name, y_pred in self.predict(X):            
            score = scorer(y, y_pred) 
            scores[name] = {'score': np.round(score, decimals=4), 'loss': 'N/A'}        
        
        if get_loss:
            log_loss = get_scorer('log_loss')            
            for name, proba in self.predict_proba(X):               
                try:
                    loss = log_loss(y, proba)
                    scores[name]['loss'] = np.round(loss, decimals=4)
                except:
                    scores[name]['loss'] = np.nan
                    warnings.warn("Could not compute loss for {}"
                                  .format(name), RuntimeWarning)                                       
        if self.verbose>0:
            for name, score in scores.items():
                print("{0:18} {1}".format(name+":", ",  ".join(
                    ["{}={}".format(k,v) for k,v in score.items()])),
                     end="\n{}\n".format("-" * 45))
        return scores
    

    def rand_optimize(self, X, y, n_iter=12, scoring='accuracy', cv=10, n_jobs=1, 
                      sample_params=False, min_params=2, get_params=False, random_state=None):
        """
        
        This method is a wrapper to cross validation using RandomizedSearchCV from scikit-learn 
        wherein we optimize each defined algorithm.
        
        Default behavior is to optimize all parameters available to each algorithm, but it is 
        possible to sample (randomly) a subset of them to optimize (sample_params=True) 
        or to choose a set of parameters (get_params=True).
        
        Parameters:
        ------------

        X : 
            Data matrix (n_samples, n_features)
        
        y : 
            Labels/ground truth (n_samples,)
        
        n_iter : integer, default: 1
            Number of iterations to use in RandomizedSearchCV method, 
            i.e. number of independent draws from parameter dictionary
        
        scoring : string or callable, default: 'accuracy'
            Type of scorer to use in optimization
        
        cv : integer or callable, default: 10
            Number of cross validation folds, or callable of correct type
        
        n_jobs : integer, default: 1
            Specify number of parallel processes to use
        
        sample_params : boolean, default: False
            Randomly sample a subset of algorithm parameters and tune these  
        
        min_params: optional, integer, default: 2
            When sample_params=True, choose number of parameters to sample
        
        get_params: optional, boolean, default: 2
            Instead of random sampling, use previously chosen set of parameters to optimize
        
        random_state: None or integer, default: None
            Used for reproducible results
        
        Returns:
        --------
            List containing (classifier name, most optimized classifier) tuples       
        
        """  
        get_key = (lambda name, i: name if i<2 else name+str(i-1))
        
        def search(clf, params):        
                        
            if not params:
                print(clf.name, 'No params')
                return (clf.estimator.fit(X, y), None)
            
            pars = params.copy()                       
            kwargs = {}
            if sample_params and (not get_params):
                kwargs = {'num_params': 
                              np.random.randint(min_params, len(pars)), 
                          'mode': 'random'}            
            elif get_params and (not sample_params) and clf.cv_params_to_tune:
                kwargs = {'keys': clf.cv_params_to_tune, 
                          'mode': 'select'}                             
            
            pars = clf.set_tune_params(pars, **kwargs) if kwargs else pars   
            niter = min(n_iter, clf.max_n_iter)        
            
            randsearch = RandomizedSearchCV(clf.estimator, pars, 
                                            n_iter=niter, scoring=scoring, cv=cv, 
                                            n_jobs=n_jobs, random_state=random_state)            
            fitted = False
            start = time.time()
            try:
                randsearch.fit(X, y)
                fitted = True
            except:
                print(clf.name, 'failed fit')
                print('error:', sys.exc_info()[1])
                return (clf.estimator.fit(X, y), None)
            else:
                print(clf.name, 'success')
                return (randsearch, randsearch.best_score_)
            
            finally:
                if self.verbose > 0 and fitted:
                    delta = (time.time()-start)/60.0
                    print("==== {} ==== \n>>> Search time: {:.1f} (min) \n>>> Best score: {:.4f}"
                          .format(clf.name, delta, randsearch.best_score_), end='\n\n') 
                
        return {get_key(name, idx): search(clf, params) 
                for name, clf in self.clf.items() for idx, params 
                in enumerate(clf.cv_params, start=1)}


    def bayes_optimize(self, X, y, n_calls=50, scoring='accuracy', 
                       greater_is_better=True, cv=10, n_jobs=1, random_state=None):
        """        
        Use package 'scikit-optimize' (github.com/scikit-optimize/scikit-optimize) 
        to do Bayesian Optimization instead of random grid search.
        
        Parameters:
        ------------
            X : matrix-like, 2d-array
                Should be a pandas dataframe or a numpy array.
                - This data is split into folds using a CV procedure
                  (cross_val_score)
            
            y : numpy array, iterable
                Training labels/ground truth.
                
            n_calls : int, default: 50
                Number of iterations/function evaluations allowed
                in search procedure.
                
            scoring : str, default: 'accuracy'
                Metric to use when optimizing the classifier.
                
            greater_is_better : bool, default: True
                If True, then a higher metric score is better, and
                if set to False, a lower score equals better classifier.
                
            cv : int, or callable, default: 10
                Number of CV folds (if integer), or data splitter which
                generates train+val splits/folds.
                
            n_jobs : int, default: 1
                Used by cross_val_score to speed up function evaluation.
                
            random_state : None, int, or callable, default: None
                Used to set the random state, for reproducibility.
                
        Returns:
        ---------
            'opts' : dict of (params, abs(best_score)) tuples: {name: (params, score),..}
            
        """           
        skopt_spaces = [(name, params) for name, clf in self.clf.items() 
                        for params in clf.cv_params]      
        skopt_spaces = skopt_space_mapping(skopt_spaces)  
        
        opts = {} 
        for name, clf in self.clf.items():                   
            
            spaces = [_space for _name, _space in skopt_spaces if _name==name]                                    
            if not spaces:
                raise ValueError("{}: spaces undefined.".format(name))
            
            for space in spaces:                
                if not space:
                    warnings.warn("{}: empty space (continue).".format(name))
                    continue                          
                names, parspace = space.keys(), space.values()
            
                def feval(params):
                    pars = {_name: param for _name, param in zip(names, params)}
                    clf.estimator.set_params(**pars) 
                    score = cross_val_score(clf.estimator, X, y, 
                                            cv=cv, 
                                            scoring=scoring, 
                                            n_jobs=n_jobs)
                    score = np.mean(score)
                    return -score if greater_is_better else score
                
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        res_gp = gp_minimize(feval, parspace, 
                                             n_calls=n_calls, 
                                             random_state=random_state)  
                    except:
                        warnings.warn("Optimization failed: {}".format(sys.exc_info()[1]))
                        continue
                        
                opts[name] = [{k:v for k, v in zip(names, res_gp.x)}, abs(res_gp.fun)]                       
                
                if self.verbose > 0:
                    print("{} \t abs(best_score): {:.4f}".format(name, abs(res_gp.fun)))                
        return opts
