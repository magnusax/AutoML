import sys
import time
import inspect
import warnings

from operator import itemgetter
from importlib import import_module  
import numpy as np

from skopt import gp_minimize
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import (cross_val_score, 
                                     RandomizedSearchCV)

from gazer import __importflags__
if __importflags__[0]: import keras
if __importflags__[1]: import xgboost

    
from .base import EnsembleBaseClassifier, BaseClassifier
from .algorithms import implemented        
from .utils import skopt_space_mapping
from .metrics import get_scorer


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
    def num_estimators(self):
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
        except:            
            try:
                module = import_module(path_to_module, package=False)
            except ImportError: 
                warnings.warn("Could not import {}".format(module_name), RuntimeWarning)
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
        return self._fit(X=X, y=y, n_jobs=n_jobs)
    
    
    def _fit(self, X, y, n_jobs=1):        
        
        for name, clf in self.clf.items():
            st = time.time()            
            if hasattr(clf, 'fit'):
                clf.fit(X, y, verbose=self.verbose)
            else:
                if n_jobs>1 and hasattr(clf.estimator, 'n_jobs'):
                    clf.estimator.set_params(**{'n_jobs': n_jobs})
                clf.estimator.fit(X, y)
            if self.verbose > 0:
                print("%s: training time = %.2f (min)" 
                      % (name, (time.time()-st)/60.))
        return self
    
    
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
        return [(name, clf.estimator.predict(X)) for name, clf in self.clf.items()]
    

    def predict_proba(self, X):
        probas = []
        for name, clf in self.clf.items():
            info = clf.get_info()
            if info['predict_probas']:
                probas.append((name, clf.estimator.predict_proba(X)))
        return probas
    

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
        A list of 3-tuples (one for each algorithm) in the following form: 
        (name, metric score, log loss). If get_loss = False, then 'log loss' 
        will be set to "N/A".
        
        """
        y_preds = self.predict(X)
        
        if get_loss:
            log_loss = get_scorer('log_loss')
            probas = self.predict_proba(X)
                
        if metric is None:
            raise ValueError("Please specify a metric")
        
        # Desired scorer
        scorer = get_scorer(metric)
               
        # Keep track of score for every algorithm
        scores = {}
        for name, y_pred in y_preds:
            
            # Use keras api for convenience
            if name == 'neuralnet':
                y_ = keras.utils.to_categorical(y.reshape(-1, 1))
                loss, score = self.clf[name].estimator.evaluate(X, y_)
                scores[name] = {"score": np.round(score, decimals=4), 
                                'loss': np.round(loss, decimals=4)}
                del y_
                continue
                
            score = scorer(y, y_pred)                        
            implements_proba = self.clf[name].get_info()['predict_probas']            
            
            if get_loss and implements_proba:
                try:
                    y_proba = [x for nme, x in probas if nme==name][0]
                    loss = log_loss(y, y_proba)  
                    scores[name] = {"score": np.round(score, decimals=4), 
                                    'loss': np.round(loss, decimals=4)}
                except:
                    scores[name] = {"score": np.round(score, decimals=4), 'loss': np.nan}
                    warnings.warn("Could not compute log-loss for {}".format(name), RuntimeWarning)
            else:                                
                scores[name] = {"score": np.round(score, decimals=4), 'loss': 'N/A'}                        
        
        if self.verbose>0:
            for name, score in scores.items():
                _score = score['score']
                _loss = score['loss']
                if isinstance(_loss, str):
                    print("%s performance:\n\t Log-loss: %s \n\t Score: %.4f" 
                      % (name, _loss, _score))
                else:
                    print("%s performance:\n\t Log-loss: %.4f \n\t Score: %.4f" 
                      % (name, _loss, _score))
        
        return scores
    

    def crossval_optimize(self, X, y, n_iter=12, scoring='accuracy', cv=10, 
                          n_jobs=1, sample_hyperparams=False, 
                          min_hyperparams=None, get_hyperparams=False, random_state=None):
        """
        
        This method is a wrapper to cross validation using RandomizedSearchCV from scikit-learn, 
        wherein we optimize each defined algorithm
        
        Default behavior is to optimize all parameters available to each algorithm, but it is 
        possible to sample (randomly) a subset of them to optimize (sample_hyperparams=True), 
        or to choose a set of parameters (get_hyperparams=True).
        
        Parameters:
        ------------

        X : 
            Data matrix (n_samples, n_features)
        
        y : 
            Labels/ground truth (n_samples,)
        
        n_iter : integer, default: 1
            Number of iterations to use in RandomizedSearchCV method, i.e. number of independent draws from parameter dictionary
        
        scoring : string or callable, default: 'accuracy'
            Type of scorer to use in optimization
        
        cv : integer or callable, default: 10
            Number of cross validation folds, or callable of correct type
        
        n_jobs : integer, default: 1
            Specify number of parallel processes to use
        
        sample_hyperparams : boolean, default: False
            Randomly sample a subset of algorithm parameters and tune these  
        
        min_hyperparams: optional, integer, default: 2
            When sample_hyperparams=True, choose number of parameters to sample
        
        get_hyperparams: optional, boolean, default: 2
            Instead of random sampling, use previously chosen set of parameters to optimize
        
        random_state: None or integer, default: None
            Used for reproducible results
        
        Ouput:
        -------
        List containing (classifier name, most optimized classifier) tuples       
        
        """
        
        def _random_grid_search(p_index, clf, clf_name, param_dict):

            clf_name = clf_name+"_v%i"%(p_index+1) if p_index>0 else clf_name

            param_dist = param_dict.copy()           
            if len(param_dist) == 0:
                return (clf_name, clf.estimator.fit(X,y))
            
            if sample_hyperparams and not get_hyperparams:
                num_params = np.random.randint(min_hyperparams, len(param_dist))
                param_dist = clf.set_tune_params(param_dist, num_params=num_params, mode='random')

            if get_hyperparams and not sample_hyperparams:
                if len(clf.cv_params_to_tune)>0:                    
                    param_dist = clf.set_tune_params(param_dist, keys=clf.cv_params_to_tune, mode='select')            
            
            n_iter_ = min(n_iter, clf.max_n_iter)        
                      
            random_search = RandomizedSearchCV(clf.estimator, param_distributions=param_dist, 
                                               n_iter=n_iter_, scoring=scoring, cv=cv, n_jobs=n_jobs, 
                                               verbose=0, error_score=0, random_state=random_state)            
            fitted = False
            start_time = time.time()
            try:
                random_search.fit(X, y)
                fitted = True
            except:
                warnings.warn("Failed to search through %s (returning basic 'fit'). \nInfo: %s" 
                              % (clf_name, sys.exc_info()[1]))
                return (clf_name, clf.estimator.fit(X,y))
            else:
                return (clf_name, random_search.best_estimator_)            
            finally:
                if self.verbose>0 and fitted:
                    print("="*50)
                    print("'%s' \tSearch time: %.2f min. \tBest score: %.5f" 
                          % (clf_name, (time.time()-start_time)/60., random_search.best_score_))
                    print("="*50)
                    
        return [_random_grid_search(i, clf, name, param) 
                for name, clf in self.clf.items() 
                for i, param in enumerate(clf.cv_params)]


    def bayes_optimize(self, X, y, n_calls=50, scoring='accuracy', greater_is_better=True, cv=10, n_jobs=1, random_state=None):
        """        
        Use package 'scikit-optimize' >=0.3 in order to do Bayesian optimization instead of random grid search.
        Package URL: https://github.com/scikit-optimize/scikit-optimize/        
        """           

        # cv_params is a list with >= 1 dicts, so we have to iterate over them
        # and add each to the repository of space
        skopt_spaces = skopt_space_mapping([
            (name, params) for name, clf in self.clf.items() for params in clf.cv_params])  
        
        results = []        
        for name, classifier in self.clf.items():       
            
            # Define search space (could get more than a single hit here), so len(space)>=1 (potentially)
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
