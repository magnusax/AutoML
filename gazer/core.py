import numpy as np
import warnings
import time
import sys

from importlib import import_module  
from operator import itemgetter

from skopt import gp_minimize
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

from .base import EnsembleBaseClassifier, BaseClassifier
from .algorithms import implemented        
from .utils import skopt_space_mapping
from .metrics import get_scorer


class GazerMetaLearner():
    """
    Class that keeps track of available classifiers 
    and implements functionality around them    
    
    ::: Importing :::
    from gazer.core import GazerMetaLearner
    learner = GazerMetaLearner()     
    
    
    Input:
    --------
    
        method : str,  default, 'random'
            random; choose num_sample random classifiers
            all;  choose all available classifiers,
            selected; send in an iterable of classifier names (arg: `estimators`)
                      
        num_sample : integer, default: 3
            choose number of classifiers to sample. Used when method='random'
       
        estimators : None or list-like, default: None
            Send a list of classifiers to initialize. Used when method='selected'

        base_estimator: None or sklearn estimator, default: None 
            Decide which classifier to use as a weak learner when using ensembles
       
        verbose : integer, default: 0 
            If verbose>0 then output feedback messages       
    
    Returns:
    ---------

        GazerMetaLearner : class instance 
            See GazerMetaLearner().clf for a dictionary of initialized learning algorithms  
            consisting of algorithm-name keys with MetaClassifier objects as values
    
    """
    def __init__(self, 
                 method='random', 
                 num_sample=3, 
                 estimators=None, 
                 base_estimator=None, 
                 exclude=None, 
                 verbose=0, 
                 random_state=None):                    
        
        options = ('random', 'all', 'selected')
        
        if method not in options:
            raise ValueError("`method` should be one of (%s)" % ",".join(options))                 
        
        self.verbose = verbose   
        self.method = method
        self.num_sample = num_sample

        # For reproducibility
        if random_state is not None:
            self.random_state = np.random.RandomState(random_state)    
        else:
            self.random_state = None
        
        # This estimator is used as 'base_estimator' in ensembling algorithms
        # It will override defaults in the individual init scripts
        self.base_estimator = base_estimator
        
        if method=='selected' and (estimators is None or len(estimators)==0):
            raise Exception(
                "Specify name of at least one algorithm in `estimators`."\
                " Valid options are:\n%s" % ", ".join(self._build_classifier_dict().keys()))
                
        if exclude is None:
            self.exclude = []
        else:
            self.exclude = exclude
        
        if method=='selected':
            self.clf = {n:c for n,c in self._build_classifier_repository() if n in estimators}
        else:
            self.clf = self._build_classifier_dict()    
        self.clf = {n:c for n,c in self.clf.items() if not (n in self.exclude)}               
        
        if self.verbose>0: print("Initialized: %s" % ", ".join(self.names))         
    
    @property
    def names(self):
        return list(self.clf.keys())
    
    @property
    def num_estimators(self):
        return len(self.names)

    def _build_classifier_dict(self):
        # Build repo
        clfs = self._build_classifier_repository()        
        
        # Do we need to downsample?
        if self.method=='random':
            if self.verbose>0: print("Sampling %i algorithms" % self.num_sample)
            clfs = [clfs[i] for i in np.random.choice(len(clfs), self.num_sample, replace=False)]        
        return {n:c for n,c in clfs}
    
    def _build_classifier_repository(self):
        # Read implemented algorithms (algorithms.py)
        to_add = implemented()
        
        # Iteratively loop and check status, then add
        algorithms = []       
        for m, c in to_add:
            name, algo = self._add_algorithm(m, c)
            if name is not None and algo is not None:
                algorithms.append((name, algo))  
        return algorithms

    def _add_algorithm(self, module_name, algorithm_name):                     
        """ Import classifier algorithms 
        """        
        module_path = ".".join((__package__, "classifiers", module_name))        
        try:           
            module = import_module(module_path, package=True)
        except ImportError: 
            warnings.warn("Could not import module %s: \n%s" 
                          % (module_name, sys.exc_info()[1]))
            return (None, None)
        
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
                clf.fit(X, y)
            else:
                if n_jobs>1 and hasattr(clf.estimator, 'n_jobs'):
                    clf.estimator.set_params(**{'n_jobs': n_jobs})
                clf.estimator.fit(X, y)
            if self.verbose > 0:
                print("%s: training time=%.2f (min)" 
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
            raise ValueError("%s not found." % name)
        return self.clf.get(name, None)


    def predict(self, X):
        return [(name, clf.estimator.predict(X)) for name, clf in self.clf.items()]
    

    def predict_proba(self, X):
        probas = []
        for name, clf in self.clf.items():
            info = clf.get_info()
            if info['predict_probas']:
                probas.append((name, clf.estimator.predict_proba(X)))
        return probas
    

    def evaluate(self, preds, y_true, metric=None, multiclass=None, **kwargs):
        """
        Evalute predictions (preds) against ground truth (y_true) using scikit-learn metrics
 
        Input:
        -------

        preds :       
            a list of (name (str), array-of-predictions (iterable)) tuples 
            (default output from 'predict' method)
            E.g.: [('my_classifier1', np.array(...)), ('my_classifier2', np.array(...)), ... ]  

        y_true :      
            an array (iterable) of ground truth labels

        metric :      
            a label (str) that indicates type of metric to use 
            (accuracy, auc, f1, recall, precision, log_loss). 
            Must be set.

        multiclass :  
            True or False. Indicate if this is a multiclass problem or not. 
            Must be set.
        
        Output:
        --------

            A list of 3-tuples (one for each algorithm) in the following form: 
            (classifier: name, classifier: metric performance, classifier: log loss)
        """
        
        if metric is None:
            raise ValueError("Please specify a metric")
        
        if multiclass is None:
            raise ValueError("Please specify if multiclass or not (bool)")
        
        # Desired scorer
        scorer = get_scorer(metric)
        
        # We need the log loss as well
        log_loss = get_scorer('log_loss')
        
        # Keep track of score for every algorithm
        scores = {}
        for name, y_pred in preds:
            
            # Handle keras model output
            if name == 'neuralnet':
                y_pred = np.array(
                    [max(enumerate(probs), key=itemgetter(1))[0] 
                     for probs in y_pred])
                
            score_ = scorer(y_true, y_pred)
            
            if multiclass and len(np.array(y_pred).shape) == 1:
                lb = LabelBinarizer()
                y_hat = lb.fit_transform(y_pred)
            else:
                y_hat = np.array(y_pred)
            lg_loss_ = log_loss(y_true, y_hat)            
            scores[name] = {metric+" score": score_, 'log loss score': lg_loss_}                        
        
        if self.verbose>0:
            for name, dscore in scores.items():
                lg_loss = dscore['log loss score']
                score = dscore[metric+" score"]
                print("%s performance:\n\t Log-loss: %.4f \n\t Score: %.4f" 
                      % (name, lg_loss, score))
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
