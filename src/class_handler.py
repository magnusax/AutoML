import seaborn as sns; sns.set()
from matplotlib import pyplot as plt


class MetaWrapperClassifier():
    """
    Docstring:
    
    Class that keeps track of available classifiers and implements functionality around them
    
    Input:
    ------------------
       method = str: ('random' : Choose num_sample random classifiers (default),
                      'complete' : Choose all available classifiers,
                      'chosen' : Send in a list (iterable will work) of classifier names and initialize)
                      
       num_sample: (int: 3) choose number of classifiers to sample. Used when method == 'random'
       estimators: (None or list/list-like: None): Send a list of classifiers to initialize. Used when method == 'chosen'
       base_estimator: (None or sklearn-type estimator: None) decide which classifier to use as a weak learner for ensemble methods
       verbose: (int, 0) if > 0 then be 'chatty'
       
    Output:
    ------------------
    List of classifiers 
       
    """
    def __init__(self, method='random', num_sample=3, estimators=None, base_estimator=None, exclude=None, verbose=0):                    
        
        if method not in ['random', 'complete', 'chosen']:
            raise ValueError("'method' should be either ['random', 'complete', 'chosen'].")        
        
        if method == 'chosen' and (estimators is None or len(estimators) == 0):
            suggestions = [name for name, _ in self.__build_classifier_repository()]
            raise ValueError("Specify name of at least one (1) classifier in list (iterable) 'clf'. \nValid options: %s" % suggestions)        
        
        self.verbose = verbose   
        self.method = method
        self.num_sample = num_sample
        self.base_estimator = base_estimator
        self.exclude = exclude
        
        if method == 'chosen':
            self.clf = [(name, c) for name, c in self.__build_classifier_repository() if name in estimators]
        else:
            self.clf = self.__build_classifier_list()           
        
        if self.verbose > 0:    
            print("Initialized classifiers:")
            for name, _ in self.clf:
                print("\t%s" % name)          
        
    def __build_classifier_list(self):
        clfs = self.__build_classifier_repository()
        
        if self.method == 'random':
            print("Sampling %i algorithms..." % self.num_sample)
            from numpy import random
            clfs = [clfs[i] for i in random.choice(len(clfs), self.num_sample, replace=False)]
        
        return clfs
        
    def __build_classifier_repository(self):
        """
        Return list of (classifier_name, classifier) tuples
        
        """        
        # This list of tuples contains all importable algorithms
        # TODO: move this to a separate module containing algorithms only
        algorithms = [
            ('adaboost', 'MetaAdaBoostClassifierAlgorithm'), 
            ('nearest_neighbors', 'MetaKNearestNeighborClassifierAlgorithm'), 
            ('logistic_regression', 'MetaLogisticRegressionClassifierAlgorithm'),
            ('naive_bayes', 'MetaGaussianNBayesClassifierAlgorithm'),
            ('naive_bayes', 'MetaMultinomialNBayesClassifierAlgorithm'),
            ('naive_bayes', 'MetaBernoulliNBayesClassifierAlgorithm'),
        ]
        if self.exclude is None:
            return [self.add_algorithm(m, c) for m, c in algorithms]
        else:
            return [self.add_algorithm(m, c) for m, c in algorithms if not m in self.exclude]

    def add_algorithm(self, module_name, algorithm_name):
        from base import EnsembleBaseClassifier      
        from importlib import reload, import_module                       
        try:
            module = import_module(module_name)
        except:
            raise ValueError("Could not import module '%s'" % module_name)                
        algorithm = getattr(module, algorithm_name)

        if issubclass(algorithm, EnsembleBaseClassifier):
            instance = algorithm(base_estimator=self.base_estimator)
        else:
            instance = algorithm()
        return instance.name, instance
    
    def fit_classifiers(self, X, y, n_jobs=1):
        import time
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
    
    def predict_classifiers(self, X):
        return [(name, clf.estimator.predict(X)) for name, clf in self.clf]
    
    def predict_proba_classifiers(self, X):
        probas = []
        for name, clf in self.clf:
            # Check if classifier implements 'predict_proba'
            d = clf.get_info()
            if d['predict_probas']:
                probas.append((name, clf.estimator.predict_proba(X)))
        return probas
    
    def classifier_performance(self, preds, y_true, metric='accuracy', multiclass=False, **kwargs):
        """
        **kwargs gives us the possibility to send extra parameters when computing various metrics
        
        """
        import numpy as np
        from operator import itemgetter
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, log_loss
        
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
                from sklearn.preprocessing import LabelBinarizer
                lb = LabelBinarizer()
                yhat = lb.fit_transform(y_pred)
            else:
                yhat = np.array(y_pred)
            loss = log_loss(y_true, yhat)            
            scores.append((name, val, loss))
                        
        scores.sort(key=itemgetter(1), reverse=True)
        
        if self.verbose > 0:
            for name, met, loss in scores: print("log_loss: %.4f \t %s: %.4f \t %s" % (loss, metric, met, name))
            ##for name, met in scores: print("%s: %.4f \t %s" % (metric, met, name))
        return scores
    
    def optimize_classifiers(self, X, y, n_iter=12, scoring='accuracy', cv=10, n_jobs=1, 
                             sample_hyperparams=False, min_hyperparams=2, get_hyperparams=False, random_state=None):
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
        import sys
        import time
        import warnings
        from sklearn.model_selection import RandomizedSearchCV
        
        optimized = []
        
        for name, classifier in self.clf:            
            estimator, param_dist = classifier.estimator, classifier.cv_params            
            
            if sample_hyperparams and not get_hyperparams:
                # Here we (by default, but other behaviors are also possible) sample randomly
                # [1, number hyperparams] to optimize in the cross-validation loop
                num_params = np.random.randint(min_hyperparams, len(param_dist))
                param_dist = classifier.sample_hyperparams(classifier.cv_params, num_params=num_params, mode='random', keys=[])
            
            if get_hyperparams and not sample_hyperparams:
                if len(classifier.cv_params_to_tune) > 0:
                    print("(%s): overriding current parameter dictionary using 'cv_params_to_tune'" % name)
                    param_dist = classifier.sample_hyperparams(classifier.cv_params, num_params=-1, mode='select', keys=classifier.cv_params_to_tune)            
            
            n_iter_ = min(n_iter, classifier.max_n_iter)
            
            if self.verbose>0:
                print("Starting grid search for '%s'" % name)
                print("Setting 'n_iter' to:",n_iter_)
            
            search = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=n_iter_, 
                                        scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=self.verbose, 
                                        error_score=0, return_train_score=True, random_state=random_state)
            start_time = time.time()
            try:
                search.fit(X, y)
            except:
                warnings.warn("Estimator '%s' failed (likey: 'n_iter' too high). \nAdding un-optimized version..." % name)
                optimized.append((name, estimator.fit(X,y)))
            else:
                optimized.append((name, search.best_estimator_))
                if isinstance(scoring, str):
                    print("(Scoring='%s')\tBest mean score: %.4f (%s)" % (scoring, search.best_score_, name))
                else:
                    print("Best mean score: %.4f (%s)" % (search.best_score_, name))  
                    
            print("Iteration time = %.2f min." % ((time.time()-start_time)/60.))            
            
                    
        
        # Rewrite later: for now just return a list of optimized estimators
        # Perhaps we should return the grids themselves
        return optimized


class CheckClassifierCorrelation():
    """
    Check correlation between classifier predictions using e.g. Pearson's formula. Initially, a repository
    of classifiers is constructed from either a random, complete, or chosen selection (set by the 'method' 
    parameter). 
    """
    def __init__(self, prediction_type=None):
        options = ['binaryclass', 'multiclass', 'regression']
        if prediction_type not in options:
            raise ValueError("Valid options for prediction_type are: %s" % ", ".join(options))            
        self.prediction_type = prediction_type
            
    def compute_correlation_matrix(self, preds):
        corr = np.zeros((len(preds), len(preds)), dtype=np.float32)
        names = []
        # Note that this is a bit "dodgy" for binary variables
        if self.prediction_type == 'binaryclass' or prediction_type == 'multiclass':
            from sklearn.metrics import matthews_corrcoef as mcc 
            for i, (nm1, y1) in enumerate(preds):
                names.append(nm1)
                for j, (nm2, y2) in enumerate(preds):
                    corr[i][j] = mcc(y1, y2)
        elif self.prediciton_type == 'regression':
            raise NotImplementedError("This method has not been implemented for regression yet.")           
        return names, corr
        
    def plot_correlation_matrix(self, names, corr, rot=0, fig_size=(9,9), font_scale=1.0, file=''):
        f = plt.figure(figsize=fig_size)
        sns.set(font_scale=font_scale)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=names, yticklabels=names)
        plt.xticks(rotation=90-rot)
        plt.tight_layout()
        if len(file)>0:
            print("Saving figure to '%s'" % file)
            try: 
                plt.savefig(file)
            except: 
                print("Could not save figure to %s." % file)
        return f
        
if __name__ == '__main__':
    import sys
    sys.exit(-1)     