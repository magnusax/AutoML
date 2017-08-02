import numpy as np
import sys
import os
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt


class Classifiers(object):
    """
    Class that keeps track of which classifiers we have available. 
    Input:
       method = str: ('random'   : Choose num_clf random classifiers (default),
                       'complete' : Choose all available classifiers,
                       'chosen'   : Send in a list of classifiers and initialize)  
       num_clf = int: Choose number of classifiers to sample (defult: 5. Used when method == 'random', else ignored)
           clf = list (list-like will also work): Send a list of classifiers to initialize (default: []. Used when method == 'chosen')
    Output:
       List of classifiers 
       
    """
    def __init__(self, method='random', num_sample=5, clf=None, verbose=0):
        self.verbose = verbose
        if method not in ['random', 'complete', 'chosen']:
            raise ValueError("'method' should be either \{'random', 'complete', 'chosen'\}.")
        if method == 'chosen' and (len(clf)<2 or type(clf) == NoneType):
            raise ValueError("Specify at least two (2) (name, classifier) tuples in 'clf'")    
        self.method = method
        self.num_sample = num_sample        
        # Depending on method, load clf list and call fit and predict method
        if method == 'chosen':
            if not type(clf[0]) == tuple:
                raise ValueError("Input format: ('clf_name' [str], clf [classifier])")
            self.clf = clf
        else:
            self.clf = self.__build_classifier_list()      
        print("Metod: '%s'" % self.method)
        if self.verbose > 0:    
            print("Initialized classifiers:", end="\n")
            for name, _ in self.clf:
                print("\t%s" % name)          
        
    def __build_classifier_list(self):
        clfs = self.__build_classifier_repository()
        if self.method == 'random':
            print("Sampling %i algorithms from complete inventory." % self.num_sample)
            from numpy import random
            clfs = random.choice(clfs, self.num_sample, replace=False)
        return clfs
        
    def __build_classifier_repository(self):
        """
        Keep a list of all classifiers
        """
        clfs = list()
        
        from importlib import reload
        
        # AdaBoost
        import adaboost; reload(adaboost)
        from adaboost import MetaAdaBoostClassifierAlgorithm 
        ada = MetaAdaBoostClassifierAlgorithm(); clfs.append((ada.name_, ada))
        
        # KNearestNeighbors
        import nearest_neighbors; reload(nearest_neighbors)
        from nearest_neighbors import MetaKNearestNeighborClassifierAlgorithm 
        knn = MetaKNearestNeighborClassifierAlgorithm(); clfs.append((knn.name_, knn)) 
        
        # LogisticRegression
        import logistic_regression; reload(logistic_regression)
        from logistic_regression import MetaLogisticRegressionClassifierAlgorithm 
        lr = MetaLogisticRegressionClassifierAlgorithm(); clfs.append((lr.name_, lr))
        
        return clfs

    def fit_classifiers(self, X, y, n_jobs=1):
        import time
        for name, clf in self.clf:
            try:
                clf.estimator.set_params(**{'n_jobs': n_jobs})
            except: pass
            st = time.time()
            clf.estimator.fit(X, y)
            if self.verbose > 0:
                print("Classifier %s trained (time: %.2f min)" % (name, (time.time()-st)/60.))
        return
    
    def predict_classifiers(self, X):
        return [(name, clf.estimator.predict(X)) for name, clf in self.clf]

    def classifier_performance(self, preds, y_true, metric='accuracy', **kwargs):
        """
        **kwargs gives us the possibility to send extra parameters when computing various metrics
        """
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
            loss = log_loss(y_true, y_pred)
            scores.append((name, val, loss))
        scores.sort(key=itemgetter(1), reverse=True)
        if self.verbose > 0:
            for name, met, loss in scores: print("log_loss: %.4f \t %s: %.4f \t %s"%(loss, metric, met, name))
        return scores
    
    def optimize_classifiers(self, X, y, n_iter=12, scoring='accuracy', cv=10, n_jobs=1, sample_hyperparams=False):
        """
        This method is a wrapper to cross validation, wherein we optimize each available algorithm
        """
        import time
        import warnings
        from sklearn.model_selection import RandomizedSearchCV
        
        optimized = list()        
        for name, algo in self.clf:
            
            estimator, param_dist = algo.estimator, algo.cv_params
            
            # TODO: fix
            if type(param_dist) == list: 
                param_dist = param_dist[0]
            
            if sample_hyperparams:
                from scipy.stats import randint
                param_dist = algo.set_hyparms(param_dist, randint(1, len(param_dist)))

            if self.verbose>0:
                print("Starting grid search for '%s'" % name)
            
            start_time = time.time()
            try:
                search = RandomizedSearchCV(estimator, param_distributions=param_dist, 
                                            n_iter=n_iter, scoring=scoring, 
                                            cv=cv, n_jobs=n_jobs, verbose=self.verbose, 
                                            error_score=0, return_train_score=True)
                search.fit(X, y)
            except:
                warnings.warn("Warning: (Estimator='%s') failed: '%s'" % (name, sys.exc_info()[1]))
            else:
                print("Best mean score: %.4f (%s)" % (search.best_score_, name))
            
            print("Iteration time = %.2f min." % ((time.time()-start_time)/60.)
            optimized.append((name, search.best_estimator_))
        
        # Re-write later: for now just return a list of optimized estimators
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
    sys.exit(-1)     