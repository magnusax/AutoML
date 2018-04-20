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
    def __init__(self, method='random', num_clf=5, clf=[], verbose=0):
        self.verbose = verbose
        if method not in ['random','complete','chosen']:
            raise ValueError("'method' should be either 'random', 'complete', 'chosen'.")
        if method == 'chosen' and len(clf)<2:
            raise ValueError("You need to choose at least two classifiers")    
        self.method = method
        self.num_clf = num_clf        
        # Depending on method, load clf list and call fit and predict method
        if method == 'chosen':
            self.clf = clf
        else:
            self.clf = self.__build_classifier_list()      
        print("Metod: '%s'" % self.method)
        if self.verbose > 0:    
            print("Initialized classifiers:", end="\n")
            for clf in self.clf:
                print("\t%s" % clf.__class__.__name__)          
        
    def __build_classifier_list(self):
        clfs = self.__build_classifier_repository()
        if self.method == 'random':
            print("Sampling %i algorithms from complete inventory." % self.num_clf)
            from numpy import random
            clfs = random.choice(clfs, self.num_clf, replace=False)
        return clfs
        
    def __build_classifier_repository(self):
        """
        Keep a list of all classifiers
        """
        clfs = []
        # Linear models
        from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
        clfs.append([LogisticRegression(), 
                     PassiveAggressiveClassifier()])
        # Trees
        from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
        clfs.append([DecisionTreeClassifier(), ExtraTreeClassifier()])
        # Discriminant analysis
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis 
        clfs.append([LinearDiscriminantAnalysis(), 
                     QuadraticDiscriminantAnalysis()])
        # Gaussian process
        from sklearn.gaussian_process import GaussianProcessClassifier
        clfs.append([GaussianProcessClassifier()])
        # Misc ensembles
        from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, \
                                        ExtraTreesClassifier, GradientBoostingClassifier
        clfs.append([AdaBoostClassifier(n_estimators=100), 
                     BaggingClassifier(n_estimators=100), 
                     RandomForestClassifier(n_estimators=1000)])
        clfs.append([ExtraTreesClassifier(n_estimators=100), 
                     GradientBoostingClassifier(learning_rate=0.01, n_estimators=200)])
        # Support vector machines
        from sklearn.svm import LinearSVC, SVC
        clfs.append([LinearSVC(), 
                     SVC(kernel='linear')])
        # Nearest Neighbors 
        from sklearn.neighbors import KNeighborsClassifier
        clfs.append([KNeighborsClassifier()])
        # Neural network
        #from sklearn.neural_network import BernoulliRBM
        #clfs.append([BernoulliRBM()])
        clfs = [clf for sl in clfs for clf in sl]        
        return clfs

    def fit_classifiers(self, X, y, n_jobs):
        import time
        for clf in self.clf:
            try:
                clf.set_params(**{'n_jobs': n_jobs})
            except: pass
            st = time.time()
            clf.fit(X, y)
            et = time.time()
            if self.verbose > 0:
                print("Classifier %s trained (time: %.2f min)" % (clf.__class__.__name__, (et-st)/60.))
            
    def predict_classifiers(self, X):
        return [(clf.__class__.__name__, clf.predict(X)) for clf in self.clf]

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
        for clf_name, y_pred in preds:
            val = scorer(y_true, y_pred)
            loss = log_loss(y_true, y_pred)
            scores.append((clf_name, val, loss))
        scores.sort(key=itemgetter(1), reverse=True)
        if self.verbose > 0:
            for name, met, loss in scores: print("log_loss: %.4f \t %s: %.4f \t %s"%(loss, metric, met, name))
        return scores
  

class CheckClassifierCorrelation():
    """
    Check correlation between classifier predictions using e.g. Pearson's formula. Initially, a repository
    of classifiers is constructed from either a random, complete, or chosen selection (set by the 'method' 
    parameter). 
    """
    def __init__(self, prediction_type=None):
        options = ['c_binary', 'c_multi', 'reg']
        if prediction_type not in options:
            raise ValueError("Valid options (prediction_type) = %s" % ", ".join(options))            
        self.prediction_type = prediction_type
            
    def compute_correlation_matrix(self, preds):
        corr = np.zeros((len(preds), len(preds)), dtype=np.float32)
        names = []
        # Note that this is a bit "dodgy" for binary variables
        if self.prediction_type == 'c_binary' or prediction_type == 'c_multi':
            from sklearn.metrics import matthews_corrcoef as mcc 
            for i, (nm1, y1) in enumerate(preds):
                names.append(nm1)
                for j, (nm2, y2) in enumerate(preds):
                    corr[i][j] = mcc(y1, y2)
        elif self.prediciton_type == 'reg':
            raise NotImplementedError("This method has not been implemented for regression yet.")           
        return (names, corr)
        
    def plot_correlation_matrix(self, names, corr, rot=0, fig_size=(9,9), font_scale=1.0, file=''):
        f = plt.figure(figsize=fig_size)
        sns.set(font_scale=font_scale)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=names, yticklabels=names)
        plt.xticks(rotation=90-rot)
        plt.tight_layout()
        if len(file)>0:
            print("Saving figure to '%s'" % file)
            try: plt.savefig(file)
            except: print("Could not save figure to %s." % file)
        return f
        
if __name__ == '__main__':
    sys.exit(-1)     