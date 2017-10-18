"""
This module provides an overview, and collects all implemented base classifiers 
(mostly from scikit-learn). But this module will contain classifiers from all
possible sources, eventually.

"""

def all_algorithms():
    
    # Hard coded list of algorithms
    implemented = [
        ('adaboost', 'MetaAdaBoostClassifier'), 
        ('nearest_neighbors', 'MetaKNearestNeighborClassifier'), 
        ('logistic_regression', 'MetaLogisticRegressionClassifier'),
        ('stochastic_gradient_descent', 'MetaSGDClassifier'),
        ('naive_bayes', 'MetaGaussianNBayesClassifier'),
        ('naive_bayes', 'MetaMultinomialNBayesClassifier'),
        ('naive_bayes', 'MetaBernoulliNBayesClassifier'),
        ('random_forest', 'MetaRandomForestClassifier'),
        ('svm', 'MetaSVMClassifier'),
        ('xgboost', 'MetaXGBoostClassifier'),
    ]    
    return implemented