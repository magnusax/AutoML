"""
This module provides an overview and collects all implemented base classifiers 
(mostly from scikit-learn). But this module will eventually contain classifiers 
from several possible sources.

The `core` module uses the below function when importing algorithms into its 
library.

To add a new algorithm simply add a new line describing the file and the class
of the new classifier: ('file_name', 'name_of_meta_class')

"""

def implemented():
    """ 
    Hard coded list of algorithms 
    """
    
    algorithms_ = [
    
        ('adaboost', 'MetaAdaBoostClassifier'), 
        ('nearest_neighbors', 'MetaKNearestNeighborClassifier'), 
        ('logistic_regression', 'MetaLogisticRegressionClassifier'),
        ('sgdescent', 'MetaSGDClassifier'),
        ('naive_bayes', 'MetaGaussianNBayesClassifier'),
        ('naive_bayes', 'MetaMultinomialNBayesClassifier'),
        ('naive_bayes', 'MetaBernoulliNBayesClassifier'),
        ('neural_network', 'MetaNeuralNetworkClassifier'),
        ('random_forest', 'MetaRandomForestClassifier'),
        ('svm', 'MetaSVMClassifier'),
        ('tree', 'MetaDecisionTreeClassifier'),
        ('xgb', 'MetaXGBoostClassifier'),
    ]    
    return algorithms_