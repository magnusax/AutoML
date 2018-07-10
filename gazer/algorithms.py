"""
This module provides an overview and collects all implemented base classifiers 
(mostly from scikit-learn). But this module will eventually contain classifiers 
from several possible sources.

The core module uses the below function when importing algorithms into its 
library.

To add a new algorithm simply add a new line describing the file and the class
of the new classifier: ('file_name', 'name_of_meta_class')

"""


def implemented(add_network=True, add_xgboost=True):
    """ 
    Hard coded list of algorithms. Certain algorithms (such as e.g. keras) 
    are only sent in if flag allows it. 
    
    """    
    algorithms = [    
        # Scikit-learn algorithms
        ('adaboost', 'MetaAdaBoostClassifier'), 
        ('logistic_regression', 'MetaLogisticRegressionClassifier'),
        ('nearest_neighbors', 'MetaKNearestNeighborClassifier'), 
        ('naive_bayes', 'MetaBernoulliNBayesClassifier'),
        ('naive_bayes', 'MetaGaussianNBayesClassifier'),
        ('naive_bayes', 'MetaMultinomialNBayesClassifier'),        
        ('random_forest', 'MetaRandomForestClassifier'),
        ('sgdescent', 'MetaSGDClassifier'),
        ('svm', 'MetaSVMClassifier'),
        ('tree', 'MetaDecisionTreeClassifier')]
    
    # Keras
    if add_network:
        algorithms.append(('neural_network', 'MetaNeuralNetworkClassifier'))
    # Xgboost
    if add_xgboost:
        algorithms.append(('xgb', 'MetaXGBoostClassifier'))        
    return algorithms

