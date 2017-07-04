import os
import sys; sys.path.append(os.getcwd()+'/src')
import numpy as np
import pandas as pd

from importlib import reload
import utils; reload(utils)
import class_handler_v2 as algorithmHandler; reload(algorithmHandler)


# Right now, only classification is supported
def main():
    
    # Generate data to play with
    from sklearn.datasets import load_breast_cancer
    x, y = load_breast_cancer(return_X_y=True)
    print("loaded toy data set")

    # Lets split into train and test
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Import classifiers through 'algorithmHandler' (verbose = 1 to receive info on what's going on)
    clfs = algorithmHandler.Classifiers(verbose=1, method='complete')
    
    # Fit on train data
    clfs.fit_classifiers(X_train, y_train, n_jobs=2)
    
    # Get predictions
    y_train_pred = clfs.predict_classifiers(X_train)
    
    # Evaulate performance
    print("Training scores:")
    train_scores = clfs.classifier_performance(y_train_pred, y_train, metric='accuracy')
    
    # Evaluate overfitting
    print("Test scores:")
    test_scores = clfs.classifier_performance(clfs.predict_classifiers(X_test), y_test, metric='accuracy')

    return

if __name__ == '__main__':
    sys.exit(main())