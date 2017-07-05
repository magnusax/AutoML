import os
import sys; sys.path.append(os.getcwd()+'/src')
import numpy as np
import pandas as pd

from importlib import reload
import utils; reload(utils)
import class_handler_v2 as algorithmHandler; reload(algorithmHandler)
import visualize; reload(visualize)
from visualize import Visualizer as viz;

from sklearn.metrics import accuracy_score, log_loss

# Right now, only classification is supported
def main():
    
    # Generate data to play with (this is a very simple classification dataset, and 
    # thus only used for demonstration purposes only)
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

    # Alternatively: optimize each classifier using a cross-validation scheme
    print("Cross validation with randomly drawn parameter realizations:")
    clfs.verbose = 0
    optims = clfs.optimize_classifiers(X_train, y_train, n_iter=4, n_jobs=-1)
    for name, optimized_clf in optims: # Classifiers are already fitted (re-training redundant)
        print(name, 
              'train score:', accuracy_score(optimized_clf.predict(X_train), y_train), 
              'test score:',  accuracy_score(optimized_clf.predict(X_test), y_test)
        )
        print()
    
    # This method expects a list of 2-tuples and can either return a matplotlib.pyplot.figure object, or 
    # as in this case, save a figure to the current working directory ('fig.png'). The plot will be in the form 
    # of a horisontal bar plot sorted according to clasifier performance.
    # See the doc string for further details
    fig = viz().show_performance([tuple([name, score]) for name, score, _ in train_scores], 
                                 fig_size=(12,6), file=os.getcwd()+"/fig.png")
    
    return

if __name__ == '__main__':
    sys.exit(main())