import os
import sys
sys.path.append("../src/")

import numpy as np

# ML-meta-wrapper project
import utils
import ml_meta_wrapper as metawrapper
from visualize import Visualizer as viz

from sklearn.metrics import accuracy_score, log_loss

# Right now, only classification is supported
def main():
    
    # Generate data to play with (this is a very simple classification dataset, and 
    # thus only used for demonstration purposes only)
    from sklearn.datasets import load_digits
    x, y = load_digits(return_X_y=True)
    print("loaded toy data set")

    # Lets split into train and test
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Import classifiers through 'metawrapper' (verbose = 1 to receive info on what's going on)
    clfs = metawrapper.MetaWrapperClassifier(verbose=1, method='complete')
    
    # Fit on train data
    clfs.fit_classifiers(X_train, y_train, n_jobs=1)
    
    # Get predictions
    y_train_pred = clfs.predict_classifiers(X_train)
    
    # How to check multi-class or not
    multiclass = True if len(np.unique(y_train_pred[0][1]))>2 else False
    print('multiclass classification? Answer: ', multiclass)
    
    # Get probabilities (handy when attempting to do ensembling later on)
    train_probas = clfs.predict_classifiers(X_train)
    
    # Get log loss when doing probabilities (e.g. metric='accuracy' will not work)
    train_scores = clfs.classifier_performance(train_probas, y_train, metric='accuracy', multiclass=True)    
        
    # Evaulate performance in terms of accuracy this time
    print("Training scores:")
    train_scores = clfs.classifier_performance(y_train_pred, y_train, metric='accuracy', multiclass=multiclass)
        
    # Evaluate overfitting
    print("Test scores:")
    test_scores = clfs.classifier_performance(clfs.predict_classifiers(X_test), y_test, metric='accuracy', multiclass=multiclass)
        
    # Alternatively: optimize each classifier using a cross-validation scheme
    print("Cross validation with randomly drawn parameter realizations:")
    clfs.verbose = 1
    optims = clfs.optimize_classifiers(X_train, y_train, n_iter=10, n_jobs=-1, random_state=1)
    for name, clf in optims: # Classifiers are already fitted (re-training redundant)
        print(name, 
              'train score:', accuracy_score(clf.predict(X_train), y_train), 
              'test score:',  accuracy_score(clf.predict(X_test), y_test)
        )
    
    # This method expects a list of 2-tuples and can either return a matplotlib.pyplot.figure object, or 
    # as in this case, save a figure to the current working directory ('fig.png'). The plot will be in the form 
    # of a horisontal bar plot sorted according to clasifier performance.
    # See the doc string for further details
    vs = viz()
    fig = vs.show_performance([tuple([name, score]) for name, score, _ in train_scores], 
                              fig_size=(12,6), 
                              file=os.getcwd()+"/fig.png")
    
    return

if __name__ == '__main__':
    import sys
    sys.exit(main())