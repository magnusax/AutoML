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
    
    # Import classifiers through 'algorithmHandler'
    clfs = algorithmHandler.Classifiers(verbose=1, method='complete')
    

if __name__ == '__main__':
    sys.exit(main())