from __future__ import print_function

import os
import sys
import copy
import warnings
import numpy as np

from tqdm import tqdm
from scipy.stats import uniform
from sklearn.externals import joblib
from sklearn.exceptions import NotFittedError

from .metrics import get_scorer
from .sampling import Loguniform
from .core import GazerMetaLearner
from .library import library_config



class GazerMetaEnsembler(object):
    
    
    def __init__(self, learner, data_shape):
        """Ensembler class

        Parameters:
        ------------
            learner : instance of GazerMetaLearner class
                Used to infer which algorithms to include in the 
                ensembling procedure

            data_shape : tuple og length 2
                Should specify input data dimensions according to
                (X.shape[0], X.shape[1]) where `X` is the canonical data-matrix
                with shape (n_samples, n_features)

        """          
        if not isinstance(data_shape, tuple) and len(data_shape)==2:
            raise TypeError("data_shape must be a 2-tuple.")
        self.data_shape = data_shape
        
        if not isinstance(learner, type(GazerMetaLearner())):
            raise TypeError("learner must be a GazerMetaLearner instance.")
        self.learner = learner
    
        # This object is later used to orchestrate 
        # hillclimbing on the validation dataset
        self.orchestrator = {}
        
        # Build ensemble dictionary
        self.ensemble = self._build()

        
    def summary(self):
        """Summarize number of fits per algorithm to expect."""
        total = 0
        for k, v in self.ensemble.items():
            total += len(v)
            print("Algorithm: {} \tFits: {}".format(k, len(v)), end="\n")
        print("\nTotal number of fits = {}".format(total))
 

    def _build(self):
        """Build ensemble from base learners contained
        in the `learner` object.

        Parameters:
        -----------
            learner : object
                instance of GazerMetaLearner class

            X : matrix-like
                input 2D matrix of shape (n_samples, n_features)
                We need some meta data to be able to make sensible choices
                on parameters.

        Returns:
        ---------
        Dictionary : (algorithm[str]: classifiers[list])
        Dictionary containing name keys with corresponding values being
        a list of possible learners with varying settings of hyperparameters.
        
        """
        lib = library_config(self.learner.names, *self.data_shape)
        return {name: self._gen_templates(name, grid) for name, grid in lib}

    
    def _gen_templates(self, name, params):    
        """Here we generate estimators to later fit."""
        learner = GazerMetaLearner(
            method='selected', 
            estimators=[name])
        clf = learner._get_algorithm(name)
        del learner

        estimators = []
        for param in params:
            par = param['param']
            premise = param['config']
            values = self._gen_grid(param['grid'])        
            for value in values:
                estimator = copy.deepcopy(clf.estimator)
                pars = {par:value}
                pars.update(premise)
                try:
                    estimator.set_params(**pars)
                except:
                    warnings.warn("Failed to set: %s" % par)
                    continue
                estimators.append(estimator)
                del estimator                    
        return estimators

    
    def _gen_grid(self, grid):
        """ Generate a config grid. """
        method = grid.get('method', None)
        assert method in ('take', 'sample') 
        
        if method=='sample':
            category = grid.get('category', None)        
            assert category in ('discrete', 'continuous')

        if method=='take':
            return grid['values']

        elif method=='sample':                
            low, high, points, prior = (
                grid['low'], grid['high'], grid['numval'], grid['prior'])

            if category=='discrete':
                raise ValueError('Discrete sampling not implemented.')                   

            elif category=='continuous':                                  
                if prior=='loguniform':
                    return Loguniform(low=low, high=high, size=points).range()
                else:
                    return np.linspace(low, high, points, endpoint=True)
    
    
    def fit(self, X, y, save_dir, scoring='accuracy', **kwargs):
        """
        Fit an ensemble of algorithms.
        
        - Models are pickled under the `save_dir`
          folder (each algorithm will have a separate folder in the tree)
        - If directory does not exist, we attempt to create it. 

        Parameters:
        ------------
            X : matrix-like
                2D matrix of shape (n_samples, n_columns)

            y : array-like
                Label vector of shape (n_samples,)
            
            save_dir : str
                A valid folder wherein pickled algorithms will be saved
                
            scoring : str or callable
                Used when obtaining training data score
                Fetches get_scorer() from local metrics.py module

            **kwargs: 
                Variables related to scikit-learn estimator.
                Used to alter estimator parameters if needed (such as e.g. n_jobs)

                Example: 
                    - Use e.g. {'random_forest': {'n_jobs': 4}} to use parallel
                      processing when fitting the random forest algorithm. 
                    - Note that the key needs to match the a key in the `ensemble` dict
                      to take effect. 
                    - The change takes place through estimator.set_params()

        Returns:
        ---------
        Dictionary with paths to fitted and pickled learners, as well as scores on 
        training data. Note that joblib is used to pickle the data.

        """         
        if (save_dir is None or len(save_dir)==0):
            raise Exception(
                "Please specify a valid directory.")

        if os.path.exists(save_dir):
            raise Exception(
                "{} already exits. Please choose a different directory."
                .format(save_dir))
        try:
            os.makedirs(save_dir)
        except:
            raise Exception(
                "Could not create folder {}.".format(save_dir))
        
        # Get the scorer method
        scorer = get_scorer(scoring)

        self.orchestrator = self._fit(X=X, y=y, 
                                      save_dir=save_dir, 
                                      scorer=scorer, 
                                      **kwargs)
        return self
    
        
    def _fit(self, X, y, save_dir, scorer, **kwargs):
        """ Implement the fitting """
        
        # Keep track of model and scores in `models`
        # All relevant data is available in `history`
        history = {}

        for name, clfs in self.ensemble.items():        
            os.makedirs(os.path.join(save_dir, name))        
            ekwargs = kwargs.get(name, {}) 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                models = []
                for idx, estimator in enumerate(tqdm(clfs, desc="{}".format(name), ncols=120)):
                    modelname = "{}_{:04d}train.pkl".format(name,(idx+1))
                    try:
                        estimator.set_params(**ekwargs).fit(X, y)
                    except:
                        _, desc, _ = sys.exc_info()
                        raise NotFittedError("Could not fit: {}".format(desc))
                    try:
                        joblib.dump(estimator, os.path.join(save_dir, name, modelname))
                    except:
                        raise Exception("Could not pickle: {}".format(modelname))
                    models.append(
                        (modelname, scorer(estimator.predict(X), y)))
            # Return sorted history dict  
            history[name] = sorted(models, key=lambda x: -x[1])
        return history
    
    
    def hillclimb(self, X_val, y_val, percent_best=10):
        """Perform hillclimbing on the validation data."""
    
    