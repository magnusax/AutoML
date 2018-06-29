from __future__ import print_function

import os
import sys
import copy
import random
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
            path = os.path.join(save_dir, name)
            os.makedirs(path)        
            ekwargs = kwargs.get(name, {}) 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                models = []
                for idx, estimator in enumerate(tqdm(clfs, desc="{}".format(name), ncols=120)):
                    modelname = "{}_{:04d}train.pkl".format(name,(idx+1))
                    modelfile = os.path.join(path, modelname)
                    try:
                        estimator.set_params(**ekwargs).fit(X, y)
                    except:
                        _, desc, _ = sys.exc_info()
                        raise NotFittedError("Could not fit: {}".format(desc))
                    try:
                        joblib.dump(estimator, modelfile)
                    except:
                        raise Exception("Could not pickle: {}".format(modelfile))
                    models.append(
                        (modelfile, scorer(estimator.predict(X), y)))
            # Return sorted history dict  
            history[name] = sorted(models, key=lambda x: -x[1])
        return history
    
    
    def _unwrap(self):
        """Convenience method. Take object containing fitted algorithms 
        and scores and return a sorted list of classifiers:       
        """
        ranked = []
        for _, v in self.orchestrator.items():
            ranked.append(v)
        return sorted([x for sublist in ranked for x in sublist], key=lambda x: -x[1])   
    
    
    def hillclimb(self, X_val, y_val, n_best=0.1, p=0.3, iterations=10, scoring='accuracy'):
        """
        Perform hillclimbing on the validation data
        
        Parameters:
        ------------
            X_val : validation data, shape (n_samples, n_features)
            
            y_val : validation labels, shape (n_samples,)
            
            n_best : int or float, default: 0.1
                Specify number (int) or fraction (float) of classifiers
                to use as initial ensemble. The best will be chosen.
                
            p : float, default: 0.3
                Fraction of classifiers to select for bootstrap
                
            iterations : int, default: 10
                Number of hillclimb iterations to perform
                
            scoring : str, default: accuracy
                The metric to use when hillclimbing
                
        """
        clfs = self._unwrap()
    
        total = len([c for c,_ in clfs])

        if isinstance(n_best, float):
            grab = int(n_best*total)
        elif isinstance(n_best, int):
            grab = n_best
        else:
            raise TypeError("n_best should be int or float.")

        scorer = get_scorer(scoring)

        # Cache predictions
        pool = []
        val_scores_check = []
        for path, _ in clfs:        
            clf = joblib.load(path)
            y_pred = clf.predict(X_val)
            val_score = scorer(y_pred, y_val)            
            pool.append((clf, y_pred, val_score))
            val_scores_check.append(val_score)
        
        del clfs
        
        print("Max validation score = ", max(val_scores_check))
        ####return val_scores_check
        
        # Sort by score on validation set. Add index.
        pool = [(str(idx), clf, y_pred) for idx, (clf, y_pred, _) 
                in enumerate(sorted(pool, key=lambda x: -x[2]))]    
    
        # Set initial ensemble
        ensemble = pool[:grab]
        print("Algorithms in initial ensemble: {}".format(len(ensemble)))
        
        weights = {str(idx): 0 for idx,_,_ in pool}    
        for t in ensemble:
            weights[t[0]] = 1

        current_score = self.score(ensemble, weights, y_val, scorer)        
        print("Initial {}-score: {:.4f}".format(scoring, current_score))
        
        # Choose a subset of algorithms to use in bootstrap 
        algs = random.choices(pool, k=int(p*len(pool)))
        
        validation_scores = []
        for it in range(1, iterations+1):

            if it==1:
                best_idx = None
                best_score = -float(1e4)

            for alg in algs:
                
                idx_ = alg[0]
                
                # Copy ensemble and add algorithm
                tst_ens = ensemble.copy()
                tst_ens.append(alg)         
                
                # Copy ensemble weights and add +1 in weight
                # to take the new addition into account
                tst_wts = weights.copy()
                tst_wts[idx_] += 1
                
                score_ = self.score(tst_ens, tst_wts, y_val, scorer)

                if score_ > best_score:
                    best_idx = idx_
                    best_score = score_
                    best_alg = [alg]

            if best_score >= current_score: 
                
                # We only add a new algorithm if not previously in ensemble
                # If it is contained in the ensemble, then increasing its
                # weight will suffice in order to take the effect into account
                if not best_alg in ensemble:
                    ensemble += best_alg

                weights[best_idx] += 1
                current_score = best_score

            print("Iteration: {} \tScore: {:.6f}".format(it, current_score))
            validation_scores.append((it, current_score))
            
        # Return the ensemble
        return validation_scores
        
        
    def score(self, ensemble, weights, y, scorer):
        """ Compute weighted majority vote """
        
        wts = np.zeros(len(ensemble))
        preds = np.zeros((len(y), len(ensemble)), dtype=int)
        
        for j, (idx, _, pred) in enumerate(ensemble):
            wts[j] = float(weights[idx])
            preds[:, j] = pred
        
        return self._weighted_vote_score(wts, preds, y, scorer)

    
    def _weighted_vote_score(self, wts, preds, y, scorer):  
        """ Score an ensemble of classifiers using weighted voting """
        
        y_hat = np.zeros(len(y))
        
        for i in range(preds.shape[0]):
            classes = np.unique(preds[i,:])

            # If all classifiers agree
            if len(classes) == 1:
                y_hat[i] = np.int8(classes[0])

            # If disagreement; then apply weighted voting
            elif len(classes)>1:
                conviction = []
                for cls in classes:
                    ind = (preds[i,:] == cls)
                    conviction.append((cls, np.sum(wts[ind])))
                label = np.int8(sorted(conviction, key=lambda x: -x[1])[0][0])
                y_hat[i] = label
                
        return scorer(y_hat, y)