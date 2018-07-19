from __future__ import print_function

import os, sys, time, copy, glob, random, warnings

import numpy as np
from scipy.stats import uniform
from sklearn.externals import joblib 
from sklearn.exceptions import NotFittedError
from tqdm import tqdm_notebook

from .metrics import get_scorer
from .sampling import Loguniform
from .core import GazerMetaLearner
from .library import library_config
from .optimization import param_search


def single_fit(estimator, scorer, X, y, path, i, **kwargs):
    modelfile = os.path.join(path, "model_{:04d}train.pkl".format(i))
    try:
        estimator.set_params(**kwargs).fit(X, y)
        joblib.dump(estimator, modelfile)
        return (modelfile, scorer(estimator.predict(X), y))
    except:
        fail = (None, float("-Inf"))
        _, desc, _ = sys.exc_info()
        warnings.warn("Could not fit and save: {}".format(desc))
        return fail
    
    
def _sklearn_score_fitted(path, X, y, scorer):
    try:
        model = joblib.load(path)
        yhat = model.predict(X) 
        score = scorer(yhat, y)        
        return (path, yhat, score)
    except:
        return None
    
    
def _keras_score_fitted(path, X, y, scorer):
    """ Load previously fitted keras model. Then predict
    on `X` and return score based on comparison to `y`.
    """
    from keras.models import load_model
    import tensorflow as tf
    config = tf.ConfigProto()
    graph  = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph, config=config)
        with sess.as_default():
            try:
                model = load_model(path)
                yhat = model.predict_classes(X)
                score = scorer(yhat, y)
                return (path, yhat, score)
            except:
                return None
            
            
class GazerMetaEnsembler(object):
    """
    Ensembler class.

    Parameters:
    ------------
        learner : instance of GazerMetaLearner class
            Used to infer which algorithms to include in the 
            ensembling procedure

        data_shape : tuple, length 2
            Should specify input data dimensions according to
            (X.shape[0], X.shape[1]) where `X` is the canonical data-matrix
            with shape (n_samples, n_features)
    
    """
    def __init__(self, learner, data_shape):

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
        """ Summarize expected number of fits (individual and total). """
        total = 0
        for k, v in self.ensemble.items():
            total += len(v)
            print("Algorithm: {} \tFits: {}".format(k, len(v)))
        print("Expected total number of fits = {}".format(total))
 

    def _build(self):
        """
        Build ensemble from base learners contained in the `learner` object.

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
        build = {}
        for name, grid in lib:
            
            # Check metadata: can we generate templates or not?
            info = self.learner.clf[name].get_info()
            external_api = info.get('external', False)    
            
            # Here we take care of algorithms from external packages
            # which we cannot follow the scikit-learn api 
            if external_api:                
                if name == 'neuralnet':
                    build[name] = grid            
            else:
                build[name] = self._gen_templates(name, grid)            
        return build
    
    
    def _gen_templates(self, name, params):    
        """ Here we generate estimators to later fit """
        
        clf = self.learner._get_algorithm(name)
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
                    warnings.warn("Failed to set {}".format(par))
                    continue
                estimators.append(estimator)
                del estimator                    
        return estimators

    
    def _gen_grid(self, grid):
        """ Generate a config grid. """
        method = grid.get('method', None)
        assert method in ('take', 'sample') 

        if method=='take':
            return grid['values']

        elif method=='sample':  
            category = grid.get('category', None)        
            assert category in ('discrete', 'continuous')
            
            low, high, points, prior = (
                grid['low'], grid['high'], grid['numval'], grid['prior'])

            if category=='discrete':
                raise NotImplementedError('Discrete sampling not implemented yet.')                   

            elif category=='continuous':                                  
                if prior=='loguniform':
                    return Loguniform(low=low, high=high, size=points).range()
                else:
                    return np.linspace(low, high, points, endpoint=True)
    
    
    def fit(self, X, y, save_dir, scoring='accuracy', n_jobs=1, verbose=0, **kwargs):
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
            
            n_jobs : integer, default: 1
                If n_jobs > 1 we use parallel processing to fit and save
                scikit-learn models. 
                Note: it is not used when training the neural network.
                
            verbose : integer, default: 0
                Control verbosity during training process.
                
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
            raise Exception("Please specify a valid directory.")

        if os.path.exists(save_dir):
            warnings.warn("Warning: overwriting existing folder {}."
                          .format(save_dir))
        else:
            try:
                os.makedirs(save_dir)
            except:
                raise Exception("Could not create folder {}."
                                .format(save_dir))

        self.orchestrator = self._fit(X=X, y=y, save_dir=save_dir, 
                                      scorer=get_scorer(scoring), 
                                      n_jobs=n_jobs, verbose=verbose, 
                                      **kwargs)
        return
    
        
    def _fit(self, X, y, save_dir, scorer, n_jobs, verbose, **kwargs):
        """ Implement the fitting """
        
        # Keep track of model and scores
        # All relevant data is available in `history`
        history = {}
               
        names = list(self.ensemble.keys())
        for name in names:
            os.makedirs(os.path.join(save_dir, name))        
        
        # Ttrain the network first, if present
        _name = 'neuralnet'
        if _name in names:
            
            args, param_grid = self.ensemble.pop(_name)            
            _modelfiles = [os.path.join(save_dir, _name, file) 
                           for file in args['modelfiles']]
            _n_iter = args['n_iter']            
            _data={'train': (X, y), 'val': None}  
            # Do parameter search (random). Note: no validation data, so sort by train scores
            _, df = param_search(self.learner, param_grid, data=_data, type_of_search='random', 
                 n_iter=_n_iter, name=_name, modelfiles=_modelfiles)
            
            history[_name] = zip(_modelfiles, df.head(len(_modelfiles))['train_score'].values)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, estimators in self.ensemble.items():            
                
                path = os.path.join(save_dir, name)
                kwargs_ = kwargs.get(name, {})                           
                
                if n_jobs != 1:
                    models = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, backend="threading")(
                        joblib.delayed(single_fit)(estimator, scorer, X, y, path, i, **kwargs_) 
                        for i, estimator in enumerate(estimators, start=1))
                else:
                    models = []
                    for i, estimator in enumerate(tqdm_notebook(estimators, desc="{}".format(name), ncols=120)):
                        this_modelfile, this_score = single_fit(estimator, scorer, X, y, path, i, **kwargs_)
                        models.append((this_modelfile, this_score))
                    
                history[name] = sorted(list(models), key=lambda x: -x[1]) 
            
        # Before returning: purge any failed fits
        for name, models in history.items():
            valid = [(file, score) for file, score in models if file is not None]
            history[name] = valid
            
        return history
    
    
    def _add_networks(self, clf, X, y, path):
        """Add to ensemble repository a set of keras neural network
        models
        
        """
        y_ = clf.y_check(y, verbose=0)
        
        # Prepare for ensembling
        os.makedirs(path)
        clf.set_param('chkpnt_dir', path) 
        
        # When 'ensemble' is set to True, 
        # checkpointing to the 'path' folder is enabled
        clf.set_param('ensemble', True) 
        
        # Train
        print("Training neural net..")
                
        start = time.time()
        clf.fit(X, y_, verbose=0)
                
        print("Train time: {:.2f} min"
              .format((time.time()-start)/60.))
        time.sleep(1)
        
        # Evaluate and save 
        patterns = ('*.hdf5','*.h5','*.h5py')
        weightfiles = []
        for pattern in patterns:
            weightfiles += glob.glob(os.path.join(path, pattern))
        
        model = clf.estimator
        models = []
               
        for weightfile in tqdm(weightfiles, desc="Net (save wts)", ncols=120): 
            model.load_weights(weightfile)
            loss, score = model.evaluate(X, y_, verbose=0)
            models.append((weightfile, np.round(loss, decimals=4)))
        del y_, model
        
        # We sort according to loss: lower is better
        return (clf, sorted(models, key=lambda x: x[1]))
   
    
    def _prep_output(self):
        """
        Internal convenience method. Take object containing fitted algorithms 
        and scores and return a sorted list of classifiers.            
        """
        # Collect and order natives    
        files = []
        for name, fs in self.orchestrator.items():
            fs = [(name, path) for path, _ in fs]
            files += fs 
        return files
    
    
    def hillclimb(self, X_val, y_val, n_best=0.1, p=0.3, iterations=10, scoring='accuracy', n_jobs=1, verbose=0):
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
            
            n_jobs : int, default: 1
                Parallel processing of files.
                
            verbose : int, default: 0
                Whether to output extra information or not.
                - Set verbose = 1 to get info.               
        
        """
        clfs = self._prep_output()        
        total = len([c for c,_ in clfs])

        if isinstance(n_best, float):
            grab = int(n_best*total)
        elif isinstance(n_best, int):
            grab = n_best
        else:
            raise TypeError("n_best should be int or float.")
            
        nets = [(name, path) for name, path in clfs 
                if (name == 'neuralnet')]    
        clfs = [(name, path) for name, path in clfs 
                if (name != 'neuralnet')]        
        scorer = get_scorer(scoring)
              
        # Cache predictions
        if len(clfs)>0:
            algs = joblib.Parallel(n_jobs=n_jobs, 
                                   verbose=verbose, 
                                   backend="threading")(
                joblib.delayed(_sklearn_score_fitted)(path, X_val, y_val, scorer) 
                for name, path in clfs)    
        else: algs = []
        del clfs
        
        if len(nets)>0:
            nalgs = joblib.Parallel(n_jobs=n_jobs, 
                                    verbose=verbose, 
                                    backend="threading")(
                joblib.delayed(_keras_score_fitted)(path, X_val, y_val, scorer) 
                for name, path in nets)
        else: nalgs = []
        del nets
        
        pool = [al for al in algs+nalgs if not al is None]  
        del algs, nalgs
        
                 
        # Sort by score on validation set. Add index.
        pool = [(str(idx), clf, y_pred) for idx, (clf, y_pred, _) 
                in enumerate(sorted(pool, key=lambda x: -x[2]))]            
        # Define weights
        weights = {idx: 0.0 for idx, *_ in pool}    

        # Set initial ensemble
        ensemble = pool[:grab]        
        for idx, *_ in ensemble:
            weights[idx] = 1.0
        current_score = self.score(ensemble, weights, y_val, scorer)        
        
        if verbose > 0:        
            print("Single model max validation score = {}"
                   .format(np.round(max([score for *_, score in pool]), decimals=4)))        
            print("Best model was: {}"
                   .format(ensemble[0][1]))
            print("Ensemble: initial {}-score: {:.5f}"
                   .format(scoring, current_score))
              
        # Sample a subset of algorithms
        algs = self._sample_algs(p, pool)
               
        for it in range(1, iterations+1):
            
            # Initialize 
            if it==1:
                best_idx = None
                best_score = float("-Inf")
                validation_scores = []
            
            for alg in algs:                
                idx = alg[0]                
                cand = ensemble.copy(); cand.append(alg)                         
                wts = weights.copy(); wts[idx] += 1.0
                this_score = self.score(cand, wts, y_val, scorer)            
                if this_score > best_score:
                    best_idx, best_score, best_alg = (idx, this_score, [alg])
            
            if best_score<=current_score:
                print("Could not improve further. Updated score was: {:.5f}"
                       .format(best_score))
                break            
            elif best_score > current_score: 
                current_score = best_score
                weights[best_idx] += 1.0                
                if not best_idx in self._get_idx(ensemble):
                    ensemble += best_alg
            
            validation_scores.append((it, current_score))
            if verbose > 0:
                print("Iteration: {} \tScore: {:.5f}"
                       .format(it, current_score))
            
        weighted_ensemble = [(path_to_model, weights[idx]) 
                             for idx, path_to_model, _ in ensemble]
                
        for path_to_model, wt in weighted_ensemble:
            if wt == 0: 
                warnings.warn("Estimator {} has weight = 0."
                               .format(path_to_model))                  
        return validation_scores, weighted_ensemble
        
        
    def score(self, ensemble, weights, y, scorer):
        """ Compute weighted majority vote 
        """        
        wts = np.zeros(len(ensemble))
        preds = np.zeros((len(y), len(ensemble)), dtype=int)        
        for j, (idx, _, pred) in enumerate(ensemble):
            wts[j] = float(weights[idx])
            preds[:, j] = pred        
        return self._weighted_vote_score(wts, preds, y, scorer)

    
    def _weighted_vote_score(self, wts, preds, y, scorer):  
        """ Score an ensemble of classifiers using weighted voting. 
        """
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
    
    
    def _sample_algs(self, p, pool):        
        idxs_mapper = {idx: (idx, clf, pr) for idx, clf, pr in pool}               
        if isinstance(p, float):
            size = int(p * float(len(pool)))
        elif isinstance(p, int):
            size = p        
        return [idxs_mapper[idx] for idx in 
                np.random.choice(self._get_idx(pool), 
                                 size=size, replace=False)]
            
    
    def _get_idx(self, item):
        return [idx for idx, *_ in item]