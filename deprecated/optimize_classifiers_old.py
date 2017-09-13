    def optimize_classifiers(self, X, y, n_iter=12, scoring='accuracy', cv=10, n_jobs=1, 
                             sample_hyperparams=False, min_hyperparams=2, get_hyperparams=False, random_state=None):
        """
        Docstring:
        
        This method is a wrapper to cross validation using RandomizedSearchCV from scikit-learn, wherein we optimize each defined algorithm
        Default behavior is to optimize all parameters available to each algorithm, but it is possible to sample (randomly) a subset of them
        to optimize (sample_hyperparams=True), or to choose a set of parameters (get_hyperparams=True).
        
        Input parameters:
        ------------------
        X: data matrix (n_samples, n_features)
        y: labels/ground truth (n_samples,)
        
        n_iter: (int: 1) number of iterations to use in RandomizedSearchCV method, i.e. number of independent draws from parameter dictionary
        scoring: (str or callable: 'accuracy') type of scorer to use in optimization
        cv: (int or callable: 10) number of cross validation folds, or callable of correct type
        n_jobs: (int: 1) specify number of parallel processes to use
        
        sample_hyperparams: (bool: False) randomly sample a subset of algorithm parameters and tune these  
        min_hyperparams: (int: 2) when sample_hyperparams=True, choose number of parameters to sample
        get_hyperparams: (bool: False) instead of random sampling, use previously chosen set of parameters to optimize (must be preceeded by ...)
        random_state: (None or int: None) used for reproducible results
        
        Ouput:
        ------------------
        List containing (classifier name, most optimized classifier) tuples
        
        """
        import sys
        import time
        import warnings
        from sklearn.model_selection import RandomizedSearchCV
        
        optimized = []
        
        for name, classifier in self.clf:            
            estimator, param_dist_list = classifier.estimator, classifier.cv_params            
            for i, param_dist in enumerate(param_dist_list):
            
                if sample_hyperparams and not get_hyperparams:
                    """ Here we (by default, but other behaviors are also possible) sample randomly
                        [1, number hyperparams] to optimize in the cross-validation loop """
                    num_params = np.random.randint(min_hyperparams, len(param_dist))
                    param_dist = classifier.sample_hyperparams(param_dist, num_params=num_params, mode='random')
            
                if get_hyperparams and not sample_hyperparams:
                    if len(classifier.cv_params_to_tune) > 0:
                        print("(%s): overriding current parameter dictionary using 'cv_params_to_tune'" % name)
                        param_dist = classifier.sample_hyperparams(param_dist, mode='select', keys=classifier.cv_params_to_tune)            
            
                n_iter_ = min(n_iter, classifier.max_n_iter)
            
                if self.verbose>0:
                    print("Starting grid search for '%s'" % name)
                    print("Setting 'n_iter' to:", n_iter_)
                    
                search = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=n_iter_, 
                                            scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=self.verbose, 
                                            error_score=0, return_train_score=True, random_state=random_state)
                name_ = name if i == 0 else "%s_%s" % (name, str(i))
                start_time = time.time()
                try:
                    search.fit(X, y)
                except:
                    warnings.warn("Estimator '%s' failed (likely: 'n_iter' too high). \nAdding un-optimized version." % name_)
                    optimized.append((name_, estimator.fit(X,y)))
                else:
                    optimized.append((name_, search.best_estimator_))
                    if isinstance(scoring, str):
                        print("(scoring='%s')\tBest mean score: %.4f (%s)" % (scoring, search.best_score_, name_))
                    else:
                        print("Best mean score: %.4f (%s)" % (search.best_score_, name_))
                    break                    
                print("Iteration time = %.2f min." % ((time.time()-start_time)/60.))            
                
        # Rewrite later: for now just return a list of optimized estimators
        return optimized
