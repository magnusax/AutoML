from ..base import BaseClassifier

import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout 
from keras.activations import relu, softmax
###from keras.optimizer import 


class MetaNeuralNetworkClassifier(BaseClassifier):
    
    def __init__(self, random_state=None):
    
         # Meta data
        self.name = 'kerasnn'
        self.max_n_iter = 1
                
        # Init params
        self.init_params = {}
        self.init_params['random_state'] = random_state
        
        self.epochs = epochs
        self.batches = batches
        self.optimizer = optimizer
        
        self.estimator = self._get_clf()    # Define estimator        
        
    def _get_clf(self):
        return flexible_model(**self.network_params)
    
        
    def flexible_model(network):
        """ 
        We use the sequential model API and introduce some flexibility
        in the number of hidden layers, hidden units, regularization, activation, 
        and learning rates.
        
        """
        
        layers_minus_final = num_hidden_layers+1
        
        input_shape = (X.shape[1],)
        
        units = network['units']
        activation = network['activation']
        add_batchnorm = network['bnorm']
        add_dropout = network['dropout']
        
        # Define model
        model = Sequential()
        
        for i in range(layers_minus_final):
            
            # Fully connected layer
            if i==0:
                model.add( Dense(units=units[i], input_shape=input_shape) )
            else:
                model.add( Dense(units=units[i]) )
                
            # Add activation before batchnorm (if BN is added)
            model.add( Activation(activation[i]) )
            
            # Adding batchnorm, if desired
            if add_batchnorm[i]:
                model.add( BatchNormalization() )    
            
            # Finally, add dropout, if desired
            if add_dropout[i]:
                model.add( Dropout(prob[i]) )
            
        model.add( Dense(units=num_classes, activation=activation[-1]) )
                
        # Compile
        model.compile(
            loss='categorical_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'])
        
        return model
    
        
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': hasattr(self.estimator, 'predict_proba')}
    
    
    def adjust_params(self, par):
        return super().adjust_params(par)    
    
    
    def check_training(X, y, train_size=0.1):
        """ 
        Perform training on x% (x<<100) of the training data to check that 
        we can include the network without having to wait for a very long 
        time to finish tranining on the full dataset.
        
        """
        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=train_size, 
                                                    random_state=self.random_state)

        start_time = time.time()
        self.estimator.fit(X_sample, y_sample)
        total_time = time.time()-start_time
        warnings.warn("Keras trained on %s%% of data. Time: %.2f (min)" 
                      % (100*train_size, total_time/60.))
        
        
        
        
        
   