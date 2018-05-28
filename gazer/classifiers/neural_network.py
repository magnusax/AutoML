from ..base import BaseClassifier

import warnings
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import (Activation, 
                          BatchNormalization, 
                          Dense, Dropout)


class MetaNeuralNetworkClassifier(BaseClassifier):
    
    def __init__(self, epochs=10, batch_size=16, optimizer='adam', 
                 learning_rate=1e-4, n_hidden=2, p=0.1, decay_units=False, input_units=250):
    
        """
        Parameters:
        ------------
        
            epochs : integer, default: 10
                The number of epochs to train for
                
            batch_size : integer, default: 16
                Batch size 
                
            optimizer : string or keras callable. Default: 'adam'
                The optimizer to use when compiling
                
            learning_rate : float, default: 1e-4
                Specify learning rate if applicable. 
                
            n_hidden : integer, default: 2
                Specify the number of hidden dense layers
                
            p : float, default: 0.1
                Specify dropout rate if applicable. 
                Only takes effect when 'dropout' is set to True in network dict.
                
            decay_units : boolean, default: False
                If False then keep the number of units constant at the value given by 'input_units'.
                If True then decay the number of units by a factor of 2 from layer to layer.
                
            input_units : integer, default: 250
                The number of units (neurons) to use in the input layer.
                
        """
        # We need to know the shape of the data, and the number of classes
        # We need to set these before constructing the network
        self.input_shape, self.output_shape = (None, None)        
        self.name = 'neuralnet'
        
        self.network = {}
        self.network['epochs'] = epochs
        self.network['batch_size'] = batch_size
        self.network['optimizer'] = optimizer
        self.network['lr'] = learning_rate
        
        # Specify architecture
        self.network['n_hidden'] = n_hidden
        self.network['activation'] = ['relu'] * (n_hidden+1)
        self.network['batchnorm'] = [False] * (n_hidden+1)
        self.network['dropout'] = [True] * (n_hidden+1)
        self.network['p'] = p
        if self.decay_units:
            self.network['units'] = [int(input_units//2**i) 
                                     for i in range(n_hidden+1)]
        else:
            self.network['units'] = [input_units] * (n_hidden+1)
            
        # Estimator is yet to be defined at this stage
        self.estimator = None
        self.ready = False
    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': True}
    
    
    def set_architecture(self, input_shape, output_shape):
        """
        Called prior to building and compiling the keras model.
        """
        # Handle different representations
        if hasattr(input_shape, len):
            self.input_shape = (*input_shape,)
        else:
            self.input_shape = (input_shape,)
            
        # The `output_shape` is simply the number of class labels    
        self.output_shape = output_shape
        self.ready = True
        return
        
    
    def _get_clf(self):

        """ 
        We use the sequential model API and introduce some flexibility
        in the number of hidden layers, hidden units, regularization, activation, 
        and learning rates.
        
        """ 
        assert self.ready==True
        
        units = self.network['units']
        activations = self.network['activation']
        add_batchnorm = self.network['batchnorm']
        add_dropout = self.network['dropout']        
        p = self.network['p']
        
        model = Sequential()
        for i in range(self.network['n_hidden']+1):
            # Fully connected layers
            if i==0:
                model.add(Dense(units=units[i], input_shape=self.input_shape))
            else:
                model.add(Dense(units=units[i]))
                
            # Add activation before batchnorm (if BN is added)
            model.add(Activation(activations[i]))
            
            # Adding batchnorm, if desired
            if add_batchnorm[i]: model.add(BatchNormalization())    
            
            # Finally, add dropout, if desired
            if add_dropout[i]: model.add(Dropout(p))
            
        model.add(Dense(units=self.output_shape, activation='softmax'))                
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.network['optimizer'],
                      metrics=['accuracy'])
        return model
    
    
    def check_training(X, y, train_size=0.1):
        """ 
        Perform training on X% (X<<100) of the training data to check that 
        we can include the algorithm without having to wait for a very long 
        time to finish training on the full dataset.
        """
        
        # We cap the size of the training fraction
        train_size = 0.5 if train_size > 0.5 else train_size
        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=train_size)

        start_time = time.time()
        self.estimator.fit(X_sample, y_sample)
        total_time = time.time()-start_time
        
        # Output a warning 
        warnings.warn("Time spent training on %s%% of data: %.2f (min)" 
                                    % (100*train_size, total_time/60.))
        
        
    def fit(self, X, y):
        
        # We keep a local copy of the labels 
        # to avoid modifying the original input data
        y_ = y.copy()
        
        if len(y_.shape)==1: y_ = y_.reshape(-1, 1)
        
        if y_.shape[1]==1:
            warnings.warn(
                """Keras expects one-hot encoded label data: your data does not seem to fit this requirement.
                   \nWill attempt to apply one-hot encoding before sending to `fit` method.""")
            y_ = keras.utils.to_categorical(y_)
        
        # Freeze architecture
        self.set_architecture(X.shape[1], y_.shape[1])
        # Define estimator
        self.estimator = self._get_clf()
        # Fit estimator
        self.estimator.fit(X, y_, 
                           batch_size=self.network['batch_size'], 
                           epochs=self.network['epochs'])     
        
        
    def _set_cv_params(self):
        """
        Define trainable parameters. In this case we specify different
        architectures.
        """
        
        
        
        
   