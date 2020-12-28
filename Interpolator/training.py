#!/usr/bin/python
#title			:training.py
#description	:This script contains the routine to train  a feedforward neural network for a grid of spectra
#author			:Nitesh Kumar
#date			:2020-10-05
#version		:0.0.2
#usage			:python training.py
#notes			:
#python_version	:3.7.5 
#==============================================================================
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from astropy.io import fits
from matplotlib import gridspec
from numpy import linalg as la


class InterpolateModel:
    """ 
    This is a class for initilisating and training of the neural network for Toy Interpolator 1. 
      
    Attributes: 
        tau (float): The precision target that we seek. 
        n1 (int)   : The number of neurons in first hidden layer.
        n2 (int)   : The number of neurons in second hidden layer.
        n3(int)    : The number of neurons in third hidden layer.
        out (int)  : The number of neurons in output layer.
        V (float)  : Matrix of Variation spectra  shape=(Number of samples, Number of Wavelength bins)
    """
    def __init__(self,tau,n1,n2, n3, out,V):
        """ 
        The constructor for InterpolateModel class. 
  
        Parameters: 
           tau (float): The precision target that we seek. 
        n1 (int)   : The number of neurons in first hidden layer.
        n2 (int)   : The number of neurons in second hidden layer.
        n3(int)    : The number of neurons in third hidden layer.
        out (int)  : The number of neurons in output layer.
        V (float)  : Matrix of Variation spectra  shape=(Number of samples, Number of Wavelength bins)
        """

        self.tau = tau
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.out = out
        self.V = V
    tf.keras.backend.set_floatx('float64')
    
    def obj_1(self,V):
        """ 
        The function returns the phi_1^2 objective function defined in "Kumar et al 20XX". 
  
        Parameters: 
            V (Matrix): Matrix of Variation spectra,
            shape     : [Number of samples, Number of Wavelength bins]
          
        Returns: 
            phi_1^2: Objective function that has to be minimised. 
        """
        def loss(model,interpolated):
            L =  model - interpolated
            zeta_1 = tf.divide(tf.reduce_sum(L*self.V ,axis=-1), tf.reduce_sum(self.V**2,axis=-1))
            tau_1 = tf.constant(self.tau**2,shape=zeta_1.shape,dtype=zeta_1.dtype)
            phi_1_square = tf.maximum(zeta_1**2,tau_1)
            return tf.reduce_mean(phi_1_square) 
        return loss

    def obj_2(self,V):
        """ 
        The function returns the phi_2^2 objective function defined in "Kumar et al 20XX". 
  
        Parameters: 
            V (Matrix): Matrix of Variation spectra,
            shape     : [Number of samples, Number of Wavelength bins]
          
        Returns: 
            phi_2^2: Objective function that has to be minimised. 
        """
        def loss(model, interpolated):
            L = model - interpolated
            zeta_2 = tf.divide(L,self.V)
            tau_2 = tf.constant(self.tau,shape=zeta_2.shape,dtype=zeta_2.dtype)
            phi_2_square = tf.reduce_sum(tf.maximum(tf.square(zeta_2),tf.square(tau_2)),axis=-1)
            return tf.reduce_mean(phi_2_square)
        return loss
    
    
    ## Least square solution for the terminal weights
    def lsq_solution_V1(self,X, y):
        w = np.dot(np.dot(la.inv(np.dot(X.T, X)), X.T), y)
        return w
    def lsq_solution_V2(self,X, y):
        w = np.dot(la.pinv(X), y)
        return w
    def lsq_solution_V3(self,X, y):
        w, residual, rank, svalues = la.lstsq(X, y,rcond=None)
        return w
    
    ## TODO  :::   write the support for custum activation functions-----
    def hidden_network(self,X,model):
        w = model.get_weights()
        for i in range(len(w)//2 -1):
            X = tf.sigmoid(tf.add(tf.matmul(X,w[2*i]),w[2*i +1]))
        return X
    
    
    def update_terminal_weights(self,X,y,model):
        ## The values of the neurons in the last hidden layer 
        H = np.insert(self.hidden_network(X=X,model=model),obj=0,values=1.,axis=-1)

        print('.'*20+'updating terminal weights'+'.'*20)
        start_time = time.time()
        p_mat = self.lsq_solution_V2(H,y)
        model.trainable_variables[-1].assign(p_mat[0])
        model.trainable_variables[-2].assign(p_mat[1:])
        print('.'*18+'DONE in {:.2f}sec'.format(time.time()-start_time)+'.'*18)
        return None
    

    def train_model(self,X,y,epochs,obj='mse',min_delta=1e-08,patience=5,verbosity=False):
        """ 
        The function trains the neural network by minimising the "Physical Loss" of the interpolation. 

    Arguments:
        X: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
            or `(inputs, targets, sample weights)`.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a dataset, generator,
          or `keras.utils.Sequence` instance, `y` should
          not be specified (since targets will be obtained from `x`).
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        obj: {'phi_1', 'phi_2','mse'} , default: 'mse'
            - if obj = 'phi_1', The objective function 'phi_1_square' will be minimized during training training. 
            - if obj = 'phi_2', The objective function 'phi_2_square' will be minimized during training training. 
            - if obj = 'mse', The objective function 'mean_square_error' will be minimized during training training. 

    Returns:
        model   : tf.keras.Model() object "The model of the training that contains information about the architecture of the network, weights of the training"
        history : A numpy array that contains the objective function values after each training epoch at each node
        flag    : If the model found the minimum, If found, the model gets stopped before completing the total epochs, and flag is True
                  otherwise False

        """
        xavier = tf.keras.initializers.glorot_normal()

        inputs = tf.keras.Input(shape=(3,), name="digits")
        l1 = tf.keras.layers.Dense(self.n1, activation="sigmoid",kernel_initializer=xavier, name="dense_1",)(inputs)
        l2 = tf.keras.layers.Dense(self.n2, activation="sigmoid",kernel_initializer=xavier, name="dense_2")(l1)
        l3 = tf.keras.layers.Dense(self.n3, activation="sigmoid",kernel_initializer=xavier, name="dense_3")(l2)
        outputs = tf.keras.layers.Dense(self.out,activation=None,kernel_initializer=xavier, name="predictions")(l3)

        nn_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Instantiate an optimizer to train he model.
        optimizer = tf.keras.optimizers.Adam()
        

        if (obj == 'phi_1'):
            loss = self.obj_1(self.V)
        elif(obj == 'phi_2'):
            loss = self.obj_2(self.V)
        elif(obj == 'mse'):
            loss = tf.keras.losses.mse
        else:
            raise NameError('The objective function does not exist!')
                 
    
        #model.compile(loss=loss,
        #          optimizer=optimizer)
        
        #history = model.fit(X, y,verbose=0, epochs=Epochs,batch_size=2908,
        #                      callbacks=[tfdocs.modeling.EpochDots(report_every=2000,dot_every=100)])
        
        history = np.zeros(shape=(epochs+1,y.shape[0]))
        delta = np.zeros(shape=(epochs+1,))
        
        
        if verbosity:print('Untrained Objective function value = {} \n'.format(np.mean(loss(y,nn_model.predict(X)))))
        
        strt_time = time.time()
        ## training loop for non terminal weights
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                y_hat = nn_model(X, training=True)
                loss_value = loss(y , y_hat)

            ## determine gradients for non terminal weights
            grads = tape.gradient(loss_value, nn_model.trainable_variables[:-2])
            ## update non terminal weights
            optimizer.apply_gradients(zip(grads, nn_model.trainable_variables[:-2]))
            
            ## value of objective function after the updation of non terminal weights 
            new_loss = loss(y , nn_model(X))
            
            history[epoch] = loss(nn_model.predict(X),y)
            delta[epoch] = np.mean(history[epoch-1]) - np.mean(history[epoch])
            if verbosity:print("Epoch {:d}/{:d} and loss={:.6f}, delta ={:.6f} ".format(epoch+1,epochs,np.mean(new_loss),delta[epoch]),end='\r')
            
            ### condition for stopping if the objective function does not decrease in given patience epochs by an amount min_delta
            for idx in range(patience):
                stop = epoch- idx > 0 and (delta[epoch-idx] <= min_delta)
                stop = stop and stop
            if stop:break    
        
        ## training of terminal weights
        self.update_terminal_weights(X,y,nn_model); 
        
        ## logging of loss after the terminal weights update
        history[epoch+1] = loss(nn_model.predict(X),y)
        
        ## Truncate the history log 
        history = history[:epoch+2,]
        delta = delta[:epoch+2,]
        flag = (epoch==epochs-1)
        print('\n Total Epochs={:d} and training time={:.2f}s, and reached to a minimum={}\n Final Objective Function value={}'
              .format(epoch+2,time.time()-strt_time,not(flag),np.mean(history[-1])))
        return nn_model,history,not(flag)
    
    def plot(self,wl,X,TGM,y,model):
        spectra_= y
        pred = model.predict(X)
        for i in np.arange(0,2908,250):
            fig = plt.figure(figsize=(12,6),facecolor='w')
            gs  = gridspec.GridSpec(2, 1,fig, height_ratios=[4, 1.5],hspace=0)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])

            ax0.plot(wl, spectra_[i],label='True')
            ax0.plot(wl, pred[i],
                     label="Prediction for Teff=%5d\nLog g=%5.2f\n[Fe/H]=%5.2f" %(TGM[i,0], 
                                                   TGM[i,1], 
                                                   TGM[i,2]))
            ax0.legend(framealpha=0.5,fontsize=12,loc='best')
            ax0.set_ylabel("Normalised Flux")
            ax0.set_title("Spectrum of Actual train and Reconstructed train")
            ax0.set_xticklabels([])

            ax1.set_frame_on(True)

            diff= spectra_[i]-pred[i]
            ax1.plot(wl,diff,label='Difference',c='k')
            zero_y=np.zeros(shape = pred.shape[1])
            ax1.plot(wl,zero_y,c='g')
            ax1.set_ylabel(r"$\Delta$")
            ax1.set_xlabel(r'Wavelength($\rm\AA$)')
            ax1.legend(loc='upper right',framealpha=0.2,fontsize=12)
            #ax0.set_ylim(0,2.5)
            #ax1.set_ylim(-.15,.15)
            plt.show()

    def save_model(self,fname,model):
        """
        Generate an Interpolator fits file with the specifications provided in "Kumar et al 20XX". 
  
        Parameters: 
            fname   : str
                      filename of the output file,
            model   : Sequential model returned by train_model,
                  
        """
        # Initialise a hdul
        new_hdul = fits.HDUList()
        
        # get the total number of layers (No. of Hidden layers + 1 )
        total_layers = int(len(model.get_weights())/2)
        
        ## For saving the weights into a hdul
        for j in reversed(range(total_layers)):
            # retrieve the weights from the Sequential model of Keras
            w = model.get_weights()[2*j]
            b = model.get_weights()[2*j+1]

            w_ = np.append(b,w).reshape(w.shape[0]+1,w.shape[1])
            if (j == total_layers - 1):
                name = 'TERM_WEIGHTS'
            else:
                name = 'HIDDEN_LAYER_%d'%(j+1)
            new_hdul.append(fits.ImageHDU(w_,name=name))
        
        ## For saving the metadata 
        for i in range(total_layers):
            if (i == 0):
                new_hdul[i].header['CRPIX1'] = 1
                new_hdul[i].header['CTYPE1'] = 'AWAV'
                new_hdul[i].header['CRVAL1'] = 4749.74
                new_hdul[i].header.comments['CRVAL1'] = 'Wavelength at first pixel \AA'
                new_hdul[i].header['CDELT1'] = 1.25
                new_hdul[i].header.comments['CDELT1'] = 'Step size in \AA'
                new_hdul[i].header['I_AFUNC'] = 'linear'
                new_hdul[i].header.comments['I_AFUNC'] = 'Activation function'
            else:
                new_hdul[i].header['I_AFUNC'] = 'logistic'
                new_hdul[i].header.comments['I_AFUNC'] = 'Activation function'
            new_hdul[i].header['I_LAYER'] = total_layers - i
            new_hdul[i].header.comments['I_LAYER'] = 'Index of the target layer'
            new_hdul[i].header['I_VERSIO'] = '0.0.1'
            new_hdul[i].header.comments['I_VERSIO'] = 'Version of the perceptron file format'
            new_hdul[i].header['I_PREPRO'] = 'Standard_Scalar'
            new_hdul[i].header.comments['I_PREPRO'] = 'Identification of the pre-processing function'
            new_hdul[i].header['I_POSTPR'] = '            '
            new_hdul[i].header.comments['I_POSTPR'] = 'Identification of the post-processing function'
            new_hdul[i].header['I_HLAYER'] = total_layers - 1
            new_hdul[i].header.comments['I_HLAYER'] = 'Number of hidden layers   '
        new_hdul.writeto(fname+'.fits',overwrite=True)
