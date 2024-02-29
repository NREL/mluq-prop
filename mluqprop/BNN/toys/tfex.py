#!/usr/bin/env python

"""
TensorFlow Bayesian Neural Network Example
================================

Follows the tutorial at:
https://towardsdatascience.com/modeling-uncertainty-in-neural-networks-with-tensorflow-probability-391b29538a7a

with the modification of using a BNN with two hidden layers
"""

# Import libraries
from time import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

tfd = tfp.distributions
tfpl = tfp.layers

# Create non-linear data
n_samples = int(5e3)
x_train = np.linspace(-1, 1, n_samples)[:, np.newaxis]
y_train = x_train**3 + 0.1*(1.5 + x_train)*np.random.randn(n_samples)[:, np.newaxis]
# y_train = 0.5 * np.power(0.5 + x_train, 2.0) * np.sin(4.0 * x_train)
# y_train += 0.05 * np.random.randn(n_samples, 1)
# y_train = y_train[:, np.newaxis]
# mu, std = np.mean(y_train), np.std(y_train)
# y_train = (y_train - mu) / std

# Plot the data
plt.figure(figsize=(6, 4))
plt.scatter(x_train, y_train, alpha=0.05)
plt.title('Non-linear training data')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

# Prior - diagonal MVN ~ N(0, 1)
def prior(kernel_size, bias_size, dtype=None):
    
    n = kernel_size + bias_size
    
    prior_model = tf.keras.Sequential([
        
        tfpl.DistributionLambda(
            # Note: Our prior is a non-trianable distribution
            lambda t: tfd.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n)))
    ])
    
    return prior_model


# Posterior
def posterior(kernel_size, bias_size, dtype=None):
    
    n = kernel_size + bias_size
    
    posterior_model = tf.keras.Sequential([
        
        tfpl.VariableLayer(tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfpl.MultivariateNormalTriL(n)
    ])
    
    return posterior_model


# Define the model
def get_full_model(x_train_shape):
    
    model = tf.keras.Sequential([

        # Epistemic uncertainty
        tfpl.DenseVariational(units=8,
                              input_shape=(1,),
                              make_prior_fn=prior,
                              make_posterior_fn=posterior,
                              kl_weight=1/x_train_shape,
                              kl_use_exact=False,
                              activation='sigmoid'),
        
        tfpl.DenseVariational(units=tfpl.IndependentNormal.params_size(1),
                              make_prior_fn=prior,
                              make_posterior_fn=posterior,
                              kl_use_exact=False,
                              kl_weight=1/x_train_shape),

        # Aleatoric uncertainty
        tfpl.IndependentNormal(1)
    ])

    def neg_loglik(y_true, y_pred):
        return -y_pred.log_prob(y_true)

    model.compile(loss=neg_loglik, optimizer='rmsprop')
    
    return model


# Define the model
def get_bnn_model(x_train_shape, D_H=5):
    
    model = tf.keras.Sequential()
    model.add(tfp.layers.DenseVariational(
        units=D_H,
        input_shape=(1,),
        activation='tanh',
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_use_exact=False,
        kl_weight=1/x_train_shape,
    ))
    model.add(tfp.layers.DenseVariational(
        units=D_H,
        activation='tanh',
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_use_exact=False,
        kl_weight=1/x_train_shape,
    ))
    model.add(tfp.layers.DenseVariational(
        units=tfp.layers.IndependentNormal.params_size(1),
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_use_exact=False,
        kl_weight=1/x_train_shape,
    ))

    # Aleatoric uncertainty
    model.add(tfp.layers.IndependentNormal(1))

    def neg_loglik(y_true, y_pred):
        return -y_pred.log_prob(y_true)

    model.compile(loss=neg_loglik, optimizer='rmsprop')
    
    return model


# Get the model
model_full = get_full_model(n_samples)
# model_full = get_bnn_model(n_samples)

# Fit
print("Training Model")
start = time()
history = model_full.fit(x_train, y_train, epochs=1000, verbose=0)
print("Elapsed training time: ", time() - start)

# Plot loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'])
plt.title('Loss')
plt.xlabel('No. of epochs')
plt.ylabel('Loss')
plt.show()

# Plot fitted regression lines
N_ITERS = 15

plt.figure(figsize=(9, 6))

#------------------------------

plt.scatter(x_train, y_train, alpha=.1, label='Training data')
for _ in range(N_ITERS):
    
    # Compute conf ints
    y_sample = model_full(x_train).sample()
    y_hat = model_full(x_train).mean()
    y_sd = model_full(x_train).stddev()
    y_hat_lower = y_hat - 2 * y_sd
    y_hat_upper = y_hat + 2 * y_sd
    

    if _ == 0:
        plt.plot(x_train, y_hat, color='red', alpha=.5, lw=1, label='Fitted reg. lines')
        plt.plot(x_train, y_hat_lower, c='g', alpha=.5, lw=1, label='95% CI')
        plt.plot(x_train, y_hat_upper, c='g', alpha=.5, lw=1)
    else:
        plt.plot(x_train, y_hat, color='red', alpha=.5, lw=1)    
        plt.plot(x_train, y_hat_lower, c='g', alpha=.5, lw=1)
        plt.plot(x_train, y_hat_upper, c='g', alpha=.5, lw=1)

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()

#------------------------------

plt.title(f'Training data + fitted lines + 95% CIs\nn={n_samples}')
plt.show()

