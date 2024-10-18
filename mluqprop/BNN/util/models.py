#!/usr/bin/env python3

"""
Bayesian Neural Network Models (TensorFlow implementation)
and Helper Functions
and Classes
====================================
NOTE:
- Cannot find TF implementation of Stein VI
"""


import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

tfpl = tfp.layers
tfd = tfp.distributions


__author__ = "Graham Pash"


#############################
# Probabilistic Helpers
#############################
# ========================================================================
# Prior - diagonal MVN ~ N(0, I)
def isotropic_prior(kernel_size, bias_size, dtype=None):
    
    n = kernel_size + bias_size
    
    prior_model = tf.keras.Sequential([
        
        tfpl.DistributionLambda(
            # Note: Our prior is a non-trainable distribution
            lambda t: tfd.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n)))
    ])
    
    return prior_model

  
# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
  ])


# ========================================================================
# Posterior - MVN(mu, Sigma)
def mvn_posterior(kernel_size, bias_size, dtype=None):
    
    n = kernel_size + bias_size
    
    posterior_model = tf.keras.Sequential([
        
        tfpl.VariableLayer(tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfpl.MultivariateNormalTriL(n)
    ])
  
    return posterior_model


# ========================================================================
# Posterior - MVN(mu, I)
def isotropic_posterior(kernel_size, bias_size, dtype=None):

    n = kernel_size + bias_size

    posterior_model = tf.keras.Sequential([
            
            tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
            tfpl.IndependentNormal(n)
    ])

    return posterior_model


# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.))
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])


# ========================================================================
# helper function for computing predictions from Bayesian net with aleatory uncertainty
def compute_prob_predictions(model, X_test:np.array, num_samples:int=100, ptiles:list=[2.5, 97.5], lowmem=False):
    """Query Bayesian neural network model to compute predictions, credible intervals, including aleatoric uncertainty.

    Args:
        model: Bayesian nueral network model.
        X_test (np.array): Test dataset.
        num_samples (int, optional): Number of epistemic model realizations. Defaults to 100.
        ptiles (list, optional): Percentiles for credible interval. Defaults to [2.5, 97.5].
        lowmem (bool, optional): Flag for applications that may cause memory pressure, uses model.predict() instead of model().

    Returns:
        predicted (np.array): The predictions.
        prediction_mean: Mean prediction.
        prediction_percentiles: Credible intervals.
    """
    # Preallocate space to hold precictions.
    predicted = np.zeros((num_samples, X_test.shape[0]))
    
    # Loop through model realizations.
    for i in range(num_samples):
        if lowmem:
            predicted[i, :] = model.predict(X_test, batch_size=2**17, verbose=0).squeeze()
        else:
            predicted[i, :] = model(X_test).sample().numpy().squeeze()

    prediction_mean = np.mean(predicted, axis=0)
    prediction_percentiles = np.percentile(predicted, ptiles, axis=0)

    return predicted, prediction_mean, prediction_percentiles


# ========================================================================
# helper function for computing predictions from Bayesian net with aleatory uncertainty
def compute_raw_epi_predictions(model, X_test:np.array, num_samples:int=100):
    """Helper function to compute epistemic credible intervals, mean predictions for Bayesian neural network.
    
    Args: 
        model: Bayesian Neural Network model.
        X_test (np.array): Test dataset.
        num_samples (int, optional): Number of epistemic model realizations. Defaults to 100.
        lowmem (bool, optional): Flag for applications that may cause memory pressure, uses model.predict() instead of model().
    
    Returns:
        prediction_mean: Mean prediction.
        aleatory: Mean aleatoric uncertainty (standard deviation).
    """
    # Preallocate space to hold precictions.
    predicted = np.zeros((num_samples, X_test.shape[0]))
    aleatory = np.zeros((num_samples, X_test.shape[0]))

    # Loop through model realizations, get mean prediction, epistemic uncertainty.
    for i in range(num_samples):
        v = model(X_test)
        predicted[i, :] = v.mean().numpy().squeeze()
        aleatory[i, :] = v.stddev().numpy().squeeze()

    return predicted, aleatory

# ========================================================================
# helper function for computing predictions from Bayesian net with aleatory uncertainty
def compute_epi_predictions_from_raw(predicted, aleatory, ptiles:list=[2.5, 97.5]):

    # Compute the mean of predictions.
    prediction_mean = np.mean(predicted, axis=0)
    # Compute the mean aleatoric uncertainty.
    aleatory = np.mean(aleatory, axis=0)
    # Compute the epistemic uncertainty.
    epi = np.std(predicted, axis=0)
    # Compute the credible intervals.
    prediction_percentiles = np.percentile(predicted, ptiles, axis=0)

    return prediction_mean, aleatory, epi, prediction_percentiles


# ========================================================================
# helper function for computing predictions from Bayesian net with aleatory uncertainty
def compute_epi_predictions(model, X_test:np.array, num_samples:int=100, ptiles:list=[2.5, 97.5]):
    """Helper function to compute epistemic credible intervals, mean predictions for Bayesian neural network.

    Args:
        model: Bayesian Neural Network model.
        X_test (np.array): Test dataset.
        num_samples (int, optional): Number of epistemic model realizations. Defaults to 100.
        ptiles (list, optional): Percentiles for credible interval. Defaults to [2.5, 97.5].
        lowmem (bool, optional): Flag for applications that may cause memory pressure, uses model.predict() instead of model().

    Returns:
        prediction_mean: Mean prediction.
        aleatory: Mean aleatoric uncertainty (standard deviation).
        epi: Epistemic uncertainty (standard deviation).
        prediction_percentiles: Epistemic credible intervals.
    """
    predicted, aleatory = compute_raw_epi_predictions(model, X_test, num_samples)
    return compute_epi_predictions_from_raw(predicted, aleatory, ptiles)


# ========================================================================
# Epistemic only prediction helper.
def compute_predictions(model, X_test, num_samples=100, ptiles=[2.5, 97.5]):
    # for use with models with deterministic outputs
    predicted = []
    for _ in range(num_samples):
        predicted.append(model(X_test).numpy())
    predicted = np.concatenate(predicted, axis=1).T

    prediction_mean = np.mean(predicted, axis=0).tolist()
    prediction_percentiles = np.percentile(predicted, ptiles, axis=0)

    return predicted, prediction_mean, prediction_percentiles

# ========================================================================
# negative log likelihood
def neg_loglik(y_true, y_pred):
    return -y_pred.log_prob(y_true)


#############################
# Models
#############################
# ========================================================================
# Deterministic Model.
def mlp(D_X:int=1, D_H:int=5, D_Y:int=1, N_H:int=2, activation_fn:str='relu'):
    """Generate a multilayer perceptron model.

    Args:
        D_X (int, optional): Input dimension. Defaults to 1.
        D_H (int, optional): Dimension of the hidden layers. Defaults to 5.
        D_Y (int, optional): Output dimension. Defaults to 1.
        N_H (int, optional): Number of hidden layers. Defaults to 2.
        activation_fn (str, optional): Activation function. Defaults to 'relu'.

    Returns:
        Multi-layer perceptron model.
    """

    # Build model.
    model = tf.keras.Sequential(name="MultilayerPerceptron")
    model.add(tf.keras.Input(shape=(D_X,)))

    # Hidden layers.
    for i in range(N_H):
        model.add(tf.keras.layers.Dense(D_H,
        activation=activation_fn,
        name="hidden"+str(i)
        ))

    model.add(tf.keras.layers.Dense(D_Y))

    return model


# ========================================================================
# Epistemic ONLY model.
def epi_bnn(kl_weight:float, D_X:int=1, D_H:int=5, D_Y:int=1, N_H:int=2, activation_fn:str='relu', prior_fn=isotropic_prior, posterior_fn=mvn_posterior):
    """Generate an epistemic Bayesian neural network model.

    Args:
        kl_weight (float): Weight for KL divergence.
        D_X (int, optional): Input dimension. Defaults to 1.
        D_H (int, optional): Number of units in hidden layer. Defaults to 5.
        D_Y (int, optional): Output dimension. Defaults to 1.
        N_H (int, optional): Number of Hidden layers. Defaults to 2.
        activation_fn (str, optional): Activation function. Defaults to 'relu'.
        prior_fn (_type_, optional): Representation for weight priors. Defaults to isotropic_prior.
        posterior_fn (_type_, optional): Representation for weight posteriors. Defaults to mvn_posterior.

    Returns:
        Epistemic Bayesian neural network model.
    """

    # Build model.
    model = tf.keras.Sequential(name="EpistemicBayesianNet")
    model.add(tf.keras.Input(shape=(D_X,)))

    # Hidden layers.
    for i in range(N_H):
        model.add(tfp.layers.DenseVariational(
            units=D_H,
            activation=activation_fn,
            make_prior_fn=prior_fn,
            make_posterior_fn=posterior_fn,
            kl_use_exact=False,
            kl_weight=kl_weight,
            name="hidden"+str(i),
        ))

    # model.add(tf.keras.layers.Dense(D_Y, name="output"))

    # Trainable output layer.
    model.add(tfpl.DenseVariational(
        units=D_Y,
        # activation=activation_fn,
        make_prior_fn=prior_fn,
        make_posterior_fn=posterior_fn,
        kl_use_exact=False,
        kl_weight=kl_weight,
        name="output"))

    return model


# ========================================================================
# Epistemic and Aleatoric Model.
def bnn(kl_weight:float, D_X:int=1, D_H:int=5, D_Y:int=1, N_H:int=2, activation_fn:str='relu', prior_fn=prior_trainable, posterior_fn=mvn_posterior):
    """Generate a Bayesian neural network.

    Args:
        kl_weight (float): Weight for KL divergence.
        D_X (int, optional): Input dimension. Defaults to 1.
        D_H (int, optional): Number of units in hidden layer. Defaults to 5.
        D_Y (int, optional): Output dimension. Defaults to 1.
        N_H (int, optional): Number of Hidden layers. Defaults to 2.
        activation_fn (str, optional): Activation function. Defaults to 'relu'.
        prior_fn (_type_, optional): Representation for weight priors. Defaults to isotropic_prior.
        posterior_fn (_type_, optional): Representation for weight posteriors. Defaults to mvn_posterior.

    Returns:
        Bayesian neural network model.
    """
    
    # Build model.
    model = tf.keras.Sequential(name="BayesianNet")
    model.add(tf.keras.Input(shape=(D_X,)))

    # Hidden layers.
    for i in range(N_H):
        model.add(tfp.layers.DenseVariational(
            units=D_H,
            activation=activation_fn,
            make_prior_fn=prior_fn,
            make_posterior_fn=posterior_fn,
            kl_use_exact=False,
            kl_weight=kl_weight,
            name="hidden"+str(i),
        ))

    # Last layer has mean, std units for stochastic output
    model.add(tfp.layers.DenseVariational(
        units=tfp.layers.IndependentNormal.params_size(D_Y),
        make_prior_fn=prior_fn,
        make_posterior_fn=posterior_fn,
        kl_use_exact=False,
        kl_weight=kl_weight,
        name="output",
    ))

    # Aleatoric uncertainty
    model.add(tfp.layers.IndependentNormal(D_Y, name="aleatoric"))

    return model


# ========================================================================
# Epistemic and Aleatoric Model. With split trunks for mean, std.
def split_bnn(kl_weight:float, D_X:int=1, D_H:int=5, D_Y:int=1, N_H:int=2, activation_fn:str='relu', prior_fn=prior_trainable, posterior_fn=mvn_posterior):
    """Generate a Bayesian neural network with a split last layer for separate mean, std outputs.

    Args:
        kl_weight (float): Weight for KL divergence.
        D_X (int, optional): Input dimension. Defaults to 1.
        D_H (int, optional): Number of units in hidden layer. Defaults to 5.
        D_Y (int, optional): Output dimension. Defaults to 1.
        N_H (int, optional): Number of Hidden layers. Defaults to 2.
        activation_fn (str, optional): Activation function. Defaults to 'relu'.
        prior_fn (_type_, optional): Representation for weight priors. Defaults to isotropic_prior.
        posterior_fn (_type_, optional): Representation for weight posteriors. Defaults to mvn_posterior.

    Returns:
        Bayesian neural network model.
    """
    
    model_inputs = tf.keras.Input(shape=(D_X,))
    
    # First dense variational layer.
    x = tfp.layers.DenseVariational(
            units=D_H,
            activation=activation_fn,
            make_prior_fn=prior_fn,
            make_posterior_fn=posterior_fn,
            kl_use_exact=False,
            kl_weight=kl_weight,
            name="hidden0",
        )(model_inputs)
    
    # Loop through remaining hidden layers.
    for i in range(N_H-1):
        x = tfp.layers.DenseVariational(
            units=D_H,
            activation=activation_fn,
            make_prior_fn=prior_fn,
            make_posterior_fn=posterior_fn,
            kl_use_exact=False,
            kl_weight=kl_weight,
            name="hidden"+str(i+1),
        )(x)
    
    # Output mean.
    z_mu = tfp.layers.DenseVariational(
        units=tfp.layers.IndependentNormal.params_size(D_Y) // 2,
        make_prior_fn=prior_fn,
        make_posterior_fn=posterior_fn,
        kl_use_exact=False,
        kl_weight=kl_weight,
        name="output_mu",
    )(x)
    
    # Output standard deviation.
    z_sigma = tfp.layers.DenseVariational(
        units=tfp.layers.IndependentNormal.params_size(D_Y) // 2,
        make_prior_fn=prior_fn,
        make_posterior_fn=posterior_fn,
        kl_use_exact=False,
        kl_weight=kl_weight,
        name="output_sigma",
    )(x)
    
    # Plumb everything through a layer to a normal distribution.
    # N(z_mu, softplus(z_sigma)))
    v = tfp.layers.DistributionLambda(lambda t: tfd.Independent
                                      (tfd.Normal(loc=t[0], scale=tf.math.softplus(t[1])), reinterpreted_batch_ndims=1), 
                                      name="aleatoric")([z_mu, z_sigma])
    
    # Build the model.
    model = tf.keras.Model(model_inputs, v, name="SplitBayesianNet")
    
    return model


# ========================================================================
# Epistemic and Aleatoric Model.
def mixed_bnn(kl_weight:float, layer_mask:list, D_X:int=1, D_H:int=5, D_Y:int=1, N_H:int=2, activation_fn:str='relu', prior_fn=prior_trainable, posterior_fn=mvn_posterior):
    """Generate a Bayesian neural network.

    Args:
        kl_weight (float): Weight for KL divergence.
        layer_mask (list): List of 0s and 1s to indicate which layers to use. 0 indicates that the layer should be deterministic.
        D_X (int, optional): Input dimension. Defaults to 1.
        D_H (int, optional): Number of units in hidden layer. Defaults to 5.
        D_Y (int, optional): Output dimension. Defaults to 1.
        N_H (int, optional): Number of Hidden layers. Defaults to 2.
        activation_fn (str, optional): Activation function. Defaults to 'relu'.
        prior_fn (_type_, optional): Representation for weight priors. Defaults to isotropic_prior.
        posterior_fn (_type_, optional): Representation for weight posteriors. Defaults to mvn_posterior.

    Returns:
        Bayesian neural network model.
    """
    
    # Build model.
    model = tf.keras.Sequential(name="BayesianNet")
    model.add(tf.keras.Input(shape=(D_X,)))

    # Hidden layers.
    for i in range(N_H):
        if layer_mask[i] == 1:
            model.add(tfp.layers.DenseVariational(
                units=D_H,
                activation=activation_fn,
                make_prior_fn=prior_fn,
                make_posterior_fn=posterior_fn,
                kl_use_exact=False,
                kl_weight=kl_weight,
                name="hidden"+str(i),
            ))
        else:
            model.add(tf.keras.layers.Dense(
                units=D_H,
                activation=activation_fn,
                name="hidden"+str(i),
            ))

    # Last layer has mean, std units for stochastic output
    model.add(tfp.layers.DenseVariational(
        units=tfp.layers.IndependentNormal.params_size(D_Y),
        make_prior_fn=prior_fn,
        make_posterior_fn=posterior_fn,
        kl_use_exact=False,
        kl_weight=kl_weight,
        name="output",
    ))

    # Aleatoric uncertainty
    model.add(tfp.layers.IndependentNormal(D_Y, name="aleatoric"))

    return model


# ========================================================================
# Epistemic and Aleatoric Model, using the Flipout estimator.
def flipout_bnn(kl_weight:int=1, D_X:int=1, D_H:int=5, D_Y:int=1, N_H:int=2, activation_fn:str="sigmoid"):
    """Bayesian neural network using Flipout estimator.

    Args:
        batch_size (int): Number of samples in each minibatch.
        D_X (int, optional): Input dimension. Defaults to 1.
        D_H (int, optional): Number of units in hidden layers. Defaults to 5.
        D_Y (int, optional): Output dimension. Defaults to 1.
        N_H (int, optional): Number of hidden layers. Defaults to 2.
        activation_fn (str, optional): Activation function. Defaults to 'relu'.
        rescale (bool, optional): Rescale KL divergence. Defaults to False.

    Returns:
        TensorFlow Bayesian neural network model.
    """
    # Kernel-divergence function. Reweight to account for batches.
    kdf = (lambda q, p, _: tfd.kl_divergence(q, p) * tf.cast(kl_weight, dtype=tf.float32))
    
    # Build model.
    model = tf.keras.Sequential(name="BayesianFlipout")
    model.add(tf.keras.Input(shape=(D_X,)))

    # Hidden layers.
    for i in range(N_H):
        model.add(tfp.layers.DenseFlipout(
            units=D_H,
            activation=activation_fn,
            name="hidden"+str(i),
            kernel_divergence_fn=kdf
        ))

    # Last layer has mean, std units for stochastic output
    model.add(tfp.layers.DenseFlipout(
        units=tfp.layers.IndependentNormal.params_size(D_Y),
        name="output",
        kernel_divergence_fn=kdf
    ))

    # Aleatoric uncertainty
    model.add(tfp.layers.IndependentNormal(D_Y, name="aleatoric"))

    return model


# ========================================================================
# MOPED Initialization - Independent Normal ONLY ~ N(w, delta*|w|)
def moped_bnn(premodel, kl_weight:float, delta:float=0.1, D_X:int=1, D_H:int=5, D_Y:int=1, N_H:int=2, activation_fn:str="sigmoid"):
    """Work in Progress implementation of MOPED BNN initialization.

    Args:
        premodel: Keras model.
        delta (float, optional): Control parameter for variance. Defaults to 0.1.
        D_X (int, optional): Input dimension. Defaults to 1.
        D_H (int, optional): Number of hidden layers. Defaults to 5.
        D_Y (int, optional): Output dimension. Defaults to 1.
        N_H (int, optional): Number of hidden layers. Defaults to 2.
        activation_fn (str, optional): Activation function for network. Defaults to 'relu'.

    Returns:
        Tensorflow/Keras MOPED model.
    """

    # Build model.
    model = tf.keras.Sequential(name="MOPED")
    model.add(tf.keras.Input(shape=(D_X,)))
    
    # Hidden layers.
    for i in range(N_H):
        model.add(tfp.layers.DenseVariational(
            units=D_H,
            activation=activation_fn,
            make_prior_fn=isotropic_prior,
            make_posterior_fn=isotropic_posterior,
            kl_use_exact=False,
            kl_weight=kl_weight,
            name="hidden"+str(i),
        ))

    # Last layer has mean, std units for stochastic output.
    model.add(tfp.layers.DenseVariational(
        units=tfp.layers.IndependentNormal.params_size(D_Y),
        make_prior_fn=isotropic_prior,
        make_posterior_fn=isotropic_posterior,
        kl_use_exact=False,
        kl_weight=kl_weight,
        name="output",
    ))
    
    # Aleatoric uncertainty
    model.add(tfp.layers.IndependentNormal(D_Y, name="aleatoric"))
    
    # Manually adjust all of the weights.
    for i in range(N_H):
        v = premodel.layers[i].weights
        vs = np.concatenate([v[0].numpy().flatten(), v[1].numpy().flatten()])
        lw = np.concatenate([vs, tfp.math.softplus_inverse(np.sqrt(delta*np.abs(vs)))])
        model.trainable_variables[i].assign(lw)
    
    # Last layer takes a little more care.
    # There is no analogue to the variance term in the last layer, so leave it at the prior.
    v = premodel.trainable_variables[-1]
    vs = np.concatenate([premodel.layers[-1].weights[0].numpy().flatten(), premodel.layers[-1].weights[1].numpy().flatten()])
    
    
    # Assuming [mu_w, mu_b, sigma_w, sigma_b]
    # model.trainable_variables[-1][:len(vs)].assign(vs)
    # model.trainable_variables[-1][len(vs)*2:len(vs)*3].assign(tfp.math.softplus_inverse(np.sqrt(delta*np.abs(vs))))
    
    # Assuming [mu_w, mu_b, mu_sw, mu_sb, sigma_w, sigma_b, sigma_sw, sigma_sb]
    model.trainable_variables[-1][:len(vs)-1].assign(vs[:-1])
    model.trainable_variables[-1][(len(vs)*2)-2].assign(vs[-1])
    model.trainable_variables[-1][(len(vs)*2):(len(vs)*3)-1].assign(tfp.math.softplus_inverse(np.sqrt(delta*np.abs(vs[:-1]))))
    model.trainable_variables[-1][-2].assign(tfp.math.softplus_inverse(np.sqrt(delta*np.abs(vs[-1]))).numpy())

    return model


# ========================================================================
# Get list of weight posteriors from split BNN model.
def get_weight_posteriors_splitBNN(splitmodel, abstractmodel):
    """Get list of weight posteriors from split BNN model.

    Args:
        splitmodel (Keras Model): The model with weight posteriors.
        abstractmodel (BNNHyperModel): Abstraciton of model containing architecture information.
    """
    # Figure out how many layers there are.
    nlayers = abstractmodel.N_H + 1
    # Alternatively... nlayers = len(splitmodel.layers) - 3 # -3 for input, output_sigma, aleatoric
    
    # Dummy for querying posterior.
    dummy = np.array([0.])
    
    # Hidden layer posteriors.
    posteriors = [splitmodel.get_layer(f"hidden{i}")._posterior(dummy) for  i in range(nlayers - 1)]
    
    # Output mean.
    posteriors.append(splitmodel.get_layer("output_mu")._posterior(dummy))
    
    return posteriors


def sample_weight_posterior(posteriors:list):
    """Sample the weight posteriors given a list of the layer posteriors.

    Args:
        posteriors (list): List of weight posteriors that can be sampled with .sample()

    Returns:
        (list): List of weight posterior samples.
    """
    return [p.sample() for p in posteriors]


def assign_weights_to_mlp(mlp, ws):
    
    # Put the weights into the MLP.
    for i in range(0, len(mlp.trainable_variables), 2):
        # Get the shapes of the weights, biases.
        nx, ny = mlp.trainable_variables[i].shape[0], mlp.trainable_variables[i].shape[1]
        
        # Assign the weights.
        mlp.trainable_variables[i].assign(np.reshape(ws[i // 2].numpy()[:nx*ny], (nx, ny)))
        
        # Assign the biases.
        mlp.trainable_variables[i+1].assign(ws[i // 2].numpy()[nx*ny:])

    return mlp

def split_bnn_to_mlp(posteriors, abstractmodel):
    
    # Create the empty MLP.
    mlp = abstractmodel.build_mlp()
    
    # Sample the BNN weights.
    ws = sample_weight_posterior(posteriors)
   
    mlp = assign_weights_to_mlp(mlp, ws)
  
    return mlp
    

# ========================================================================
class BNNHyperModel():
    """A class to define and build a variety of BNN models.
    """
    def __init__(self, dx:int, dh:int, dy:int, nh:int, kl_weight:float, model_type:str, activation_fn:str="sigmoid", posterior_model:str="independent", prior_model:str="trainable", split:bool=False, layer_mask:list=None):
        """Initialize the BNN model with necessary architecture info.

        Args:
            dx (int): Input dimension.
            dh (int): Number of units in hidden dimension.
            dy (int): Output dimension.
            nh (int): Number of hidden layers.
            kl_weight (float): Weighting for KL rescaling.
            model_type (str): Type of BNN model. Either "epi", "variational", or "flipout".
            activation_fn (str): Activation function of the model. Defaults to 'relu'.
            posterior_model (str): Posterior model. Either "mvn" or "independent". Only for use with DenseVariatioanl layers, i.e. "variational" model type. Defaults to "independent".
            prior_model (str): Prior model. Either "isotropic" or "trainable". Only for use with DenseVariational layers, i.e. "variational" model type. Defaults to "trainable".
            split (Optional, bool): Whether to split the last layer of the model into mean and std. Defaults to False.
            layer_mask (list): List of 0s and 1s to indicate which layers to use. 0 indicates that the layer should be deterministic. Defaults to "None", i.e. not to be used.
        """
        self.D_X = dx
        self.D_H = dh
        self.D_Y = dy
        self.N_H = nh
        self.kl_weight = kl_weight
        self.model_type = model_type
        self.activation_fn = activation_fn
        self.posterior_model = posterior_model
        self.prior_model = prior_model
        self.split = split
        self.layer_mask = layer_mask
    
    def build(self):
        if self.model_type == "epi":
            # Epistemic ONLY model.
            model = epi_bnn(
                    kl_weight=self.kl_weight,
                    D_X=self.D_X,
                    D_H=self.D_H,
                    N_H=self.N_H,
                    activation_fn=self.activation_fn,
                    posterior_fn=mvn_posterior if self.posterior_model == "mvn" else isotropic_posterior,
                    prior_fn=isotropic_prior if self.prior_model == "isotropic" else prior_trainable
                )
        elif self.model_type == "variational":
                # Using DenseVariational layers.
            if self.split:
                model = split_bnn(
                        kl_weight=self.kl_weight,
                        D_X=self.D_X,
                        D_H=self.D_H,
                        N_H=self.N_H,
                        activation_fn=self.activation_fn,
                        posterior_fn=mvn_posterior if self.posterior_model == "mvn" else isotropic_posterior,
                        prior_fn=isotropic_prior if self.prior_model == "isotropic" else prior_trainable
                    )
            else:
                model = bnn(
                        kl_weight=self.kl_weight,
                        D_X=self.D_X,
                        D_H=self.D_H,
                        N_H=self.N_H,
                        activation_fn=self.activation_fn,
                        posterior_fn=mvn_posterior if self.posterior_model == "mvn" else isotropic_posterior,
                        prior_fn=isotropic_prior if self.prior_model == "isotropic" else prior_trainable
                    )
        elif self.model_type == "flipout":
            # Using DenseFlipout layers.
            model = flipout_bnn(
                kl_weight=self.kl_weight,
                D_X=self.D_X,
                D_H=self.D_H,
                D_Y=self.D_Y,
                N_H=self.N_H,
                activation_fn=self.activation_fn   
            )
        else:
            # Catch invalid model type.
            raise ValueError("Invalid model type. Must be 'epi', 'variational', or 'flipout'.")
        
        # Rebuild the model as mixed if the layer mask is set.
        if self.layer_mask is not None:
            model = mixed_bnn(
                kl_weight=self.kl_weight,
                layer_mask=self.layer_mask,
                D_X=self.D_X,
                D_H=self.D_H,
                N_H=self.N_H,
                activation_fn=self.activation_fn,
                posterior_fn=mvn_posterior if self.posterior_model == "mvn" else isotropic_posterior,
                prior_fn=isotropic_prior if self.prior_model == "isotropic" else prior_trainable
            )
        
        return model
    
    def build_mlp(self):
        mlpmodel = mlp(D_X=self.D_X,
                       D_H=self.D_H,
                       D_Y=self.D_Y,
                       N_H=self.N_H,
                       activation_fn=self.activation_fn)
        return mlpmodel


#############################
# General model helper functions.
#############################
# ========================================================================
# Load weights from checkpoint file into specified net
def load_model(fpath:str, D_X:int, D_H:int, D_Y:int, N_H:int, kl_weight:int=1, model_type:str='variational', activation_fn='sigmoid', posterior_model="independent", split:bool=False, layer_mask:list=None):
    """Helper function to load model from checkpointed weights. This is necessary as TensorFlow Probability does not support saving/loading models with `model.save` method, if you would like to inspect the weight posteriors distributions.

    Args:
        fpath (str): Filepath to the checkpoint file.
        D_X (int): Input dimension.
        D_H (int): Number of units in hidden dimension.
        D_Y (int): Output dimension.
        N_H (int): Number of hidden layers.
        batch_size (int, optional): Number of training data in each minibatch for KL rescaling. Defaults to 1.
        model_type (str, optional): Type of model to load. Either "epi", "variational", or "flipout". Defaults to 'variational'.
        activation_fn (str, optional): Activation function of the model. Defaults to 'relu'.
        posterior_model (str, optional): Posterior model. Either "mvn" or "independent". Only for use with DenseVariatioanl layers, i.e. "variational" model type. Defaults to "mvn".
        split (Optional, bool): Whether to split the last layer of the model into mean and std. Defaults to False.
            layer_mask (list): List of 0s and 1s to indicate which layers to use. 0 indicates that the layer should be deterministic. Defaults to "None", i.e. not to be used.

    Returns:
        TensorFlow model.
    """
    # Specify the architecture.
    abstract_model = BNNHyperModel(dx=D_X, dh=D_H, dy=D_Y, nh=N_H, kl_weight=kl_weight, model_type=model_type, activation_fn=activation_fn, posterior_model=posterior_model, split=split, layer_mask=layer_mask)
    
    # Build model.
    model = abstract_model.build()

    # Load weights into model.
    model.load_weights(fpath)

    return model


# ========================================================================
# Load weights from checkpoint file into specified deterministic net (MLP)
def load_mlp_model(fpath:str, D_X:int, D_H:int, D_Y:int, N_H:int, kl_weight:int=1, model_type:str='variational', activation_fn='sigmoid', posterior_model="independent", split:bool=False, layer_mask:list=None):
    """Helper function to load model from checkpointed weights. This is necessary as TensorFlow Probability does not support saving/loading models with `model.save` method, if you would like to inspect the weight posteriors distributions.

    Args:
        fpath (str): Filepath to the checkpoint file.
        D_X (int): Input dimension.
        D_H (int): Number of units in hidden dimension.
        D_Y (int): Output dimension.
        N_H (int): Number of hidden layers.
        batch_size (int, optional): Number of training data in each minibatch for KL rescaling. Defaults to 1.
        model_type (str, optional): Type of model to load. Either "epi", "variational", or "flipout". Defaults to 'variational'.
        activation_fn (str, optional): Activation function of the model. Defaults to 'relu'.
        posterior_model (str, optional): Posterior model. Either "mvn" or "independent". Only for use with DenseVariatioanl layers, i.e. "variational" model type. Defaults to "mvn".
        split (Optional, bool): Whether to split the last layer of the model into mean and std. Defaults to False.
            layer_mask (list): List of 0s and 1s to indicate which layers to use. 0 indicates that the layer should be deterministic. Defaults to "None", i.e. not to be used.

    Returns:
        TensorFlow model.
    """
    # Specify the architecture.
    abstract_model = BNNHyperModel(dx=D_X, dh=D_H, dy=D_Y, nh=N_H, kl_weight=kl_weight, model_type=model_type, activation_fn=activation_fn, posterior_model=posterior_model, split=split, layer_mask=layer_mask)
    
    # Build model.
    model = abstract_model.build_mlp()

    # Load weights into model.
    model.load_weights(fpath)

    return model


# ========================================================================
# Load weights from checkpoint file into specified net
def load_model_from_unsplit_weights(fpath:str, D_X:int, D_H:int, D_Y:int, N_H:int, kl_weight:int=1, model_type:str='variational', activation_fn='sigmoid', posterior_model="independent", split:bool=False, layer_mask:list=None):
    """Helper function to load model from unsplit checkpointed weights. This is necessary as TensorFlow Probability does not support saving/loading models with `model.save` method, if you would like to inspect the weight posteriors distributions.

    Args:
        fpath (str): Filepath to the checkpoint file.
        D_X (int): Input dimension.
        D_H (int): Number of units in hidden dimension.
        D_Y (int): Output dimension.
        N_H (int): Number of hidden layers.
        batch_size (int, optional): Number of training data in each minibatch for KL rescaling. Defaults to 1.
        model_type (str, optional): Type of model to load. Either "epi", "variational", or "flipout". Defaults to 'variational'.
        activation_fn (str, optional): Activation function of the model. Defaults to 'relu'.
        posterior_model (str, optional): Posterior model. Either "mvn" or "independent". Only for use with DenseVariatioanl layers, i.e. "variational" model type. Defaults to "mvn".
        split (Optional, bool): Whether to split the last layer of the model into mean and std. Defaults to False.
            layer_mask (list): List of 0s and 1s to indicate which layers to use. 0 indicates that the layer should be deterministic. Defaults to "None", i.e. not to be used.

    Returns:
        TensorFlow model.
    """

    # Create split and unsplit arch
    abstract_model_split = BNNHyperModel(dx=D_X, dh=D_H, dy=D_Y, nh=N_H, kl_weight=kl_weight, model_type=model_type, activation_fn=activation_fn, posterior_model=posterior_model, split=True, layer_mask=layer_mask)
    abstract_model_unsplit = BNNHyperModel(dx=D_X, dh=D_H, dy=D_Y, nh=N_H, kl_weight=kl_weight, model_type=model_type, activation_fn=activation_fn, posterior_model=posterior_model, split=False, layer_mask=layer_mask)
    
    # Create models with split and unsplit arch
    model_split = abstract_model_split.build()
    model_unsplit = abstract_model_unsplit.build()

    # Load weights into unsplit model
    model_unsplit.load_weights(fpath)

    # inspect gradients to know how to split    
    #x = tf.ones((1, 10))
    #with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
    #    tape.watch(model_unsplit.variables)
    #    y_mean = model_unsplit(x).mean()
    #    y_std = model_unsplit(x).stddev()
    #breakpoint()
 
    # Assign unsplit weights to split model except at the last layer.
    for i_l in range(int(2*N_H)):
        model_split.variables[i_l].assign(model_unsplit.variables[i_l])
    
    # Special treatment for last layer
    ind_split = [int(2*N_H), int(2*N_H)+1]
    for inds in ind_split:
        n_unspl = model_unsplit.variables[inds].shape[0]
        n_spl = n_unspl // 2
        model_split.variables[inds].assign(model_unsplit.variables[inds][::2])
        model_split.variables[inds+2].assign(model_unsplit.variables[inds][1::2])
        

    return model_split

# ========================================================================
# Save model training history to file.
def save_history(history, metrics, fpath:str):
    """Save model training history to file.

    Args:
        history (dict): History dictionary from model.fit()
        metrics (dict): Dictionary of metrics from training.
        fpath (str): File path to save history to.
    """
    np.savez(fpath,
                loss=history.history["loss"],
                epochs=metrics.epochs,
                wasserstein=metrics.wasserstein,
                medsnr=metrics.medsnr,
                meandist=metrics.meandist,
                stddist=metrics.stddist
            )


# ========================================================================
# Apply binary mask to the weights of a BNN model.
def apply_weight_mask(model, mask:np.array):
    
    nw0 = 0  # index of layer's first weight in the mask.
    for i in range(len(model.layers) - 1):
        # Get the layer's weights.
        ll = model.layers[i].trainable_variables[0]
        
        # Number of "real" weights.
        nw = ll.shape[0] // 2
        
        # Apply mask to weights, set standard deviation to zero.
        tmp = tf.math.multiply(ll[nw:], mask[nw0:nw0+nw])
        tmp = tf.where(tf.equal(tmp,0), tfp.math.softplus_inverse(0.), tmp)
        ll[nw:].assign(tmp)
        
        # Reset the index in the mask.
        nw0 += nw
    
    return model
