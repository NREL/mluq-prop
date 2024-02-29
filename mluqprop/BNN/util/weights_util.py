import tensorflow as tf
import numpy as np
import sys
import os

def reshape_list_of_tensors(list_of_tensors):
    temp = [None] * len(list_of_tensors)
    for i, g in enumerate(list_of_tensors):
        temp[i] = tf.reshape(g, (tf.math.reduce_prod(tf.shape(g)),))
    return tf.concat(temp, axis=0)


def get_np_weights(ws=None, mlp=None):
    if ws is None and mlp is None:
        sys.exit("ws or mlp should be passed")
    elif ws is None:
        return np.array(reshape_list_of_tensors(mlp.trainable_weights))
    elif mlp is None:
        return np.array(reshape_list_of_tensors(ws))


def get_layer_size(model_hyper):
    n_layers = 1 + (model_hyper["N_H"])
    layer_size = [None] * n_layers
    for i in range(n_layers):
        if i == 0:
            layer_size[i] = (
                model_hyper["D_X"] * model_hyper["D_H"] + model_hyper["D_H"],
            )
        elif i > 0 and i < n_layers - 1:
            layer_size[i] = (
                model_hyper["D_H"] * model_hyper["D_H"] + model_hyper["D_H"],
            )
        else:
            layer_size[i] = (
                model_hyper["D_H"] * model_hyper["D_Y"] + model_hyper["D_Y"],
            )
    return layer_size, n_layers


def from_np_weights_to_tf_weights(weights_tensor, model_hyper):
    weights_list = []
    layer_size, n_layers = get_layer_size(model_hyper)
    temp = [None] * n_layers
    w_min = None
    for i in range(n_layers):
        if w_min is None:
            w_min = 0
        else:
            w_min = w_max
        w_max = tf.math.reduce_prod(layer_size[i]) + w_min
        temp[i] = tf.reshape(
            weights_tensor[w_min:w_max], tf.math.reduce_prod(layer_size[i])
        )
    return temp


def from_inputDeck_to_modelHyper(simparams, exact_path=False):
    if not exact_path:
        path = os.path.join(simparams.checkpath, "best", "best")
    else:
        path = simparams.checkpath
    model_hyper = {"D_X": 10, "D_H": simparams.hidden_dim, "D_Y": 1, 
                   "N_H": simparams.num_layers,
                   "path": path,
                   "type": simparams.model_type,
                   "act": simparams.nonlin, 
                   "post": simparams.posterior_model,
                   "split": True if simparams.split.lower()=="true" else False}
    try:
        layer_mask_spl = simparams.layer_mask.split()
        layer_mask = [int(entry) for entry in layer_mask_spl]
        model_hyper["layer_mask"] = layer_mask
    except AttributeError:
        model_hyper["layer_mask"] = None


    return model_hyper
