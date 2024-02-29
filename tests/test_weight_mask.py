from mluqprop.BNN.util import BNNHyperModel, load_model, compute_snr, apply_weight_mask
import numpy as np

def test_compute_snr():
    model = load_model("./ckpt/best/best",
                        D_X=12,
                        D_H=20,
                        D_Y=1,
                        N_H=4,
                        kl_weight=1.,
                        model_type="variational",
                        activation_fn="sigmoid",
                        posterior_model="independent",
                        split=False)

    compute_snr(model)

def test_apply_weight_mask():
    model = load_model("./ckpt/best/best",
                        D_X=12,
                        D_H=20,
                        D_Y=1,
                        N_H=4,
                        kl_weight=1.,
                        model_type="variational",
                        activation_fn="sigmoid",
                        posterior_model="independent",
                        split=False)

    snr = compute_snr(model)

    mask = 0*snr

    maskedmodel = apply_weight_mask(model, mask)

    # If all weights have no variance, then you should get the same thing every time.
    dummy = np.array([0.])
    check = np.allclose(maskedmodel.layers[0]._posterior(dummy).sample(), maskedmodel.layers[0]._posterior(dummy).sample())

    assert check
