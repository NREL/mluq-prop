from mluqprop.BNN.util import BNNHyperModel, load_model

def test_model_specification():
    BNNHyperModel(12,
                    20,
                    1,
                    4,
                    1.,
                    'variational',
                    'sigmoid',
                    'independent',
                    'trainable',
                    split=True)
    
def test_load_model():
    load_model("./ckpt/best/best",
                        D_X=12,
                        D_H=20,
                        D_Y=1,
                        N_H=4,
                        kl_weight=1.,
                        model_type="variational",
                        activation_fn="sigmoid",
                        posterior_model="independent",
                        split=False)
