# echo options
python train_kse_bnn.py --help

# train model specifying path to where model should be checkpointed
python train_kse_bnn.py --datapath "/Users/gpash/Documents/uq-prop/data/leanKSE.npz" --checkpath "/Users/graham/uq-prop/models/leanKSE"

# NOTE: the training dataset was subsetted by hand. If you wish to use the full DNS data, you will need to subset it again manually, awful - I know.
