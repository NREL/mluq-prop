# plot the data and the scalings
# NOTE: you will need to change the paths within the header of the file.

python viz_data.py

# vizualize the results from the uncertainty propagation simulation.
# NOTE: you will need to point the script to where the data lives.

python viz_kse_prop.py --help
python viz_kse_prop.py --inputfpath "/Users/gpash/Documents/uq-prop/KSE/input_src" --datadir "/Users/gpash/Documents/uq-prop/data/prop_log"

# visualize the pointwise uncertainty of model predictions.
# NOTE you will need to change the paths within the header of the file.
python viz_kse_pointwise.py --help
python viz_kse_pointwise.py

# script for processing a single KSE solution
python viz_kse_single.py --help
