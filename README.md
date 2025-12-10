# mluq-prop

## Installation
### From `conda` (recommended)

1. `conda create --name uqprop python=3.10`
2. `pip install -e .`

### For NLR HPC users
1. `module load openmpi/4.1.0/gcc-8.4.0`
2. `conda activate /projects/mluq/condaEnvs/uqprop`

## Code Description
Bayesian neural networks are an attractive method to estimate and predict modeling uncertainty due to their ability to ingest large amounts of data, relatively fast inference cost (compared to Gaussian processes), rigorous characterization of uncertainty, and expressivity. Recent advancements employing variational inference have made the training of BNNs tractable for large models and amenable to large datasets. BNNs reformulate deterministic deep learning models as point-estimators and emulate the construction of an ensemble of neural nets by assigning a probability distribution to each network parameter. Thus, they generate a predictive distribution by sampling the parameter distributions and collecting the resulting distribution of point estimates. We explore the use of BNNs for quantifying both _epistemic_ and _aleatoric_ uncertainties stemming from the adopted model form and training dataset. In particular, we model the sub-filter progress variable dissipation rate of premixed turbulent flames.

## Reference
If you find this repository useful in your research, please consider citing the following:
```
@misc{pash2024priori,
      title={A Priori Uncertainty Quantification of Reacting Turbulence Closure Models using Bayesian Neural Networks}, 
      author={Graham Pash and Malik Hassanaly and Shashank Yellapantula},
      year={2024},
      eprint={2402.18729},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn}
}
```
The NREL software record for this repository is SWR-24-36

## Acknowledgement
> This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Department of Energy Computational Science Graduate Fellowship under Award Number DE-SC0021110.

> This work was authored in part by the National Laboratory of the Rockies (NLR), operated by Alliance for Energy Innovation, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported as part of DEGREES, an Energy Earthshot Research Center (EERC) funded by the U.S. Department of Energy, Office of Science, Basic Energy Sciences. The research was performed using computational resources sponsored by the Department of Energy's Office of Critical Minerals and Energy Innovation (CMEI) and located at the National Laboratory of the Rockies.

The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
