# BNN: Bayesian Neural Networks
This directory contains utilities that build off of `tensorflow` and `tensorflow-probability` to quickly construct Bayesian neural network models.

# Getting Started
The standard workflow for this project is to set up an input deck that provides the necessary architecture information to build a Bayesian neural network. This is then supplied to the training script to train the model. The post-processing scripts also use this input deck to load in the correct model and generate results. When submitting jobs for distributed training, we make use of the TACC Launcher and have utilities to generate the jobfiles that will then be called upon, one model per line.

> Note: scripts will provide additional information if called with the `--help` or `-h` flags.

## Implementation Details
Codes to develop and use BNN models are based (largely) upon the TensorFlow Probability library. There are a set of (less well maintained) codes that were developed in parallel using PyTorch in the `pytorch` directory.

The main functions are `epi_bnn` which implements epistemic only BNNs and `bnn` which implements a full Bayesian neural network with both epistemic and aleatoric uncertainty. The `flipout_bnn` implements a full Bayesian neural network with the Flipout estimator. These are implemented in `util/models.py`. A `BNNHyperModel` class has been provided to simplify the creation of these models. Simply pass all of the required constructor arguments and build the model with the `build()` method.

The wrapper functions to generate model predictions are `compute_predictions` and `compute_prob_predictions`, respectively.

Data wrangling helpers specific to the DNS datasets are found in `util/dns.py`.

Plotting utilities are available in `util/plotting.py`

Useful training callbacks are implemented through the `BNNCallbacks` class in `util/callbacks.py`.

Various metrics for assessing the performance of the model are provided in `util/metrics.py`.

Methods for parsing for the input decks and CLI arguments are in `util/input_parser.py`.

## Model Training
The main training script is `trainer.py` and can be customized via a text input deck (examples are provided in the `inputs` directory). This training is easily deployed to a cluster with `Slurm`, and examples are provided in the `jobs` directory.

### Assessing model performance.
The `viz_model_preds.py` script is provided to generate analysis plots for the model in the context of closure modeling for reacting flows.

### Hyperparameter Optimization
Hyperparameters are tuned via `scikit-learn` grid search. Jobs are written to a job file for `Slurm` job submission in tandem with the TACC [launcher](https://github.com/TACC/launcher) utility to distribute the candidate models across multiple processes. A convenience script is provided to generate the jobfile: `write_hypertune_jobfile.py`. Post-processing of the results can be handled with the script: `postproc_hypertune.py`.

To generate the necessary file with the individual jobs to be distributed, run:

`python write_hypertune_jobfile.py --filename FILENAME`

A shared input deck is provided in `inputs/hypertune.in` which sets parameters to be shared. Note that the architecture and training parameters are specified through CLI arguments when calling `trainer.py`. If one wishes to expand the parameters that are tuned, new interfaces _may_ need to be written.

The job may be submitted to the Eagle queues using the SLURM script `jobs/hypertune.job`.

### Out-of-Distribution Extrapolation
One may wish to append out-of-distribution data to the original dataset. We have provided an example for the 10-dimensional combustion dataset: `write_10d_extrap_jobfile.py`. Post-processing can be performed directly with the script: `postproc_10d_extrap.py`, or with the convenience script: `run_extrap_postproc.sh`.

To perform distributed training with extrapolatory data, run:

`python write_extrap_jobfile.py --extraptype EXTRAPTYPE --extrap_data_path FPATH`

The `--append` option will append runs to the current jobfile to allow one to run both normalizing flow and soft brownian offset data together.

### Dimension Reduction
To perform distributed training with mixed probabilistic and determinisitc models, run:

`python write_dimreduce_jobfile.py --filename FNAME --num_layers N --refmodel_path FPATH --npredict NP`

This creates the jobfile to allow one to run a distributed training of all of the combinations of probabilistic / deterministic layers for a given architecture depth.

## Understanding Bayesian Neural Networks
This repository also contains scripts in the `toys` directory that demonstrate different aspects of BNNs:

- The `extrap.py` script demonstrates how to use out-of-distribution data to address extrapolation uncertainty in the model predictions using a low-fidelity model.

- The `collider.py` script demonstrates how uncertainty changes when you have a region of high epistemic uncertainty and low aleatoric uncertainty and another region of low epistemic uncertainty and high aleatoric uncertainty.

- The `uncertainties.py` script demonstrates how BNNs compare to Gaussian Processes and how the individual realizations look.

## Bayesian Neural Networks as Data-Driven Closure Term Models
We build on the work of Yellapantula et al. (2021) using deep learning based models for the progress variable dissipation rate in premixed flames.

```
 @article{title={Deep learning-based model for progress variable dissipation rate in turbulent premixed flames},
 volume={38},
 ISSN={15407489},
 DOI={10.1016/j.proci.2020.06.205},
 journal={Proceedings of the Combustion Institute},
 author={Yellapantula, Shashank and Perry, Bruce A. and Grout, Ray W.},
 year={2021}
 }
```

We utilize the same direct numerical simulation (DNS) dataset as Yellapantula et. al. to train our Bayesian neural network. 

```
@inproceedings{Wilson:2014:LSF:2616498.2616534,
Acmid = {2616534},
Address = {New York, NY, USA},
Articleno = {40},
Author = {Wilson, Lucas A. and Fonner, John M.},
Booktitle = {Proceedings of the 2014 Annual Conference on Extreme Science and Engineering Discovery Environment},
Date-Added = {2015-08-05 19:28:28 +0000},
Date-Modified = {2015-08-05 19:28:28 +0000},
Doi = {10.1145/2616498.2616534},
Isbn = {978-1-4503-2893-7},
Keywords = {Parametric studies, Scalable applications, Software frameworks},
Location = {Atlanta, GA, USA},
Numpages = {8},
Pages = {40:1--40:8},
Publisher = {ACM},
Series = {XSEDE '14},
Title = {Launcher: A Shell-based Framework for Rapid Development of Parallel Parametric Studies},
Url = {http://doi.acm.org/10.1145/2616498.2616534},
Year = {2014},
Bdsk-Url-1 = {http://doi.acm.org/10.1145/2616498.2616534},
Bdsk-Url-2 = {http://dx.doi.org/10.1145/2616498.2616534}}
```
