This directory contains input decks that specify
- Model architecture
- Training parameters
- I/O for data, model checkpointing, etc.

## Data Paths
We record a few data paths for the combustion problem:

Data Folder
```
/projects/safvto/mhassana/data/
```

Inducing points (in distribution)
```
/projects/safvto/mhassana/inducing_LRM_100000/inducingLRM_160.npz
```

Out-of-Distribution Data (Normalizing Flow)
```
/projects/safvto/mhassana/NF_genBack/
```
- `backgroundComb_highD_labeled_fixed.npz`: ~10 million datapoints.
- `frechetDistRef_synth_XX_fixed.npz`: inducing points, `XX` is the number of inducing points.

Out-of-Distribution Data (Soft Brownian Offset)
```
/projects/safvto/mhassana/SBO_genBack/
```
- `background_dXX_nYY_labeled_fixed.npz`: data with `XX` denoting offset distance and `YY` is the number of datapoints.
- `frechetDistRef_background_dXX_nYY_100_fixed.npz`: Corresponding inducing points (all use 100).

