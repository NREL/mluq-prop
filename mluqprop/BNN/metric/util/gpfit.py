import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
import matplotlib.pyplot as plt

def gpfit_simple(x ,y, ndim, n_restart=100, constant_value_bounds=(1e-4, 1e3), length_scale_bounds=(1e-3, 1e3)):
    
    kernel = ConstantKernel(constant_value=[1], constant_value_bounds=[constant_value_bounds]) * RBF(
        length_scale=[1e1]*ndim, length_scale_bounds=[length_scale_bounds]*ndim
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=0.0, n_restarts_optimizer=n_restart
    )
    gpr.fit(x, y)
    y_mean, y_std = gpr.predict(x, return_std=True)
    like = gpr.log_marginal_likelihood(gpr.kernel_.theta)
    return y_mean, y_std, gpr, like

def gpfit_homo(
    x,
    y,
    ndim=1,
    n_restart=10,
    constant_value_bounds=(1e-4, 1e3),
    length_scale_bounds=(1e-5, 1e5),
    noise_level_bounds=(1e-12, 1e2),
):
    kernel = ConstantKernel(
        constant_value=[1], constant_value_bounds=[constant_value_bounds]
    ) * RBF(
        length_scale=[1e1] * ndim,
        length_scale_bounds=[length_scale_bounds] * ndim,
    ) + WhiteKernel(
        noise_level=1, noise_level_bounds=noise_level_bounds
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=0.0, n_restarts_optimizer=n_restart
    )
    gpr.fit(x, y)
    y_mean, y_std = gpr.predict(x, return_std=True)
    like = gpr.log_marginal_likelihood(gpr.kernel_.theta)
    return y_mean, y_std, gpr, like

