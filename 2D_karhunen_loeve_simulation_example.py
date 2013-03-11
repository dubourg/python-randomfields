#!/usr/bin/python
# -*- coding: utf8 -*-

"""
==============================================================
Simulation of a 2D random field using Karhunen-Loeve expansion
==============================================================

This script simulates a Gaussian random field using Karhunen-Loeve expansion.
The randomfield is defined over the square domain [0; 10]^2. It is supposed to
have a zero mean and unit variance and a stationary squared exponential
covariance function whose correlation lengths are 3 and 2.
"""

# Author: Marine Marcilhac <marcilhac@phimeca.com>
#         Vincent Dubourg <dubourg@phimeca.com>
# License: BSD

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import openturns as ot
from randomfields import KarhunenLoeveExpansion

# Parameters (see KarhunenLoeveExpansion's docstring)
integral_range = np.array([3., 2.])
domain_expand_factor = 1.1
verbose = True
truncation_order = 30
galerkin_scheme = 'legendre'
legendre_galerkin_order = 8
legendre_quadrature_order = 17
n_sample_paths = 1000
n_index_values = 50

# Rectangular domain definition
lower_bound = np.array([0.] * 2)
upper_bound = np.array([10.] * 2)

def mean(x):
    x = np.atleast_2d(x)
    return np.zeros(x.shape[0])

def covariance(xx):
    xx = np.atleast_2d(xx)
    dd = np.abs(xx[:, :2] - xx[:, 2:])
    ll = np.atleast_2d(integral_range)
    return np.exp(- np.sum((dd / ll) ** 2., axis=1))

# Discretization of the random field using Karhunen-Loeve expansion
random_field = KarhunenLoeveExpansion(
                        mean,
                        covariance,
                        truncation_order,
                        [lower_bound, upper_bound],
                        domain_expand_factor=domain_expand_factor,
                        verbose=verbose,
                        galerkin_scheme=galerkin_scheme,
                        legendre_galerkin_order=legendre_galerkin_order,
                        legendre_quadrature_order=legendre_quadrature_order)
truncation_order = random_field._truncation_order

# Plot the relative variance discretization error
res = 50
x1, x2 = np.meshgrid(np.linspace(lower_bound[0], upper_bound[0], res),
                     np.linspace(lower_bound[1], upper_bound[1], res))
xx = np.vstack([x1.ravel(), x2.ravel()]).T
xxxx = np.hstack([xx, xx])
approximated_variance = random_field.compute_approximated_covariance(xxxx)
variance = covariance(xxxx)
variance_error = variance - approximated_variance
variance_relative_error = 100. * variance_error / variance
pl.figure()
im = pl.imshow(np.flipud(variance_relative_error.reshape((res, res))),
               extent=(lower_bound[0], upper_bound[0],
                       lower_bound[1], upper_bound[1]),
               cmap=pl.matplotlib.cm.jet)
cb = pl.colorbar(im)
cb.set_label('Relative variance discretization error (%)')
pl.title('min = %.2f %%, max = %.2f %%, avg = %.2f %%'
         % (variance_relative_error.min(),
            variance_relative_error.max(),
            variance_relative_error.mean()))
pl.xlabel("$x$")
pl.ylabel("$x'$")
pl.grid()
pl.savefig('2D_simulation_discretization_error.png')
pl.close()

# Simulation of the (Gaussian) random field
res = n_index_values
x1, x2 = np.meshgrid(np.linspace(lower_bound[0], upper_bound[0], res),
                     np.linspace(lower_bound[0], upper_bound[0], res))
x = np.vstack([x1.ravel(), x2.ravel()]).T
xi_distribution = ot.Normal(truncation_order)
xi = xi_distribution.getNumericalSample(n_sample_paths)
sample_paths_values = random_field(x, xi)

# Plot a sample path
fig = pl.figure()
ax = Axes3D(fig)
ax.plot_surface(x1, x2, sample_paths_values[0].reshape((res, res)),
                rstride=1, cstride=1, cmap=pl.matplotlib.cm.jet)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$h(\mathbf{x})$')
pl.savefig('2D_simulation_sample_path.png')
pl.close()

# Save the sample paths to ascii file
np.savetxt('2D_sample_paths.txt',
           np.vstack([x1.ravel(), x2.ravel(), sample_paths_values]))
