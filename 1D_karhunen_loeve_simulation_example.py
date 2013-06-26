#!/usr/bin/python
# -*- coding: utf8 -*-

"""
==============================================================
Simulation of a 1D random field using Karhunen-Loeve expansion
==============================================================

This script simulates a Gaussian random field using Karhunen-Loeve expansion.
The randomfield is defined over the interval [0; 10]. It is supposed to have a
zero mean, unit variance and a squared exponential covariance function with
correlation length 2.
"""

# Author: Marine Marcilhac <marcilhac@phimeca.com>
#         Vincent Dubourg <dubourg@phimeca.com>
# License: BSD

import numpy as np
import pylab as pl
import openturns as ot
from randomfields import KarhunenLoeveExpansion

# Parameters (see KarhunenLoeveExpansion's docstring)
integral_range = 2.
domain_expand_factor = 1.1
verbose = True
truncation_order = 10
galerkin_scheme = 'legendre'
legendre_galerkin_order = 20
legendre_quadrature_order = 41
n_sample_paths = 1000
n_index_values = 100

# Rectangular domain definition
lower_bound = np.array([0.] * 1)
upper_bound = np.array([10.] * 1)

def mean(x):
    x = np.asanyarray(x)
    if x.ndim <= 1:
        x = np.atleast_2d(x).T
    x = np.atleast_2d(x)
    return np.zeros(x.shape[0])

def covariance(xx):
    xx = np.atleast_2d(xx)
    dd = xx[:, :1] - xx[:, 1:]
    ll = np.atleast_2d(integral_range)
    return np.exp(- np.sum((dd / ll) ** 2., axis=1))

# Discretization of the random field using Karhunen-Loeve expansion
# from its given theoretical moments
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

# Plot the relative covariance discretization error
res = 50
x1, x2 = np.meshgrid(np.linspace(lower_bound, upper_bound, res),
                     np.linspace(lower_bound, upper_bound, res))
xx = np.vstack([x1.ravel(), x2.ravel()]).T
approximated_covariance = \
    random_field.compute_approximated_covariance(xx)
true_covariance = covariance(xx)
covariance_error = true_covariance - approximated_covariance
covariance_relative_error = 100. * covariance_error / true_covariance
pl.figure()
im = pl.imshow(np.flipud(covariance_relative_error.reshape((res, res))),
               extent=(lower_bound[0], upper_bound[0], ) * 2,
               cmap=pl.matplotlib.cm.jet)
cb = pl.colorbar(im)
cb.set_label('Covariance relative error (%)')
pl.title('min = %.2f %%, max = %.2f %%, avg = %.2f %%'
         % (covariance_relative_error.min(),
            covariance_relative_error.max(),
            covariance_relative_error.mean()))
pl.xlabel("$x$")
pl.ylabel("$x'$")
pl.grid()
pl.savefig('1D_simulation_covariance_discretization_error.png')
pl.close()

# Plot the relative variance discretization error
res = 1000
x = np.linspace(lower_bound[0], upper_bound[0], res)
xx = np.vstack([x, x]).T
approximated_variance = \
    random_field.compute_approximated_covariance(xx)
true_variance = covariance(xx)
variance_error = true_variance - approximated_variance
variance_relative_error = 100. * variance_error / true_variance
pl.figure()
pl.plot(x, variance_relative_error)
pl.title('min = %.2f %%, max = %.2f %%, avg = %.2f %%'
         % (variance_relative_error.min(),
            variance_relative_error.max(),
            variance_relative_error.mean()))
pl.xlabel("$x$")
pl.ylabel('Variance relative error (%)')
pl.grid()
pl.savefig('1D_simulation_variance_discretization_error.png')
pl.close()

# Simulation of the (Gaussian) random field
res = n_index_values
x = np.linspace(lower_bound[0], upper_bound[0], res)
xi_theoretical_distribution = ot.Normal(truncation_order)
xi = xi_theoretical_distribution.getSample(n_sample_paths)
sample_paths_values = random_field(x[:, np.newaxis], xi)

# Plot a few sample paths
pl.figure()
pl.title('A few sample paths')
for i in range(min(n_sample_paths, 10)):
    pl.plot(x, sample_paths_values[i], label='$h^{(%d)}$' % i)
pl.xlabel('$x$')
pl.ylabel('$H(x)$')
pl.grid()
pl.legend(loc='lower center', ncol=5)
pl.savefig('1D_simulation_sample_paths.png')
pl.close()

# Save the sample paths to ascii file
np.savetxt('1D_sample_paths.txt',
           np.vstack([x, sample_paths_values]))
