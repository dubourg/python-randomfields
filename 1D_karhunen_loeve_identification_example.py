#!/usr/bin/python
# -*- coding: utf8 -*-

"""
==================================================================
Identification of a 1D random field using Karhunen-Loeve expansion
==================================================================

This scripts shows how to identify the parameters characterizing a random
field H(x) from a set of sample paths using the Karhunen-Loeve expansion
method.

The set of sample paths is loaded from an ascii file assuming the first line
contains the indexing variable values of the 1D random field and the subsequent
lines contain the corresponding sample paths values.
These assumptions hold for this script only, note that it is possible to use
the KarhunenLoeveExpansion in other contexts such as different unstructured
grid formats for each sample path by first building a callable version of each
sample paths using ANY suitable interpolation technique (interp1d is used
here).

From `randomfields`, it uses `KarhunenLoeveExpansion` for:

 - solving the spectral decomposition problem of the estimated covariance
   function of the field H.

 - calculating the coefficients of the Karhunen-Loeve expansion truncated
   to a given order depending on the decrease of the formerly computed
   eigenvalues. These coefficients are uncorrelated, zero mean and unit
   variance random variables.

Their joint distribution is then estimated (here, using kernel smoothing) and
new sample paths are eventually generated using the discretized random field
computed by the instanciated KarhunenLoeveExpansion.

The identification procedure consists in the following steps:

 1. Modelling of the sample paths by some interpolation technique.

 2. Estimation of the mean and covariance functions using the usual statistical
    estimators on the interpolated sample paths.

 3. Resolution of the integral eigenvalue problem of the covariance function
    (i.e. discretization of the random field into a Karhunen-Loeve expansion).

 4. Computation of the coefficients of the Karhunen-Loeve expansion by
    functional projection using Gauss-Legendre quadrature.

 5. Identification of the joint distribution of these coefficients.
    Here, this resorts to a kernel smoothing non-parametric technique for the
    marginal distributions and a Gaussian copula is assumed for illustration.

 6. Generation of new sample paths.
"""

# Author: Marine Marcilhac <marcilhac@phimeca.com>
#         Vincent Dubourg <dubourg@phimeca.com>
# License: BSD

import numpy as np
import pylab as pl
from scipy.interpolate import interp1d
import openturns as ot
from randomfields import KarhunenLoeveExpansion, \
                         matrix_plot

# Name of the input file that contains the indexing variable and sample paths
# values
input_file_name = '1D_sample_paths.txt'

# Parameters (see KarhunenLoeveExpansion's docstring)
verbose = True
truncation_order = 10
galerkin_scheme = 'legendre'
legendre_galerkin_order = 20
legendre_quadrature_order = 41

# Load input data from txt file
x = np.loadtxt(input_file_name)[0]
sample_paths_values = np.loadtxt(input_file_name)[1:]
n_sample_paths, res = sample_paths_values.shape

# Rectangular domain definition
lower_bound = np.array([0.] * 1)
upper_bound = np.array([10.] * 1)

# Interpolation of the sample paths
sample_paths = [interp1d(x, sample_paths_values[i], kind='linear')
                for i in xrange(n_sample_paths)]

def estimated_mean(x):
    y = np.vstack([sample_paths[i](x) for i in xrange(n_sample_paths)])
    return np.mean(y, axis=0)

def estimated_covariance(xx):
    xx = np.atleast_2d(xx)
    y1 = np.vstack([sample_paths[i](xx[:, 0])
                    for i in xrange(n_sample_paths)])
    y2 = np.vstack([sample_paths[i](xx[:, 1])
                    for i in xrange(n_sample_paths)])
    cov = np.sum((y1 - y1.mean(axis=0)) * (y2 - y2.mean(axis=0)),
                 axis=0) / (n_sample_paths - 1.)
    return cov

# Discretization of the random field using Karhunen-Loeve expansion
# from its estimated theoretical moments
estimated_random_field = KarhunenLoeveExpansion(
                        estimated_mean,
                        estimated_covariance,
                        truncation_order,
                        [lower_bound, upper_bound],
                        domain_expand_factor=1.,
                        verbose=verbose,
                        galerkin_scheme=galerkin_scheme,
                        legendre_galerkin_order=legendre_galerkin_order,
                        legendre_quadrature_order=legendre_quadrature_order)
truncation_order = estimated_random_field._truncation_order

# Plot eigenvalues and eigenfunctions
pl.figure()
pl.title('Eigensolutions')
for i in xrange(truncation_order):
    pl.plot(x, estimated_random_field._eigenfunctions[i](x),
            label='$\lambda_{%d} = %.2f$' % (i,
            estimated_random_field._eigenvalues[i]))
pl.xlabel('$x$')
pl.ylabel('$\\varphi_i(x)$')
pl.grid()
pl.legend(loc='lower center', ncol=3)
pl.savefig('1D_identification_eigensolutions.png')
pl.close()

# Calculation of the KL coefficients by functional projection using
# Gauss-Legendre quadrature
xi = estimated_random_field.compute_coefficients(sample_paths)

# Statistical inference of the KL coefficients' distribution
kernel_smoothing = ot.KernelSmoothing(ot.Normal())
xi_marginal_distributions = ot.DistributionCollection(
                            [kernel_smoothing.build(xi[:, i][:, np.newaxis])
                             for i in xrange(truncation_order)])
try:
    xi_copula = ot.NormalCopulaFactory().build(xi)
except RuntimeError:
    print('ERR: The normal copula correlation matrix built from the given\n'
         + 'Spearman correlation matrix is not definite positive.\n'
         + 'This would require expert judgement on the correlation\n'
         + 'coefficients significance (using e.g. Spearman test).\n'
         + 'Assuming an independent copula in the sequel...')
    xi_copula = ot.IndependentCopula(truncation_order)
xi_estimated_distribution = ot.ComposedDistribution(xi_marginal_distributions,
                                                    xi_copula)

# Matrix plot of the empirical KL coefficients & their estimated distribution
matrix_plot(xi, ot_distribution=xi_estimated_distribution,
            labels=[('$\\xi_{%d}$' % i) for i in xrange(truncation_order)])
pl.suptitle('Karhunen-Loeve coefficients '
          + '(observations and estimated distribution)')
pl.savefig('1D_identification_KL_coefficients_joint_distribution.png')
pl.close()

# Plot the ten first observed sample paths reconstructed from the estimated
# random field and an adequation plot with respect to the original observed
# sample paths
reconstructed_sample_paths_values = estimated_random_field(x[:, np.newaxis],
                                                           xi[:10])
pl.figure()
pl.title('Adequation: observation vs. model')
for i in xrange(10):
    pl.plot(sample_paths_values[i], reconstructed_sample_paths_values[i], '.',
            label='$h^{(%d)}(x)$' % i)
pl.plot([reconstructed_sample_paths_values.min(),
         reconstructed_sample_paths_values.max()],
        [reconstructed_sample_paths_values.min(),
         reconstructed_sample_paths_values.max()], 'k--', lw=1.5)
for lim in [pl.xlim, pl.ylim]:
    lim(reconstructed_sample_paths_values.min(),
        reconstructed_sample_paths_values.max())
pl.axis('equal')
pl.xlabel('$H(x)$')
pl.ylabel('$\hat{H}(x)$')
pl.grid()
pl.legend(loc='lower right', ncol=2)
pl.savefig('1D_identification_adequation_plot.png')
pl.close()
