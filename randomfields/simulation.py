#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.core.umath_tests import inner1d
from .utils import get_available_memory
from scipy import linalg
import openturns as ot


class KarhunenLoeveExpansion:
    """
    A class that implements the Karhunen-Loeve expansion (KLE) representation
    of second-order random fields with arbitrary mean and covariance functions
    over a rectangular domain.

    The Karhunen-Loeve expansion of a random field H reads:

    .. math::

        \hat{H}(x) = \mu(x) + \sum_i {\sqrt(\lambda_i) \Xi_i \phi_i(x)}

    where:
        - x is the indexing variable which spans a given rectangular domain.
        - \mu is the given mean of the random field.
        - \lambda_i and \phi_i are solutions of the spectral decomposition of
          the given autocovariance function C of the random field.
        - \Xi_i are uncorrelated, zero-mean and unit-variance random variables
          whose distribution depends on the desired random field distribution.

    NB: If the random field is assumed to be Gaussian, then the \Xi
    random vector is a standard Gaussian random vector. Otherwise, the
    user may use any distribution matching the requirements enunced above.

    Parameters
    ----------

    mean : callable
        A mean function for the random field. The function is assumed to take a
        unique input X in array_like format with shape (size, dimension).

    covariance : callable
        An autocovariance function for the random field. The function is
        assumed to take an input XX in array_like format with shape
        (size, 2 * dimension).

    truncation_order : integer
        The number of terms to keep in the truncated KL expansion.

    domain : array_like with shape (2, dimension)
        The boundaries of the rectangular domain on which the random field is
        defined [lower_bound, upper_bound].

    domain_expand_factor : double >= 1., optional
        A factor used for expanding the domain at the discretization step.
        Indeed, the relative mean squared error usually gets large at the
        border. Augmenting the size of the domain prevents this error from
        being large in the domain of interest.
        Default does not alter the domain (domain_expand_factor = 1.).

    verbose : boolean, optional
        A boolean specifying the verbose level.
        Default assumes verbose = False.

    galerkin_scheme : string, optional
        A string specifying the basis that should be used in the Galerkin
        discretization scheme of the Fredholm integral equation.
        Default uses Legendre polynomial basis (galerkin_scheme='legendre').
        TODO: Implement 'haar_wavelet' based Galekin scheme.

    Depending on the chosen Galerkin scheme, other optional parameters might be
    specified:

      - For the Legendre Galerkin scheme (galerkin_scheme='legendre'):

        legendre_galerkin_order : integer, optional
            An integer specifying the maximum order of the tensorized Legendre
            polynomials used for approximating eigenfunctions.
            Default uses tensorized Legendre polynoms of maximal order 10.

        legendre_quadrature_order : integer, optional
            An integer specifying the required quadrature order for estimating
            the integrals using Gauss-Legendre quadrature.
            Default uses 2 * legendre_galerkin_order + 1.

    Returns
    -------

    discretized_random_field : callable
        The discretized random field taking two inputs X and xi:
          - x : array_like with shape (n_points, dimension)
                The points on which the random field should be evaluated.

          - xi : array_like with shape (n_samples, truncation_order)
                The Karhunen-Loeve coefficients. The truncation_order must
                match the one specified at instanciation.
        The sample paths are returned as an array with shape
        (n_samples, n_points).

    See also
    --------

    self.__call__?
    self.compute_coefficients?
    self.compute_approximated_covariance?
    """

    _implemented_galerkin_schemes = ['legendre']

    def __init__(self, mean, covariance, truncation_order, domain,
                 domain_expand_factor=1., verbose=False,
                 galerkin_scheme='legendre', **kwargs):

        # Input checks and storage
        if not callable(mean):
            raise ValueError('mean must be a callable function.')
        else:
            self._mean = mean

        if not callable(covariance):
            raise ValueError('covariance must be a callable function.')
        else:
            self._covariance = covariance

        domain = np.atleast_2d(domain)
        if domain.shape[0] != 2:
            raise ValueError('domain must contain exactly 2 rows')
        else:
            self._lower_bound = domain[0]
            self._upper_bound = domain[1]

        if truncation_order <= 0:
            raise ValueError('truncation_order must be a positive integer.')
        else:
            self._truncation_order = truncation_order

        if type(verbose) is not bool:
            raise ValueError('verbose should be of type bool.')
        else:
            self.verbose = verbose

        if galerkin_scheme not in self._implemented_galerkin_schemes:
            raise ValueError('The Galerkin scheme should be selected amongst'
            + ' the implemented Galerkin schemes: %s.'
            % self._implemented_galerkin_schemes
            + ' Got %s instead.' % galerkin_scheme)
        else:
            self._galerkin_scheme = galerkin_scheme

        # Expand the domain for reducing the discretization error at the
        # borders
        center = (self._lower_bound + self._upper_bound) / 2.
        width = self._upper_bound - self._lower_bound
        self._lower_bound = center - width * domain_expand_factor / 2.
        self._upper_bound = center + width * domain_expand_factor / 2.

        # Discretize the random field using the chosen Galerkin scheme
        if self._galerkin_scheme == 'legendre':
            self._legendre_galerkin_scheme(**kwargs)

    def __call__(self, x, xi):
        """
        Calculates sample paths values of the discretized random field.

        Parameters
        ----------

        x: array_like with shape (n_index_values, dimension)
            The index values at which the sample path(s) should be calculated.

        xi: array_like with shape (n_sample_paths, truncation_order)
            The sample of Karhunen-Loeve coefficients values whose distribution
            depends on that of the random field (e.g. for a Gaussian random
            field, Xi is a standard Gaussian random vector).

        Returns
        -------

        sample_paths_values: array with shape (n_sample_paths, n_index_values)
            The required sample paths values.
        """

        # Input checks
        dimension = self._lower_bound.size
        truncation_order = self._truncation_order
        x = np.atleast_2d(x)
        xi = np.atleast_2d(xi)
        if x.shape[1] != dimension:
            raise ValueError('The number of columns in x must equal the '
                               + 'dimension of the random field which is %d.'
                               % dimension)
        if xi.shape[1] != truncation_order:
            raise ValueError('The number of columns in xi must equal the '
                               + 'truncation order of the random field which '
                               + 'is %d.' % truncation_order)

        PHI = np.vstack([np.sqrt(self._eigenvalues[k])
                         * self._eigenfunctions[k](x)
                         for k in xrange(self._truncation_order)])
        sample_paths_values = np.ravel(self._mean(x)) + np.dot(xi, PHI)

        return sample_paths_values

    def compute_approximated_covariance(self, xx):
        """
        Calculates the values of the covariance function of the discretized
        random field.

        Parameters
        ----------

        xx: array_like with shape (n_index_values, 2 * dimension)
            The couples of index values at which the covariance should be
            calculated.

        Returns
        -------

        covariance_values: array with shape (n_index_values, )
            The required sample paths values.
        """

        # Input checks
        dimension = self._lower_bound.size
        xx = np.atleast_2d(xx)
        if xx.shape[1] != 2 * dimension:
            raise ValueError('The number of columns in xx must be %d.'
                               % (2 * dimension))

        x1, x2 = xx[:, :dimension], xx[:, dimension:]
        PHI1 = np.vstack([np.sqrt(self._eigenvalues[k])
                          * self._eigenfunctions[k](x1)
                          for k in xrange(self._truncation_order)])
        PHI2 = np.vstack([np.sqrt(self._eigenvalues[k])
                          * self._eigenfunctions[k](x2)
                          for k in xrange(self._truncation_order)])
        covariance_values = inner1d(PHI1.T, PHI2.T)

        return covariance_values

    def compute_coefficients(self, sample_paths, **kwargs):
        """
        Calculates the coefficients in the Karhunen-Loeve expansion by
        projection of the observed sample path(s) onto the eigenfunctions.
        This uses the appropriate method depending on the chosen Galerkin
        scheme.

        Parameters
        ----------

        sample_paths: callable or collection of callables
            The observed sample path(s).

        Depending on the chosen Galerkin scheme, other optional parameters
        might be specified:

          - For the Legendre Galerkin scheme (galerkin_scheme='legendre'):

            legendre_quadrature_order : integer, optional
                An integer specifying the required quadrature order for
                estimating the integrals using Gauss-Legendre quadrature.
                Default uses the same order as the one used for discretizing
                the random field.

        Returns
        -------

        xi : array with shape (n_sample_paths, truncation_order)
            The Karhunen-Loeve coefficients associated to the given
            sample path(s).
        """

        # Input checks
        if not(callable(sample_paths) or
                hasattr(sample_paths, '__getitem__')):
            raise ValueError('sample_paths must be callable or a '
            + 'collection of callables.')

        if hasattr(sample_paths, '__getitem__'):
            if not callable(sample_paths[0]):
                raise ValueError('sample_paths must be callable or a '
                + 'collection of callables.')
        else:
            sample_paths = [sample_paths]

        # Compute the coefficients using the chosen Galerkin scheme
        if self._galerkin_scheme == 'legendre':
            return self._compute_coefficients_legendre(sample_paths, **kwargs)

    def _legendre_galerkin_scheme(self,
                                  legendre_galerkin_order=10,
                                  legendre_quadrature_order=None):

        # Input checks
        if legendre_galerkin_order <= 0:
            raise ValueError('legendre_galerkin_order must be a positive '
            + 'integer!')

        if legendre_quadrature_order is not None:
            if legendre_quadrature_order <= 0:
                raise ValueError('legendre_quadrature_order must be a '
                + 'positive integer!')

        # Settings
        dimension = self._lower_bound.size
        truncation_order = self._truncation_order
        galerkin_size = ot.EnumerateFunction(
            dimension).getStrataCumulatedCardinal(legendre_galerkin_order)
        if legendre_quadrature_order is None:
            legendre_quadrature_order = 2 * legendre_galerkin_order + 1

        # Check if the current settings are compatible
        if truncation_order > galerkin_size:
            raise ValueError('The truncation order must be less than or '
            + 'equal to the size of the functional basis in the chosen '
            + 'Legendre Galerkin scheme. Current size of the galerkin basis '
            + 'only allows to get %d terms in the KL expansion.'
            % galerkin_size)

        # Construction of the Galerkin basis: tensorized Legendre polynomials
        tensorized_legendre_polynomial_factory = \
            ot.PolynomialFamilyCollection([ot.LegendreFactory()] * dimension)
        tensorized_legendre_polynomial_factory = \
            ot.OrthogonalProductPolynomialFactory(
                tensorized_legendre_polynomial_factory)
        tensorized_legendre_polynomials = \
            [tensorized_legendre_polynomial_factory.build(i)
             for i in xrange(galerkin_size)]

        # Compute matrix C coefficients using Gauss-Legendre quadrature
        polyColl = ot.PolynomialFamilyCollection(
            [ot.LegendreFactory()] * dimension * 2)
        polynoms = ot.OrthogonalProductPolynomialFactory(polyColl)
        W = ot.NumericalPoint(1, 0.)
        U = polynoms.getNodesAndWeights(
            ot.Indices([legendre_quadrature_order] * dimension * 2), W)
        W = np.ravel(W)
        scale = (self._upper_bound - self._lower_bound) / 2.
        shift = (self._upper_bound + self._lower_bound) / 2.
        U = np.array(U)
        X = np.repeat(scale, 2) * U + np.repeat(shift, 2)

        if self.verbose:
            print 'Computing matrix C...'

        try:
            available_memory = int(.9 * get_available_memory())
        except:
            if self.verbose:
                print('WRN: Available memory estimation failed! '
                       'Assuming 1Gb is available (first guess).')
            available_memory = 1024 ** 3
        max_size = int(available_memory / 8 / galerkin_size ** 2)
        batch_size = min(W.size, max_size)
        if self.verbose and batch_size < W.size:
            print 'RAM: %d Mb available' % (available_memory / 1024 ** 2)
            print 'RAM: %d allocable terms / %d total terms' % (max_size,
                                                                 W.size)
            print 'RAM: %d loops required' % np.ceil(float(W.size) / max_size)
        while True:
            C = np.zeros((galerkin_size, galerkin_size))
            try:
                n_done = 0
                while n_done < W.size:
                    covariance_at_X = self._covariance(
                        X[n_done:(n_done + batch_size)])
                    H1 = np.vstack([np.ravel(
                        tensorized_legendre_polynomials[i](
                        U[n_done:(n_done + batch_size), :dimension]))
                        for i in xrange(galerkin_size)])
                    H2 = np.vstack([np.ravel(
                        tensorized_legendre_polynomials[i](
                        U[n_done:(n_done + batch_size), dimension:]))
                        for i in xrange(galerkin_size)])
                    C += np.sum(W[np.newaxis, np.newaxis,
                                    n_done:(n_done + batch_size)]
                                * covariance_at_X[np.newaxis, np.newaxis, :]
                                * H1[np.newaxis, :, :]
                                * H2[:, np.newaxis, :], axis=-1)
                    del covariance_at_X, H1, H2
                    n_done += batch_size
                break
            except MemoryError:
                batch_size /= 2
        C *= np.prod(self._upper_bound - self._lower_bound) ** 2.

        # Matrix B is orthonormal up to some constant
        B = np.diag(np.repeat(np.prod(self._upper_bound - self._lower_bound),
                              galerkin_size))

        # Solve the generalized eigenvalue problem C D = L B D in L, D
        if self.verbose:
            print 'Solving generalized eigenvalue problem...'
        eigenvalues, eigenvectors = linalg.eigh(C, b=B, lower=True)
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real

        # Sort the eigensolutions in the descending order of eigenvalues
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Truncate the expansion
        eigenvalues = eigenvalues[:truncation_order]
        eigenvectors = eigenvectors[:, :truncation_order]

        # Eliminate unsignificant negative eigenvalues
        if eigenvalues.min() <= 0.:
            if eigenvalues.min() > .01 * eigenvalues.max():
                raise Exception('The smallest significant eigenvalue seems '
                + 'to be negative... Check the positive definiteness of the '
                + 'covariance function.')
            else:
                truncation_order = np.nonzero(eigenvalues <= 0)[0][0]
                eigenvalues = eigenvalues[:truncation_order]
                eigenvectors = eigenvectors[:, :truncation_order]
                self._truncation_order = truncation_order
                print 'WRN: truncation_order was too large.'
                print 'It has been reset to: %d' % truncation_order

        # Define eigenfunctions
        class LegendrePolynomialsBasedEigenFunction():
            def __init__(self, vector):
                self._vector = vector

            def __call__(self, x):
                x = np.asanyarray(x)
                if x.ndim <= 1:
                    x = np.atleast_2d(x).T
                u = (x - shift) / scale
                return np.sum([np.ravel(tensorized_legendre_polynomials[i](u))
                                * self._vector[i]
                                for i in xrange(truncation_order)], axis=0)

        # Set attributes
        self._eigenvalues = eigenvalues
        self._eigenfunctions = [LegendrePolynomialsBasedEigenFunction(vector)
                                for vector in eigenvectors.T]
        self._legendre_galerkin_order = legendre_galerkin_order
        self._legendre_quadrature_order = legendre_quadrature_order

    def _compute_coefficients_legendre(self, sample_paths,
                                       legendre_quadrature_order=None):

        dimension = self._lower_bound.size
        truncation_order = self._truncation_order
        if legendre_quadrature_order is None:
            legendre_quadrature_order = self._legendre_quadrature_order
        elif type(legendre_quadrature_order) is not int \
            or legendre_quadrature_order <= 0:
            raise ValueError('legendre_quadrature_order must be a positive '
            + 'integer.')
        n_sample_paths = len(sample_paths)

        # Gauss-Legendre quadrature nodes and weights
        polyColl = ot.PolynomialFamilyCollection(
            [ot.LegendreFactory()] * dimension)
        polynoms = ot.OrthogonalProductPolynomialFactory(polyColl)
        W = ot.NumericalPoint(1, 0.)
        U = polynoms.getNodesAndWeights(
            ot.Indices([legendre_quadrature_order] * dimension), W)
        W = np.ravel(W)
        U = np.array(U)
        scale = (self._upper_bound - self._lower_bound) / 2.
        shift = (self._upper_bound + self._lower_bound) / 2.
        X = scale * U + shift

        # Compute coefficients
        try:
            available_memory = int(.9 * get_available_memory())
        except:
            if self.verbose:
                print('WRN: Available memory estimation failed! '
                       'Assuming 1Gb is available (first guess).')
            available_memory = 1024 ** 3
        max_size = int(available_memory
                       / 8 / truncation_order / n_sample_paths)
        batch_size = min(W.size, max_size)
        if self.verbose and batch_size < W.size:
            print 'RAM: %d Mb available' % (available_memory / 1024 ** 2)
            print 'RAM: %d allocable terms / %d total terms' % (max_size,
                                                                 W.size)
            print 'RAM: %d loops required' % np.ceil(float(W.size) / max_size)
        while True:
            coefficients = np.zeros((n_sample_paths, truncation_order))
            try:
                n_done = 0
                while n_done < W.size:
                    sample_paths_values = np.vstack([np.ravel(sample_paths[i](
                        X[n_done:(n_done + batch_size)]))
                        for i in xrange(n_sample_paths)])
                    mean_values = np.ravel(self._mean(
                        X[n_done:(n_done + batch_size)]))[np.newaxis, :]
                    centered_sample_paths_values = \
                        sample_paths_values - mean_values
                    del sample_paths_values, mean_values
                    eigenelements_values = np.vstack([self._eigenfunctions[k](
                        X[n_done:(n_done + batch_size)])
                        / np.sqrt(self._eigenvalues[k])
                        for k in xrange(truncation_order)])
                    coefficients += np.sum(
                        W[np.newaxis, np.newaxis, n_done:(n_done + batch_size)]
                        * centered_sample_paths_values[:, np.newaxis, :]
                        * eigenelements_values[np.newaxis, :, :], axis=-1)
                    del centered_sample_paths_values, eigenelements_values
                    n_done += batch_size
                break
            except MemoryError:
                batch_size /= 2
        coefficients *= np.prod(self._upper_bound - self._lower_bound)

        return coefficients
