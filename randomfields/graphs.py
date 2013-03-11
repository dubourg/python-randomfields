#!/usr/bin/env python
#-*- coding: utf-8 -*-

import pylab as pl
import numpy as np
import openturns as ot


def matrix_plot(X, ot_distribution=None, ot_kernel=None,
                labels=None, res=1000, grid=False):
    """
    Return a handle to a matplotlib figure containing a 'matrix plot'
    representation of the sample in X. It plots:
       - the marginal distributions on the diagonal terms,
       - the dependograms on the lower terms,
       - scatter plots on the upper terms.
    One may also add representation of the original distribution provided it
    is known, and/or a kernel smoothing (based on OpenTURNS).

    Parameters
    ----------
    X: array_like
        The sample to plot with shape (n_samples, n_features).
    ot_distribution: OpenTURNS Distribution of dimension n_features, optional.
        The underlying multivariate distribution if known.
        Default is set to None.
    ot_kernel: A list of n_features OpenTURNS KernelSmoothing's ready for
        build, optional.
        Kernel smoothing for the margins.
        Default is set to None.
    labels: A list of n_features strings, optional.
        Variates' names for labelling X & Y axes.
        Default is set to None.
    res: int, optional.
        Number of points used for plotting the marginal PDFs.
        Default is set to 1000.
    grid: bool, optional.
        Whether a grid should be added or not.
        Default is set to False (no grid).

    Returns
    -------
    ax: matplotlib.Axes instance.
        A handle to the matplotlib figure.

    Example
    -------
    >>> import pylab as pl
    >>> from phimeca.graphs import plot_matrix
    >>> import openturns as ot
    >>> probabilistic_model = ot.Normal(3)
    >>> sample = probabilistic_model.getSample(100)
    >>> ax = plot_matrix(sample,
                         ot_distribution=X,
                         ot_kernel=[ot.KernelSmoothing(ot.Epanechnikov())] * 3,
                         labels=[('$X_%d$' % i) for i in xrange(3)],
                         grid=True)
    >>> pl.show()
    """

    X = np.array(X)
    n_samples, n_features = X.shape
    if ot_distribution is None:
        ranks = np.array(ot.NumericalSample(X).rank())
    else:
        ranks = np.zeros_like(X)
        for i in xrange(n_features):
            ranks[:, i] = np.ravel(ot_distribution.getMarginal(i).computeCDF(
                            np.atleast_2d(X[:, i]).T))
            ranks[:, i] *= n_samples

    pl.figure(figsize=(8, 8))
    n = 0
    for i in xrange(n_features):
        for j in xrange(n_features):
            n += 1
            pl.subplot(n_features, n_features, n)
            if i == j:
                n_bins = int(1 + np.log2(n_samples)) + 1
                pl.hist(X[:, j], bins=n_bins, normed=True,
                        cumulative=False, bottom=None,
                        edgecolor='grey', color='grey', alpha=.25)
                if ot_distribution is not None:
                    Xi = ot_distribution.getMarginal(i)
                    a = Xi.getRange().getLowerBound()[0]
                    b = Xi.getRange().getUpperBound()[0]
                    middle = (a + b) / 2.
                    width = b - a
                    if Xi.computePDF(a - .1 * width / 2.) == 0.:
                        a = middle - 1.1 * width / 2.
                    if Xi.computePDF(b + .1 * width / 2.) == 0.:
                        b = middle + 1.1 * width / 2.
                    support = np.linspace(a, b, res)
                    pdf = Xi.computePDF(np.atleast_2d(support).T)
                    pl.plot(support, pdf, color='b', alpha=.5, lw=1.5)
                if ot_kernel is not None:
                    Xi = ot_kernel[i].build(np.atleast_2d(X[:, i]).T)
                    if ot_distribution is None:
                        a = Xi.getRange().getLowerBound()[0]
                        b = Xi.getRange().getUpperBound()[0]
                        support = np.linspace(a, b, res)
                    pdf = Xi.computePDF(np.atleast_2d(support).T)
                    pl.plot(support, pdf, color='r', alpha=.5, lw=1.5)
                pl.xticks([pl.xlim()[0], np.mean(pl.xlim()), pl.xlim()[1]])
                pl.yticks([])
            elif i < j:
                pl.plot(X[:, j], X[:, i],
                        'o', color='grey', alpha=0.25)
                pl.xticks([pl.xlim()[0], np.mean(pl.xlim()), pl.xlim()[1]],
                          ('', ) * 3)
                pl.yticks([pl.ylim()[0], np.mean(pl.ylim()), pl.ylim()[1]],
                          ('', ) * 3)
            else:
                pl.plot(ranks[:, j].astype(float) / n_samples,
                        ranks[:, i].astype(float) / n_samples,
                        'o', color='grey', alpha=0.25)
                pl.xticks([0., 1.])
                pl.yticks([0., 1.])

            if j == 0 and labels is not None:
                pl.ylabel(labels[i])

            if i == n_features - 1 and labels is not None:
                pl.xlabel(labels[j])

            if grid:
                pl.grid()

    return pl.gcf()
