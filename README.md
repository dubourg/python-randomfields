python-randomfields
===================

A Python module that implements tools for the simulation and identification of
random fields using the Karhunen-Loeve expansion representation.

Folder description
------------------

This folder contains:
    - a Python module named randomfields,
    - 4 Python scripts implementing basic examples, showing the ways the module
      functionalities can be used.
      NB: *_simulation_* scripts must be run before their corresponding
      *_identification_* counterpart because the simulation scripts generate
      input randomfield data (dumped to ascii file) for identification.

Requirements
------------

The present Python module and its examples rely on:
    - OpenTURNS (>= 1.1)
    - Numpy (>= 1.6)
    - Scipy (>= 0.9)
    - Matplotlib (>= 1.0)

Installation
------------

The example scripts can be run from this folder for testing. They'll import the
randomfields module from the local folder.

In order to make the randomfields module installation systemwide, you may
either:
    - copy the randomfields module (directory) in the "site-package" directory
      of your Python distribution (e.g. /usr/local/lib/python2.7/site-package).
      NB: You might need admin rights to do so.
    - append the parent directory of the randomfields module (directory) to
      your PYTHONPATH environment variable.

Documentation
-------------

The randomfields module uses Python docstrings. Use either "help(object)" in a
classic Python shell or "object?" in an improved Python (IPython) shell.

Authors and terms of use
------------------------

This module was implemented by Phimeca Engineering SA, EdF and Institut Navier
(ENPC). It is shipped as is without any warranty of any kind. Contributions are
welcome.

Todo list
---------

    - Implement other Galerkin schemes such as the Haar wavelet Galerkin scheme
      proposed by Phoon et al. (2002). More advanced (smoother) wavelets could
      also be used.
    - Call for data: if you have any, please contribute, possibly along with
      an identification example.

References
----------

    * Phoon, K.; Huang, S. & Quek, S.
      `Implementation of Karhunen-Loeve expansion for simulation using a
      wavelet-Galerkin scheme
      <http://www.eng.nus.edu.sg/civil/people/cvepkk/JPaper_2002_vol17.pdf>`_
      Prob. Eng. Mech., 2002, 17, 293-303

    * Ghanem, R. & Spanos, P.
      Stochastic Finite Elements: A Spectral Approach (Revised edition)
      Dover Publications Inc., 2003, 224

    * Sudret, B. & Der Kiureghian, A.
      `Stochastic Finite Element Methods and Reliability, A State-of-the-Art
      report
      <http://www.ibk.ethz.ch/su/publications/Reports/SFE-report-Sudret.pdf>`_
      University of California, Berkeley, 2000
