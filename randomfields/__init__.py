#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
============
Randomfields
============

A module that implements tools for the simulation and identification of
random fields using the Karhunen-Loeve expansion representation.
"""

# Author: Marine Marcilhac <marcilhac@phimeca.com>
#         Vincent Dubourg <dubourg@phimeca.com>
# License: BSD
#   This module was implemented by Phimeca Engineering SA, EdF and Institut
#   Navier (ENPC). It is shipped as is without any warranty of any kind.

from .simulation import KarhunenLoeveExpansion
from .graphs import matrix_plot

__version__ = 0
