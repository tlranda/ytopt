"""
``ytopt`` is a machine learning-based autotuning software package that uses Bayesian Optimization
to find the best input parameter configurations for a given kernel, miniapp, or application.
"""

from ytopt.__version__ import __version__
name = 'ytopt'
version = __version__

from . import benchmark
from . import core
from . import evaluator
from . import problem
from . import search

