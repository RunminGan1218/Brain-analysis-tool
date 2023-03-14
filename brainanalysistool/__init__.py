from __future__ import division, print_function, absolute_import

from .plotting import *
from .preprocessing import *
from .WaveAnalysisFun import *
from .WaveletFun import *


__all__ = [s for s in dir() if not s.startswith('_')]
try:
    # In Python 2.x the name of the tempvar leaks out of the list
    # comprehension.  Delete it to not make it show up in the main namespace.
    del s
except NameError:
    pass



__name__ = 'BrainAnalysisTool'
__version__ = '0.0.3'