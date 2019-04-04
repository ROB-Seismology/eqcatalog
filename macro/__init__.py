"""
macro submodule
"""

from __future__ import absolute_import, division, print_function, unicode_literals


## Reloading mechanism
try:
	reloading
except NameError:
	## Module is imported for the first time
	reloading = False
else:
	## Module is reloaded
	reloading = True
	try:
		## Python 3
		from importlib import reload
	except ImportError:
		## Python 2
		pass



## Import submodules

## intensity (no internal dependencies)
if not reloading:
	from . import intensity
else:
	reload(intensity)
from .intensity import *

## macro_info (no internal dependencies)
if not reloading:
	from . import macro_info
else:
	reload(macro_info)
from .macro_info import *

## dyfi (no internal dependencies)
if not reloading:
	from . import dyfi
else:
	reload(dyfi)
from .dyfi import *

## imax (no internal dependencies)
if not reloading:
	from . import imax
else:
	reload(imax)
from .imax import *

## isoseismal (no internal dependencies)
if not reloading:
	from . import isoseismal
else:
	reload(isoseismal)
from .isoseismal import *
