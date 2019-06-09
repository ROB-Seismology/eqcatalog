"""
Plotting submodule of eqcatalog
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

## plot_generic
if not reloading:
  from . import plot_generic
else:
  reload(plot_generic)

## plot_catalog (depends on plot_generic)
if not reloading:
  from . import plot_catalog
else:
  reload(plot_catalog)
from .plot_catalog import *

## plot_macro_map
if not reloading:
	from . import plot_macro_map
else:
	reload(plot_macro_map)
from .plot_macro_map import *
