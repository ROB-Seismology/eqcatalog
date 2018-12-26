"""
ROB-specific submodule of eqcatalog
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

## completeness (depends on ..completeness)
if not reloading:
	from . import completeness
else:
	reload(completeness)
from .completeness import *

## source_models (no internal dependencies)
if not reloading:
	from . import source_models
else:
	reload(source_models)
from .source_models import (read_source_model,
	rob_source_models_dict as source_models_dict)

## seismodb (depends on ..eqrecord, ..macrorecord, ..eqcatalog)
if not reloading:
	from . import seismodb
else:
	reload(seismodb)
