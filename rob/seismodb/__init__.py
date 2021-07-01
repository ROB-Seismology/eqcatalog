"""
seismodb
Python module to retrieve information from the ROB seismology database
======================================================================
Author: Kris Vanneste, Royal Observatory of Belgium.
Date: Apr 2008. - 2021
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

## base (no dependencies)
if not reloading:
	from . import base
else:
	reload(base)
from .base import *

## historical (depends on base)
if not reloading:
	from . import historical
else:
	reload(historical)
from .historical import *

## local_eq (depends on base)
if not reloading:
	from . import local_eq
else:
	reload(local_eq)
from .local_eq import *

## focmec (depends on base)
if not reloading:
	from . import focmec
else:
	reload(focmec)
from .focmec import *

## communes (depends on base)
if not reloading:
	from . import communes
else:
	reload(communes)
from .communes import *

## phases (depends on base)
if not reloading:
	from . import phases
else:
	reload(phases)
from .phases import *

## stations (depends on base, local_eq)
if not reloading:
	from . import stations
else:
	reload(stations)
from .stations import *

## macro (depends on base, local_eq, communes)
if not reloading:
	from . import macro
else:
	reload(macro)
from .macro import *
