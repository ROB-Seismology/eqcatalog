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


## Directories with MapInfo tables for named catalogs and source models
# TODO: should become seismo-gis folder
import platform
if platform.uname()[0] == "Windows":
	GIS_ROOT = "D:\\GIS-data"
else:
	GIS_ROOT = os.path.join(os.environ.get("HOME", ""), "gis-data")

from mapping.seismo_gis.seismo_gis import SEISMOGIS_ROOT


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

## eqrecord (depends on ..eqrecord)
if not reloading:
	from . import eqrecord
else:
	reload(eqrecord)
from .eqrecord import ROBLocalEarthquake

## seismodb (depends on ..eqrecord, ..macrorecord, ..eqcatalog)
if not reloading:
	from . import seismodb
else:
	reload(seismodb)
from .seismodb import *
