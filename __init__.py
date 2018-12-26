"""
eqcatalog

Module to process earthquake catalogs

Author: Kris Vanneste, Royal Observatory of Belgium
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

## time_functions (no internal dependencies)
if not reloading:
	from . import time_functions_np as time_functions
else:
	reload(time_functions)

## completeness (no internal dependencies)
if not reloading:
	from . import completeness
else:
	reload(completeness)
from .completeness import *

## calcGR, depends on time_functions
if not reloading:
	from . import calcGR
else:
	reload(calcGR)

## msc (no internal dependencies)
if not reloading:
	from . import msc
else:
	reload(msc)

## declustering (no internal dependencies)
if not reloading:
	from . import declustering
else:
	reload(declustering)

"""
## source_models (no internal dependencies)
if not reloading:
	from . import source_models
else:
	reload(source_models)
from .source_models import (read_source_model, rob_source_models_dict)
"""

## rob (depends on completeness, indirectly on eqrecord, eqcatalog, macrorecord)
if not reloading:
	from . import rob
else:
	reload(rob)

## macrorecord (no internal dependencies)
if not reloading:
	from . import macrorecord
else:
	reload(macrorecord)
from .macrorecord import (MacroseismicInfo, MacroseismicEnquiryEnsemble,
						MacroseismicDataPoint, get_roman_intensity)

## eqrecord (depends on time_functions, msc)
if not reloading:
	from . import eqrecord
else:
	reload(eqrecord)
from .eqrecord import (LocalEarthquake, FocMecRecord)

## eqcatalog (depends on time_functions, completeness eqrecord, declustering, source_models)
if not reloading:
	from . import eqcatalog
else:
	reload(eqcatalog)
from .eqcatalog import (EQCatalog, read_catalog_sql,
	read_catalog_gis, read_catalog_csv, read_named_catalog, plot_catalogs_map,
	concatenate_catalogs, get_catalogs_map, plot_catalogs_magnitude_time,
	plot_depth_statistics)

## composite_catalog (depends on completeness, eqcatalog)
if not reloading:
	from . import composite_catalog
else:
	reload(composite_catalog)
from .composite_catalog import CompositeEQCatalog

"""
## seismodb (depends on eqrecord, macrorecord, eqcatalog)
if not reloading:
	from . import seismodb
else:
	reload(seismodb)
"""
