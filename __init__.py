"""
eqcatalog

Module to process earthquake catalogs

Author: Kris Vanneste, Royal Observatory of Belgium
"""

from __future__ import absolute_import, division, print_function, unicode_literals



## Make relative imports work in Python 3
import importlib


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
	time_functions = importlib.import_module('.time_functions', package=__name__)
else:
	reload(time_functions)

## completeness (no internal dependencies)
if not reloading:
	completeness = importlib.import_module('.completeness', package=__name__)
else:
	reload(completeness)
from .completeness import *

## calcGR, depends on time_functions
if not reloading:
	calcGR = importlib.import_module('.calcGR', package=__name__)
else:
	reload(calcGR)

## msc (no internal dependencies)
if not reloading:
	msc = importlib.import_module('.msc', package=__name__)
else:
	reload(msc)

## declustering (no internal dependencies)
if not reloading:
	declustering = importlib.import_module('.declustering', package=__name__)
else:
	reload(declustering)

## source_models (no internal dependencies)
if not reloading:
	source_models = importlib.import_module('.source_models', package=__name__)
else:
	reload(source_models)
from .source_models import (read_source_model, rob_source_models_dict)

## macrorecord (no internal dependencies)
if not reloading:
	macrorecord = importlib.import_module('.macrorecord', package=__name__)
else:
	reload(macrorecord)
from .macrorecord import (MacroseismicInfo, MacroseismicEnquiryEnsemble,
						MacroseismicDataPoint, get_roman_intensity)

## eqrecord (depends on time_functions, msc)
if not reloading:
	eqrecord = importlib.import_module('.eqrecord', package=__name__)
else:
	reload(eqrecord)
from .eqrecord import (LocalEarthquake, FocMecRecord)

## eqcatalog (depends on time_functions, completeness eqrecord, declustering, source_models)
if not reloading:
	eqcatalog = importlib.import_module('.eqcatalog', package=__name__)
else:
	reload(eqcatalog)
from .eqcatalog import (EQCatalog, read_catalogSQL, read_catalogGIS,
	concatenate_catalogs, read_named_catalog, read_catalogTXT, plot_catalogs_map,
	get_catalogs_map, plot_catalogs_magnitude_time, plot_depth_statistics)

## composite_catalog (depends on completeness, eqcatalog)
if not reloading:
	composite_catalog = importlib.import_module('.composite_catalog', package=__name__)
else:
	reload(composite_catalog)
from .composite_catalog import CompositeEQCatalog

## seismodb (depends on eqrecord, macrorecord, eqcatalog)
if not reloading:
	seismodb = importlib.import_module('.seismodb', package=__name__)
else:
	reload(seismodb)
