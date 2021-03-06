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

## moment (no internal dependencies)
if not reloading:
	from . import moment
else:
	reload(moment)
from .moment import (moment_to_mag, mag_to_moment)

## earthquake_types (no internal dependencies)
if not reloading:
	from . import earthquake_types
else:
	reload(earthquake_types)
from .earthquake_types import *

## time (no internal dependencies)
if not reloading:
	from . import time as time
else:
	reload(time)

## gen_id (depends on time)
if not reloading:
	from . import gen_id as gen_id
else:
	reload(gen_id)

## completeness (depends on time)
if not reloading:
	from . import completeness
else:
	reload(completeness)
from .completeness import *

## calcGR (depends on time)
if not reloading:
	from . import calcGR
else:
	reload(calcGR)

## calcGR_MLE (depends on completeness)
if not reloading:
	from . import calcGR_MLE
else:
	reload(calcGR_MLE)

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

## omori (no internal dependencies)
if not reloading:
	from . import omori
else:
	reload(omori)

## source_models (no internal dependencies)
if not reloading:
	from . import source_models
else:
	reload(source_models)
#from .source_models import (read_source_model, rob_source_models_dict)

## rob (depends on completeness, indirectly on eqrecord, eqcatalog, macrorecord)
if not reloading:
	from . import rob
else:
	reload(rob)

## harvard_cmt (depends on moment, rob)
if not reloading:
	from . import harvard_cmt
else:
	reload(harvard_cmt)

## macro (dependends on rob)
if not reloading:
	from . import macro
else:
	reload(macro)

## eqrecord (depends on time, msc)
if not reloading:
	from . import eqrecord
else:
	reload(eqrecord)
from .eqrecord import (LocalEarthquake, FocMecRecord)

## eqcatalog (depends on time, completeness eqrecord, declustering, source_models)
if not reloading:
	from . import eqcatalog
else:
	reload(eqcatalog)
from .eqcatalog import (EQCatalog, concatenate_catalogs)

## merged_catalog (depends on eqcatalog)
if not reloading:
	from . import merged_catalog
else:
	reload(merged_catalog)
from .merged_catalog import *

## composite_catalog (depends on completeness, eqcatalog)
if not reloading:
	from . import composite_catalog
else:
	reload(composite_catalog)
from .composite_catalog import CompositeEQCatalog

## io submodule (depends on eqrecord, eqcatalog, time)
if not reloading:
	from . import io
else:
	reload(io)
from .io import (read_named_catalog, read_catalog_sql, read_catalog_csv,
				read_catalog_gis)

## plot submodule (depends on macro, rob)
if not reloading:
	from . import plot
else:
	reload(plot)
