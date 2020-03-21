"""
Input/output of earthquake catalogs
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


__all__ = ["HY4", "read_named_catalog", "read_catalog_sql", "read_catalog_csv",
			"read_catalog_gis"]


## Import submodules

## HY4
if not reloading:
	from . import HY4
else:
	reload(HY4)

## read_catalog (depends on ..eqrecord, ..eqcatalog, ..time)
if not reloading:
	from . import read_catalog
else:
	reload(read_catalog)
from .read_catalog import *

## parse_php_vars
if not reloading:
	from . import parse_php_vars
else:
	reload(parse_php_vars)
