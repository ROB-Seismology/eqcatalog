"""
ROB source model definitions
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import os
from collections import OrderedDict

# TODO: add read function?

class SourceModelDefinition:
	def __init__(self, name, gis_filespec, column_map):
		self.name = name
		self.gis_filespec = gis_filespec
		self.column_map = column_map

	def __getitem__(self, key):
		return getattr(self, key)


def read_source_model(gis_filespec, ID_colname="", fix_mi_lambert=True,
					verbose=True):
	"""
	Read source-zone model stored in a GIS (MapInfo) table.

	:param gis_filespec:
		String, full path to GIS file containing seismic sources
	:param ID_colname:
		String, name of GIS column containing record ID
		(default: "")
	:param fix_mi_lambert:
		bool, whether or not to apply spatial reference system fix for
		old MapInfo files in Lambert 1972 system
		(default: True)
	:param verbose:
		Boolean, whether or not to print information while reading
		GIS table (default: True)

	:return:
		ordered dict {String sourceID: dict {String column_name: value}}
		Note: special key 'obj' contains instance of :class:`osgeo.ogr.Geometry`}
	"""
	from mapping.geotools.read_gis import read_gis_file

	zone_records = read_gis_file(gis_filespec, verbose=verbose, encoding=None)
	if ID_colname:
		zone_ids = [rec[ID_colname] for rec in zone_records]
	else:
		zone_ids = range(1, len(zone_records)+1)

	zone_data = OrderedDict()
	for id, rec in zip(zone_ids, zone_records):
		if rec["obj"].GetGeometryName() == "POLYGON":
			rec["obj"].CloseRings()
		zone_data[id] = rec

	return zone_data
