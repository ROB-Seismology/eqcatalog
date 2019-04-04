"""
Interface to GIS data containing commune polygons
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os

from . import SEISMOGIS_ROOT

__all__ = ["get_communes_gis_file", "read_commune_polygons"]


GIS_FOLDER = os.path.join(SEISMOGIS_ROOT, "collections", "Bel_administrative_ROB", "TAB")


def get_communes_gis_file(country='BE', main_communes=False):
	"""
	Get path to GIS file containing commune polygons

	:param country:
		2-char string, country code.
		Only 'BE' is supported currently.

	:return:
		str, full path to GIS file
	"""
	country = country.upper()
	if country == 'BE':
		if main_communes == True:
			gis_filename = "Bel_villages_polygons_fixed.TAB"
		else:
			gis_filename = "Bel_communes_avant_fusion.TAB"
	else:
		raise NotImplementedError

	#gis_filespec = "http://seishaz.oma.be:8080/geoserver/rob/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=rob:bel_villages_polygons&outputFormat=application%2Fjson"

	gis_filespec = os.path.join(GIS_FOLDER, gis_filename)
	return gis_filespec


def read_commune_polygons(country='BE', main_communes=False):
	"""
	Read commune polygons

	:param country:
	:param main_communes:
		see :func:`get_communes_gis_file`

	:return:
		dict, mapping commune IDs to instances of
		:class:`layeredbasemap.PolygonData`
		or :class:`layeredbasemap.MultiPolygonData`
	"""
	import mapping.layeredbasemap as lbm

	gis_file = get_communes_gis_file(country, main_communes=main_communes)
	gis_data = lbm.GisData(gis_file)
	_, _, pg_data = gis_data.get_data()

	## Some communes consist of more than 1 polygon
	commune_dict = {}
	for pg in pg_data:
		id_com = pg.value['ID_ROB']
		if not id_com in commune_dict:
			commune_dict[id_com] = pg
		else:
			if isinstance(commune_dict[id_com], lbm.PolygonData):
				commune_dict[id_com] = commune_dict[id_com].to_multi_polygon()
			commune_dict[id_com].append(pg)

	return commune_dict
