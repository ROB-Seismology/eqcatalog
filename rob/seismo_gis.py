"""
Interface to GIS data on seismogis
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os


try:
	from mapping.seismogis import SeismoGisCatalog as SEISMOGIS
except ImportError:
	SEISMOGIS = None



__all__ = ["SEISMOGIS", "get_dataset_file_on_seismogis",
			"get_communes_gis_file", "read_commune_polygons"]



def get_dataset_file_on_seismogis(collection_name, dataset_name, strict=False):
	"""
	Determine full path to dataset file on seismogis

	:param collection_name:
		str, name of collection the dataset belongs to
	:param dataset_name:
		str, name of dataset
		If name does not include an extension, the file corresponding
		to native format will be returned.
	:param strict:
		bool, whether or not name matching should be strict
		If False and more than 1 dataset matches the given pattern,
		the first result will be returned
		(default: False)

	:return:
		str, full path to GIS file containing dataset
	"""
	ds_file = ''
	dataset_name, format = os.path.splitext(dataset_name)
	format = format.replace('.', '').upper()
	if SEISMOGIS is not None:
		try:
			ds = SEISMOGIS.find_datasets(dataset_name, collection_name,
											strict=strict)[0]
		except IndexError:
			print('%s not found in seismogis!' % dataset_name)
		else:
			ds_file = ds.get_gis_filespec(format=format)
	else:
		print('Please install mapping.seismogis module!')

	return ds_file


def get_communes_gis_file(country='BE', main_communes=False):
	"""
	Get path to GIS file in seismogis containing commune polygons

	:param country:
		2-char string, country code.
		Only 'BE' is supported currently.

	:return:
		str, full path to GIS file
	"""
	country = country.upper()
	if country == 'BE':
		collection_name = 'Bel_administrative_ROB'
		if main_communes:
			gis_filespec = get_dataset_file_on_seismogis(collection_name,
													"Bel_villages_polygons")
		else:
			gis_filespec = get_dataset_file_on_seismogis(collection_name,
													"Bel_communes_avant_fusion")
	else:
		raise NotImplementedError

	#gis_filespec = "http://seishaz.oma.be:8080/geoserver/rob/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=rob:bel_villages_polygons&outputFormat=application%2Fjson"
	#gis_filespec = os.path.join(GIS_FOLDER, gis_filename)

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
