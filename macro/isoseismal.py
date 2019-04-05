"""
Macroseismic isoseismals
"""

from __future__ import absolute_import, division, print_function#, unicode_literals

import os

from ..rob import seismodb, SEISMOGIS_ROOT

#GIS_FOLDER = os.path.join(SEISMOGIS_ROOT, "collections", "Bel_administrative_ROB", "TAB")
GIS_FOLDER = "D:\\GIS-data\\KSB-ORB\\HistoricalEQ"


__all__ = []


EQID_DICT = {89: '1692',
			57: '1382',
			63: '1449',
			78: '1580',
			125: '1756',
			509: '1938',
			987: '1992'}


def get_eq_isoseismals_file(eq_id, filled=True):
	"""
	Get path to GIS file containing isoseismals for given earthquake

	:param eq_id:
		int or str, ID of earthquake in ROB database
	:param filled:
		bool, whether isoseismals are filled areas (True) or contours
		(False)
		(default: True)

	:return:
		str, full path to GIS file containing isoseismal geometries
	"""
	iso_id = EQID_DICT.get(eq_id)
	gis_filespec = ''

	if iso_id:
		if filled:
			gis_filename = "Filled_Isoseismals_%s.TAB" % iso_id
		else:
			gis_filename = "%s_isoseismal.TAB" % iso_id
		gis_filespec = os.path.join(GIS_FOLDER, iso_id, gis_filename)

	return gis_filespec


def get_available_isoseismals():
	"""
	Fetch list of earthquake IDs for which isoseismals are available
	"""
	return EQID_DICT.keys()


def read_isoseismals(eq_id, filled=True):
	"""
	Read isoseismals for given earthquake

	:param eq_id:
	:param filled:
		see :func:`get_eq_isoseismals_file`

	:return:
		instance of :class:`layeredbasemap.MultiPolygonData`
	"""
	import mapping.layeredbasemap as lbm

	gis_file = get_eq_isoseismals_file(eq_id, filled=filled)
	gis_data = lbm.GisData(gis_file)
	_, _, pg_data = gis_data.get_data()
	return pg_data


def get_commune_intensities_from_isoseismals(eq_id, main_communes=False,
											as_points=True):
	"""
	Assign intensities to communes based on isoseismals

	:param eq_id:
	:param filled:
		see :func:`get_eq_isoseismals_file`
	:param as_points:
		bool, whether  points (True) or polygons (False) should
		be used for testing whether communes are within a particular
		isoseismal
		(default: True)

	:return:
		dict, mapping commune IDs to intensities
	"""
	import mapping.layeredbasemap as lbm
	from ..rob.communes import read_commune_polygons

	isoseismals = read_isoseismals(eq_id, filled=True)
	id_com_intensity_dict = {}

	if as_points:
		comm_recs = seismodb.get_communes('BE', main_communes=main_communes)
		lons = [rec['longitude'] for rec in comm_recs]
		lats = [rec['latitude'] for rec in comm_recs]
		commune_ids = [rec['id'] for rec in comm_recs]
		pt_data = lbm.MultiPointData(lons, lats)

		for isoseismal in isoseismals:
			intensity = isoseismal.value['Intensity']
			is_inside = isoseismal.contains(pt_data)
			for i in range(len(is_inside)):
				if is_inside[i]:
					id_com = commune_ids[i]
					id_com_intensity_dict[id_com] = intensity

	else:
		comm_dict = read_commune_polygons('BE', main_communes=main_communes)
		for id_com, pg_data in comm_dict.items():
			geom = pg_data.to_ogr_geom()

			for isoseismal in isoseismals:
				intensity = isoseismal.value['Intensity']
				overlap = isoseismal.get_overlap_ratio(geom)
				if overlap >= 0.5:
					id_com_intensity_dict[id_com] = intensity

	return id_com_intensity_dict
