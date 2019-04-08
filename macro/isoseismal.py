"""
Macroseismic isoseismals
"""

from __future__ import absolute_import, division, print_function#, unicode_literals

import os

from ..rob import seismodb, SEISMOGIS_ROOT

#GIS_FOLDER = os.path.join(SEISMOGIS_ROOT, "collections", "Bel_administrative_ROB", "TAB")
GIS_FOLDER = "D:\\GIS-data\\KSB-ORB\\HistoricalEQ"


__all__ = ["get_eq_isoseismals_file", "get_available_isoseismals",
			"read_isoseismals", "get_commune_intensities_from_isoseismals",
			"get_isoseismal_macro_info"]


EQID_DICT = {89: '1692',
			57: '1382',
			63: '1449',
			78: '1580',
			125: '1756',
			509: '1938',
			987: '1992'}


def get_eq_isoseismals_file(id_earth, filled=True):
	"""
	Get path to GIS file containing isoseismals for given earthquake

	:param id_earth:
		int or str, ID of earthquake in ROB database
	:param filled:
		bool, whether isoseismals are filled areas (True) or contours
		(False)
		(default: True)

	:return:
		str, full path to GIS file containing isoseismal geometries
	"""
	iso_id = EQID_DICT.get(id_earth)
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


def read_isoseismals(id_earth, filled=True):
	"""
	Read isoseismals for given earthquake

	:param id_earth:
	:param filled:
		see :func:`get_eq_isoseismals_file`

	:return:
		instance of :class:`layeredbasemap.MultiPolygonData`
	"""
	import mapping.layeredbasemap as lbm

	gis_file = get_eq_isoseismals_file(id_earth, filled=filled)
	if gis_file:
		gis_data = lbm.GisData(gis_file)
		_, _, pg_data = gis_data.get_data()
		return pg_data


def get_commune_intensities_from_isoseismals(id_earth, main_communes=False,
											as_points=True):
	"""
	Assign intensities to communes based on isoseismals

	:param id_earth:
		int, earthquake ID
	:param main_communes:
		bool, whether or not to use main communes
		(default: False)
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

	isoseismals = read_isoseismals(id_earth, filled=True)
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


def get_isoseismal_macro_info(id_earth, main_communes=False, as_points=True):
	"""
	Similar to :func:`get_commune_intensities_from_isoseismals`,
	but return instance of :class:`MacroInfoCollection`
	"""
	from .macro_info import MacroseismicInfo, MacroInfoCollection

	id_com_intensity_dict = get_commune_intensities_from_isoseismals(id_earth,
								main_communes=main_communes, as_points=as_points)

	comm_recs = seismodb.get_communes('BE', main_communes=main_communes)
	lons = [rec['longitude'] for rec in comm_recs]
	lats = [rec['latitude'] for rec in comm_recs]
	commune_ids = [rec['id'] for rec in comm_recs]

	macro_infos = []
	agg_type = {True: 'main commune', False: 'commune'}[main_communes]
	for id_com, intensity in id_com_intensity_dict.items():
		try:
			idx = commune_ids.index(id_com)
		except:
			lon, lat = None, None
		else:
			lon, lat = lons[idx], lats[idx]
		mi = MacroseismicInfo(id_earth, id_com, intensity, agg_type,
							'isoseismal', lon, lat)
		macro_infos.append(mi)

	proc_info = dict(as_points=as_points)
	macro_info_col = MacroInfoCollection(macro_infos, agg_type, 'isoseismal',
										proc_info=proc_info)
	return macro_info_col
