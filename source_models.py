import os
from collections import OrderedDict


## Directories with MapInfo tables for source models
GIS_root = r"D:\GIS-data"

GIS_areasource_directory = os.path.join(GIS_root, "KSB-ORB", "Source Zone Models")
GIS_faultsource_directory = os.path.join(GIS_root, "SHARE", "Task 3.2", "DATA")


## Dictionary with data for ROB source models
rob_source_models_dict = {}

## Seismotectonic
rob_source_model = {}
rob_source_model['name'] = 'Seismotectonic'
rob_source_model['tab_filespec'] = os.path.join(GIS_areasource_directory, 'seismotectonic zones 1.2.TAB')
rob_source_model['column_map'] = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'min_mag': 3.5,
	'max_mag': 'MS_max_evaluated',
	'upper_seismogenic_depth': 0.0,
	'lower_seismogenic_depth': 25.0,
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'dip': 45.0,
	'rake': 0.0,
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth'}
rob_source_models_dict[rob_source_model['name']] = rob_source_model

## TwoZone
rob_source_model = {}
rob_source_model['name'] = 'TwoZone'
rob_source_model['tab_filespec'] = os.path.join(GIS_areasource_directory, 'SLZ+RVG.TAB')
rob_source_model['column_map'] = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'min_mag': 3.5,
	'max_mag': 'MS_max_evaluated',
	'upper_seismogenic_depth': 0.0,
	'lower_seismogenic_depth': 25.0,
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'dip': 45.0,
	'rake': 0.0,
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth'}
rob_source_models_dict[rob_source_model['name']] = rob_source_model

## TwoZone_split
rob_source_model = {}
rob_source_model['name'] = 'TwoZone_split'
rob_source_model['tab_filespec'] = os.path.join(GIS_areasource_directory, 'SLZ+RVG_split.TAB')
rob_source_model['column_map'] = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'min_mag': 3.5,
	'max_mag': 'MS_max_evaluated',
	'upper_seismogenic_depth': 0.0,
	'lower_seismogenic_depth': 25.0,
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'dip': 45.0,
	'rake': 0.0,
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth'}
rob_source_models_dict[rob_source_model['name']] = rob_source_model

## Leynaud
rob_source_model = {}
rob_source_model['name'] = 'Leynaud'
rob_source_model['tab_filespec'] = os.path.join(GIS_areasource_directory, 'ROB Seismic Source Model (Leynaud, 2000).TAB')
rob_source_model['column_map'] = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'a',
	'b_val': 'b',
	'min_mag': 3.5,
	'max_mag': 'MS_max_evaluated',
	'upper_seismogenic_depth': 0.0,
	'lower_seismogenic_depth': 25.0,
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'dip': 45.0,
	'rake': 0.0,
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth'}
rob_source_models_dict[rob_source_model['name']] = rob_source_model

## Leynaud_updated
rob_source_model = {}
rob_source_model['name'] = 'Leynaud_updated'
rob_source_model['tab_filespec'] = os.path.join(GIS_areasource_directory, 'Leynaud updated.TAB')
rob_source_model['column_map'] = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLEfix',
	'b_val': 'bMLEfix',
	'min_mag': 3.5,
	'max_mag': 'MS_max_evaluated',
	'upper_seismogenic_depth': 0.0,
	'lower_seismogenic_depth': 25.0,
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'dip': 45.0,
	'rake': 0.0,
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth'}
rob_source_models_dict[rob_source_model['name']] = rob_source_model

## RVRS_area
rob_source_model = {}
rob_source_model['name'] = 'RVRS_area'
rob_source_model['tab_filespec'] = os.path.join(GIS_areasource_directory, 'RVRS_area.TAB')
rob_source_model['column_map'] = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type': 'Active Shallow Crust',
	'a_val': 2.4,
	'b_val': 0.9,
	'min_mag': 3.5,
	'max_mag': 'MS_max_evaluated',
	'upper_seismogenic_depth': 0.0,
	'lower_seismogenic_depth': 25.0,
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'dip': 55.0,
	'rake': -90.0,
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth'}
rob_source_models_dict[rob_source_model['name']] = rob_source_model

## RVRS_CSS
rob_source_model = {}
rob_source_model['name'] = 'RVRS_CSS'
rob_source_model['tab_filespec'] = os.path.join(GIS_faultsource_directory, 'CSS_SurfaceTrace.TAB')
rob_source_model['column_map'] = {
	'id': 'IDSource',
	'name': 'SourceName',
	'tectonic_region_type': 'Active Shallow Crust',
	'a_val': 2.4,
	'b_val': 0.9,
	'min_mag': 3.5,
	'max_mag': 'MaxMag',
	'upper_seismogenic_depth': 'MinDepth',
	'lower_seismogenic_depth': 'MaxDepth',
	'dip': 'Dip',
	'rake': 'Rake',
	'slip_rate': 'SlipRate'}
rob_source_models_dict[rob_source_model['name']] = rob_source_model

## RVRS SHARE AS model
rob_source_model = {}
rob_source_model['name'] = 'RVRS_SHARE_AS'
rob_source_model['tab_filespec'] = os.path.join(GIS_areasource_directory, 'RVRS_SHARE_v4alpha.TAB')
rob_source_model['column_map'] = {
	'id': 'Idas',
	'name': 'Idas',
	'tectonic_region_type': 'Active Shallow Crust',
	'a_val': 'A',
	'b_val': 'B',
	'min_mag': 3.5,
	'max_mag': 'Maxmag03',
	'upper_seismogenic_depth': 0.0,
	'lower_seismogenic_depth': 25.0,
	'min_strike': 135,
	'max_strike': 315,
	'dip': 55.0,
	'rake': -90.0,
	'min_hypo_depth': 'Mindepth',
	'max_hypo_depth': 'Maxdepth'}
rob_source_models_dict[rob_source_model['name']] = rob_source_model



def read_source_model(source_model_name, verbose=True):
	"""
	Read source-zone model stored in a GIS (MapInfo) table.

	:param source_model_name:
		String, name of source-zone model containing area sources
	:param verbose:
		Boolean, whether or not to print information while reading
		GIS table (default: True)

	:return:
		ordered dict {String sourceID: instande of :class:`osgeo.ogr.Geometry`}
	"""
	from mapping.geo.readGIS import read_GIS_file

	## Read zone model from MapInfo file
	#source_model_table = ZoneModelTables[source_model_name.lower()]
	#tab_filespec = os.path.join(GIS_root, "KSB-ORB", "Source Zone Models", source_model_table + ".TAB")
	tab_filespec = rob_source_models_dict[source_model_name]["tab_filespec"]
	ID_colname = rob_source_models_dict[source_model_name]["column_map"]["id"]

	zone_records = read_GIS_file(tab_filespec, verbose=verbose)
	zone_ids = [rec[ID_colname] for rec in zone_records]

	zone_data = OrderedDict()
	for id, rec in zip(zone_ids, zone_records):
		if rec["obj"].GetGeometryName() == "POLYGON":
			rec["obj"].CloseRings()
		zone_data[id] = rec

	return zone_data

