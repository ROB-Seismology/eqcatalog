import os
from collections import OrderedDict


## Directories with MapInfo tables for source models
GIS_root = r"D:\GIS-data"

GIS_areasource_directory = os.path.join(GIS_root, "KSB-ORB", "Source Zone Models")
SHARE_directory = os.path.join(GIS_root, "SHARE")
GIS_faultsource_directory = os.path.join(SHARE_directory, "Task 3.2", "DATA")


class SourceModelDefinition:
	def __init__(self, name, gis_filename, gis_folder):
		self.name = name
		self.gis_filename = gis_filename
		self.gis_folder = gis_folder


class AreaSourceModelDefinition(SourceModelDefinition):
	pass


class FaultSourceModelDefinition(SourceModelDefinition):
	pass


class HybridSourceModelDefinition:
	pass


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
	'upper_seismogenic_depth': 'upper_rupture_depth',
	'lower_seismogenic_depth': 'lower_rupture_depth',
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth',
	'mean_hypo_depth': 'mean_hypo_depth',
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'min_dip': 'min_dip',
	'max_dip': 'max_dip',
	'Ss': 'Ss',
	'Nf': 'Nf',
	'Tf': 'Tf'}
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

## TwoZonev2
rob_source_model = {}
rob_source_model['name'] = 'TwoZonev2'
rob_source_model['tab_filespec'] = os.path.join(GIS_areasource_directory, 'TwoZone_v2.TAB')
rob_source_model['column_map'] = {
	'id': 'ShortName',
	'name': 'Name',
	'area': 'Area',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'min_mag': 4.0,
	'max_mag': 'Mmax_evaluated',
	'upper_seismogenic_depth': 'upper_rupture_depth',
	'lower_seismogenic_depth': 'lower_rupture_depth',
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth',
	'mean_hypo_depth': 'mean_hypo_depth',
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'min_dip': 'min_dip',
	'max_dip': 'max_dip',
	'Ss': 'Ss',
	'Nf': 'Nf',
	'Tf': 'Tf'}
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

## Leynaud_extended
rob_source_model = {}
rob_source_model['name'] = 'Leynaud_extended'
rob_source_model['tab_filespec'] = os.path.join(GIS_areasource_directory, 'Leynaud extended.TAB')
rob_source_model['column_map'] = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'a',
	'b_val': 'b',
	'min_mag': 4.0,
	'max_mag': 'Mmax_evaluated',
	'upper_seismogenic_depth': 'upper_rupture_depth',
	'lower_seismogenic_depth': 'lower_rupture_depth',
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth',
	'mean_hypo_depth': 'mean_hypo_depth',
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'min_dip': 'min_dip',
	'max_dip': 'max_dip',
	'Ss': 'Ss',
	'Nf': 'Nf',
	'Tf': 'Tf'}
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
rob_source_model['tab_filespec'] = os.path.join(GIS_areasource_directory, 'RVRS_area_v2.TAB')
rob_source_model['column_map'] = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type': 'SHARE_TRT',
	'a_val': 2.4,
	'b_val': 0.9,
	'min_mag': 3.5,
	'max_mag': 'Mmax_evaluated',
	'upper_seismogenic_depth': 'upper_rupture_depth',
	'lower_seismogenic_depth': 'lower_rupture_depth',
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth',
	'mean_hypo_depth': 'mean_hypo_depth',
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'min_dip': 'min_dip',
	'max_dip': 'max_dip',
	'Ss': 'Ss',
	'Nf': 'Nf',
	'Tf': 'Tf'}
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

## Seismotectonic Hybrid
rob_source_model = {}
rob_source_model['name'] = 'Seismotectonic_Hybrid'
rob_source_model['tab_filespec'] = os.path.join(GIS_faultsource_directory, 'Seismotectonic Hybrid.TAB')
rob_source_model['column_map'] = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type': 'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'min_mag': 4.0,
	'max_mag': 'Mmax_evaluated',
	'upper_seismogenic_depth': 'upper_rupture_depth',
	'lower_seismogenic_depth': 'lower_rupture_depth',
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth',
	'mean_hypo_depth': 'mean_hypo_depth',
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'min_dip': 'min_dip',
	'max_dip': 'max_dip',
	'Ss': 'Ss',
	'Nf': 'Nf',
	'Tf': 'Tf',
	'length': 'Length',
	'min_rake': 'min_rake',
	'max_rake': 'max_rake',
	'min_slip_rate': 'min_slip_rate',
	'max_slip_rate': 'max_slip_rate'}
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
	'min_mag': 4.0,
	'max_mag': 'Maxmag02',
	'upper_seismogenic_depth': 'Mindepth',
	'lower_seismogenic_depth': 'Maxdepth',
	'min_strike': 135,
	'max_strike': 315,
	'min_dip': 55.0,
	'max_dip': 55.0,
	'Ss': 30,
	'Nf': 70,
	'Tf': 0,
	'min_hypo_depth': 'Mindepth',
	'max_hypo_depth': 'Maxdepth'}
rob_source_models_dict[rob_source_model['name']] = rob_source_model

## SHARE AS model
rob_source_model = {}
rob_source_model['name'] = 'SHARE_AS'
rob_source_model['tab_filespec'] = os.path.join(SHARE_directory, 'ASModel', 'Ver6.1', 'ASModelVer61.shp')
rob_source_model['column_map'] = {
	'id': 'IDAS',
	'name': 'IDAS',
	'tectonic_region_type': 'TECTONICS',
	'a_val': 'A',
	'b_val': 'B',
	'min_mag': 4.0,
	'max_mag': 'MAXMAG02',
	'upper_seismogenic_depth': 'MINDEPTH',
	'lower_seismogenic_depth': 'MAXDEPTH',
	'min_strike': 'STRIKE1',
	'max_strike': 'STRIKE3',
	'min_hypo_depth': 'HYPODEPTH1',
	'max_hypo_depth': 'HYPODEPTH3',
	'min_dip': 'DIP1',
	'max_dip': 'DIP3',
	'Ss': 'SS',
	'Nf': 'NF',
	'Tf': 'TF'}
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

