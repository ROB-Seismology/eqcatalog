"""
ROB source model definitions
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import os
import platform
from collections import OrderedDict



## Directories with GIS tables for source models
if platform.uname()[0] == "Windows":
	GIS_ROOT = "D:\\GIS-data"
else:
	GIS_ROOT = os.path.join(os.environ.get("HOME", ""), "gis-data")

ROB_directory = os.path.join(GIS_ROOT, "KSB-ORB", "Source Zone Models")
SHARE_directory = os.path.join(GIS_ROOT, "SHARE")
SHARE_CSS_directory = os.path.join(SHARE_directory, "Task 3.2", "DATA")


# TODO: move SourceModelDefinition to eqcatalog.source_models
# TODO: add read function?

class SourceModelDefinition:
	def __init__(self, name, gis_filespec, column_map):
		self.name = name
		self.gis_filespec = gis_filespec
		self.column_map = column_map

	def __getitem__(self, key):
		return getattr(self, key)


## Dictionary with data for ROB source models
rob_source_models_dict = {}

## Seismotectonic
name = 'Seismotectonic'
gis_filespec = os.path.join(ROB_directory, 'seismotectonic zones 1.2.TAB')
column_map = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'a_sigma': 0.,
	'b_sigma': 0.,
	'min_mag': 4.0,
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
	'Tf': 'Tf',
	'area': 'Area'}
Seismotectonic = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = Seismotectonic


## TwoZone
name = 'TwoZone'
gis_filespec = os.path.join(ROB_directory, 'SLZ+RVG.TAB')
column_map = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'a_sigma': 0.,
	'b_sigma': 0.,
	'min_mag': 4.0,
	'max_mag': 'MS_max_evaluated',
	'upper_seismogenic_depth': 'upper_rupture_depth',
	'lower_seismogenic_depth': 'lower_rupture_depth',
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth',
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'min_dip': 'min_dip',
	'max_dip': 'max_dip',
	'Ss': 'Ss',
	'Nf': 'Nf',
	'Tf': 'Tf',
	'area': 'Area'}
TwoZone = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = TwoZone

## TwoZone_split
name = 'TwoZone_split'
gis_filespec = os.path.join(ROB_directory, 'SLZ+RVG_split.TAB')
column_map = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'a_sigma': 0.,
	'b_sigma': 0.,
	'min_mag': 4.0,
	'max_mag': 'MS_max_evaluated',
	'upper_seismogenic_depth': 0.0,
	'lower_seismogenic_depth': 25.0,
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth',
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'min_dip': 45.0,
	'max_dip': 45.0,
	'Ss': 34,
	'Nf': 33,
	'Tf': 33}
TwoZone_split = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = TwoZone_split

## TwoZonev2
name = 'TwoZonev2'
gis_filespec = os.path.join(ROB_directory, 'TwoZone_v2.TAB')
column_map = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'a_sigma': 0.,
	'b_sigma': 'stdbMLE',
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
	'area': 'Area'}
TwoZonev2 = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = TwoZonev2

## Leynaud
name = 'Leynaud'
gis_filespec = os.path.join(ROB_directory, 'ROB Seismic Source Model (Leynaud, 2000).TAB')
# Note: leave min_mag at 3.5 (Eurocode 8)
column_map = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'a',
	'b_val': 'b',
	'a_sigma': 0.,
	'b_sigma': 0.,
	'min_mag': 3.5,
	'max_mag': 'MS_max_evaluated',
	'upper_seismogenic_depth': 0.0,
	'lower_seismogenic_depth': 25.0,
	'min_hypo_depth': 'Source_Depth',
	'max_hypo_depth': 'Source_Depth',
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'min_dip': 45.0,
	'max_dip': 45.0,
	'Ss': 34,
	'Nf': 33,
	'Tf': 33}
Leynaud = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = Leynaud

## Leynaud_extended
name = 'Leynaud_extended'
gis_filespec = os.path.join(ROB_directory, 'Leynaud extended.TAB')
column_map = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'a_sigma': 0.,
	'b_sigma': 'stdbMLE',
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
	'area': 'Area'}
Leynaud_extended = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = Leynaud_extended

## Leynaud_updated
name = 'Leynaud_updated'
gis_filespec = os.path.join(ROB_directory, 'Leynaud updated.TAB')
column_map = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLEfix',
	'b_val': 'bMLEfix',
	'a_sigma': 0.,
	'b_sigma': 0.,
	'min_mag': 4.0,
	'max_mag': 'MS_max_evaluated',
	'upper_seismogenic_depth': 0.0,
	'lower_seismogenic_depth': 25.0,
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth',
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'min_dip': 45.0,
	'max_dip': 45.0,
	'Ss': 34,
	'Nf': 33,
	'Tf': 33}
Leynaud_updated = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = Leynaud_updated

## RVRS_area
name = 'RVRS_area'
gis_filespec = os.path.join(ROB_directory, 'RVRS_area_v2.TAB')
column_map = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type': 'SHARE_TRT',
	'a_val': 2.4,
	'b_val': 0.9,
	'a_sigma': 0.,
	'b_sigma': 'stdbMLE',
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
RVRS_area = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = RVRS_area

## RVRS_CSS
name = 'RVRS_CSS'
gis_filespec = os.path.join(SHARE_CSS_directory, 'CSS_SurfaceTrace.TAB')
column_map = {
	'id': 'IDSource',
	'name': 'SourceName',
	'tectonic_region_type': 'Active Shallow Crust',
	'a_val': 2.4,
	'b_val': 0.9,
	'a_sigma': 0.,
	'b_sigma': 0.,
	'min_mag': 4.0,
	'max_mag': 'MaxMag',
	'upper_seismogenic_depth': 'MinDepth',
	'lower_seismogenic_depth': 'MaxDepth',
	'dip_distribution': [('DipMin', 0.5),
							('DipMax', 0.5)],
	'rake_distribution': [('RakeMin', 0.5),
							('RakeMax', 0.5)],
	'slip_rate_distribution': [('SlipRateMin', 0.5),
								('SlipRateMax', 0.5)],
	'bg_zone': None}
RVRS_CSS = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = RVRS_CSS

## Seismotectonic Hybrid
name = 'Seismotectonic_Hybrid'
gis_filespec = os.path.join(ROB_directory, 'Seismotectonic Hybrid.TAB')
column_map = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type': 'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'a_sigma': 0.,
	'b_sigma': 'stdbMLE',
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
	'dip_distribution': [('min_dip', 0.5),
							('max_dip', 0.5)],
	'rake_distribution': [('min_rake', 0.5),
							('max_rake', 0.5)],
	'slip_rate_distribution': [('min_slip_rate', 0.5),
								('max_slip_rate', 0.5)],
	'bg_zone': 'BG_zone',
	'area': 'Area'}
Seismotectonic_Hybrid = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = Seismotectonic_Hybrid

## Seismotectonic Hybrid (including Hockai fault)
name = 'Seismotectonic_Hybrid_v2'
gis_filespec = os.path.join(ROB_directory, 'Seismotectonic Hybrid + Hockai.TAB')
column_map = {
	'id': 'ShortName',
	'name': 'Name',
	'tectonic_region_type': 'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'a_sigma': 0.,
	'b_sigma': 'stdbMLE',
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
	'dip_distribution': [('min_dip', 0.5),
							('max_dip', 0.5)],
	'rake_distribution': [('min_rake', 0.5),
							('max_rake', 0.5)],
	'slip_rate_distribution': [('min_slip_rate', 0.5),
								('max_slip_rate', 0.5)],
	'bg_zone': 'BG_zone',
	'area': 'Area'}
Seismotectonic_Hybrid_v2 = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = Seismotectonic_Hybrid_v2

## RVRS SHARE AS model
name = 'RVRS_SHARE_AS'
gis_filespec = os.path.join(ROB_directory, 'RVRS_SHARE_v4alpha.TAB')
column_map = {
	'id': 'Idas',
	'name': 'Idas',
	'tectonic_region_type': 'Active Shallow Crust',
	'a_val': 'A',
	'b_val': 'B',
	'a_sigma': 0.,
	'b_sigma': 0.,
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
RVRS_SHARE_AS = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = RVRS_SHARE_AS

## SHARE AS model
name = 'SHARE_AS'
gis_filespec = os.path.join(SHARE_directory, 'ASModel', 'Ver6.1', 'ASModelVer61.shp')
#gis_filespec = os.path.join(SHARE_directory, 'ASModel', 'Ver4alpha', 'AsModel_ver4_alpha.shp')
column_map = {
	'id': 'IDAS',
	'name': 'IDAS',
	'tectonic_region_type': 'TECTONICS',
	'a_val': 'A',
	'b_val': 'B',
	'a_sigma': 0.,
	'b_sigma': 0.,
	'min_mag': 4.5,
	'max_mag': 'MAXMAG01',
	'max_mag_distribution': [('MAXMAG01', 'WMAXMAG01'),
								('MAXMAG02', 'WMAXMAG02'),
								('MAXMAG03', 'WMAXMAG03'),
								('MAXMAG04', 'WMAXMAG04')],
	'upper_seismogenic_depth': 'MINDEPTH',
	'lower_seismogenic_depth': 'MAXDEPTH',
	#'min_strike': 'STRIKE1',
	#'max_strike': 'STRIKE3',
	'hypo_distribution': [('HYPODEPTH1', 'WHDEPTH1'),
							('HYPODEPTH2', 'WHDEPTH2'),
							('HYPODEPTH3', 'WHDEPTH3')],
	'min_hypo_depth': 'MINDEPTH',
	'max_hypo_depth': 'MAXDEPTH',
	'strike_dip_distribution': [('STRIKE1', 'DIP1', 'DIRWEIGHT1'),
									('STRIKE2', 'DIP2', 'DIRWEIGHT2'),
									('STRIKE3', 'DIP3', 'DIRWEIGHT3'),
									('STRIKE4', 'DIP4', 'DIRWEIGHT4')],
	#'min_dip': 'DIP1',
	#'max_dip': 'DIP3',
	'Ss': 'SS',
	'Nf': 'NF',
	'Tf': 'TF'}
SHARE_AS = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = SHARE_AS

## SHARE AS model around Belgium
name = 'SHARE_AS_Belgium'
gis_filespec = os.path.join(ROB_directory, 'SHARE_ASv61_Belgium.TAB')
column_map = {
	'id': 'Idas',
	'name': 'Idas',
	'tectonic_region_type': 'SHARE_TRT',
	'a_val': 'A',
	'b_val': 'B',
	'a_sigma': 0.,
	'b_sigma': 'stdbMLE',
	'min_mag': 4.5,
	'max_mag': 'Mmax_evaluated',
	'upper_seismogenic_depth': 'upper_rupture_depth',
	'lower_seismogenic_depth': 'lower_rupture_depth',
	'min_strike': 'min_strike',
	'max_strike': 'max_strike',
	'min_hypo_depth': 'min_hypo_depth',
	'max_hypo_depth': 'max_hypo_depth',
	'min_dip': 'min_dip',
	'max_dip': 'max_dip',
	'Ss': 'Ss',
	'Nf': 'Nf',
	'Tf': 'Tf'}
SHARE_AS_Belgium = SourceModelDefinition(name, gis_filespec, column_map)
rob_source_models_dict[name] = SHARE_AS_Belgium


def read_source_model(source_model_name, ID_colname="", fix_mi_lambert=True,
					verbose=True):
	"""
	Read source-zone model stored in a GIS (MapInfo) table.

	:param source_model_name:
		String, name of source-zone model containing area sources
		or else full path to GIS file containing area sources
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

	## Read zone model from MapInfo file
	#source_model_table = ZoneModelTables[source_model_name.lower()]
	#gis_filespec = os.path.join(GIS_ROOT, "KSB-ORB", "Source Zone Models",
	# 							source_model_table + ".TAB")
	try:
		gis_filespec = rob_source_models_dict[source_model_name]["gis_filespec"]
	except:
		gis_filespec = source_model_name
	else:
		if not ID_colname:
			ID_colname = rob_source_models_dict[source_model_name]["column_map"]["id"]

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
