import os


## Directories with MapInfo tables for source models
GIS_areasource_directory = r'D:\GIS-data\KSB-ORB\Source Zone Models'
GIS_faultsource_directory = r'D:\GIS-data\SHARE\Task 3.2\DATA'


## Dictionary with data for ROB source models
rob_source_models_dict = {}

## Seismotectonic
rob_source_model = {}
rob_source_model['name'] = 'Seismotectonic'
rob_source_model['tab_filespec'] = os.path.join(GIS_areasource_directory, 'seismotectonic zones 1.2.TAB')
rob_source_model['column_map'] = {
	'id': 'shortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'min_mag': 1.8,
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
	'id': 'shortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'min_mag': 1.8,
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
	'id': 'shortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLE',
	'b_val': 'bMLE',
	'min_mag': 1.8,
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
	'id': 'shortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'a',
	'b_val': 'b',
	'min_mag': 1.8,
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
	'id': 'shortName',
	'name': 'Name',
	'tectonic_region_type':'SHARE_TRT',
	'a_val': 'aMLEfix',
	'b_val': 'bMLEfix',
	'min_mag': 1.8,
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
	'id': 'ID',
	'name': 'Name',
	'tectonic_region_type': 'Active Shallow Crust',
	'a_val': 2.4,
	'b_val': 0.9,
	'min_mag': 1.8,
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
	'min_mag': 1.8,
	'max_mag': 'MaxMag',
	'upper_seismogenic_depth': 'MinDepth',
	'lower_seismogenic_depth': 'MaxDepth',
	'dip': 'Dip',
	'rake': 'Rake',
	'slip_rate': 'SlipRate'}
rob_source_models_dict[rob_source_model['name']] = rob_source_model

