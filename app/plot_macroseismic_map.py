# -*- coding: iso-Latin-1 -*-

"""
Plot macroseismic maps
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import os
import numpy as np

import mapping.layeredbasemap as lbm
from eqcatalog.macro import (aggregate_traditional_macro_info,
						aggregate_online_macro_info, get_isoseismal_macro_info)


## Earthquake to plot
## 1692
#id_earth = 89
## 1938 earthquake
#id_earth = 509
## 2002 Alsdorf
id_earth = 1306
## Kinrooi 25/05/2018
#id_earth = 6625


## Macroseismic paramerers
#enq_type = 'online'
enq_type = 'official'
#enq_type = 'isoseismal'

## Common
min_fiability = 80

## DYFI
query_info = "cii"
#query_info = "num_replies"
min_replies = 3
filter_floors = None
#filter_floors = (0, 4)
agg_method_online = 'mean'
#agg_method_online = 'dyfi'
#agg_method_online = 'mean-dyfi'
fix_records = True
include_other_felt = True
include_heavy_appliance = False
remove_outliers = (0, 100)
#remove_outliers = (5, 95)

## Official/historical enquiries
min_or_max = 'max'
agg_subcommunes = 'mean'


## How to aggregate
## No aggregation = plot at individual locations (if geocoded)
#aggregate_by = None
## Aggregate by commune or main commune
#aggregate_by = 'commune'
aggregate_by = 'main commune'
## Aggregate by grid cell
#aggregate_by = 'grid_5'


if enq_type in ('internet', 'online', 'dyfi'):
	macro_info = aggregate_online_macro_info(id_earth, min_replies=min_replies,
				query_info=query_info, min_fiability=min_fiability,
				filter_floors=filter_floors, aggregate_by=aggregate_by,
				agg_method=agg_method_online, fix_records=fix_records,
				include_other_felt=include_other_felt,
				include_heavy_appliance=include_heavy_appliance,
				remove_outliers=remove_outliers)
elif enq_type == 'isoseismal' and 'commune' in aggregate_by:
	main_communes = {'main commune': True, 'commune': False}[aggregate_by]
	macro_info = get_isoseismal_macro_info(id_earth, main_communes=main_communes,
											as_points=True)
else:
	if enq_type == 'traditional':
		data_type = ''
	else:
		data_type = enq_type
	macro_info = aggregate_traditional_macro_info(id_earth, data_type=data_type,
				min_fiability=min_fiability, min_or_max=min_or_max,
				aggregate_by=aggregate_by, agg_method=agg_subcommunes)


plot_info = 'intensity'
#plot_info = 'num_replies'
#plot_info = 'residual'


## Choose symbols/polygons for aggregation by commune or main commune
#symbol_style = lbm.PointStyle(shape='D', size=4)
symbol_style = None

line_style = "default"
#line_style = None

## Grid interpolation (only if aggregate_by is None or symbol_style is not None)
#interpolate_grid = {'num_cells': (100, 100),
#					'method': 'idw',
#					'params': {'max_dist': 5.}}
interpolate_grid = {}
## resolution or cell size, interpolation method, max_distance
## TODO: Need to set lon, lat when grid cells are used, add x, y and srs props


## Color options
colorbar_style = "default"
#colorbar_style = lbm.LegendStyle('Intensity', location=4)
color_gradient = "discrete"
#color_gradient = "continuous"
cmap = "rob"
#cmap = "usgs"
#cmap = "jet"

thematic_num_replies = False

## Extras
#radii = [10, 25, 50]
radii = []
#plot_pie = dict(prop='asleep', min_replies=25, size_scaling=2, legend_location=3)
plot_pie = {}


## Map parameters
region = (2, 7, 49.25, 51.75)
#region = (4.9, 6, 50.5, 51.5)
projection = "tmerc"
graticule_interval = (2, 1)
#graticule_interval = (0.5, 1)


## Title and output file
#title = "Kinrooi 25/05/2018"
title = ""


## Copyright / text box
copyright = ''
text = macro_info.get_proc_info_text()
text_box = {'pos': 'bl', 'text': text}
#text_box = {}


#out_folder = "D:\\Earthquake Reports\\20180525\\plots"
out_folder = "E:\\Home\\_kris\\Meetings\\2019 - Afdelingsvergadering"
#fig_filename = "Kinrooi_grid_agg_cii_filter_floors_disc.PNG"
#fig_filename = "Kinrooi_dyfi_mean_agg=%s_pie.PNG" % aggregate_by
fig_filename = "2002_%s_agg=%s_minormax=%s_aggfunc=%s.PNG" % (enq_type, aggregate_by, min_or_max, agg_subcommunes)
fig_filespec = os.path.join(out_folder, fig_filename)
#fig_filespec = None


if macro_info:
	#region = "auto"

	## Plot
	#dpi = "default"
	if fig_filespec:
		dpi = 200
	else:
		dpi = 90
	macro_info.plot_map(region=region, projection=projection,
				graticule_interval=graticule_interval, plot_info=plot_info,
				int_conversion='round', symbol_style=symbol_style,
				line_style=line_style, thematic_num_replies=thematic_num_replies,
				interpolate_grid=interpolate_grid, cmap=cmap,
				color_gradient=color_gradient, admin_level='province',
				colorbar_style=colorbar_style, radii=radii, plot_pie=plot_pie,
				title=title, fig_filespec=fig_filespec, copyright=copyright,
				text_box=text_box, dpi=dpi, verbose=True)

	## Export to geojson
	#print(macro_info.to_geojson())

	## Export to vector GIS format
	#gis_file = os.path.splitext(fig_filespec)[0] + ".TAB"
	#macro_info.export_gis('MapInfo File', gis_file)

	## Export to GeoTiff
	#geotiff_file = os.path.splitext(fig_filespec)[0] + ".TIF"
	geotiff_file = "C:\\Temp\\macromap.TIF"
	dpi = 300
	#macro_info.export_geotiff(geotiff_file, region=region, projection=projection,
	#			plot_info=plot_info, int_conversion='round',
	#			symbol_style=symbol_style, thematic_num_replies=thematic_num_replies,
	#			cmap=cmap, color_gradient=color_gradient, dpi=dpi)
