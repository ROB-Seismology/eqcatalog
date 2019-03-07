# -*- coding: iso-Latin-1 -*-

"""
Plot macroseismic maps
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import os

import mapping.layeredbasemap as lbm
from eqcatalog.plot import plot_web_macroseismic_map, plot_official_macroseismic_map


## Earthquake to plot
## 1692
#id_earth = 89
## 1938 earthquake
#id_earth = 509
## 2002 Alsdorf
id_earth = 1306
## Kinrooi 25/05/2018
#id_earth = 6625

## Map parameters
region = (2, 7, 49.25, 51.75)
#region = (4.9, 6, 50.5, 51.5)
projection = "tmerc"
graticule_interval = (2, 1)
#graticule_interval = (0.5, 1)

## DYFI parameters
query_info = "cii"
#query_info = "num_replies"
min_replies = 3
#filter_floors = (-100, 900)
filter_floors = (0, 4)
min_fiability = 20
recalc = False

## How to aggregate
## No aggregation = plot at individual locations (if geocoded)
#aggregate_by = None
## Aggregate by commune or main commune
#aggregate_by = 'commune'
aggregate_by = 'main commune'
## Aggregate by grid cell
#aggregate_by = 'grid_5'

## Choose symbols/polygons for aggregation by commune or main commune
#symbol_style = lbm.PointStyle(shape='D', size=5)
symbol_style = None

## Color options
color_gradient = "discrete"
#color_gradient = "continuous"
cmap = "rob"
#cmap = "usgs"
#cmap = "jet"

## Extras
#radii = [10, 25, 50]
radii = []
#plot_pie = 'asleep'
plot_pie = None


## Title and output file
#title = "Kinrooi 25/05/2018"
title = ""

out_folder = "D:\\Earthquake Reports\\20180525\\plots"
fig_filename = "Kinrooi_grid_agg_cii_filter_floors_disc.PNG"
#fig_filespec = os.path.join(out_folder, fig_filename)
fig_filespec = None


## Plot
"""
plot_web_macroseismic_map(id_earth, region=region, projection=projection,
				graticule_interval=graticule_interval, min_replies=min_replies,
				query_info=query_info, min_fiability=min_fiability, filter_floors=filter_floors,
				cmap=cmap, color_gradient=color_gradient, symbol_style=symbol_style,
				aggregate_by=aggregate_by, radii=radii,
				plot_pie=plot_pie, title=title, fig_filespec=fig_filespec,
				recalc=recalc)
"""

plot_official_macroseismic_map(id_earth, region=region, projection=projection,
				graticule_interval=graticule_interval, min_fiability=min_fiability,
				min_or_max='min', aggregate_by=aggregate_by, agg_function='average',
				int_conversion='round', symbol_style=symbol_style, cmap=cmap,
				color_gradient=color_gradient, radii=radii, title=title,
				fig_filespec=fig_filespec, verbose=True)

