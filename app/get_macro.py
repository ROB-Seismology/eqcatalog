# -*- coding: iso-Latin-1 -*-
"""
Version of Thomas' get_macro.py script based on eqcatalog

Intended to be run as cron job on seisweb server
Set up crontab as follows:

## Run cronjobs via bash instead of sh
SHELL=/bin/bash
## Source conda snippet to bash run by crontab
## This is necessary to activate the right conda environment
BASH_ENV=~/.conda_env
## Send mail if an error occurs
MAILTO=seismo.info@seismology.be
MAILFROM=seisweb3

*/5 * * * * conda activate eqcatalog; python ~/python/seismo/eqcatalog/app/get_macro.py  >> ~/log/do_macro.log 2>&1
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import datetime
import shutil
import argparse
from distutils.util import strtobool

import numpy as np

## Avoid 'Invalid DISPLAY variable' error on headless server
import matplotlib
matplotlib.use('Agg')

import mapping.layeredbasemap as lbm
import eqcatalog


#TODO:
# - administrative boundaries for DE / NL / LU / FR
# - more command-line options


if sys.platform == 'win32':
	BASE_FOLDER = 'C:\\Temp\\macro_maps'
else:
	BASE_FOLDER = '/home/seismo/lib/macro_maps'

MIN_FIABILITY = 80
MIN_NUM_REPLIES = 3


## Command-line options
parser = argparse.ArgumentParser(description="ROB macroseismic map generator")

parser.add_argument("--id_earth", help="Force creating maps for given earthquake ID",
					nargs=1, type=int)
parser.add_argument("--data_type", help="Type of macroseismic data",
					choices=["dyfi", "official", "historical", "traditional"],
					default="dyfi")
parser.add_argument("--aggregate_by", help="How to aggregate macroseismic data points",
					choices=["", "commune", "main commune", "grid5", "grid10"], default="commune")
parser.add_argument("--agg_method", help="Aggregation method (availability depends on data_type option!)",
                    choices=["mean", "dyfi", "min", "max", "median"], default="mean")
parser.add_argument("--commune_marker", help="Which symbol/geometry to use for commune intensities",
                    choices=["o", "s", "v", "^", "p", "*", "h", "D", "polygon"], default="D")
parser.add_argument("--admin_source", help="Source for administrtive boundaries",
                    choices=["gadm", "statbel", "rob"], default="gadm")
parser.add_argument("--admin_level", help="Level of administrtive boundaries ('region', 'province', 'arrondissement', 'main commune', 'commune', 'sector' or any combination or 'auto'",
#                    choices=["", "region", "province", "commune", "default"], default="default")
                    type=str, default="auto")
parser.add_argument("--show_main_cities", help="Whether or not to plot main cities",
                    type=strtobool, default="true")
parser.add_argument("--epicenter_marker", help="Which symbol to use for epicenter",
                    choices=["", "o", "s", "v", "^", "p", "*", "h", "D"], default="*")
parser.add_argument("--cmap", help="Intensity color map",
					choices=["rob", "usgs"], default="rob")
parser.add_argument("--color_gradient", help="Color gradient",
                    choices=["discrete", "continuous"], default="discrete")
parser.add_argument("--map_region", help="Size of map region wrt intensity data",
                    choices=["wide", "tight", "both"], default="both")
parser.add_argument("--copyright_label", help="Label to use as copyright",
                    type=str, default="Collaborative project of ROB and BNS")
parser.add_argument("--base_folder", help="Base folder for generated maps",
                    type=str, default=BASE_FOLDER)
parser.add_argument("--dry_run", help="Run script without actually plotting anything",
					type=strtobool, default="false")
parser.add_argument("--verbose", help="Whether or not to print progress information",
					type=strtobool, default="true")
args = parser.parse_args()


## Retrieve earthquakes
if args.id_earth:
	catalog = eqcatalog.rob.query_local_eq_catalog_by_id(args.id_earth)
	overwrite_map = True
else:
	catalog = eqcatalog.rob.query_local_eq_catalog(has_open_enquiry=True, start_date=2000)
	overwrite_map = False


for eq in catalog:
	id_earth = eq.ID
	## Fetch macroseismic enquiries for event
	if args.verbose:
		print("Getting data for event #%d" % id_earth)
	if args.data_type == "dyfi":
		macro_data = eq.get_online_macro_enquiries(min_fiability=MIN_FIABILITY)
	else:
		macro_data = eq.get_aggregated_traditional_macro_info(data_type=args.data_type,
							aggregate_by=args.aggregate_by or "commune",
							agg_method=args.agg_method, min_fiability=MIN_FIABILITY)
		MIN_NUM_REPLIES = 1

	if len(macro_data) >= MIN_NUM_REPLIES:
		if args.data_type == "dyfi":
			if np.isnan(macro_data.latitudes).all():
				macro_data.set_locations_from_communes()
			## Note: NULL submit_time values are automatically replaced with event time
			lastmod = macro_data.submit_times
			maxlastmod = np.max(lastmod)
		else:
			# TODO
			maxlastmod = np.datetime64('now')
		minlon, maxlon, minlat, maxlat = macro_data.get_region()

		if args.verbose:
			print("id_earth = %i | ML=%.1f | %s " % (id_earth, eq.ML, eq.name))
			print("  datetime = %s" % eq.datetime)
			print("  lastmod = %s" % maxlastmod)
			print("  count = %i" % len(macro_data))
			print("  bounds : %.4f/%.4f/%.4f/%.4f" % (minlat, maxlat, minlon, maxlon))

		## Determine if map has to be generated
		plot_map = False
		map_filespec = os.path.join(args.base_folder, 'ROB', 'large',
									'%i.png' % id_earth)
		if os.path.exists(map_filespec) and not overwrite_map:
			## Convert creation_time to np.datetime64
			creation_time = os.stat(map_filespec)[-1]
			creation_time = datetime.datetime.fromtimestamp(creation_time)
			creation_time = eqcatalog.time.as_np_datetime(creation_time)
			if maxlastmod > creation_time:
				if args.verbose:
					print("New data available for mapping !")
				plot_map = True
			else:
				print("No new data available !")
				plot_map = False
		else:
			plot_map = True

		if plot_map:
			if args.verbose:
				print("Creating maps!")

			## Aggregate DYFI
			if args.data_type == "dyfi":
				macro_info = macro_data.aggregate(aggregate_by=args.aggregate_by,
					filter_floors=(0, 4), agg_info='cii', agg_method=args.agg_method,
					fix_records=True, include_other_felt=True,
					include_heavy_appliance=False, remove_outliers=(2.5, 97.5))
				if args.verbose and len(macro_info):
					print('Imin/max: %.1f/%.1f' % macro_info.Iminmax())
			else:
				macro_info = macro_data

			if not(len(macro_info)):
				continue

			minlon, maxlon, minlat, maxlat = macro_info.get_region()

			## Plot parameters
			projection = 'merc'
			graticule_interval = 'auto'
			plot_info = 'intensity'
			if "grid" in args.aggregate_by or args.commune_marker == "polygon":
				symbol_style = None
				thematic_num_replies = False
			else:
				symbol_style = lbm.PointStyle(shape=args.commune_marker, size=4)
				thematic_num_replies = True

			line_style = "default"
			cmap = args.cmap
			color_gradient = args.color_gradient
			colorbar_style = "default"
			event_style = lbm.PointStyle(shape=args.epicenter_marker, size=14,
										fill_color='magenta', line_color='k')
			country_style = "default"
			admin_style = lbm.LineStyle(line_width=0.5)
			if args.show_main_cities:
				city_style = "default"
			else:
				city_style = None
			copyright = args.copyright_label
			if copyright:
				copyright = '© ' + copyright
			dpi = 200

			map_sizes = []
			if args.map_region in ('wide', 'both'):
				map_sizes.append('large')
			if args.map_region in ('tight', 'both'):
				map_sizes.append('small')

			for size in map_sizes:
				for agency in ('ROB', 'BNS'):
					if agency == 'ROB':
						map_filename = '%i.png' % id_earth
					elif agency == 'BNS':
						dt_string = str(eq.datetime).replace('T', '_').replace(':', '')[:15]
						map_filename = '%s.png' % dt_string
					map_filespec = os.path.join(args.base_folder, agency, size, map_filename)
					if size == 'large':
						region = [minlon-1, maxlon+1, minlat-.5, maxlat+.5]
						## Ensure epicenter is inside map if marker is specified
						if args.epicenter_marker:
							region[0] = min(region[0], eq.lon)
							region[1] = max(region[1], eq.lon)
							region[2] = min(region[2], eq.lat)
							region[3] = max(region[3], eq.lat)
						if args.admin_level == 'default':
							admin_level = 'region'
						else:
							admin_level = args.admin_level
					else:
						#region = (eq.lon-1, eq.lon+1, eq.lat-.5, eq.lat+.5)
						region = (minlon-0.25, maxlon+0.25, minlat-0.1, maxlat+0.1)
						if args.admin_level == 'default':
							admin_level = 'region,province'
						else:
							admin_level = args.admin_level

					## Plot map
					if not args.dry_run:
						if agency == 'ROB':
							rob_filespec = map_filespec
							macro_info.plot_map(region=region, projection=projection,
								graticule_interval=graticule_interval, plot_info=plot_info,
								int_conversion='round', symbol_style=symbol_style,
								line_style=line_style, thematic_num_replies=thematic_num_replies,
								interpolate_grid={}, cmap=cmap,
								color_gradient=color_gradient, event_style=event_style,
								country_style=country_style, admin_source=args.admin_source,
								admin_level=admin_level, admin_style=admin_style,
								city_style=city_style, colorbar_style=colorbar_style,
								radii=[], plot_pie=None,
								title='', fig_filespec=map_filespec, copyright=copyright,
								text_box={}, dpi=dpi, verbose=args.verbose)
						else:
							## Sync ROB and BNS maps rather than plotting them both
							shutil.copyfile(rob_filespec, map_filespec)

	else:
		print("Not enough data to draw a map (<%d replies)" % MIN_NUM_REPLIES)
	if args.verbose:
		print("------------------------------------------")
