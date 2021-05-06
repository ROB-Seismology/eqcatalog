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
# - more command-line options


if sys.platform == 'win32':
	BASE_FOLDER = 'C:\\Temp\\macro_maps'
else:
	BASE_FOLDER = '/home/seismo/lib/macro_maps'

MIN_FIABILITY = 80


## Command-line options
parser = argparse.ArgumentParser(description="ROB macroseismic map generator")

parser.add_argument("--id_earth", help="Force creating maps for given earthquake ID / all / missing",
					nargs=1)
parser.add_argument("--data_type", help="Type of macroseismic data",
					choices=["dyfi", "official", "historical", "traditional"],
					default="dyfi")
parser.add_argument("--aggregate_by", help="How to aggregate macroseismic data points",
					choices=["", "commune", "main commune", "grid5", "grid10"],
					default="main commune")
parser.add_argument("--agg_method", help="Aggregation method (availability depends on data_type option!)",
					choices=["mean", "dyfi", "min", "max", "median"], default="mean")
parser.add_argument("--min_num_replies", help="Min. number of replies per aggregate",
					type=int, default=2)
parser.add_argument("--mdp_marker", help="Which symbol/geometry to use for aggregated MDPs",
					choices=["o", "s", "v", "^", "p", "*", "h", "D", "polygon"], default="D")
parser.add_argument("--admin_source", help="Source for administrtive boundaries",
					choices=["gadm", "statbel", "rob"], default="gadm")
parser.add_argument("--admin_level", help="Level of administrtive boundaries ('region', 'province', 'arrondissement', 'main commune', 'commune', 'sector' or any combination or 'auto'",
#					choices=["", "region", "province", "commune", "default"], default="default")
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
parser.add_argument("--min_lon", help="Minimum map longitude", type=float, default=None)
parser.add_argument("--max_lon", help="Maximum map longitude", type=float, default=None)
parser.add_argument("--min_lat", help="Minimum map latitude", type=float, default=None)
parser.add_argument("--max_lat", help="Maximum map latitude", type=float, default=None)
parser.add_argument("--copyright_label", help="Label to use as copyright",
					type=str, default="Collaborative project of ROB and BNS")
parser.add_argument("--base_folder", help="Base folder for generated maps",
					type=str, default=BASE_FOLDER)
parser.add_argument("--export_json", help="Whether or not to export json file as well",
					type=strtobool, default="false")
parser.add_argument("--dry_run", help="Run script without actually plotting anything",
					type=strtobool, default="false")
parser.add_argument("--verbose", help="Whether or not to print progress information",
					type=strtobool, default="true")
args = parser.parse_args()


if args.verbose:
	print('get-macro running at %s' % datetime.datetime.now())

## Retrieve earthquakes
if args.id_earth in (['all'], ['missing']):
	if args.data_type == 'dyfi':
		start_date = 2000
	else:
		start_date = 1000
	if args.data_type == 'dyfi':
		catalog = eqcatalog.rob.seismodb.get_earthquakes_with_online_enquiries()
	else:
		catalog = eqcatalog.rob.seismodb.get_earthquakes_with_traditional_enquiries()
	catalog = catalog.subselect(start_date=start_date, attr_val=('event_type', ['ke', 'ki']),
										region=(-1, 9, 48.5, 52.5))
	if args.id_earth == ['all']:
		overwrite_map = True
	else:
		overwrite_map = False
elif args.id_earth:
	catalog = eqcatalog.rob.query_local_eq_catalog_by_id(args.id_earth)
	overwrite_map = True
elif args.data_type == 'dyfi':
	catalog = eqcatalog.rob.query_local_eq_catalog(has_open_enquiry=True, start_date=2000)
	overwrite_map = False
	if args.verbose and len(catalog) == 0:
		print('Nothing to do: no open enquiries...')
else:
	print('Nothing to do: please check command-line options!')
	exit()

if not args.aggregate_by:
	args.min_num_replies = 1

for eq in catalog:
	id_earth = eq.ID
	## Fetch macroseismic enquiries for event
	if args.verbose:
		print("Getting data for event #%s" % id_earth)
	if args.data_type == "dyfi":
		macro_data = eq.get_online_macro_enquiries(min_fiability=MIN_FIABILITY)
	else:
		macro_data = eq.get_traditional_macroseismic_info(data_type=args.data_type,
												min_fiability=MIN_FIABILITY)
		args.min_num_replies = 1
	#if args.verbose:
	#	print('N = %d' % len(macro_data))

	if len(macro_data) >= args.min_num_replies:
		if args.data_type == "dyfi":
			macro_data = macro_data.fix_all()
			if np.isnan(macro_data.latitudes).all():
				macro_data.set_locations_from_communes()
			## Note: NULL submit_time values are automatically replaced with event time
			lastmod = macro_data.submit_times
			maxlastmod = np.max(lastmod)
		else:
			maxlastmod = eq.datetime

		try:
			minlon, maxlon, minlat, maxlat = macro_data.get_region()
		except:
			## No valid data
			print('No valid locations found for event #%s' % id_earth)
			if args.verbose:
				print("------------------------------------------")
			continue

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

			## Aggregate
			if args.data_type == "dyfi":
				min_replies = args.min_num_replies
				if len(macro_data) < 20:
					## If there are only few enquiries, set min_replies to 1
					min_replies = 1
					print('Few data: lowering min_num_replies to 1!')
				macro_info, rejected_macro_info = macro_data.aggregate(
					aggregate_by=args.aggregate_by,
					filter_floors=(0, 4), agg_info='cii', agg_method=args.agg_method,
					min_replies=min_replies, keep_not_felt=True,
					min_fiability=MIN_FIABILITY,
					fix_commune_ids=False, fix_felt=False, remove_duplicates=False,
					include_other_felt=True, include_heavy_appliance=False,
					remove_outliers=2.0, return_rejected=True)
				if len(macro_info) == 0 and len(rejected_macro_info) > 0:
					## If there are no aggregates with required number of replies,
					## plot rejected aggregates instead
					print('No aggregates: lowering min_num_replies to 1!')
					macro_info, rejected_macro_info = rejected_macro_info, macro_info
			else:
				macro_info = macro_data.aggregate(aggregate_by=args.aggregate_by or "commune",
												agg_function=args.agg_method)
				rejected_macro_info = None

			if not(len(macro_info)):
				print("Not enough data to draw a map (<%d replies)"
						% args.min_num_replies)
				if args.verbose:
					print("------------------------------------------")
				continue

			if args.verbose and len(macro_info):
				print('Imin/max: %.1f/%.1f' % macro_info.Iminmax())

			minlon, maxlon, minlat, maxlat = macro_info.get_region(percentile_width=98)

			## Plot parameters
			projection = 'merc'
			graticule_interval = 'auto'
			graticule_style = lbm.GraticuleStyle()
			graticule_style.line_style.line_width = 0
			plot_info = 'intensity'
			if "grid" in args.aggregate_by or args.mdp_marker == "polygon":
				symbol_style = None
				thematic_num_replies = False
			else:
				symbol_style = lbm.PointStyle(shape=args.mdp_marker, size=4,
													line_width=0.5)
				thematic_num_replies = True

			line_style = "default"
			cmap = args.cmap
			color_gradient = args.color_gradient
			colorbar_style = "default"
			label_style = lbm.TextStyle(font_size=9, vertical_alignment="top",
												horizontal_alignment='center', offset=(0, -5))
			event_style = lbm.PointStyle(shape=args.epicenter_marker, size=14,
										fill_color='Fuchsia', line_color='w',
										label_style=label_style)
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
							dlon = region[1] - region[0]
							dlat = region[3] - region[2]
							if args.min_lon is None:
								region[0] = min(region[0], eq.lon - dlon/5.)
							else:
								region[0] = args.min_lon
							if args.max_lon is None:
								region[1] = max(region[1], eq.lon + dlon/5.)
							else:
								region[1] = args.max_lon
							if args.min_lat is None:
								region[2] = min(region[2], eq.lat - dlat/5.)
							else:
								region[2] = args.min_lat
							if args.max_lat is None:
								region[3] = max(region[3], eq.lat + dlat/5.)
							else:
								region[3] = args.max_lat
						if args.admin_level == 'default':
							admin_level = 'region'
						else:
							admin_level = args.admin_level
					else:
						region = [minlon-0.25, maxlon+0.25, minlat-0.1, maxlat+0.1]
						## Ensure epicenter is inside map if marker is specified
						if args.epicenter_marker:
							dlon = region[1] - region[0]
							dlat = region[3] - region[2]
							if args.min_lon is None:
								region[0] = min(region[0], eq.lon - dlon/10.)
							else:
								region[0] = args.min_lon
							if args.max_lon is None:
								region[1] = max(region[1], eq.lon + dlon/10.)
							else:
								region[1] = args.max_lon
							if args.min_lat is None:
								region[2] = min(region[2], eq.lat - dlat/10.)
							else:
								region[2] = args.min_lat
							if args.max_lat is None:
								region[3] = max(region[3], eq.lat + dlat/10.)
							else:
								region[3] = args.max_lat
						if args.admin_level == 'default':
							admin_level = 'region,province'
						else:
							admin_level = args.admin_level

					## Plot map
					if not args.dry_run:
						if agency == 'ROB':
							rob_filespec = map_filespec
							macro_info.plot_map(region=region, projection=projection,
								graticule_interval=graticule_interval,
								graticule_style=graticule_style, plot_info=plot_info,
								int_conversion='round', symbol_style=symbol_style,
								line_style=line_style, thematic_num_replies=thematic_num_replies,
								interpolate_grid={}, cmap=cmap,
								color_gradient=color_gradient, event_style=event_style,
								country_style=country_style, admin_source=args.admin_source,
								admin_level=admin_level, admin_style=admin_style,
								city_style=city_style, colorbar_style=colorbar_style,
								radii=[], plot_pie=None,
								title='', fig_filespec=map_filespec, copyright=copyright,
								text_box={}, dpi=dpi, verbose=args.verbose,
								export_json=args.export_json,
								rejected_macro_info_coll=rejected_macro_info)
						else:
							## Sync ROB and BNS maps rather than plotting them both
							shutil.copyfile(rob_filespec, map_filespec)

	else:
		print("Not enough data to draw a map (<%d replies)" % args.min_num_replies)
	if args.verbose:
		print("------------------------------------------")
