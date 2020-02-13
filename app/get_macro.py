"""
Version of Thomas' get_macro.py script based on eqcatalog
"""

import os
#import sys
import datetime
import shutil
import argparse
from distutils.util import strtobool

import numpy as np

import mapping.layeredbasemap as lbm
import eqcatalog


#TODO:
# - administrative boundaries for DE / NL / LU / FR
# - more command-line options



#BASE_FOLDER = '/home/seismo/lib/macro_maps'
BASE_FOLDER = 'C:\\Temp\\macro_maps'

MIN_FIABILITY = 80
MIN_NUM_REPLIES = 3


parser = argparse.ArgumentParser(description="ROB macroseismic map generator")

parser.add_argument("--id_earth", help="Force creating maps for given earthquake ID",
					nargs=1, type=int)
parser.add_argument("--data_type", help="Type of macroseismic data",
					choices=["dyfi", "official", "historical", "traditional"],
					default="dyfi")
parser.add_argument("--aggregate_by", help="How to aggregate macroseismic data points",
					choices=["", "commune", "main commune", "grid5", "grid10"], default="commune")
parser.add_argument("--agg_method", help="Aggregation method",
                    choices=["mean", "aggregated", "min", "max", "median"], default="mean")
parser.add_argument("--commune_marker", help="Which symbol/geometry to use for commune intensities",
                    choices=["o", "s", "v", "^", "p", "*", "h", "D", "polygon"], default="D")
parser.add_argument("--bel_admin_level", help="Level of administrtive boundaries in Belgium",
                    choices=["","country", "region", "province", "commune"], default="default")
parser.add_argument("--show_main_cities", help="Whether or not to plot main cities",
                    type=strtobool, default="true")
#parser.add_argument("--epicenter_marker", help="Which symbol to use for epicenter",
#                    choices=["", "o", "s", "v", "^", "p", "*", "h", "D"], default="*")
parser.add_argument("--show_epicenter", help="Whether or not to plot epicenter",
					type=strtobool, default="true")
parser.add_argument("--cmap", help="Color map",
					choices=["rob", "usgs"], default="rob")
parser.add_argument("--color_gradient", help="Color gradient",
                    choices=["discrete", "continuous"], default="discrete")
parser.add_argument("--copyright_label", help="Label to use as copyright",
                    type=str, default="Collaborative project of ROB and BNS")
args = parser.parse_args()

## Retrieve earthquakes
if args.id_earth:
	catalog = eqcatalog.rob.query_local_eq_catalog_by_id(args.id_earth)
	overwrite_map = True
else:
	catalog = eqcatalog.rob.query_local_eq_catalog(has_open_enquiry=True)
	overwrite_map = False


for eq in catalog:
	id_earth = eq.ID
	## Fetch macroseismic enquiries for event
	print("Getting data for event #%d" % id_earth)
	if args.data_type == "dyfi":
		macro_data = eq.get_online_macro_enquiries(min_fiability=MIN_FIABILITY)
	else:
		macro_data = eq.get_aggregated_traditional_macro_info(data_type=args.data_type,
							aggregate_by=args.aggregate_by or "commune",
							agg_method=args.agg_method, min_fiability=MIN_FIABILITY)
		MIN_NUM_REPLIES = 0

	if len(macro_data) > MIN_NUM_REPLIES:
		if args.data_type == "dyfi":
			if np.isnan(macro_data.latitudes).all():
				macro_data.set_locations_from_communes()
			## Note: NULL submit_time values are automatically replaced with event time
			lastmod = macro_data.get_datetimes()
			maxlastmod = np.max(lastmod)
		else:
			# TODO
			maxlastmod = np.datetime64('now')
		minlon, maxlon, minlat, maxlat = macro_data.get_region()

		print("id_earth = %i | ML=%.1f | %s " % (id_earth, eq.ML, eq.name))
		print("  datetime = %s" % eq.datetime)
		print("  lastmod = %s" % maxlastmod)
		print("  count = %i" % len(macro_data))
		print("  Bounds : %.4f/%.4f/%.4f/%.4f" % (minlat, maxlat, minlon, maxlon))

		## Determine if map has to be generated
		plot_map = False
		map_filespec = os.path.join(BASE_FOLDER, 'ROB', 'large', '%i.png' % id_earth)
		if os.path.exists(map_filespec) and not overwrite_map:
			## Convert creation_time to np.datetime64
			creation_time = os.stat(map_filespec)[-1]
			creation_time = datetime.datetime.fromtimestamp(creation_time)
			creation_time = eqcatalog.time_functions.as_np_datetime(creation_time)
			if maxlastmod > creation_time:
				print("New data available for mapping !")
				plot_map = True
			else:
				print("No new data available !")
				plot_map = False
		else:
			plot_map = True

		if plot_map:
			print("Creating maps!")

			## Aggregate DYFI
			if args.data_type == "dyfi":
				macro_info = macro_data.get_aggregated_info(aggregate_by=args.aggregate_by,
					filter_floors=(0, 4), agg_info='cii', agg_method=args.agg_method,
					fix_records=True, include_other_felt=True,
					include_heavy_appliance=False, remove_outliers=(2.5, 97.5))
			else:
				macro_info = macro_data

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
			if args.show_epicenter:
				event_style = "default"
			else:
				event_style = None
			country_style = "default"
			admin_style = lbm.LineStyle(line_width=0.5)
			if args.show_main_cities:
				city_style = "default"
			else:
				city_style = None
			copyright = args.copyright_label
			dpi = 200

			for agency in ('ROB', 'BNS'):
				if agency == 'ROB':
					map_filename = '%i.png' % id_earth
				elif agency == 'BNS':
					dt_string = str(eq.datetime).replace('T', '_').replace(':', '')[:15]
					map_filename = '%s.png' % dt_string
				for size in ('large', 'small'):
					map_filespec = os.path.join(BASE_FOLDER, agency, size, map_filename)
					if size == 'large':
						region = [minlon-1, maxlon+1, minlat-.5, maxlat+.5]
						## Ensure epicenter is inside map
						if args.show_epicenter:
							region[0] = min(region[0], eq.lon)
							region[1] = max(region[1], eq.lon)
							region[2] = min(region[2], eq.lat)
							region[3] = max(region[3], eq.lat)
						if args.bel_admin_level == 'default':
							admin_level = 'region'
						else:
							admin_level = args.bel_admin_level
					else:
						#region = (eq.lon-1, eq.lon+1, eq.lat-.5, eq.lat+.5)
						region = (minlon-0.25, maxlon+0.25, minlat-0.1, maxlat+0.1)
						if args.bel_admin_level == 'default':
							admin_level = 'province'
						else:
							admin_level = args.bel_admin_level

					## Plot map
					if agency == 'ROB':
						rob_filespec = map_filespec
						macro_info.plot_map(region=region, projection=projection,
							graticule_interval=graticule_interval, plot_info=plot_info,
							int_conversion='round', symbol_style=symbol_style,
							line_style=line_style, thematic_num_replies=thematic_num_replies,
							interpolate_grid={}, cmap=cmap,
							color_gradient=color_gradient, event_style=event_style,
							country_style=country_style,
							admin_level=admin_level, admin_style=admin_style,
							city_style=city_style, colorbar_style=colorbar_style,
							radii=[], plot_pie=None,
							title='', fig_filespec=map_filespec, copyright=copyright,
							text_box={}, dpi=dpi, verbose=False)
					else:
						## Sync ROB and BNS maps rather than plotting them both
						shutil.copyfile(rob_filespec, map_filespec)

	else:
		print("Not enough data to draw a map (<3 replies)")
	print("------------------------------------------")
