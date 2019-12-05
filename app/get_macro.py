"""
Version of Thomas' get_macro.py script based on eqcatalog
"""

import os
#import sys
import datetime
import shutil
import argparse

import numpy as np

import mapping.layeredbasemap as lbm
import eqcatalog


#TODO:
# - automatic graticule interval
# - administrative boundaries for DE / NL / LU / FR
# - cities
# - more command-line options



#BASE_FOLDER = '/home/seismo/lib/macro_maps'
BASE_FOLDER = 'C:\\Temp\\macro_maps'

MIN_FIABILITY = 80
MIN_NUM_REPLIES = 3


parser = argparse.ArgumentParser(description="ROB macroseismic map generator")

parser.add_argument("--id_earth", help="Force creating maps for given earthquake ID",
					nargs=1, type=int)
parser.add_argument("--aggregate-by", help="How to aggregate macroseismic data points",
					choices=["", "commune", "main commune", "grid"], default="commune")
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
	dyfi = eq.get_online_macro_enquiries(min_fiability=MIN_FIABILITY)
	if len(dyfi) > MIN_NUM_REPLIES:
		if np.isnan(dyfi.latitudes).all():
			dyfi.set_locations_from_communes()
		minlon, maxlon, minlat, maxlat = dyfi.get_region()
		## Note: NULL submit_time values are automatically replaced with event time
		lastmod = dyfi.get_datetimes()
		maxlastmod = np.max(lastmod)

		print("id_earth = %i | ML=%.1f | %s " % (id_earth, eq.ML, eq.name))
		print("  datetime = %s" % eq.datetime)
		print("  lastmod = %s" % maxlastmod)
		print("  count = %i" % len(dyfi))
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
			print("New data found since maps were created! Creating maps!")

			## Aggregate
			macro_info = dyfi.get_aggregated_info(aggregate_by=args.aggregate_by,
					filter_floors=(0, 4), agg_info='cii', agg_method='mean',
					fix_records=True, include_other_felt=True,
					include_heavy_appliance=False, remove_outliers=(2.5, 97.5))

			minlon, maxlon, minlat, maxlat = macro_info.get_region()

			## Plot parameters
			projection = 'merc'
			graticule_interval = (None, None) # auto?
			plot_info = 'intensity'
			symbol_style = lbm.PointStyle(shape='D', size=4)
			line_style = "default"
			thematic_num_replies = True
			cmap = "rob"
			color_gradient = "discrete"
			colorbar_style = "default"
			event_style = "default"
			admin_level = "province"
			admin_style = "default"
			copyright = 'Collaborative project of ROB and BNS'
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
						region = (minlon-1, maxlon+1, minlat-.5, maxlat+.5)
					else:
						region = (eq.lon-1, eq.lon+1, eq.lat-.5, eq.lat+.5)

					## Plot map
					if agency == 'ROB':
						rob_filespec = map_filespec
						macro_info.plot_map(region=region, projection=projection,
							graticule_interval=graticule_interval, plot_info=plot_info,
							int_conversion='round', symbol_style=symbol_style,
							line_style=line_style, thematic_num_replies=thematic_num_replies,
							interpolate_grid={}, cmap=cmap,
							color_gradient=color_gradient, event_style=event_style,
							admin_level=admin_level, admin_style=admin_style,
							colorbar_style=colorbar_style, radii=[], plot_pie=None,
							title='', fig_filespec=map_filespec, copyright=copyright,
							text_box={}, dpi=dpi, verbose=False)
					else:
						## Sync ROB and BNS maps rather than plotting them both
						shutil.copyfile(rob_filespec, map_filespec)

					"""
					E.EQMapper('/home/seismo/lib/macro_maps/ROB/large/%i'%id_earth,
						E.GMT_region(minlon-1,maxlon+1,minlat-.5,maxlat+.5),"Mercator",
						1000,want_macro=True,macro_ID=id_earth,macro_symbol_style='diamond',
						want_event=True,event_ID=id_earth,event_label=name,want_cities=True,
						city_labels=True,city_label_size=6,copyright_label='Collaborative project of ROB and BNS',
						macro_web_symbol_size_fixed=False,macro_cpt='KVN_intensity_ROB',
						macro_web_minreplies=1, macro_web_minfiability=10.0,
						land_color=RGB(255,255,255), ocean_color=RGB(255,255,255) )
					E.EQMapper('/home/seismo/lib/macro_maps/ROB/small/%i'%id_earth,
						E.GMT_region(longitude-1,longitude+1,latitude-.5,latitude+.5),"Mercator",
						1000,want_macro=True,macro_ID=id_earth,macro_symbol_style='diamond',
						want_event=True,event_ID=id_earth,event_label=name,want_cities=True,
						city_labels=True,city_label_size=6,copyright_label='Collaborative project of ROB and BNS',
						macro_web_symbol_size_fixed=False,macro_cpt='KVN_intensity_ROB',
						macro_web_minreplies=1, macro_web_minfiability=10.0,
						land_color=RGB(255,255,255), ocean_color=RGB(255,255,255) )
					"""

	else:
		print("Not enough data to draw a map (<3 replies)")
	print("--------------------------------------")
