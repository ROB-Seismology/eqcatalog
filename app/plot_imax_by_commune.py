"""
Plot map of Imax by commune
"""


import os
from eqcatalog import macro
from eqcatalog.plot import plot_macroseismic_map


#print(macro.get_eq_intensities_for_commune_official(6, min_or_max='min', as_main_commune=False))
#print(macro.get_eq_intensities_for_commune_official(6, min_or_max='max', as_main_commune=False))
#print(macro.get_eq_intensities_for_commune_web(6, as_main_commune=False, include_other_felt=False))
#exit()

enq_type = 'official'
#enq_type = 'online'
by_main_commune = True
macro_info_coll = macro.get_imax_by_commune(enq_type=enq_type, include_other_felt=False,
									by_main_commune=by_main_commune, verbose=False)
print(sum(macro.num_replies for macro in macro_info_coll))
print([macro.I for macro in macro_info_coll])
#for id_com in comm_macro_dict:
#	macro_info = comm_macro_dict[id_com]
#	print("%d: Imax=%d (n=%d)" % (id_com, macro_info.I, macro_info.num_replies))

region = (2, 7, 49.25, 51.75)
projection = "merc"
graticule_interval = (2, 1)
title = "Maximum intensity by commune (%s)" % enq_type
fig_folder = "C:\\Temp"
if by_main_commune:
	fig_filename = "Imax_by_main_commune_%s.PNG"
else:
	fig_filename = "Imax_by_commune_%s.PNG"
fig_filename %= enq_type
#fig_filespec = os.path.join(fig_folder, fig_filename)
fig_filespec = None


macro_info_coll.plot_map(region=region, projection=projection,
				graticule_interval=graticule_interval,
				event_style=None, cmap="usgs", title=title,
				fig_filespec=fig_filespec)

#print(macro_info_coll.to_geojson())

#gis_file = os.path.splitext(fig_filespec)[0] + ".TAB"
#macro_info_coll.export_gis('MapInfo File', gis_file)

#geotiff_file = os.path.splitext(fig_filespec)[0] + ".TIF"
#macro_info_coll.export_geotiff(geotiff_file)
