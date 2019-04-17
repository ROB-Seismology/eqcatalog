"""
Plot map of Imax by commune
"""


import os
import mapping.layeredbasemap as lbm
from eqcatalog import macro
from eqcatalog.plot import plot_macroseismic_map


#print(macro.get_eq_intensities_for_commune_official(6, min_or_max='min', as_main_commune=False))
#print(macro.get_eq_intensities_for_commune_official(6, min_or_max='max', as_main_commune=False))
#print(macro.get_eq_intensities_for_commune_web(6, as_main_commune=False, include_other_felt=False))
#exit()

#data_type = 'all'
#data_type = 'isoseismal+traditional'
#data_type = "traditional"
data_type = 'online'
#data_type = 'isoseismal'
by_main_commune = True
#count_exceedances = None
count_exceedances = 3
macro_info_coll = macro.get_imax_by_commune(data_type=data_type, include_other_felt=True,
									by_main_commune=by_main_commune,
									count_exceedances=count_exceedances, verbose=True)
print(sum(macro.num_replies for macro in macro_info_coll))
#print([macro.I for macro in macro_info_coll])
#for id_com in comm_macro_dict:
#	macro_info = comm_macro_dict[id_com]
#	print("%d: Imax=%d (n=%d)" % (id_com, macro_info.I, macro_info.num_replies))

if count_exceedances:
	fig_filename = "NumExI=%d" % count_exceedances
	title = "Exceedances (I>=%s) by commune (%s)" % (count_exceedances, data_type)
	plot_info = 'num_replies'
	colorbar_style = lbm.ColorbarStyle('Number of exceedances')
	cmap = "jet"
else:
	fig_filename = "Imax"
	title = "Maximum intensity by commune (%s)" % data_type
	plot_info = 'intensity'
	colorbar_style = "default"
	#cmap = "rob"
	cmap = "usgs"
color_gradient = "discrete"
#color_gradient = "continuous"


region = (2, 7, 49.25, 51.75)
projection = "merc"
graticule_interval = (2, 1)
#fig_folder = "C:\\Temp"
fig_folder = "E:\\Home\\_kris\\Meetings\\2019 - Afdelingsvergadering"
if by_main_commune:
	fig_filename += "_by_main_commune_%s.PNG"
else:
	fig_filename += "_by_commune_%s.PNG"
fig_filename %= data_type
fig_filespec = os.path.join(fig_folder, fig_filename)
#fig_filespec = None


macro_info_coll.plot_map(region=region, projection=projection,
				graticule_interval=graticule_interval,
				plot_info=plot_info, colorbar_style=colorbar_style,
				color_gradient=color_gradient, cmap=cmap,
				event_style=None, title=title,
				fig_filespec=fig_filespec)
