"""
Plot intensity map
"""

import os
import numpy as np

import eqcatalog
import mapping.layeredbasemap as lbm


GIS_FOLDER = r"D:\seismo-gis\collections\Bel_administrative_ROB\TAB"


def plot_macroseismic_map(id_earth, region=(2, 7, 49.25, 51.75), projection="tmerc",
				graticule_interval=(1, 1), min_replies=3, query_info="cii", min_val=1,
				min_fiability=20.0, group_by_main_village=False, agg_function="average",
				round_function="floor", symbol_style=None, cmap="rob",
				color_gradient="discontinuous", title="", verbose=True):
	"""
	:param round_func:
		str, "floor", "round" or "ceil"
	"""

	[eq] = eqcatalog.seismodb.query_ROB_LocalEQCatalogByID(id_earth)
	recs = eq.get_macroseismic_data_aggregated_web(min_replies=min_replies,
			query_info=query_info, min_val=min_val, min_fiability=min_fiability,
			group_by_main_village=group_by_main_village, agg_function=agg_function,
			verbose=verbose)

	unassigned = recs.pop(None)
	print("Note: %d enquiries are not assigned to a commune" % unassigned.num_replies)

	if verbose:
		tot_num_replies = np.sum([rec.num_replies for rec in recs.values()])
		print("Found %d records (%d replies) for event %d:" % (len(recs), tot_num_replies, id_earth))
		intensities = [rec.I for rec in recs.values()]
		idxs = np.argsort(intensities)
		for idx in idxs:
			print("  %4d : %.2f (n=%d)" % (recs.keys()[idx], recs.values()[idx].I, recs.values()[idx].num_replies))

	if symbol_style:
		gis_filename = "Bel_villages_points.TAB"
	else:
		if group_by_main_village:
			gis_filename = "Bel_villages_polygons.TAB"
		else:
			gis_filename = "Bel_communes_avant_fusion.TAB"
	gis_filespec = os.path.join(GIS_FOLDER, gis_filename)

	cmap_name = cmap
	if cmap_name.lower() in ("usgs", "rob"):
		cmap = lbm.cm.get_cmap("macroseismic", cmap_name)
	else:
		cmap = matplotlib.cm.get_cmap(cmap_name)

	if query_info == 'cii':
		intensities = np.array([rec.I for rec in recs.values()])
		if color_gradient == "discontinuous":
			#intensities = np.floor(intensities).astype('int')
			intensities = getattr(np, round_function)(intensities).astype('int')
		for r, rec in enumerate(recs.values()):
			rec.cii = intensities[r]


	key = "ID_ROB"
	joined_attributes = {}
	joined_attributes[query_info] = {'key': key,
		'values': {rec.id_com: getattr(rec, query_info) for rec in recs.values()}}
	joined_attributes['num_replies'] = {'key': key,
		'values': {rec.id_com: rec.num_replies for rec in recs.values()}}
	print(joined_attributes)
	if verbose:
		Imax = np.max(joined_attributes[query_info]['values'].values())
		print("Max Intensity: %s" % Imax)

	layers = []

	## Commune layer
	gis_data = lbm.GisData(gis_filespec, joined_attributes=joined_attributes)
	#_, _, polygon_data = gis_data.get_data()
	#print [val for val in polygon_data.values['cii'] if val != None]


	if query_info == 'cii':
		classes = np.arange(1, cmap.N + 1)
		if color_gradient == "discontinuous":
			#ts = lbm.ThematicStyleColormap(cmap, value_key=query_info, vmin=1, vmax=7)
			tfc = lbm.ThematicStyleIndividual(classes, cmap, value_key=query_info,
											labels=["%d" % val for val in classes],
											style_under='w')
			cb_title = "Community Internet Intensity"

		elif color_gradient == "continuous":
			tfc = lbm.ThematicStyleGradient(classes, cmap, value_key=query_info,
									labels=["%d" % val for val in intensities],
									style_under='w')
			cb_title = "Community Decimal Intensity"

			#values = {'cii': [0, 1, 7, 9, np.nan]}
			#print ts(values)
			#print ts.color_under
			#exit()

	colorbar_style = lbm.ColorbarStyle(location="bottom", format="%d", title=cb_title)
	tfc.colorbar_style = colorbar_style
	thematic_legend_style = lbm.LegendStyle()

	if not symbol_style:
		## Plot polygons
		num_replies = [1, 3, 5, 10, 20, 50, 500]
		tfh = lbm.ThematicStyleRanges(num_replies, ['', '.....', '....', '...', '..', ''],
										value_key="num_replies")
		tfh = None
		ta = lbm.ThematicStyleGradient(num_replies, [0.1, 0.3, 0.5, 0.625, 0.75, 0.875, 1.],
										value_key="num_replies")
		polygon_style = lbm.PolygonStyle(fill_color=tfc, line_width=0.1,
					fill_hatch=tfh, alpha=ta, thematic_legend_style=thematic_legend_style)
		style = lbm.CompositeStyle(polygon_style=polygon_style)
	else:
		## Plot points
		## Symbol size in function of number of replies
		num_replies = np.array([1, 3, 5, 10, 20, 50, 100, 500])
		num_replies = num_replies[num_replies >= min_replies]
		symbol_size = symbol_style.size
		sizes = symbol_size + np.log10(num_replies-min_replies+1) * symbol_size
		ts = lbm.ThematicStyleGradient(num_replies, sizes, value_key="num_replies")
		symbol_style.size = ts
		symbol_style.fill_color = tfc
		#symbol_style.thematic_legend_style = thematic_legend_style
		style = lbm.CompositeStyle(point_style=symbol_style)

	commune_layer = lbm.MapLayer(gis_data, style, legend_label={"polygons": ""})
	layers.append(commune_layer)

	## Province layer
	data = lbm.GisData(os.path.join(GIS_FOLDER, "Bel_provinces.TAB"))
	polygon_style = lbm.PolygonStyle(line_width=1, fill_color='none')
	gis_style = lbm.CompositeStyle(polygon_style=polygon_style)
	province_layer = lbm.MapLayer(data, gis_style, legend_label={"polygons": ""})
	layers.append(province_layer)

	map = lbm.LayeredBasemap(layers, title, projection, region=region,
							graticule_interval=graticule_interval)
	fig_filespec = None
	#fig_filespec = r"C:\Temp\seismodb_%s.png" % attribute
	map.plot(fig_filespec=fig_filespec)

#def plot_macro_web(psf, region, proj, event_ID, min_replies=3, min_fiability=20.0, plot_info="Intensity", cpt="KVN_intensity_USGS", legend=True, group_by_main_village=False, agg_function="", min_intensity=1, symbol_style=None, symbol_size=0.25, symbol_size_fixed=True, verbose=True, errf=None):


if __name__ == "__main__":
	#region = (2, 7, 49.25, 51.75)
	region = (4.9, 6, 50.5, 51.5)
	projection = "tmerc"
	graticule_interval = (0.5, 1)
	id_earth = 6625
	min_replies = 1
	query_info = "cii"
	min_fiability = 20
	group_by_main_village = False
	color_gradient = "discontinuous"
	symbol_style = lbm.PointStyle(shape='D', size=5)
	#symbol_style = None
	title = "Kinrooi 25/05/2018"
	plot_macroseismic_map(id_earth, region=region, projection=projection,
					graticule_interval=graticule_interval, min_replies=min_replies,
					query_info=query_info, min_fiability=min_fiability,
					color_gradient=color_gradient, symbol_style=symbol_style,
					group_by_main_village=group_by_main_village, title=title)
