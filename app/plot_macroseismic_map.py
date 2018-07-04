"""
Plot intensity map
"""

import os
import numpy as np
import matplotlib

import eqcatalog
import mapping.layeredbasemap as lbm


GIS_FOLDER = r"D:\seismo-gis\collections\Bel_administrative_ROB\TAB"


def plot_macroseismic_map(id_earth, region=(2, 7, 49.25, 51.75), projection="tmerc",
				graticule_interval=(1, 1), min_replies=3, query_info="cii", min_val=1,
				min_fiability=20.0, filter_floors=(0, 4), group_by_main_village=False, agg_function="average",
				int_conversion="round", symbol_style=None, cmap="rob",
				color_gradient="discontinuous", event_style="default",
				radii=[], recalc=False, plot_pie=None, title="", fig_filespec=None,
				ax=None, verbose=True):
	"""
	Plot macroseismic map for given earthquake

	:param int_conversion:
		str, "floor", "round" or "ceil"
		(default: "round", corresponding to the convention of Wald et al.)
	"""
	# TODO: plot individual points, replace group_by_main_village with
	# group_by option, allowing None
	query_info = query_info.lower()
	if query_info == 'num_replies':
		min_replies = 1

	[eq] = eqcatalog.seismodb.query_ROB_LocalEQCatalogByID(id_earth)

	if not recalc:
		# TODO: filter floors!
		qi = query_info if query_info in ('cii', 'cdi') else 'cii'
		recs = eq.get_macroseismic_data_aggregated_web(min_replies=min_replies,
				query_info=qi, min_val=min_val, min_fiability=min_fiability,
				group_by_main_village=group_by_main_village, agg_function=agg_function,
				verbose=verbose)
		unassigned = recs.pop(None)
		macro_recs = recs.values()

	else:
		# TODO: agg_function
		from eqcatalog.macrorecord import MacroseismicRecord

		ensemble = eqcatalog.seismodb.query_ROB_Web_enquiries(id_earth,
						min_fiability=min_fiability, verbose=verbose)
		ensemble = ensemble.filter_floors(*filter_floors, keep_nan_values=True)
		ensemble.fix_felt_is_none()
		ensemble.fix_commune_ids()
		ensemble.set_main_commune_ids()
		comm_key = {False: 'id_com', True: 'id_main'}[group_by_main_village]
		ensemble.set_locations_from_communes(comm_key=comm_key, keep_unmatched=False)
		comm_ensemble_dict = ensemble.aggregate_by_commune(comm_key=comm_key)
		unassigned = comm_ensemble_dict.pop(0)
		macro_recs = []
		for key in comm_ensemble_dict.keys():
			if comm_ensemble_dict[key].num_replies < min_replies:
				comm_ensemble_dict.pop(key)
				continue
			id_com = comm_ensemble_dict[key].recs[0][comm_key]
			num_replies = len(comm_ensemble_dict[key])
			lon = comm_ensemble_dict[key].recs[0]['longitude']
			lat = comm_ensemble_dict[key].recs[0]['latitude']
			web_ids = comm_ensemble_dict[key].get_prop_values('id_web')
			if query_info == 'cii':
				I = comm_ensemble_dict[key].calc_cii()
				#I = comm_ensemble_dict[key].calc_mean_cii(filter_floors=None,
				#				include_other_felt=False, remove_outliers=(0,100))
			elif query_info == 'cdi':
				I = comm_ensemble_dict[key].calc_cdi()
			elif query_info == "num_replies":
				I = 1
			else:
				print("Don't know how to recompute %s" % query_info)
				exit()
			rec = MacroseismicRecord(id_earth, id_com, I, num_replies, lon, lat, web_ids)
			macro_recs.append(rec)

	print("Note: %d enquiries are not assigned to a commune" % unassigned.num_replies)

	tot_num_replies = np.sum([rec.num_replies for rec in macro_recs])
	if verbose:
		print("Found %d records (%d replies) for event %d:"
				% (len(macro_recs), tot_num_replies, id_earth))
		intensities = [rec.I for rec in macro_recs]
		idxs = np.argsort(intensities)
		for idx in idxs:
			print("  %4d : %.2f (n=%d)" % (macro_recs[idx].id_com,
							macro_recs[idx].I, macro_recs[idx].num_replies))
			#if recs.keys()[idx] == 3815:
			#	print macro_recs[idx].web_ids

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

	if query_info in ('cii', 'cdi'):
		intensities = np.array([rec.I for rec in macro_recs])

		if color_gradient == "discontinuous":
			#intensities = np.floor(intensities).astype('int')
			intensities = getattr(np, int_conversion)(intensities).astype('int')
		for r, rec in enumerate(macro_recs):
			setattr(rec, query_info, intensities[r])
		idxs = np.argsort(intensities)
		for idx in idxs:
			print("  %4d : %.2f (n=%d)" % (macro_recs[idx].id_com, macro_recs[idx].cii,
												macro_recs[idx].num_replies))

	layers = []

	## Commune layer
	if symbol_style:
		lons = [rec.lon for rec in macro_recs]
		lats = [rec.lat for rec in macro_recs]
		values = {}
		values[query_info] = [getattr(rec, query_info) for rec in macro_recs]
		if query_info != 'num_replies':
			values['num_replies'] = [rec.num_replies for rec in macro_recs]
		commune_data = lbm.MultiPointData(lons, lats, values=values)
	else:
		key = "ID_ROB"
		joined_attributes = {}
		joined_attributes[query_info] = {'key': key,
			'values': {rec.id_com: getattr(rec, query_info) for rec in macro_recs}}
		if query_info != 'num_replies':
			joined_attributes['num_replies'] = {'key': key,
				'values': {rec.id_com: rec.num_replies for rec in macro_recs}}
		#print(joined_attributes)
		if verbose:
			Imax = np.nanmax(joined_attributes[query_info]['values'].values())
			print("Max %s: %s" % (query_info, Imax))

		commune_data = lbm.GisData(gis_filespec, joined_attributes=joined_attributes)
		#_, _, polygon_data = commune_data.get_data()
		#print [val for val in polygon_data.values['cii'] if val != None]


	if query_info in ('cii', 'cdi'):
		classes = np.arange(1, cmap.N + 1)
		cb_title = {'cii': "Community Internet Intensity",
					'cdi': "Community Decimal Intensity"}[query_info]
	elif query_info == 'num_replies':
		classes = np.array([1, 3, 5, 10, 20, 50, 100, 200, 500, 1000])
		cb_title = "Number of replies"

	if color_gradient == "discontinuous":
		if query_info in ('cii', 'cdi'):
			tfc = lbm.ThematicStyleIndividual(classes, cmap, value_key=query_info,
										labels=["%d" % val for val in classes],
										style_under='w')
		elif query_info == 'num_replies':
			tfc = lbm.ThematicStyleRanges(classes, cmap, value_key=query_info,
										labels=["%d" % val for val in classes],
										style_under='w', style_bad='w')
	elif color_gradient == "continuous":
		tfc = lbm.ThematicStyleGradient(classes, cmap, value_key=query_info,
								labels=["%d" % val for val in classes],
								style_under='w')

	colorbar_style = lbm.ColorbarStyle(location="bottom", format="%d", title=cb_title,
										spacing="uniform")
	#colorbar_style = None
	tfc.colorbar_style = colorbar_style
	thematic_legend_style = lbm.LegendStyle(location=4)

	if not symbol_style:
		## Plot polygons
		ta = 1.
		tfh = None
		if query_info != 'num_replies':
			## Set transparency (or hatching) in function of number of replies
			num_replies = [1, 3, 5, 10, 20, 50, 500]
			#tfh = lbm.ThematicStyleRanges(num_replies, ['', '.....', '....', '...', '..', ''],
			#								value_key="num_replies")
			if color_gradient == "discontinuous":
				ta = lbm.ThematicStyleGradient(num_replies, [0.1, 0.3, 0.5, 0.625, 0.75, 0.875, 1.],
											value_key="num_replies")
		polygon_style = lbm.PolygonStyle(fill_color=tfc, line_width=0.1,
					fill_hatch=tfh, alpha=ta, thematic_legend_style=thematic_legend_style)
		commune_style = lbm.CompositeStyle(polygon_style=polygon_style)
		legend_label = {"polygons": ""}
	else:
		## Plot points
		## Symbol size in function of number of replies
		if query_info != 'num_replies':
			num_replies = np.array([1, 3, 5, 10, 20, 50, 100, 500])
			num_replies = num_replies[num_replies >= min_replies]
			symbol_size = symbol_style.size
			sizes = symbol_size + np.log10(num_replies-min_replies+1) * symbol_size
			ts = lbm.ThematicStyleGradient(num_replies, sizes, value_key="num_replies")
			symbol_style.size = ts
		symbol_style.fill_color = tfc
		#symbol_style.thematic_legend_style = thematic_legend_style
		#commune_style = lbm.CompositeStyle(point_style=symbol_style)
		commune_style = symbol_style
		legend_label = ""

	commune_layer = lbm.MapLayer(commune_data, commune_style, legend_label=legend_label)
	layers.append(commune_layer)

	## Province layer
	data = lbm.GisData(os.path.join(GIS_FOLDER, "Bel_provinces.TAB"))
	polygon_style = lbm.PolygonStyle(line_width=1, fill_color='none')
	gis_style = lbm.CompositeStyle(polygon_style=polygon_style)
	province_layer = lbm.MapLayer(data, gis_style, legend_label={"polygons": ""})
	layers.append(province_layer)

	## Pie charts
	if plot_pie:
		prop = "asleep"
		lons, lats = [], []
		ratios = []
		sizes = []
		for rec in macro_recs:
			if rec.num_replies >= 25:
				if not recalc:
					enq_ensemble = rec.get_enquiries()
				else:
					enq_ensemble = comm_ensemble_dict[rec.id_com]
				lons.append(rec.lon)
				lats.append(rec.lat)
				sizes.append(np.sqrt(rec.num_replies)*2)
				bins, counts = enq_ensemble.bincount(prop, [0,1,2])
				ratios.append(counts)
				#print rec.id_com, rec.num_replies, bins, counts
				#print rec.id_com, rec.I, enq_ensemble.CII.mean()
		#print enq_ensemble.CII
		#print 3.40 * np.log(enq_ensemble.CWS) - 4.38

		pie_data = lbm.PiechartData(lons, lats, ratios, sizes)
		colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
		pie_style = lbm.PiechartStyle(colors, start_angle=90, alpha=0.75)
		pie_layer = lbm.MapLayer(pie_data, pie_style)
		layers.append(pie_layer)

	## Plot event
	if event_style == "default":
		label_style = lbm.TextStyle(font_size=9)
		event_style = lbm.PointStyle('*', size=14, fill_color='magenta',
								line_color=None, label_style=label_style)
	if event_style:
		#label = "%s - ML=%.1f" % (eq.date.isoformat(), eq.ML)
		label = ""
		event_data = lbm.PointData(eq.lon, eq.lat, label=label)
		event_layer = lbm.MapLayer(event_data, event_style, legend_label="Epicenter")
		layers.append(event_layer)

	## Plot radii around epicenter
	if radii:
		color = "brown"
		if event_style and event_style.fill_color:
			color = event_style.fill_color
		n = len(radii)
		circle_data = lbm.CircleData([eq.lon]*n, [eq.lat]*n, radii)
		circle_style = lbm.LineStyle(line_width=1, line_color=color)
		circle_layer = lbm.MapLayer(circle_data, circle_style,
									legend_label="%s-km radius" % radii)
		layers.append(circle_layer)

	label_style = lbm.TextStyle(font_size=10)
	legend_style = lbm.LegendStyle(location=1, label_style=label_style)
	map = lbm.LayeredBasemap(layers, title, projection, region=region,
				graticule_interval=graticule_interval, legend_style=legend_style,
				ax=ax)
	if fig_filespec:
		dpi = 300
	else:
		dpi = 90
	map.plot(fig_filespec=fig_filespec, dpi=dpi)

#def plot_macro_web(psf, region, proj, event_ID, min_replies=3, min_fiability=20.0, plot_info="Intensity", cpt="KVN_intensity_USGS", legend=True, group_by_main_village=False, agg_function="", min_intensity=1, symbol_style=None, symbol_size=0.25, symbol_size_fixed=True, verbose=True, errf=None):


# TODO: plot number of replies, aggregate by grid


if __name__ == "__main__":
	#region = (2, 7, 49.25, 51.75)
	region = (4.9, 6, 50.5, 51.5)
	projection = "tmerc"
	graticule_interval = (0.5, 1)
	id_earth = 6625
	min_replies = 3
	#query_info = "cii"
	query_info = "num_replies"
	min_fiability = 20
	group_by_main_village = False
	color_gradient = "discontinuous"
	#cmap = "rob"
	cmap = "jet"
	#symbol_style = lbm.PointStyle(shape='D', size=5)
	symbol_style = None
	#radii = [10, 25, 50]
	radii = []
	plot_pie = False
	#title = "Kinrooi 25/05/2018"
	title = ""

	out_folder = r"D:\Earthquake Reports\20180525\plots"
	fig_filename = "Kinrooi_id_com_num_replies.PNG"
	fig_filespec = os.path.join(out_folder, fig_filename)
	#fig_filespec = None

	plot_macroseismic_map(id_earth, region=region, projection=projection,
					graticule_interval=graticule_interval, min_replies=min_replies,
					query_info=query_info, min_fiability=min_fiability, filter_floors=(-100,900),
					cmap=cmap, color_gradient=color_gradient, symbol_style=symbol_style,
					group_by_main_village=group_by_main_village, radii=radii,
					plot_pie=plot_pie, title=title, fig_filespec=fig_filespec,
					recalc=True)
