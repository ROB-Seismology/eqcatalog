# -*- coding: iso-Latin-1 -*-

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
				min_fiability=20.0, filter_floors=(0, 4), aggregate_by="commune",
				agg_function="average", int_conversion="round", symbol_style=None,
				cmap="rob", color_gradient="discontinuous", event_style="default",
				radii=[], recalc=False, plot_pie=None, title="", fig_filespec=None,
				ax=None, copyright=u"© ROB", verbose=True):
	"""
	Plot macroseismic map for given earthquake

	:param aggregate_by:
		str, 'commune', 'main commune', 'grid_X' (where X is grid spacing
		in km) or '' or None, specifying how macroseismic data should
		be aggregated in the map.
		Empty string or None means no aggregation, i.e. all replies are
		plotted individually (potentially on top of each other)
		(default: 'commune')
	:param int_conversion:
		str, "floor", "round" or "ceil"
		(default: "round", corresponding to the convention of Wald et al.)
	"""
	query_info = query_info.lower()
	if query_info == 'num_replies':
		min_replies = 1

	if aggregate_by == 'commune':
		aggregate_by = 'id_com'
	elif aggregate_by == 'main commune':
		aggregate_by = 'id_main'
	elif not aggregate_by:
		min_replies = 1

	[eq] = eqcatalog.seismodb.query_ROB_LocalEQCatalogByID(id_earth)

	if not recalc:
		if aggregate_by == 'id_com':
			group_by_main_village = False
		elif aggregate_by == 'id_main':
			group_by_main_village = True
		else:
			print("Use recalc=True for aggregate_by=%s" % aggregate_by)
			exit()
		qi = query_info if query_info in ('cii', 'cdi') else 'cii'
		recs = eq.get_macroseismic_data_aggregated_web(min_replies=min_replies,
				query_info=qi, min_val=min_val, min_fiability=min_fiability,
				group_by_main_village=group_by_main_village, filter_floors=filter_floors,
				agg_function=agg_function, verbose=verbose)
		try:
			unassigned = recs.pop(None)
		except KeyError:
			unassigned = None

		macro_recs = recs.values()

	else:
		# TODO: agg_function?
		from eqcatalog.macrorecord import MacroseismicRecord

		ensemble = eqcatalog.seismodb.query_ROB_Web_enquiries(id_earth,
						min_fiability=min_fiability, verbose=verbose)
		ensemble = ensemble.filter_floors(*filter_floors, keep_nan_values=True)
		ensemble.fix_felt_is_none()
		ensemble.fix_commune_ids()
		ensemble.set_main_commune_ids()

		if aggregate_by in ('id_com', 'id_main'):
			comm_key = aggregate_by
			ensemble.set_locations_from_communes(comm_key=comm_key, keep_unmatched=False)
			agg_ensemble_dict = ensemble.aggregate_by_commune(comm_key=comm_key)
		elif not aggregate_by:
			if np.isnan(ensemble.get_prop_values('longitude')).all():
				ensemble.set_locations_from_communes(comm_key='id_com')
			agg_ensemble_dict = {}
			for subensemble in ensemble:
				lon = subensemble.recs[0]['longitude']
				if lon:
					lat = subensemble.recs[0]['latitude']
					id_web = subensemble.recs[0]['id_web']
					agg_ensemble_dict[id_web] = subensemble
		elif aggregate_by[:4] == 'grid':
			if np.isnan(ensemble.get_prop_values('longitude')).all():
				ensemble.set_locations_from_communes(comm_key='id_com')
			if '_' in aggregate_by:
				_, grid_spacing = aggregate_by.split('_')
				grid_spacing = float(grid_spacing)
			else:
				grid_spacing = 5
			aggregate_by = 'grid'
			agg_ensemble_dict = ensemble.aggregate_by_grid(grid_spacing)

		try:
			unassigned = agg_ensemble_dict.pop(0)
		except KeyError:
			unassigned = None

		macro_recs = []
		for key in agg_ensemble_dict.keys():
			num_replies = agg_ensemble_dict[key].num_replies
			if num_replies < min_replies:
				agg_ensemble_dict.pop(key)
				continue
			if aggregate_by in ('id_com', 'id_main'):
				id_com = agg_ensemble_dict[key].recs[0][comm_key]
			else:
				id_com = key
			if aggregate_by in ('id_com', 'id_main') or not aggregate_by:
				lon = agg_ensemble_dict[key].recs[0]['longitude']
				lat = agg_ensemble_dict[key].recs[0]['latitude']
			elif aggregate_by == 'grid':
				lon, lat = key
			web_ids = agg_ensemble_dict[key].get_prop_values('id_web')
			if query_info == 'cii':
				I = agg_ensemble_dict[key].calc_cii()
				#I = agg_ensemble_dict[key].calc_mean_cii(filter_floors=None,
				#				include_other_felt=False, remove_outliers=(0,100))
			elif query_info == 'cdi':
				I = agg_ensemble_dict[key].calc_cdi()
			elif query_info == "num_replies":
				I = 1
			else:
				print("Don't know how to recompute %s" % query_info)
				exit()
			rec = MacroseismicRecord(id_earth, id_com, I, num_replies, lon, lat, web_ids)
			macro_recs.append(rec)

	if unassigned:
		print("Note: %d enquiries are not assigned to a commune" % unassigned.num_replies)

	tot_num_replies = np.sum([rec.num_replies for rec in macro_recs])
	if verbose:
		print("Found %d aggregates (%d replies) for event %d:"
				% (len(macro_recs), tot_num_replies, id_earth))
		intensities = [rec.I for rec in macro_recs]
		idxs = np.argsort(intensities)
		for idx in idxs:
			print("  %s : %.2f (n=%d)" % (macro_recs[idx].id_com,
							macro_recs[idx].I, macro_recs[idx].num_replies))
			#if recs.keys()[idx] == 3815:
			#	print macro_recs[idx].web_ids

	#if symbol_style:
	#	gis_filename = "Bel_villages_points.TAB"
	gis_filename = ""
	if not symbol_style:
		if aggregate_by == 'id_main':
			gis_filename = "Bel_villages_polygons.TAB"
		elif aggregate_by == 'id_com':
			gis_filename = "Bel_communes_avant_fusion.TAB"
	if gis_filename:
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
			print("  %s : %.2f (n=%d)" % (macro_recs[idx].id_com, macro_recs[idx].cii,
												macro_recs[idx].num_replies))

	layers = []

	## Commune layer
	if aggregate_by == 'grid':
		import mapping.geotools.coordtrans as ct
		symbol_style = None
		X_left = np.array([rec.lon for rec in macro_recs])
		Y_bottom = np.array([rec.lat for rec in macro_recs])
		X_right = X_left + grid_spacing * 1000
		Y_top = Y_bottom + grid_spacing * 1000
		all_lons, all_lats = [], []
		for i in range(len(macro_recs)):
			X = [X_left[i], X_right[i], X_right[i], X_left[i], X_left[i]]
			Y = [Y_bottom[i], Y_bottom[i], Y_top[i], Y_top[i], Y_bottom[i]]
			lons, lats = ct.transform_array_coordinates(ct.lambert1972, ct.wgs84, X, Y)
			all_lons.append(lons)
			all_lats.append(lats)
		values = {}
		values[query_info] = [getattr(rec, query_info) for rec in macro_recs]
		if query_info != 'num_replies':
			values['num_replies'] = [rec.num_replies for rec in macro_recs]
		commune_data = lbm.MultiPolygonData(all_lons, all_lats, values=values)

	elif symbol_style:
		lons = [rec.lon for rec in macro_recs]
		lats = [rec.lat for rec in macro_recs]
		values = {}
		values[query_info] = [getattr(rec, query_info) for rec in macro_recs]
		if query_info != 'num_replies':
			values['num_replies'] = [rec.num_replies for rec in macro_recs]
		if len(lons):
			commune_data = lbm.MultiPointData(lons, lats, values=values)
		else:
			commune_data = None
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
		#commune_style = lbm.CompositeStyle(polygon_style=polygon_style)
		commune_style = polygon_style
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

	if commune_data:
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
					enq_ensemble = agg_ensemble_dict[rec.id_com]
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

	## Plot copyright box
	if copyright:
		text_style = lbm.TextStyle(font_size=10, color='w', background_color='k',
					horizontal_alignment='left', vertical_alignment='bottom',
					font_weight='bold', alpha=1)
		offset = text_style.border_pad * text_style.font_size
		text_data = lbm.TextData(offset, offset, copyright, coord_frame="axes points")
		copyright_layer = lbm.MapLayer(text_data, text_style)
		layers.append(copyright_layer)

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



if __name__ == "__main__":
	#region = (2, 7, 49.25, 51.75)
	region = (4.9, 6, 50.5, 51.5)
	projection = "tmerc"
	graticule_interval = (0.5, 1)
	#id_earth = 509
	id_earth = 6625
	min_replies = 1
	filter_floors = (-100, 900)
	query_info = "cii"
	#query_info = "num_replies"
	min_fiability = 20
	aggregate_by = 'commune'
	color_gradient = "continuous"
	cmap = "rob"
	#cmap = "jet"
	symbol_style = lbm.PointStyle(shape='D', size=5)
	#symbol_style = None
	#radii = [10, 25, 50]
	radii = []
	plot_pie = False
	#title = "Kinrooi 25/05/2018"
	title = ""

	out_folder = r"D:\Earthquake Reports\20180525\plots"
	fig_filename = "Kinrooi_id_com_num_replies.PNG"
	#fig_filespec = os.path.join(out_folder, fig_filename)
	fig_filespec = None

	plot_macroseismic_map(id_earth, region=region, projection=projection,
					graticule_interval=graticule_interval, min_replies=min_replies,
					query_info=query_info, min_fiability=min_fiability, filter_floors=filter_floors,
					cmap=cmap, color_gradient=color_gradient, symbol_style=symbol_style,
					aggregate_by=aggregate_by, radii=radii,
					plot_pie=plot_pie, title=title, fig_filespec=fig_filespec,
					recalc=True)
