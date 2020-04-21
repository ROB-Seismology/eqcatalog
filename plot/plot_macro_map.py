# -*- coding: iso-Latin-1 -*-

"""
Plot macroseismic maps
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import os

import numpy as np
import matplotlib


from ..rob import SEISMOGIS



__all__ = ["plot_macroseismic_map"]


def plot_macroseismic_map(macro_info_coll, region=(2, 7, 49.25, 51.75),
				projection="merc", graticule_interval=(1, 1), plot_info="intensity",
				int_conversion="round", symbol_style=None, line_style="default",
				thematic_num_replies=False, interpolate_grid={},
				cmap="rob", color_gradient="discontinuous", event_style="default",
				country_style="default", admin_level="province", admin_style="default",
				city_style="default", colorbar_style="default", radii=[],
				plot_pie={}, title="", fig_filespec=None,
				ax=None, copyright=u"© ROB", text_box={}, dpi="default",
				border_width=0.2, verbose=True):
	"""
	Plot macroseismic map for given earthquake

	:param macro_info_coll:
		instance of :class:`MacroInfoCollection`, representing
		(aggregated) macroseismic information to plot
	:param region:
		(lonmin, lonmax, latmin, latmax) tuple or str
		If "auto", region will be determined automatically from 95th
		percentile extent of :param:`macro_info_coll`
		(default: (2, 7, 49.25, 51.75))
	:param projection:
		str, name of projection supported in layeredbasemap
		(default: "merc")
	:param graticule_interval:
		(lon_spacing, lat_spacing) tuple
		(default: (1, 1))
	:param plot_info:
		str, information that should be plotted, either 'intensity',
		'num_replies' or 'residual'
		(default: 'intensity')
	:param int_conversion:
		str, "floor", "round" or "ceil", how to convert intensities
		to integers if :param:`color_gradient` == "discontinuous"
		(default: "round", corresponding to the convention of Wald et al.)
	:param symbol_style:
		instance of :class:`mapping.layeredbasemap.SymbolStyle`,
		point style for macroseismic data. If None, data will be plotted
		as commune polygons
		(default: None)
	:param line_style:
		instance of :class:`mapping.layeredbasemap.LineStyle`,
		line style for macroseismic data polygons or contour lines.
		(default: "default").
	:param thematic_num_replies:
		bool, whether or not thematic style should be applied for
		number of replies (symbol size for points, transparency for
		polygons if  color gradient` is discontinuous)
		(default: False)
	:param interpolate_grid:
		dict, containing interpolation parameters (resolution or cell size,
		method, max_dist, and possibly method-dependent parameters)
		Is only taken into account if:
		- agg_type of :param:`macro_info_coll` is None
		- symbol_style is not (None or '') if :param:`macro_info_coll` is None
		(default: {})
	:param cmap:
		str, color map, either "rob" or "usgs" (for intensity)
		or the name of a matplotlib colormap (for num_replies)
		(default: "rob")
	:param color_gradient:
		str, either "continuous" or "discontinuous"
		(default: "discontinuous")
	:param event_style:
		instance of :class:`mapping.layeredbasemap.SymbolStyle`,
		point style for earthquake epicenter
		(default: "default")
	:param country_style:
		instance of :class:`mapping.layeredbasemap.LineStyle`,
		line style for country borders
		(default: "default")
	:param admin_level:
		str, administrative level to plot over intensity map,
		one of 'province', 'region' or 'country'
		(default: 'province')
	:param admin_style:
		instance of :class:`mapping.layeredbasemap.LineStyle`,
		line style for administrative boundaries
		(default: "default")
	:param city_style:
		instance of :class:`mapping.layeredbasemap.PointStyle`,
		point style for main cities
		(default: "default")
	:param colorbar_style:
		instance of :class:`mapping.layeredbasemap.ColorbarStyle`
		or instance of :class:`mapping.layeredbasemap.LegendStyle`
		(default: "default")
	:param radii:
		list of floats, raddii of circles (in km) to plot around
		epicenter
		(default: [])
	:param plot_pie:
		dict, containing parameters to plot pie charts
		('prop', 'min_replies', 'size_scaling', 'pie_style',
		'legend_location',...)
		'prop' (property to plot as pie charts on the map) is required.
		Only applies to internet macroseismic data
		(default: {})
	:param title:
		str, map title
		(default: "")
	:param fig_filespec:
		str, full path to output file, or 'hold' to return map object
		instead of plotting it.
		(default: None, will plot map on screen)
	:param ax:
		matplotlib axes instance in which map should be plotted
		(default: None)
	:param copyright:
		str, copyright label to plot in lower left corner of map
		(default: u"© ROB")
	:param text_box:
		dict, containing text box parameters (pos, text and style)
		(default: {})
	:param dpi:
		int, resolution of output image
		(default: "default" = 90 if :param:`fig_filespec` is None,
		else 300)
	:param border_width:
		float, width of border around map frame in cm
		If None, white space will not be removed
		(default: 0.2)
	:param verbose:
		bool, whether or not to plot some useful information

	:return:
		None
		or instance of :class:`LayeredBasemap` if :param:`ax` is not
		None or if fig_filespec == 'hold'
	"""
	import mapping.layeredbasemap as lbm

	if len(macro_info_coll) == 0:
		print("No macroseismic information provided! Nothing to plot.")
		return

	if verbose:
		tot_num_replies = np.sum([rec.num_replies for rec in macro_info_coll])
		print("Found %d aggregates (%d replies) for event %s:"
				% (len(macro_info_coll), tot_num_replies, macro_info_coll[0].id_earth))
		if verbose > 1:
			idxs = np.argsort(macro_info_coll.intensities)
			for idx in idxs:
				macro_rec = macro_info_coll[idx]
				print("  %s : %.2f (n=%d)" % (macro_rec.id_com,
							macro_rec.I, macro_rec.num_replies))

	if region == "auto":
		lonmin, lonmax, latmin, latmax = macro_info_coll.get_region(95)
		dlon = lonmax - lonmin
		dlat = latmax - latmin
		lonmin -= (dlon * 0.1)
		lonmax += (dlon * 0.1)
		latmin -= (dlat * 0.1)
		latmax += (dlat * 0.1)
		region = (lonmin, lonmax, latmin, latmax)

	plot_polygons_as_points = False
	if symbol_style:
		plot_polygons_as_points = True
	aggregate_by = macro_info_coll.agg_type
	if aggregate_by is None:
		if not interpolate_grid:
			symbol_style = symbol_style or lbm.PointStyle(shape='D', size=5)

	if plot_info == 'intensity' and color_gradient[:4] == "disc":
		## Round intensities
		intensities = macro_info_coll.intensities
		intensities = getattr(np, int_conversion)(intensities).astype('int')
		for r, rec in enumerate(macro_info_coll):
			setattr(rec, 'intensity', intensities[r])

	elif plot_info == 'residual' and aggregate_by is None:
		print("Residuals are zero if data are not aggregated!")
		return

	if line_style == "default":
		line_style = lbm.LineStyle(line_width=0.3, line_color='0.33')

	layers = []

	## Set up thematic style for macroseismic data
	cmap_name = cmap
	if cmap_name.lower() in ("usgs", "rob"):
		cmap = lbm.cm.get_cmap("macroseismic", cmap_name)
	else:
		cmap = matplotlib.cm.get_cmap(cmap_name)

	if plot_info == 'intensity':
		classes = np.arange(1, min(cmap.N, 12) + 1, dtype='int')
		data_type = macro_info_coll.data_type
		cb_title = {'internet': "Community Internet Intensity",
					'online': "Community Internet Intensity",
					'traditional': "Macroseismic Intensity"}.get(data_type,
													"Macroseismic Intensity")
	elif plot_info == 'num_replies':
		classes = np.array([1, 3, 5, 10, 20, 50, 100, 200, 500, 1000], dtype='int')
		cb_title = "Number of replies"

	elif plot_info == 'residual':
		max_residual = np.abs(macro_info_coll.residuals).max()
		if max_residual <= 1.25:
			classes = np.arange(-1.25, 1.3, 0.25)
		elif max_residual <= 2.5:
			classes = np.arange(-2.5, 2.6, 0.5)
		else:
			classes = np.arange(-4, 5, 1)
		cb_title = "Residual Intensity"
		cmap = matplotlib.cm.get_cmap('bwr')

	if color_gradient[:4] == "disc":
		if plot_info == 'intensity':
			tfc = lbm.ThematicStyleIndividual(classes, cmap, value_key=plot_info,
										#labels=["%s" % val for val in classes],
										style_under='w', style_over=cmap(1.),
										style_bad='w')
		elif plot_info in ('num_replies', 'residual'):
			tfc = lbm.ThematicStyleRanges(classes, cmap, value_key=plot_info,
										#labels=["%s" % val for val in classes],
										style_under='w', style_over=cmap(1.),
										style_bad='w')
	elif color_gradient[:4] == "cont":
		tfc = lbm.ThematicStyleGradient(classes, cmap, value_key=plot_info,
								#labels=["%s" % val for val in classes],
								style_under='w', style_over=cmap(1.),
								style_bad='w')

	## Color bar / thematic legend
	if colorbar_style == "default":
		colorbar_style = lbm.ColorbarStyle(location="bottom", format="%d",
							title=cb_title, spacing="uniform", extend='both')
		if plot_info == "num_replies":
			colorbar_style.extend = 'neither'
	if isinstance(colorbar_style, lbm.ColorbarStyle):
		tfc.colorbar_style = colorbar_style
		tfc.labels = tfc.gen_labels(as_ranges=False)
		thematic_legend_style = None
	elif colorbar_style is None:
		thematic_legend_style = lbm.LegendStyle(title=cb_title, location=4)
	elif isinstance(colorbar_style, lbm.LegendStyle):
		thematic_legend_style = colorbar_style
		tfc.labels = tfc.gen_labels(as_ranges=True)
		colorbar_style = None

	## Interpolate grid
	if interpolate_grid and (aggregate_by is None or symbol_style):
		num_cells = interpolate_grid.get('num_cells', 100)
		interpol_method = interpolate_grid.get('method', 'idw')
		interpol_params = interpolate_grid.get('params', {})
		grid_data = macro_info_coll.interpolate_grid(num_cells, region=region,
											prop=plot_info,
											interpolation_method=interpol_method,
											interpolation_params=interpol_params)
		vmin, vmax = classes[0], classes[-1] + 1
		#if cmap_name.lower() in ("usgs", "rob"):
		#	vmax = {'rob': 8, 'usgs': 10}[cmap_name]
		#else:
		#	vmax = 13

		if plot_info == 'intensity':
			if color_gradient == 'continuous':
				if cmap.N < 256:
					cmap = matplotlib.colors.LinearSegmentedColormap(cmap.name,
														cmap._segmentdata, N=256)
			elif color_gradient[:4] == 'disc':
				if cmap.N != (vmax - 1):
					cmap = matplotlib.colors.LinearSegmentedColormap(cmap.name,
														cmap._segmentdata, N=vmax-1)

		tcm = lbm.ThematicStyleColormap(cmap, vmin=vmin, vmax=vmax, style_under='w',
										style_bad='w', colorbar_style=None)
		## Only add colorbar if no other information is plotted
		if not symbol_style:
			if color_gradient[:4] == 'disc':
				colorbar_style.ticks = np.arange(0.5, vmax+1)
				colorbar_style.tick_labels = range(1, vmax)
			else:
				colorbar_style.ticks = range(1, vmax)
				vmax -= 1
			tcm.colorbar_style = colorbar_style
		Imin, Imax = macro_info_coll.Iminmax()
		contour_levels = np.arange(Imin, Imax + 1)
		grid_style = lbm.GridStyle(tcm, color_gradient, contour_levels=contour_levels,
									line_style=line_style)
		grid_layer = lbm.MapLayer(grid_data, grid_style)
		layers.append(grid_layer)

	else:
		interpolate_grid = {}

	if not symbol_style:
		## Plot polygons
		ta = 1.
		tfh = None
		if plot_info != 'num_replies' and thematic_num_replies:
			## Set transparency (or hatching) in function of number of replies
			num_replies = [1, 3, 5, 10, 20, 50, 500]
			#tfh = lbm.ThematicStyleRanges(num_replies, ['', '.....', '....', '...', '..', ''],
			#								value_key="num_replies")
			if color_gradient[:4] == "disc":
				ta = lbm.ThematicStyleGradient(num_replies, [0.1, 0.3, 0.5, 0.625, 0.75, 0.875, 1.],
											value_key="num_replies")
		polygon_style = lbm.PolygonStyle(line_width=0, fill_color=tfc, fill_hatch=tfh,
						alpha=ta, thematic_legend_style=thematic_legend_style)
		if line_style:
			polygon_style.line_pattern = line_style.line_pattern
			polygon_style.line_width = line_style.line_width
			polygon_style.line_color = line_style.line_color
			polygon_style.dash_pattern = line_style.dash_pattern
		macro_style = polygon_style
		legend_label = {"polygons": ""}
	else:
		symbol_style = symbol_style.copy()
		## Plot points
		## Symbol size in function of number of replies
		if plot_info != 'num_replies' and thematic_num_replies:
			num_replies = np.array([1, 3, 5, 10, 20, 50, 100, 500])
			#num_replies = num_replies[num_replies >= min_replies]
			symbol_size = symbol_style.size
			#sizes = symbol_size + np.log10(num_replies-min_replies+1) * symbol_size
			sizes = symbol_size + np.log10(num_replies) * symbol_size
			ts = lbm.ThematicStyleGradient(num_replies, sizes, value_key="num_replies")
			symbol_style.size = ts
		symbol_style.fill_color = tfc
		#symbol_style.thematic_legend_style = thematic_legend_style
		#macro_style = lbm.CompositeStyle(point_style=symbol_style)
		macro_style = symbol_style
		legend_label = ""

	## Macro layer
	if interpolate_grid and not symbol_style:
		macro_geom_data = None
	else:
		macro_geom_data = macro_info_coll.get_geometries(plot_polygons_as_points)
	if macro_geom_data:
		macro_layer = lbm.MapLayer(macro_geom_data, macro_style, legend_label=legend_label)
		layers.append(macro_layer)

	## Country layer
	if country_style == 'default':
		country_style = lbm.LineStyle(line_width=1.0)
	if country_style:
		for feature in ('coastlines', 'countries'):
			country_data = lbm.BuiltinData(feature)
			country_layer = lbm.MapLayer(country_data, country_style)
			layers.append(country_layer)

	## Admin layer
	admin_data = None
	if admin_level:
	if SEISMOGIS is not None:
		coll = SEISMOGIS.read_collection('Bel_administrative_ROB')
		if admin_level.lower() == 'province':
			ds_name ="Bel_provinces.TAB"
		elif admin_level.lower() == 'region':
			ds_name = "Bel_regions.TAB"
		elif admin_level.lower() == 'country':
			ds_name = "Bel_border.TAB"
		elif admin_level.lower() == 'commune':
			ds_name = "Bel_communes_avant_fusion.TAB"
		elif admin_level.lower() == 'main commune':
			ds_name = "Bel_villages_polygons.TAB"
		try:
			[ds] = coll.find_datasets(ds_name)
		except:
			print("%s dataset not found in seismogis!" % ds_name
		else:
			admin_data = lbm.GisData(ds.get_gis_filespec())
	else:
		print('Please install mapping.seismogis module for admin data!')

	if admin_data:
		if admin_style == "default":
			admin_style = lbm.PolygonStyle(line_width=0.5, fill_color='none')
		#gis_style = lbm.CompositeStyle(polygon_style=admin_style)
		admin_layer = lbm.MapLayer(admin_data, admin_style, legend_label={"polygons": ""})
		layers.append(admin_layer)

	if city_style:
		# TODO: label style, thematic size
		if city_style == "default":
			city_symbol_size = 4
			city_label_style = lbm.TextStyle(font_size=6, vertical_alignment="top",
											offset=(0, -4))
			population = np.array([10000, 50000, 100000, 200000, 500000,
						1000000, 2000000, 5000000, 10000000, 20000000])
			sizes = city_symbol_size + np.log10(population / 500000) * city_symbol_size
			ts = lbm.ThematicStyleGradient(population, sizes, value_key="Population_agglomeration")
			city_style = lbm.PointStyle('s', size=ts, fill_color='k', label_style=city_label_style)
		if SEISMOGIS is not None:
			coll = SEISMOGIS.read_collection('UN_Cities')
			try:
				[ds] = coll.find_datasets('UN Cities')
			except:
				print('UN_Cities dataset not found in seismogis!')
			else:
				gis_file = ds.get_gis_filespec()
				city_data = lbm.GisData(gis_file, label_colname="Name")
				city_layer = lbm.MapLayer(city_data, city_style)
				layers.append(city_layer)
		else:
			print('Please install mapping.seismogis module for city data!')

	## Pie charts
	# TODO: legend
	if plot_pie:
		prop = plot_pie['prop']
		pie_min_replies = plot_pie.get('min_replies', 25)
		size_scaling = plot_pie.get('size_scaling', 2)
		pie_style = plot_pie.get('pie_style')
		pie_legend_loc = plot_pie.get('legend_location', 2)
		lons, lats = [], []
		ratios = []
		sizes = []
		for rec in macro_info_coll:
			if rec.num_replies >= pie_min_replies:
				enq_ensemble = rec.get_online_enquiries()
				lons.append(rec.lon)
				lats.append(rec.lat)
				sizes.append(np.sqrt(rec.num_replies) * size_scaling)
				bins, counts = enq_ensemble.bincount(prop)
				ratios.append(counts)

		if len(ratios):
			pie_data = lbm.PiechartData(lons, lats, ratios, sizes)
			if not pie_style:
				colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
				_, pie_labels = enq_ensemble.get_prop_title_and_labels(prop)
				thematic_legend_style = lbm.LegendStyle(prop, location=pie_legend_loc,
											label_style=lbm.TextStyle(font_size=10))
				pie_style = lbm.PiechartStyle(colors, pie_labels, start_angle=90,
								alpha=0.75, thematic_legend_style=thematic_legend_style)
			pie_layer = lbm.MapLayer(pie_data, pie_style)
			layers.append(pie_layer)

	## Plot event
	if event_style == "default":
		label_style = lbm.TextStyle(font_size=9)
		event_style = lbm.PointStyle('*', size=14, fill_color='magenta',
								line_color=None, label_style=label_style)
	if event_style:
		eq = macro_info_coll[0].get_eq()
		legend_label = "%s - ML=%.1f" % (eq.date, eq.ML)
		label = ""
		event_data = lbm.PointData(eq.lon, eq.lat, label=label)
		event_layer = lbm.MapLayer(event_data, event_style, legend_label=legend_label)
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
		text_style = lbm.TextStyle(font_size=8, color='w', background_color='k',
					horizontal_alignment='left', vertical_alignment='bottom',
					font_weight='bold', alpha=1)
		offset = text_style.border_pad * text_style.font_size
		text_data = lbm.TextData(offset, offset, copyright, coord_frame="axes points")
		copyright_layer = lbm.MapLayer(text_data, text_style)
		layers.append(copyright_layer)

	label_style = lbm.TextStyle(font_size=10)
	legend_style = lbm.LegendStyle(location=1, label_style=label_style)
	if graticule_interval:
		graticule_style = lbm.GraticuleStyle()
	else:
		graticule_style = None

	if dpi == 'default':
		if fig_filespec:
			dpi = 300
		else:
			dpi = 90

	map = lbm.LayeredBasemap(layers, title, projection, region=region,
				graticule_interval=graticule_interval, legend_style=legend_style,
				graticule_style=graticule_style, resolution='auto', dpi=dpi, ax=ax)

	## Text box
	if text_box:
		pos = text_box.get('pos', (0, 0))
		text = text_box.get('text', '')
		text_style = text_box.get('style')
		if not text_style:
			text_style = lbm.TextStyle(font_size=10, horizontal_alignment='left',
						vertical_alignment='bottom', multi_alignment='left',
						background_color='w', border_color='k', border_pad=0.5)
		map.draw_text_box(pos, text, text_style, zorder=1000)

	return map.plot(fig_filespec=("hold" if ax else fig_filespec), dpi=dpi,
					border_width=border_width)
