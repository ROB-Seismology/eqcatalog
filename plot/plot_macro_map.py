# -*- coding: iso-Latin-1 -*-

"""
Plot macroseismic maps
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import os

import numpy as np
import matplotlib


from ..rob import SEISMOGIS_ROOT
GIS_FOLDER = os.path.join(SEISMOGIS_ROOT, "collections", "Bel_administrative_ROB", "TAB")
#GIS_FOLDER = "D:\\seismo-gis\\collections\\Bel_administrative_ROB\\TAB"



__all__ = ["plot_macroseismic_map"]


def plot_macroseismic_map(macro_info_coll, region=(2, 7, 49.25, 51.75),
				projection="merc", graticule_interval=(1, 1), plot_info="intensity",
				int_conversion="round", symbol_style=None, thematic_num_replies=False,
				cmap="rob", color_gradient="discontinuous", event_style="default",
				admin_level="province", admin_style="default", colorbar_style="default",
				radii=[], plot_pie=None, title="", fig_filespec=None,
				ax=None, copyright=u"� ROB", verbose=True):
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
		str, "floor", "round" or "ceil"
		(default: "round", corresponding to the convention of Wald et al.)
	:param symbol_style:
		instance of :class:`mapping.layeredbasemap.SymbolStyle`,
		point style for macroseismic data. If None, data will be plotted
		as commune polygons
		(default: None)
	:param thematic_num_replies:
		bool, whether or not thematic style should be applied for
		number of replies (symbol size for points, transparency for
		polygons if  color gradient` is discontinuous)
		(default: False)
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
	:param admin_level:
		str, administrative level to plot over intensity map,
		one of 'province', 'region' or 'country'
		(default: 'province')
	:param admin_style:
		instance of :class:`mapping.layeredbasemap.LineStyle`,
		line style for administrative boundaries
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
		str, name of property to plot as pie charts on the map,
		only applies to internet macroseismic data
		(default: None)
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
		(default: u"� ROB")
	:param verbose:
		bool, whether or not to plot some useful information

	:return:
		None
		or instance of :class:`LayeredBasemap` if :param:`ax` is not
		None or if fig_filespec == 'hold'
	"""
	# TODO: commune_linestyle, etc.
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

	plot_communes_as_points = False
	if symbol_style:
		plot_communes_as_points = True
	aggregate_by = macro_info_coll.agg_type
	if aggregate_by is None:
		symbol_style = symbol_style or lbm.PointStyle(shape='D', size=5)
	elif aggregate_by == 'grid':
		symbol_style = None

	if plot_info == 'intensity' and color_gradient[:4] == "disc":
		## Round intensities
		intensities = macro_info_coll.intensities
		intensities = getattr(np, int_conversion)(intensities).astype('int')
		for r, rec in enumerate(macro_info_coll):
			setattr(rec, 'intensity', intensities[r])

	layers = []

	## Set up thematic style for macroseismic data
	cmap_name = cmap
	if cmap_name.lower() in ("usgs", "rob"):
		cmap = lbm.cm.get_cmap("macroseismic", cmap_name)
	else:
		cmap = matplotlib.cm.get_cmap(cmap_name)

	if plot_info == 'intensity':
		classes = np.arange(1, cmap.N + 1)
		enq_type = macro_info_coll.enq_type
		cb_title = {'internet': "Community Internet Intensity",
					'online': "Community Internet Intensity",
					'official': "Macroseismic Intensity"}.get(enq_type,
													"Macroseismic Intensity")
	elif plot_info == 'num_replies':
		classes = np.array([1, 3, 5, 10, 20, 50, 100, 200, 500, 1000])
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
	elif color_gradient == "continuous":
		tfc = lbm.ThematicStyleGradient(classes, cmap, value_key=plot_info,
								#labels=["%s" % val for val in classes],
								style_under='w', style_over=cmap(1.),
								style_bad='w')

	if colorbar_style == "default":
		colorbar_style = lbm.ColorbarStyle(location="bottom", format="%d",
										title=cb_title, spacing="uniform")
	thematic_legend_style = lbm.LegendStyle(title=cb_title, location=4)
	if isinstance(colorbar_style, lbm.ColorbarStyle):
		tfc.colorbar_style = colorbar_style
		tfc.labels = tfc.gen_labels(as_ranges=False)
	elif isinstance(colorbar_style, lbm.LegendStyle):
		thematic_legend_style = colorbar_style
		tfc.labels = tfc.gen_labels(as_ranges=True)

	if not symbol_style:
		## Plot polygons
		ta = 1.
		tfh = None
		if plot_info != 'num_replies' and thematic_num_replies:
			## Set transparency (or hatching) in function of number of replies
			num_replies = [1, 3, 5, 10, 20, 50, 500]
			#tfh = lbm.ThematicStyleRanges(num_replies, ['', '.....', '....', '...', '..', ''],
			#								value_key="num_replies")
			if color_gradient == "discontinuous":
				ta = lbm.ThematicStyleGradient(num_replies, [0.1, 0.3, 0.5, 0.625, 0.75, 0.875, 1.],
											value_key="num_replies")
		polygon_style = lbm.PolygonStyle(fill_color=tfc, line_width=0.1,
					fill_hatch=tfh, alpha=ta, thematic_legend_style=thematic_legend_style)
		#macro_style = lbm.CompositeStyle(polygon_style=polygon_style)
		macro_style = polygon_style
		legend_label = {"polygons": ""}
	else:
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
	macro_geom_data = macro_info_coll.get_geometries(plot_communes_as_points)
	if macro_geom_data:
		macro_layer = lbm.MapLayer(macro_geom_data, macro_style, legend_label=legend_label)
		layers.append(macro_layer)

		if region == "auto":
			lonmin, lonmax, latmin, latmax = macro_info_coll.get_region(95)
			dlon = lonmax - lonmin
			dlat = latmax - latmin
			lonmin -= (dlon * 0.1)
			lonmax += (dlon * 0.1)
			latmin -= (dlat * 0.1)
			latmax += (dlat * 0.1)
			region = (lonmin, lonmax, latmin, latmax)

	## Admin layer
	admin_data = None
	if admin_level.lower() == 'province':
		admin_data = lbm.GisData(os.path.join(GIS_FOLDER, "Bel_provinces.TAB"))
	elif admin_level.lower() == 'region':
		admin_data = lbm.GisData(os.path.join(GIS_FOLDER, "Bel_regions.TAB"))
	elif admin_level.lower() == 'country':
		admin_data = lbm.GisData(os.path.join(GIS_FOLDER, "Bel_border.TAB"))
	elif admin_level.lower() == 'commune':
		admin_data = lbm.GisData(os.path.join(GIS_FOLDER, "Bel_communes_avant_fusion.TAB"))
	elif admin_level.lower() == 'main commune':
		admin_data = lbm.GisData(os.path.join(GIS_FOLDER, "Bel_villages_polygons.TAB"))

	if admin_data:
		if admin_style == "default":
			admin_style = lbm.PolygonStyle(line_width=1, fill_color='none')
		#gis_style = lbm.CompositeStyle(polygon_style=admin_style)
		admin_layer = lbm.MapLayer(admin_data, admin_style, legend_label={"polygons": ""})
		layers.append(admin_layer)

	## Pie charts
	# TODO: legend
	if plot_pie:
		prop = "asleep"
		lons, lats = [], []
		ratios = []
		sizes = []
		for rec in macro_info_coll:
			if rec.num_replies >= 25:
				enq_ensemble = rec.get_enquiries()
				lons.append(rec.lon)
				lats.append(rec.lat)
				sizes.append(np.sqrt(rec.num_replies)*2)
				bins, counts = enq_ensemble.bincount(prop, [0,1,2])
				ratios.append(counts)
				#print rec.id_com, rec.num_replies, bins, counts
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
		eq = macro_info_coll[0].get_eq()
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
	if graticule_interval:
		graticule_style = lbm.GraticuleStyle()
	else:
		graticule_style = None
	map = lbm.LayeredBasemap(layers, title, projection, region=region,
				graticule_interval=graticule_interval, legend_style=legend_style,
				graticule_style=graticule_style, ax=ax)

	if fig_filespec:
		dpi = 300
	else:
		dpi = 90
	return map.plot(fig_filespec=("hold" if ax else fig_filespec), dpi=dpi)
