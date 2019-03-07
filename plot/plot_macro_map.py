# -*- coding: iso-Latin-1 -*-

"""
Plot macroseismic maps
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import os
from collections import OrderedDict

import numpy as np
import matplotlib

import mapping.layeredbasemap as lbm


from ..rob import SEISMOGIS_ROOT
GIS_FOLDER = os.path.join(SEISMOGIS_ROOT, "collections", "Bel_administrative_ROB", "TAB")
#GIS_FOLDER = "D:\\seismo-gis\\collections\\Bel_administrative_ROB\\TAB"



__all__ = ["MacroInfoCollection", "plot_macroseismic_map",
			"plot_official_macroseismic_map", "plot_web_macroseismic_map"]


class MacroInfoCollection():
	"""
	Class representing a collection of aggregated macroseismic records

	:param macro_infos:
		list with instances of :class:`MacroseismicInfo`
	:param agg_type:
		str, aggregation type
	:param enq_type:
		str, type of enquirey, one of:
		- 'internet' or 'online'
		- 'official'
	"""
	def __init__(self, macro_infos, agg_type, enq_type):
		self.macro_infos = macro_infos
		self.agg_type = agg_type
		if not enq_type:
			enq_type = macro_infos[0].enq_type
		self.enq_type = enq_type

	def __len__(self):
		return len(self.macro_infos)

	def __iter__(self):
		return self.macro_infos.__iter__()

	def __getitem__(self, item):
		return self.macro_infos.__getitem__(item)

	@property
	def intensities(self):
		return np.array([rec.I for rec in self])

	@property
	def id_earth(self):
		id_earths = set([rec.id_earth for rec in self])
		if len(id_earths) == 1:
			return id_earths.pop()
		else:
			return sorted(id_earths)

	def to_com_info_dict(self):
		com_info_dict = {}
		for rec in self:
			com_info_dict[rec.id_com] = rec
		return com_info_dict

	def get_geometries(self, communes_as_points=False):
		"""
		Transform aggregated macroseismic information to layeredbasemap
		geometries.

		:param communes_as_points:
			bool, whether to represent communes as points (True)
			or polygons (False)
			(default: False)

		:return:
			instance of :class:`lbm.MultiPolygonData`
			or :class:`lbm.MultiPointData`
		"""
		if len(self) == 0:
			return []

		## Determine aggregation type
		aggregate_by = self.agg_type
		if aggregate_by == 'commune':
			aggregate_by = 'id_com'
		elif aggregate_by == 'main commune':
			aggregate_by = 'id_main'
		elif not aggregate_by:
			communes_as_points = True
		elif aggregate_by[:4] == 'grid':
			if '_' in aggregate_by:
				_, grid_spacing = aggregate_by.split('_')
				grid_spacing = float(grid_spacing)
			else:
				grid_spacing = 5
			aggregate_by = 'grid'

		## Construct attribute dictionary
		values = OrderedDict()
		attributes = ['id_earth', 'intensity', 'num_replies', 'agg_type',
					'enq_type']
		if aggregate_by == 'grid' or communes_as_points:
			if aggregate_by in ('id_main', 'id_com'):
				attributes += ['id_com']
			attributes += ['lon', 'lat']
			for attrib in attributes:
				values[attrib] = [getattr(rec, attrib) for rec in self]
		else:
			join_key = "ID_ROB"
			for attrib in attributes:
				values[attrib] = {'key': join_key, 'values':
							{rec.id_com: getattr(rec, attrib) for rec in self}}

		## Select GIS file with commune polygons in function of aggregation type
		#if communes_as_points:
		#	gis_filename = "Bel_villages_points.TAB"
		gis_filename = ""
		if not communes_as_points:
			if aggregate_by == 'id_main':
				gis_filename = "Bel_villages_polygons.TAB"
			elif aggregate_by == 'id_com':
				gis_filename = "Bel_communes_avant_fusion.TAB"
		if gis_filename:
			gis_filespec = os.path.join(GIS_FOLDER, gis_filename)
			#gis_filespec = "http://seishaz.oma.be:8080/geoserver/rob/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=rob:bel_villages_polygons&outputFormat=application%2Fjson"

		## Grid
		if aggregate_by == 'grid':
			import mapping.geotools.coordtrans as ct
			X_left = np.array([rec.lon for rec in self])
			Y_bottom = np.array([rec.lat for rec in self])
			X_right = X_left + grid_spacing * 1000
			Y_top = Y_bottom + grid_spacing * 1000
			all_lons, all_lats = [], []
			for i in range(len(self)):
				X = [X_left[i], X_right[i], X_right[i], X_left[i], X_left[i]]
				Y = [Y_bottom[i], Y_bottom[i], Y_top[i], Y_top[i], Y_bottom[i]]
				lons, lats = ct.transform_array_coordinates(ct.LAMBERT1972, ct.WGS84, X, Y)
				all_lons.append(lons)
				all_lats.append(lats)
			macro_geoms = lbm.MultiPolygonData(all_lons, all_lats, values=values)

		## Points
		elif communes_as_points:
			lons = [rec.lon for rec in self]
			lats = [rec.lat for rec in self]
			macro_geoms = lbm.MultiPointData(lons, lats, values=values)

		## Commune polygons
		else:
			gis_data = lbm.GisData(gis_filespec, joined_attributes=values)
			_, _, polygon_data = gis_data.get_data()
			macro_geoms = polygon_data

		return macro_geoms

	def to_geojson(self, communes_as_points=False):
		"""
		Convert to GeoJSON.

		:param communes_as_points:
			see :meth:`get_geometries`

		:return:
			dict
		"""
		multi_data = self.get_geometries(communes_as_points=communes_as_points)
		return multi_data.to_geojson()

	def export_gis(self, format, filespec, encoding='latin-1',
					communes_as_points=False):
		"""
		Export to GIS file

		:param format:
			str, OGR format specification (e.g., 'ESRI Shapefile',
			'MapInfo File', 'GeoJSON', 'MEMORY', ...)
		:param out_filespec:
			str, full path to output file, will also be used as layer name
		:param encoding:
			str, encoding to use for non-ASCII characters
			(default: 'latin-1')
		:param communes_as_points:
			see :meth:`get_geometries`

		:return:
			instance of :class:`ogr.DataSource` if :param:`format`
			== 'MEMORY', else None
		"""
		multi_data = self.get_geometries(communes_as_points=communes_as_points)
		return multi_data.export_gis(format, filespec, encoding=encoding)

	def plot_map(self, region=(2, 7, 49.25, 51.75), projection="merc",
				graticule_interval=(1, 1), plot_info="intensity",
				int_conversion="round", symbol_style=None,
				cmap="rob", color_gradient="discontinuous", event_style="default",
				province_style="default", colorbar_style="default",
				radii=[], plot_pie=None, title="", fig_filespec=None,
				ax=None, copyright=u"� ROB", verbose=True):
		return plot_macroseismic_map(self, region=region, projection=projection,
				graticule_interval=graticule_interval, plot_info=plot_info,
				int_conversion=int_conversion, symbol_style=symbol_style,
				cmap=cmap, color_gradient=color_gradient, event_style=event_style,
				province_style=province_style, colorbar_style=colorbar_style,
				radii=radii, plot_pie=plot_pie, title=title, fig_filespec=fig_filespec,
				ax=ax, copyright=copyright, verbose=verbose)

	def export_geotiff(self, out_filespec, plot_info="intensity",
				int_conversion="round", symbol_style=None, cmap="rob",
				color_gradient="discontinuous", copyright="", dpi=120):
		region = (2, 7, 49.25, 51.75)
		projection = "tmerc"
		graticule_interval = ()
		event_style = None
		province_style = None
		colorbar_style = None
		radii = []
		plot_pie = None
		title = ""
		fig_filespec = "hold"
		ax = None

		map = self.plot_map(region=region, projection=projection,
				graticule_interval=graticule_interval, plot_info=plot_info,
				int_conversion=int_conversion, symbol_style=symbol_style,
				cmap=cmap, color_gradient=color_gradient, event_style=event_style,
				province_style=province_style, colorbar_style=colorbar_style,
				radii=radii, plot_pie=plot_pie, title=title, fig_filespec=fig_filespec,
				ax=ax, copyright=copyright, verbose=False)
		map.export_geotiff(out_filespec, dpi=dpi)


def plot_macroseismic_map(macro_info_coll, region=(2, 7, 49.25, 51.75),
				projection="merc", graticule_interval=(1, 1), plot_info="intensity",
				int_conversion="round", symbol_style=None,
				cmap="rob", color_gradient="discontinuous", event_style="default",
				province_style="default", colorbar_style="default",
				radii=[], plot_pie=None, title="", fig_filespec=None,
				ax=None, copyright=u"� ROB", verbose=True):
	"""
	Plot macroseismic map for given earthquake

	:param macro_info_coll:
		instance of :class:`MacroInfoCollection`, representing
		(aggregated) macroseismic information to plot
	:param region:
		(lonmin, lonmax, latmin, latmax) tuple or str
		(default: (2, 7, 49.25, 51.75))
	:param projection:
		str, name of projection supported in layeredbasemap
		(default: "merc")
	:param graticule_interval:
		(lon_spacing, lat_spacing) tuple
		(default: (1, 1))
	:param plot_info:
		str, information that should be plotted, either 'intensity'
		or 'num_replies'
		(default: 'intensity')
	:param int_conversion:
		str, "floor", "round" or "ceil"
		(default: "round", corresponding to the convention of Wald et al.)
	:param symbol_style:
		instance of :class:`mapping.layeredbasemap.SymbolStyle`,
		point style for macroseismic data. If None, data will be plotted
		as commune polygons
		(default: None)
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
	:param province_style:
		instance of :class:`mapping.layeredbasemap.LineStyle`,
		line style for provinces
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
	# TODO: country_style, region_style, commune_linestyle, etc.
	#import mapping.layeredbasemap as lbm

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

	if color_gradient[:4] == "disc":
		if plot_info == 'intensity':
			tfc = lbm.ThematicStyleIndividual(classes, cmap, value_key=plot_info,
										labels=["%d" % val for val in classes],
										style_under='w', style_over=cmap(1.))
		elif plot_info == 'num_replies':
			tfc = lbm.ThematicStyleRanges(classes, cmap, value_key=plot_info,
										labels=["%d" % val for val in classes],
										style_under='w', style_over=cmap(1.),
										style_bad='w')
	elif color_gradient == "continuous":
		tfc = lbm.ThematicStyleGradient(classes, cmap, value_key=plot_info,
								labels=["%d" % val for val in classes],
								style_under='w', style_over=cmap(1.))

	if colorbar_style == "default":
		colorbar_style = lbm.ColorbarStyle(location="bottom", format="%d",
										title=cb_title, spacing="uniform")
	tfc.colorbar_style = colorbar_style
	thematic_legend_style = lbm.LegendStyle(location=4)

	if not symbol_style:
		## Plot polygons
		ta = 1.
		tfh = None
		if plot_info != 'num_replies':
			## Set transparency (or hatching) in function of number of replies
			num_replies = [1, 3, 5, 10, 20, 50, 500]
			#tfh = lbm.ThematicStyleRanges(num_replies, ['', '.....', '....', '...', '..', ''],
			#								value_key="num_replies")
			if color_gradient == "discontinuous":
				ta = lbm.ThematicStyleGradient(num_replies, [0.1, 0.3, 0.5, 0.625, 0.75, 0.875, 1.],
											value_key="num_replies")
				#ta = 1.
		polygon_style = lbm.PolygonStyle(fill_color=tfc, line_width=0.1,
					fill_hatch=tfh, alpha=ta, thematic_legend_style=thematic_legend_style)
		#macro_style = lbm.CompositeStyle(polygon_style=polygon_style)
		macro_style = polygon_style
		legend_label = {"polygons": ""}
	else:
		## Plot points
		## Symbol size in function of number of replies
		if plot_info != 'num_replies':
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

	## Province layer
	if province_style:
		data = lbm.GisData(os.path.join(GIS_FOLDER, "Bel_provinces.TAB"))
		polygon_style = lbm.PolygonStyle(line_width=1, fill_color='none')
		gis_style = lbm.CompositeStyle(polygon_style=polygon_style)
		province_layer = lbm.MapLayer(data, gis_style, legend_label={"polygons": ""})
		layers.append(province_layer)

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


def plot_official_macroseismic_map(id_earth, region=(2, 7, 49.25, 51.75),
				projection="merc", graticule_interval=(1, 1), min_or_max='max',
				aggregate_by="commune", agg_function="average", min_val=1,
				min_fiability=20, int_conversion="round",
				symbol_style=None, cmap="rob", color_gradient="discontinuous",
				event_style="default", radii=[], title="",
				fig_filespec=None, ax=None, copyright=u"� ROB", verbose=True):
	"""
	Plot "official" macroseismic map

	:param id_earth:
		int or str, ID of earthquake in ROB database for which to plot map
	:param region:
	:param projection:
	:param graticule_interval:
		see :func:`plot_macroseismic_map`

	:param min_or_max:
		str, one of 'min', 'mean' or 'max' to select between
		intensity_min and intensity_max values in database
		(default: 'max')
	:param aggregate_by:
		str, type of aggregation, specifying how macroseismic data should
		be aggregated in the map, one of:
		- 'id_com' or 'commune'
		- 'id_main' or 'main commune'
		(default: 'commune')
	:param agg_function:
		str, aggregation function to use if :param:`aggregate_by`
		is 'main commune', one of "minimum", "maximum" or "average"
		(default: "average")
	:param min_val:
		float, minimum intensity to return
		(default: 1)
	:param min_fiability:
		float, minimum fiability of macroseismic record
		(default: 20.)

	:param int_conversion:
	:param symbol_style:
	:param cmap:
	:param color_gradient:
	:param event_style:
	:param radii:
	:param plot_pie:
	:param title:
	:param fig_filespec:
	:param ax:
	:param copyright:
	:param verbose:
		see :func:`plot_macroseismic_map`

	:return:
		None
		or instance of :class:`LayeredBasemap` if :param:`ax` is not
		None or if fig_filespec == 'hold'
	"""
	from ..rob import query_local_eq_catalog_by_id

	if aggregate_by in ('main commune', 'id_main'):
		group_by_main_village = True
	else:
		group_by_main_village = False

	## Retrieve macroseismic information from database
	[eq] = query_local_eq_catalog_by_id(id_earth)

	comm_rec_dict = eq.get_macroseismic_data_aggregated_official(min_or_max=min_or_max,
			min_val=min_val, group_by_main_village=group_by_main_village,
			agg_function=agg_function, verbose=verbose)

	for id_com, rec in comm_rec_dict.items():
		if rec.lon is None:
			print("Commune #%s has no location!" % id_com)
			comm_rec_dict.pop(id_com)
	macro_recs = comm_rec_dict.values()

	return plot_macroseismic_map(macro_recs, id_earth, region=region,
				projection=projection, graticule_interval=graticule_interval,
				plot_info='intensity', int_conversion=int_conversion,
				symbol_style=symbol_style, cmap=cmap, color_gradient=color_gradient,
				event_style=event_style, radii=radii, plot_pie=None, title=title,
				fig_filespec=fig_filespec, ax=ax, copyright=copyright,
				verbose=verbose)


# TODO: add agg_function for plot_web_macroseismic_map too?

def plot_web_macroseismic_map(id_earth, region=(2, 7, 49.25, 51.75), projection="merc",
				graticule_interval=(1, 1), min_replies=3, query_info="cii", min_val=1,
				min_fiability=20, filter_floors=(0, 4), aggregate_by="commune",
				recalc=False, int_conversion="round",
				symbol_style=None, cmap="rob", color_gradient="discontinuous",
				event_style="default", radii=[], plot_pie=None, title="",
				fig_filespec=None, ax=None, copyright=u"� ROB", verbose=True):
	"""
	Plot internet macroseismic map for given earthquake

	:param id_earth:
		int or str, ID of earthquake in ROB database for which to plot map
	:param region:
	:param projection:
	:param graticule_interval:
		see :func:`plot_macroseismic_map`

	:param min_replies:
		int, minimum number of replies to use for plotting macroseismic data
	:param query_info:
		str, info to query from the database, either 'cii', 'cdi' or
		'num_replies'
		(default: 'cii')
	:param min_val:
		int, minimum intensity value to plot
		(default: 1)
	:param min_fiability:
		int, minimum fiability of enquiries to include in plot
		(default: 20)
	:param filter_floors:
		(lower_floor, upper_floor) tuple, floors outside this range
			(basement floors and upper floors) are filtered out
			(default: (0, 4))
	:param aggregate_by:
		str, type of aggregation, specifying how macroseismic data should
		be aggregated in the map, one of:
		- 'id_com' or 'commune'
		- 'id_main' or 'main commune'
		- 'grid_X' (where X is grid spacing in km)
		- None or '' (= no aggregation, i.e. all replies are
		plotted individually (potentially on top of each other)
		(default: 'commune')
	:param recalc:
		bool, whether or not to recalculate aggregated intensity
		(default: False)

	:param int_conversion:
	:param symbol_style:
	:param cmap:
	:param color_gradient:
	:param event_style:
	:param radii:
	:param plot_pie:
	:param title:
	:param fig_filespec:
	:param ax:
	:param copyright:
	:param verbose:
		see :func:`plot_macroseismic_map`

	:return:
		None
		or instance of :class:`LayeredBasemap` if :param:`ax` is not
		None or if fig_filespec == 'hold'
	"""
	from ..rob import query_local_eq_catalog_by_id, query_web_macro_enquiries

	query_info = query_info.lower()
	if query_info == 'num_replies':
		min_replies = 1

	if aggregate_by == 'commune':
		aggregate_by = 'id_com'
	elif aggregate_by == 'main commune':
		aggregate_by = 'id_main'
	elif not aggregate_by:
		min_replies = 1

	## Retrieve macroseismic information from database
	[eq] = query_local_eq_catalog_by_id(id_earth)
	dyfi_ensemble = eq.get_macroseismic_enquiries(min_fiability)

	## Aggregate
	macro_recs = dyfi_ensemble.get_aggregated_info(aggregate_by, min_replies,
							agg_info=query_info, min_fiability=min_fiability,
							filter_floors=filter_floors, recalc=recalc)
	# TODO: remove
	"""
	if not recalc:
		if aggregate_by == 'id_com':
			group_by_main_village = False
		elif aggregate_by == 'id_main':
			group_by_main_village = True
		else:
			# TODO !
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
		from ..macrorecord import MacroseismicInfo

		ensemble = query_web_macro_enquiries(id_earth, min_fiability=min_fiability,
											verbose=verbose)
		ensemble = ensemble.fix_all()
		ensemble = ensemble.filter_floors(*filter_floors, keep_nan_values=True)

		if aggregate_by in ('id_com', 'id_main'):
			comm_key = aggregate_by
			ensemble.set_locations_from_communes(comm_key=comm_key, keep_unmatched=False,
												max_quality=10)
			agg_ensemble_dict = ensemble.aggregate_by_commune(comm_key=comm_key)
		elif not aggregate_by:
			if np.isnan(ensemble.get_prop_values('longitude')).all():
				ensemble.set_locations_from_communes(comm_key='id_com')
			agg_ensemble_dict = {}
			for subensemble in ensemble:
				lon = subensemble.recs[0]['longitude']
				if lon:
					#lat = subensemble.recs[0]['latitude']
					id_web = subensemble.recs[0]['id_web']
					agg_ensemble_dict[id_web] = subensemble
		elif aggregate_by[:4] == 'grid':
			if np.isnan(ensemble.get_prop_values('longitude')).any():
				ensemble.set_locations_from_communes(comm_key='id_com',
													keep_unmatched=False)
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
			rec = MacroseismicInfo(id_earth, id_com, I, aggregate_by, 'internet',
									num_replies, lon, lat, web_ids)
			macro_recs.append(rec)

	if unassigned:
		print("Note: %d enquiries are not assigned to a commune" % unassigned.num_replies)
	"""

	plot_info = 'num_replies' if query_info == 'num_replies' else 'intensity'
	return plot_macroseismic_map(macro_recs, id_earth, region=region, projection=projection,
				graticule_interval=graticule_interval, plot_info=plot_info,
				int_conversion=int_conversion, symbol_style=symbol_style,
				cmap=cmap, color_gradient=color_gradient, event_style=event_style,
				radii=radii, plot_pie=plot_pie, title=title,
				fig_filespec=fig_filespec, ax=ax, copyright=copyright,
				verbose=verbose)
