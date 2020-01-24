# -*- coding: iso-Latin-1 -*-
"""
Various earthquake catalog plotting functions
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int


import os

import numpy as np
import pylab
import matplotlib
from matplotlib.dates import AutoDateLocator, DateFormatter

from plotting.generic_mpl import plot_xy
import eqcatalog.time_functions_np as tf


__all__ = ['plot_time_magnitude', 'plot_cumulated_moment', 'plot_depth_statistics',
			'plot_depth_histogram', 'plot_poisson_test', 'plot_map']


def plot_time_magnitude(catalogs, Mtype, Mrelation, rel_time_unit=None,
						Mrange=(None, None), start_date=None, end_date=None,
						markers=['o'], marker_sizes=[8], colors=[], fill_colors=[],
						edge_widths=[0.5], labels=[], x_is_time=True,
						completeness=None, completeness_color='r',
						lang='en', **kwargs):
	"""
	Magnitude vs time or time vs magnitude plot for different earthquake
	catalogs

	:param catalogs:
		list with instances of :class:`EQCatalog`
	:param Mtype:
		str, magnitude type for magnitude scaling
		or list of strs, magnitude type for each catalog (in which case
		:param:`Mrelation` will be ignored)
	:param Mrelation:
		{str: str} ordered dict, mapping magnitude type ("MW", "MS" or "ML")
		to name of magnitude conversion relation for :param:`Mtype`
		(default: {})
	:param rel_time_unit:
		str, relative time unit ('Y', 'W', 'D', 'h', 'm' or 's')
		(default: None = plot absolute time)
	:param Mrange:
		(min_mag, max_mag) tuple of floats, magnitude range in Y axis
		(default: (None, None))
	:param start_date:
		datetime spec, start date in X axis:
		- int (year)
		- str (timestamp)
		- datetime.datetime
		- np.datetime64
		(default: None = auto-determine from catalogs)
	:param end_date:
		datetme spec, end date in X axis:
		see :param:`start_date` for options
		(default: None = auto-determine from catalogs)
	:param markers:
		list of chars, marker symbols to cycle over for each catalog
		(default: ['o'])
	:param marker_sizes:
		list of marker sizes to cycle over for each catalog
		(default: [8])
	:param colors:
		list of marker edge colors to cycle over for each catalog
		(default: [], will use default colors for :param:`style_sheet`)
	:param fill_colors:
		list of marker fill colors to cycle over for each catalog
		(default: [], will not apply fill color)
	:param edge_widths:
		list of marker line widths to cycle over for each catalog
		(default: [0.5])
	:param labels:
		list of labels to cycle over for each catalog
		(default: [], will not label catalogs)
	:param x_is_time:
		bool, whether or not time should be in X axis
		(default: True)
	:param completeness:
		instance of :class:`eqcatalog.Completeness`,
		catalog completeness to draw as a line over the catalog events
		(default: None)
	:param completeness_color:
		str, color to plot completeness line
		(default: 'r')
	:param lang:
		str, language of plot labels
		(default: 'en')

	See :func:`plotting.generic_mpl.plot_xy` for additional
	keyword arguments
	"""
	if start_date is None:
		start_date = np.min([cat.start_date for cat in catalogs])
	else:
		if isinstance(start_date, int):
			start_date = str(start_date)
		start_date = tf.as_np_datetime(start_date)
	if end_date is None:
		end_date = np.max([cat.end_date for cat in catalogs])
	else:
		if isinstance(end_date, int):
			end_date = str(end_date)
		end_date = tf.as_np_datetime(end_date)

	## Define axis ranges
	if rel_time_unit:
		key = {True: 'xmin', False: 'ymin'}[x_is_time]
		kwargs[key] = 0
		key = {True: 'xmax', False: 'ymax'}[x_is_time]
		kwargs[key] = tf.timespan(start_date, end_date, rel_time_unit)
	else:
		key = {True: 'xmin', False: 'ymin'}[x_is_time]
		kwargs[key] = tf.to_py_datetime(start_date)
		key = {True: 'xmax', False: 'ymax'}[x_is_time]
		kwargs[key] = tf.to_py_datetime(end_date)

	min_mag, max_mag = Mrange
	if min_mag is not None:
		key = {True: 'ymin', False: 'xmin'}[x_is_time]
		kwargs[key] = min_mag
	if max_mag is not None:
		key = {True: 'ymax', False: 'xmax'}[x_is_time]
		kwargs[key] = max_mag

	## Catalogs
	if isinstance(Mtype, list):
		Mtypes = Mtype
		Mrelation = {}
	else:
		Mtypes = [Mtype] * len(catalogs)

	datasets = []
	for i, catalog in enumerate(catalogs):
		Mtype = Mtypes[i]
		mags = catalog.get_magnitudes(Mtype, Mrelation)
		if rel_time_unit:
			dates = tf.timespan(start_date, catalog.get_datetimes(), rel_time_unit)
			idxs = (dates >= 0)
			dates = dates[idxs]
			mags = mags[idxs]
		else:
			dates = tf.to_py_datetime(catalog.get_datetimes())
		if x_is_time:
			datasets.append((dates, mags))
		else:
			datasets.append((mags, dates))
		if isinstance(labels, list) and len(labels) <= i:
			labels.append(catalog.name)

	kwargs['marker_edge_colors'] = colors or list('bgrcmyk')
	kwargs['colors'] = ['None'] * len(datasets)
	kwargs['marker_fill_colors'] = fill_colors
	kwargs['fill_colors'] = []
	kwargs['markers'] = markers
	kwargs['marker_sizes'] = marker_sizes
	kwargs['marker_edge_widths'] = edge_widths

	## Completeness
	if completeness:
		clabel = {"en": "Completeness",
				"nl": "Compleetheidsmagnitude",
				"fr": u"Magnitude de complétude",
				"nlfr": u"Compleetheid / Complétude"}[lang]
		dates, mags = completeness.min_dates, completeness.min_mags
		#dates = np.concatenate([dates[:1], np.repeat(dates[1:], 2), [tf.to_fractional_year(end_date)]])
		dates = np.concatenate([dates[:1], np.repeat(dates[1:], 2), [tf.to_py_datetime(end_date)]])
		if rel_time_unit:
			dates = tf.timespan(start_date, dates, rel_time_unit)
		mags = np.repeat(mags, 2)
		if x_is_time:
			datasets.append((dates, mags))
		else:
			datasets.append((mags, dates))

		if not isinstance(labels, list):
			labels = ['_nolegend_'] * len(catalogs)
		labels.append(clabel)
		kwargs['colors'].append(completeness_color)
		kwargs['markers'].append('')
		kwargs['marker_edge_colors'].append('None')
		kwargs['marker_sizes'].append(0)
		kwargs['linewidths'] = [0] * len(catalogs) + [2]

	kwargs['labels'] = labels

	## Axis labels
	default_date_label = {"en": "Time",
							"nl": "Tijd",
							"fr": "Temps",
							"nlfr": "Tijd / Temps"}[lang]
	time_unit_str = {'Y': 'years', 'W': 'weeks', 'D': 'days',
					'h': 'hours', 'm': 'minutes', 's': 'seconds'}.get(rel_time_unit)
	if time_unit_str:
		default_date_label += ' (%s)' % time_unit_str
	default_mag_label = "Magnitude ($M_%s$)" % Mtype[1]

	key = {True: 'xlabel', False: 'ylabel'}[x_is_time]
	kwargs[key] = kwargs.get(key, default_date_label)
	key = {True: 'ylabel', False: 'xlabel'}[x_is_time]
	kwargs[key] = kwargs.get(key, default_mag_label)

	## Default tick intervals
	default_date_tick_interval = (AutoDateLocator(),
								matplotlib.ticker.AutoMinorLocator())
	default_mag_tick_interval = (1, 0.1)

	key = {True: 'xtick_interval', False: 'ytick_interval'}[x_is_time]
	kwargs[key] = kwargs.get(key, default_date_tick_interval)
	key = {True: 'ytick_interval', False: 'xtick_interval'}[x_is_time]
	kwargs[key] = kwargs.get(key, default_mag_tick_interval)

	## Convert date ticks to datetimes if necessary
	key = {True: 'xticks', False: 'yticks'}[x_is_time]
	if key in kwargs:
		date_ticks = []
		for tick in kwargs[keys]:
			if isinstance(tick, int):
				tick = str(tick)
			tick = tf.as_np_datetime(tick)
			date_ticks.append(tf.to_py_datetime(tick))
		kwargs[key] = date_ticks
		## If ticks are datetimes, we need to explicitly set a date formatter
		## for the labels
		key = {True: 'xticklabels', False: 'yticklabels'}[x_is_time]
		kwargs[key] = kwargs.get(key, DateFormatter('%Y'))

	kwargs['xgrid'] = kwargs.get('xgrid', 1)
	kwargs['ygrid'] = kwargs.get('ygrid', 1)

	return plot_xy(datasets, **kwargs)

plot_time_magnitude.__doc__ += plot_xy.__doc__.split("\n", 4)[4]


def plot_cumulated_moment(catalogs,
						start_date=None, end_date=None, rel_time_unit=None,
						Mrelation={}, M0max=None,
						**kwargs):
	"""
	Plot cumulated moment versus time for different earthquake catalogs

	:param catalogs:
		list with instances of :class:`EQCatalog`
	:param start_date:
		datetime spec, datetime where to start the plot
		(default: None)
	:param end_date:
		datetime spec, datetime where to end the plot
		(default: None)
	:param rel_time_unit:
		str, relative time unit ('Y', 'W', 'D', 'h', 'm' or 's')
		(default: None = plot absolute time)
	:param Mrelation:
		OrderedDict or str, magnitude relations to use for conversion
		to seismic moment
		(default: {})
	:param M0max:
		float, maximum moment value in Y axis
		(default: None)

	See :func:`plotting.generic_mpl.plot_xy` for additional
	keyword arguments
	"""
	if start_date is None:
		start_date = np.min([cat.start_date for cat in catalogs])
	else:
		if isinstance(start_date, (int)):
			start_date = str(start_date)
		start_date = tf.as_np_datetime(start_date)
	if end_date is None:
		end_date = np.max([cat.end_date for cat in catalogs])
	else:
		if isinstance(end_date, (int)):
			end_date = str(end_date)
		end_date = tf.as_np_datetime(end_date)

	datasets = []
	for i, catalog in enumerate(catalogs):
		catalog = catalog.get_sorted()
		M0 = catalog.get_M0(Mrelation=Mrelation)
		## Remove NaN values
		nan_idxs = np.isnan(M0)
		catalog = catalog[~nan_idxs]
		M0 = M0[~nan_idxs]

		if rel_time_unit:
			dates = tf.timespan(start_date, catalog.get_datetimes(), rel_time_unit)
			idxs = (dates >= 0)
			dates = dates[idxs]
			M0 = M0[idxs]
		else:
			dates = tf.to_py_datetime(catalog.get_datetimes())

		M0_cumul = np.cumsum(M0)

		## Construct arrays with duplicate points in order to plot horizontal
		## lines between subsequent points
		M0_cumul2 = np.concatenate([[0.], np.repeat(M0_cumul, 2)[:-1]])
		dates2 = np.repeat(dates, 2)

		datasets.append((dates2, M0_cumul2))

	kwargs['marker_intervals'] = kwargs.get('marker_intervals', [2])

	xlabel = "Time"
	time_unit_str = {'Y': 'years', 'W': 'weeks', 'D': 'days',
					'h': 'hours', 'm': 'minutes', 's': 'seconds'}.get(rel_time_unit)
	if time_unit_str:
		xlabel += ' (%s)' % time_unit_str
	kwargs['xlabel'] = kwargs.get('xlabel', xlabel)
	kwargs['ylabel'] = kwargs.get('ylabel', "Seismic Moment (N.m)")

	if rel_time_unit:
		xmin = 0
		xmax = tf.timespan(start_date, end_date, rel_time_unit)
	else:
		xmin = tf.to_py_datetime(start_date)
		xmax = tf.to_py_datetime(end_date)
	kwargs['xmin'] = kwargs.get('xmin', xmin)
	kwargs['xmax'] = kwargs.get('xmax', xmax)

	kwargs['ymin'] = kwargs.get('ymin', 0)
	kwargs['ymax'] = kwargs.get('ymax', M0max)

	if not 'marker_edge_colors' in kwargs:
		#kwargs['marker_edge_colors'] = ['k']
		kwargs['marker_edge_colors'] = kwargs.get('colors', [])[:]
	if not 'marker_fill_colors' in kwargs:
		#kwargs['marker_fill_colors'] = kwargs.get('colors', [])[:]
		kwargs['marker_fill_colors'] = ['None']

	## Default tick intervals
	if rel_time_unit:
		default_date_tick_interval = (matplotlib.ticker.AutoLocator(),
								matplotlib.ticker.AutoMinorLocator())
	else:
		default_date_tick_interval = (AutoDateLocator(),
								matplotlib.ticker.AutoMinorLocator())

	kwargs['xtick_interval'] = kwargs.get('xtick_interval', default_date_tick_interval)

	return plot_xy(datasets, **kwargs)

plot_cumulated_moment.__doc__ += plot_xy.__doc__.split("\n", 4)[4]


def plot_cumul_num_events(catalogs):
	pass


def plot_map(catalogs,
			Mtype="MW", Mrelation={},
			labels=[], catalog_styles=[],
			symbols=[], edge_colors=[], fill_colors=[], edge_widths=[0.5],
			symbol_sizes=[9], mag_size_inc=4,
			coastline_style={}, country_style={}, river_style=None, continent_style=None,
			source_model=None, sm_label_colname="ShortName",
			sm_style={"line_color": 'k', "line_pattern": '-', "line_width": 2},
			sites=[], site_style={"shape": 's', "fill_color": 'b', "size": 10}, site_legend="",
			circles=[], circle_styles=[],
			projection="merc", region=None, origin=(None, None),
			graticule_interval=(1., 1.), graticule_style={"annot_axes": "SE"},
			resolution="i",
			title=None, legend_style={}, border_style={},
			fig_filespec=None, fig_width=0, dpi=None):
	"""
	Construct map of multiple catalogs.

	:param catalogs:
		List containing instances of :class:`EQCatalog`
	:param Mtype:
		str, magnitude type for magnitude scaling
		or list of strs, magnitude type for each catalog (in which case
		:param:`Mrelation` will be ignored)
		(default: "MW")
	:param Mrelation:
		{str: str} ordered dict, mapping magnitude type ("MW", "MS" or "ML")
		to name of magnitude conversion relation for :param:`Mtype`
		(default: {})
	:param labels:
		List containing plot labels, one for each catalog
		(default: [])
	:param catalog_styles:
		List with styles (instances of :class:`PointStyle` or dictionaries
		with subset of PointStyle attributes as keys) for each catalog.
		If list contains only 1 element, the same style will be used for
		all catalogs. If list is empty, a default style will be used.
		Point size refers to a magnitude-3 earthquake if :param:`mag_size_inc`
		is set
		(default: [])
	:param symbols:
		List containing point symbols (matplotlib marker specifications)
		for each catalog, overriding style given by :param:`catalog_styles`
		(default: [])
	:param edge_colors:
		List containing symbol edge colors (matplotlib color specifications)
		for each catalog, overriding style given by :param:`catalog_styles`
		(default: [])
	:param fill_colors:
		List containing symbol fill colors (matplotlib color specifications)
		for each catalog, overriding style given by :param:`catalog_styles`
		(default: [])
	:param edge_widths:
		List containing symbol edge width for each catalog
		(default: [0.5])
	:param symbol_sizes:
		list of inst or floats, base (M=3) symbol sizes in points for
		each catalog
		(default: [9])
	:param mag_size_inc:
		Int or Float, symbol size increment per magnitude relative to M=3
		(default: 4)
	:param coastline_style:
		instance of :class:`LineStyle` or dictionary with subset of
		LineStyle attributes as keys, used to plot coastlines. If None,
		coastlines will not be drawn
		(default: {}, equivalent to default line style)
	:param country_style:
		instance of :class:`LineStyle` or dictionary with subset of
		LineStyle attributes as keys, used to plot country borders. If None,
		country borders will not be drawn
		(default: {}, equivalent to default line style)
	:param river_style:
		instance of :class:`LineStyle` or dictionary with subset of
		LineStyle attributes as keys, used to plot rivers. If None, rivers
		will not be drawn
		(default: None)
	:param continent_style:
		instance of :class:`PolygonStyle` or dictionary with subset of
		PolygonStyle attributes as keys, used to plot continents/oceans.
		If None, continents/oceans will not be drawn
		(default: None)
	:param source_model:
		String, name of source model to overlay on the plot
		or full path to GIS file containing source model
		(default: None)
	:param sm_label_colname:
		Str, column name of GIS table to use as label
		(default: "ShortName")
	:param sm_style:
		instance of :class:`LineStyle`, :class:`PolygonStyle`,
		:class:`CompositeStyle` or dictionary with subset of attributes
		of LineStyle or PolygonStyle as keys, used to plot source model.
		(default: {"line_color": 'k', "line_style": '-', "line_width": 2, "fill_color": "None"}
	:param sites:
		List of (lon, lat) tuples or instance of :class:`PSHASite`
		(default: [])
	:param site_style:
		instance of :class:`PointStyle` or dictionary containing subset of
		PointStyle attributes as keys, used to plot sites
		(default: {"shape": 's', "fill_color": 'b', "size": 10})
	:param site_legend:
		String, common text referring to all sites to be placed in legend
		(default: "")
	:param circles:
		list with (lon, lat, radius) tuples defining center and radius
		(in km) of circles to plot
		(default: [])
	:param circle_styles:
		List with styles (instances of :class:`LineStyle` or dictionaries
		with subset of LineStyle attributes as keys) for each circle.
		If list contains only 1 element, the same style will be used for
		all circles. If list is empty, a default style will be used
		(default: [])
	:param projection:
		String, map projection. See Basemap documentation
		(default: "merc")
	:param region:
		(w, e, s, n) tuple specifying rectangular region to plot in
		geographic coordinates
		(default: None)
	:param origin:
		(lon, lat) tuple defining map origin. Needed for some
		projections
		(default: None)
	:param graticule_interval:
		(dlon, dlat) tuple defining meridian and parallel interval in
		degrees
		(default: (1., 1.)
	:param graticule_style:
		instance of :class:`GraticuleStyle` or dictionary containing
		GraticuleStyle attributes as keys, defining graticule style of map
		(default: {"annot_axes": "SE"}
		annot_axes: string, containing up to 4 characters ('W', 'E', 'S' and/or 'N'),
		defining which axes should be annotated
		(default: {"annot_axes": "SE"})
	:param resolution:
		String, map resolution (coastlines and country borders):
		'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high), 'f' (full)
		(default: 'i')
	:param title:
		String, plot title
		(default: None)
	:param legend_style:
		instance of :class:`LegendStyle` or dictionary containing
		LegendStyle attributes as keys, defining style of map legend
		(default: {})
	:param border_style:
		instance ov :class:`MapBorderStyle` or dictionary containing
		MapBorderStyle attributes as keys, defining style of map border
		(default: {})
	:param fig_filespec:
		str, full path to output file
		or None (plot on screen)
		or "wait" (do not show plot, return map object)
	:param fig_width:
		float, figure width in inches
		(default: 0, use default figure width)
	:param dpi:
		int, image resolution in dots per inch
		(default: None)

	:return:
		None, or instance of :class:`LayeredBasemap`
		(if :param:`fig_filespec` is 'wait')
	"""
	import mapping.layeredbasemap as lbm

	layers = []

	## Continents/oceans
	if continent_style != None:
		data = lbm.BuiltinData("continents")
		if isinstance(continent_style, dict):
			style = lbm.PolygonStyle.from_dict(continent_style)
		else:
			style = continent_style
		layers.append(lbm.MapLayer(data, style, name="continents"))

	## Coastlines
	if coastline_style != None:
		data = lbm.BuiltinData("coastlines")
		if isinstance(coastline_style, dict):
			style = lbm.LineStyle.from_dict(coastline_style)
		else:
			style = coastline_style
		layers.append(lbm.MapLayer(data, style, name="coastlines"))

	## Country borders
	if country_style != None:
		data = lbm.BuiltinData("countries")
		if isinstance(country_style, dict):
			style = lbm.LineStyle.from_dict(country_style)
		else:
			style = country_style
		layers.append(lbm.MapLayer(data, style, name="countries"))

	## Rivers
	if river_style != None:
		data = lbm.BuiltinData("rivers")
		if isinstance(river_style, dict):
			style = lbm.LineStyle.from_dict(river_style)
		else:
			style = country_style
		layers.append(lbm.MapLayer(data, style, name="rivers"))

	## Source model
	if source_model:
		from ..rob.source_models import rob_source_models_dict
		try:
			gis_filespec = rob_source_models_dict[source_model]["gis_filespec"]
		except:
			if isinstance(source_model, basestring):
				gis_filespec = source_model
				source_model_name = os.path.splitext(os.path.split(source_model)[1])[0]
			else:
				import hazard.rshalib as rshalib
				if isinstance(source_model, rshalib.source.SourceModel):
					gis_filespec = None
					source_model_name = source_model.name
		else:
			source_model_name = source_model

		if gis_filespec:
			data = lbm.GisData(gis_filespec, label_colname=sm_label_colname)
		else:
			# TODO: implement line and point sources too
			point_lons, point_lats, point_labels = [], [], []
			line_lons, line_lats, line_labels = [], [], []
			polygon_lons, polygon_lats, polygon_labels = [], [], []
			for src in source_model:
				if isinstance(src, rshalib.source.AreaSource):
					polygon_lons.append(src.polygon.lons)
					polygon_lats.append(src.polygon.lats)
					polygon_labels.append(getattr(src, sm_label_colname, ""))
				elif isinstance(src, rshalib.source.PointSource):
					point_lons.append(src.location.longitude)
					point_lats.append(src.location.latitude)
					point_labels.append(getattr(src, sm_label_colname, ""))
				elif isinstance(src, rshalib.source.SimpleFaultSource):
					line_lons.append([pt.lon for pt in src.fault_trace.points])
					line_lats.append([pt.lat for pt in src.fault_trace.points])
					line_labels.append(getattr(src, sm_label_colname, ""))
			point_data = lbm.MultiPointData(point_lons, point_lats, labels=point_labels)
			line_data = lbm.MultiLineData(line_lons, line_lats, labels=line_labels)
			polygon_data = lbm.MultiPolygonData(polygon_lons, polygon_lats, labels=polygon_labels)
			data = lbm.CompositeData(points=point_data, lines=line_data, polygons=polygon_data)

		if isinstance(sm_style, dict):
			if "fill_color" in sm_style and not sm_style.get("fill_color") in ("None", None):
				polygon_style = lbm.PolygonStyle.from_dict(sm_style)
				line_style = None
			else:
				line_style = lbm.LineStyle.from_dict(sm_style)
				polygon_style = None
		elif isinstance(sm_style, lbm.CompositeStyle):
			line_style = sm_style.line_style
			polygon_style = sm_style.polygon_style
		elif isinstance(sm_style, lbm.LineStyle):
			line_style = sm_style
			polygon_style = None
		elif isinstance(sm_style, lbm.PolygonStyle):
			polygon_style = sm_style
			line_style = None
		if line_style and not line_style.label_style:
			line_style.label_style = lbm.TextStyle(color=line_style.line_color, font_size=8)
		elif polygon_style and not polygon_style.label_style:
			polygon_style.label_style = lbm.TextStyle(color=polygon_style.line_color, font_size=8)
		style = lbm.CompositeStyle(line_style=line_style, polygon_style=polygon_style)
		legend_label = {'lines': source_model_name + " faults", 'polygons': source_model_name + " zones"}
		layer = lbm.MapLayer(data, style, legend_label=legend_label, name="source model")
		layers.append(layer)

	## Earthquakes
	if not labels:
		labels = [None] * len(catalogs)
	if catalog_styles in ([], None):
		catalog_styles = lbm.PointStyle(shape='o', size=9)
	if isinstance(catalog_styles, (lbm.PointStyle, dict)):
		catalog_styles = [catalog_styles]
	if len(catalog_styles) == 1:
		base_style = catalog_styles[0]
		if isinstance(base_style, dict):
			base_style = lbm.PointStyle.from_dict(base_style)
		if not symbols:
			symbols = ["o"]
		if not edge_colors:
			edge_colors = 'bgrcmyk'
		if not fill_colors:
			fill_colors = ["None"]
		if not symbol_sizes:
			symbol_sizes = [9]
		if not edge_widths:
			edge_widths = [0.5]
		catalog_styles = []
		for i in range(len(catalogs)):
			style = lbm.PointStyle.from_dict(base_style.__dict__)
			style.shape = symbols[i%len(symbols)]
			style.line_color = edge_colors[i%len(edge_colors)]
			style.fill_color = fill_colors[i%len(fill_colors)]
			style.size = symbol_sizes[i%len(symbol_sizes)]
			style.line_width = edge_widths[i%len(edge_widths)]
			catalog_styles.append(style)

	if isinstance(Mtype, list):
		Mtypes = Mtype
		Mrelation = {}
	else:
		Mtypes = [Mtype] * len(catalogs)

	for i in range(len(catalogs)):
		Mtype = Mtypes[i]
		catalog = catalogs[i]
		style = catalog_styles[i]
		if isinstance(style, dict):
			style = lbm.PointStyle.from_dict(style)
		values = {}
		if mag_size_inc:
			## Magnitude-dependent size
			if i == 0:
				min_mag = np.floor(catalog.get_Mmin(Mtype, Mrelation))
				max_mag = np.ceil(catalog.get_Mmax(Mtype, Mrelation))
				mags = np.linspace(min_mag, max_mag, min(5, max_mag-min_mag+1))
				sizes = style.size + (mags - 3) * mag_size_inc
				sizes = sizes.clip(min=1)
				style.thematic_legend_style = lbm.LegendStyle(title="Magnitude", location=3, shadow=True, fancy_box=True, label_spacing=0.7)
			values['magnitude'] = catalog.get_magnitudes(Mtype, Mrelation)
			style.size = lbm.ThematicStyleGradient(mags, sizes, value_key="magnitude")

		# TODO: color by depth
		#values['depth'] = catalog.get_depths()
		#colorbar_style = ColorbarStyle(title="Depth (km)", location="bottom", format="%d")
		#style.fill_color = ThematicStyleRanges([0,1,10,25,50], ['red', 'orange', 'yellow', 'green'], value_key="depth", colorbar_style=colorbar_style)

		# TODO: color by age
		#values['year'] = catalog.get_fractional_years()
		#style.fill_color = ThematicStyleRanges([1350,1910,2050], ['green', (1,1,1,0)], value_key="year")

		point_data = lbm.MultiPointData(catalog.get_longitudes(), catalog.get_latitudes(), values=values)

		layer = lbm.MapLayer(point_data, style, legend_label=labels[i], name="earthquakes")
		layers.append(layer)

	## Sites
	if sites:
		if isinstance(site_style, dict):
			site_style = lbm.PointStyle.from_dict(site_style)
			site_style.label_style = lbm.TextStyle()
		site_lons, site_lats, site_labels = [], [], []
		for i, site in enumerate(sites):
			try:
				lon, lat = site.longitude, site.latitude
				name = site.name
			except:
				lon, lat, name = site[:3]
			site_lons.append(lon)
			site_lats.append(lat)
			site_labels.append(name)

		point_data = lbm.MultiPointData(site_lons, site_lats, labels=site_labels)
		layer = lbm.MapLayer(point_data, site_style, site_legend, name="sites")
		layers.append(layer)

	## Circles
	if circles:
		if circle_styles == []:
			circle_styles = {}
		if isinstance(circle_styles, dict):
			circle_styles = lbm.LineStyle.from_dict(circle_styles)
		if isinstance(circle_styles, lbm.LineStyle):
			circle_styles = [circle_styles]
		for circle in circles:
			lon, lat, radius = circle
			circle_data = lbm.CircleData([lon], [lat], [radius])
			layer = lbm.MapLayer(circle_data, circle_styles[i%len(circle_styles)], name="circles")
			layers.append(layer)

	if isinstance(legend_style, dict):
		legend_style = lbm.LegendStyle.from_dict(legend_style)
	if isinstance(border_style, dict):
		border_style = lbm.MapBorderStyle.from_dict(border_style)
	if isinstance(graticule_style, dict):
		graticule_style = lbm.GraticuleStyle.from_dict(graticule_style)

	## Determine map extent if necessary
	if not region:
		west, east, south, north = 180, -180, 90, -90
		for catalog in catalogs:
			if catalog.region:
				w, e, s, n = list(catalog.region)
			else:
				w, e, s, n = list(catalog.get_region())
			if w < west:
				west = w
			if e > east:
				east = e
			if s < south:
				south = s
			if n > north:
				north = n
		region = (west, east, south, north)

	map = lbm.LayeredBasemap(layers, title, projection, region=region, origin=origin,
						graticule_interval=graticule_interval, resolution=resolution,
						graticule_style=graticule_style, legend_style=legend_style)

	return map.plot(fig_filespec=fig_filespec, fig_width=fig_width, dpi=dpi)


def plot_depth_statistics(
	catalogs,
	labels=[],
	dmin=0,
	dmax=30,
	Mmin=None,
	Mtype="MW",
	Mrelation={},
	title="",
	fig_filespec="",
	fig_width=0,
	dpi=300,
	ax=None):
	"""
	Plot depth statistics for different catalogs.

	:param catalogs:
		List containing instances of :class:`EQCatalog`
	:param labels:
		List containing catalog labels
		(default: [])
	:param dmin:
		float, minimum depth
		(default: 0)
	:param dmax:
		float, maximum depth
		(default: 30)
	:param Mtype:
		String, magnitude type for magnitude scaling (default: "MW")
	:param Mrelation:
		{str: str} dict, mapping name of magnitude conversion relation
		to magnitude type ("MW", "MS" or "ML")
		(default: {})
	:param title:
		String, plot title (default: None)
	:param fig_filespec:
		String, full path of image to be saved.
		If None (default), map is displayed on screen.
	:param fig_width:
		Float, figure width in cm, used to recompute :param:`dpi` with
		respect to default figure width
		(default: 0)
	:param dpi:
		Int, image resolution in dots per inch
		(default: 300)
	:param ax:
		matplotlib Axes instance
		(default: None)
	"""
	if ax is None:
		ax = pylab.axes()
	else:
		fig_filespec = "wait"

	if not labels:
		labels = [catalog.name for catalog in catalogs]
	for i, catalog in enumerate(catalogs):
		if Mmin != None:
			catalog = catalog.subselect(Mmin=Mmin, Mtype=Mtype, Mrelation=Mrelation)
		depths = catalog.get_depths()

		ax.boxplot(depths, positions=[i], widths=0.15)
		ax.plot([i], [np.nanmean(depths)], marker='d', mfc='g', ms=8)
		ax.plot([i], [np.percentile(depths, 2.5)], marker='v', mfc='g', ms=8)
		ax.plot([i], [np.percentile(depths, 97.5)], marker='^', mfc='g', ms=8)

	ax.set_ylim(dmin, dmax)
	ax.invert_yaxis()
	ax.set_xlim(-0.5, len(catalogs) - 0.5)
	ax.set_xticks(np.arange(len(catalogs)))
	ax.set_xticklabels(labels)
	ax.set_xlabel("Catalog", fontsize="x-large")
	ax.set_ylabel("Depth (km)", fontsize="x-large")
	if title is None:
		title = 'Depth statistics  (M>=%.1f)' % Mmin
	ax.set_title(title)

	if fig_filespec == "wait":
		return
	elif fig_filespec:
		default_figsize = pylab.rcParams['figure.figsize']
		#default_dpi = pylab.rcParams['figure.dpi']
		if fig_width:
			fig_width /= 2.54
			dpi = dpi * (fig_width / default_figsize[0])
		pylab.savefig(fig_filespec, dpi=dpi)
		pylab.clf()
	else:
		pylab.show()


def plot_depth_histogram(catalogs, labels=[], colors=[], stacked=False,
	min_depth=0, max_depth=30, bin_width=2,
	depth_error=None,
	normalized=False,
	title=None, legend_location=0,
	fig_filespec="", dpi=300,
	ax=None,
	**kwargs):
	"""
	Plot histogram with number of earthquakes versus depth.

	:param catalogs:
		list containing instances of :class:`EQCatalog`
	:param labels:
		list containing plot labels, one for each catalog
		(default: [])
	:param colors:
		list containing matplotlib color specifications for histogram bars
		corresponding to each catalog
		(default: [])
	:param stacked:
		bool, whether or not histograms should be stacked
		(default: False)
	:param min_depth:
		float, minimum depth in km
		(default: 0)
	:param max_depth:
		float, maximum depth in km
		(default: 30)
	:param bin_width:
		float, bin width in km
		(default: 2)
	:param depth_error:
		float, maximum depth uncertainty
		(default: None)
	:param normalized:
		bool, whether or not bin numbers should be normalized
		If :param:`stacked` is True, bin numbers are normalized with
		respect to total number of events in combined catalogs, else
		bin numbers are normalized with respect to number of events in
		each separate catalog
		(default: False)
	:param title:
		string, title (None = default title, empty string = no title)
		(default: None)
	:param legend_location:
		int, matplotlib legend location code
		(default: 0)
	:param fig_filespec:
		string, full path to output image file, if None plot to screen
		(default: None)
	:param dpi:
		int, image resolution in dots per inch (default: 300)
	:param ax:
		matplotlib Axes instance
		(default: None)
	"""
	from plotting.generic_mpl import plot_histogram

	bins_N = []
	for catalog in catalogs:
		bins_n, bins_depth = catalog.bin_by_depth(min_depth, max_depth, bin_width,
				depth_error, include_right_edge=True)
		bins_N.append(bins_n)

	if normalized:
		if stacked:
			## Normalize to total number of events in all catalogs
			total_num = np.sum(map(np.sum, bins_N)) * 1.0
			bins_N = [bins_n.astype('f') / total_num for bins_n in bins_N]
		else:
			## Normalize to number of events in each separate catalog
			bins_N = [bins_n.astype('f') / np.sum(bins_n) for bins_n in bins_N]

	if not labels:
		labels = [catalog.name for catalog in catalogs]

	xlabel = kwargs.pop('xlabel', "Number of events")
	if normalized:
		xlabel += " (%)"
	ylabel = kwargs.pop('ylabel', "Depth (km)")

	if title is None:
		title = 'Depth histogram'

	orientation = kwargs.pop('orientation', 'horizontal')
	yscaling = kwargs.pop('yscaling', '-lin')

	return plot_histogram(bins_N, bins_depth, labels=labels, data_is_binned=True,
						stacked=stacked, orientation=orientation,
						xlabel=xlabel, ylabel=ylabel, yscaling=yscaling,
						ymin=min_depth, ymax=max_depth,
						title=title, legend_location=legend_location,
						fig_filespec=fig_filespec, dpi=dpi, ax=ax, **kwargs)


def plot_poisson_test(catalogs,
	Mmin, interval=100,
	Mtype='MW', Mrelation={},
	completeness=None,
	labels=[], bar_colors=None, line_colors=None, title=None,
	fig_filespec=None, dpi=300, ax=None,
	verbose=True,
	**kwargs):
	"""
	Plot catalog distribution versus Poisson distribution
	p(n, t, tau) = (t / tau)**n * exp(-t/tau) / n!

	First, the specified completeness constraint is applied to the catalog.
	The completeness-constrained catalog is then truncated to the
	specified minimum magnitude and corresponding year of completeness.
	The resulting catalog is divided into intervals of the specified
	length, the number of events in each interval is counted, and a
	histogram is computed of the number of intervals having the same
	number of events up to nmax.
	This histogram is compared to the theoretical Poisson distribution.
	It seems to work best if :param:`interval` is larger (2 to 4 times)
	than tau, the average return period.

	:param catalogs:
		list containing instances of :class:`EQCatalog`
	:param Mmin:
		Float, minimum magnitude to consider in analysis (ideally
		corresponding to one of the completeness magnitudes)
	:param interval:
		Int, length of interval (number of days)
		(default: 100)
	:param Mtype:
		String, magnitude type: "ML", "MS" or "MW"
		(default: "MW")
	:param Mrelation:
		{str: str} dict, mapping name of magnitude conversion relation
		to magnitude type ("MW", "MS" or "ML")
		(default: {})
	:param completeness:
		instance of :class:`Completeness` containing initial years of
		completeness and corresponding minimum magnitudes.
		If None, use start year of catalog
		(default: None)
	:param labels:
		list of strings, labels corresponding to each catalog
		(default: [])
	:param bar_colors:
		list of matplotlib color specs, histogram color for each catalog
		(default: None)
	:param line_colors:
		list of matplotlib color specs, line color for each catalog
		(default: None, will use same colors as :param:`bar_colors`)
	:param title:
		String, plot title. (None = default title, "" = no title)
		(default: None)
	:param fig_filespec:
		String, full path of image to be saved.
		If None (default), histogram is displayed on screen.
		:param dpi:
			int, resolution of output figure in DPI
			(default: 300)
	:param ax:
		matplotlib axes instance in which graph should be plotted
		(default: None)
	:param verbose:
		Bool, whether or not to print additional information
		(bool True)
	:param **kwargs:
		additional keyword arguments to be passed to
		:func:`plotting.generic_mpl.plot_ax_frame`

	:return:
		matplotlib axes instance
	"""
	from plotting.generic_mpl import plot_histogram, plot_xy
	from hazard.rshalib.poisson import PoissonT

	PT = PoissonT(interval)
	time_delta = np.timedelta64(interval, 'D')

	histogram_datasets = []
	xy_datasets = []
	catalog_taus = []
	nmax = 0
	for catalog in catalogs:
		## Apply completeness constraint, and truncate result to
		## completeness year for specified minimum magnitude
		completeness = completeness or catalog.default_completeness
		if completeness:
			min_date = completeness.get_initial_completeness_date(Mmin)
			cc_catalog = catalog.subselect_completeness(Mtype=Mtype, Mrelation=Mrelation,
													completeness=completeness)
		else:
			min_date = catalog.start_date
			cc_catalog = catalog
		catalog = cc_catalog.subselect(start_date=min_date, Mmin=Mmin,
									Mtype=Mtype, Mrelation=Mrelation)

		num_events = len(catalog)
		td = catalog.get_time_delta(from_events=False)
		catalog_num_days = tf.fractional_time_delta(td, 'D')
		num_intervals = np.floor(catalog_num_days / interval)

		## Compute number of events in each interval
		start_date, end_date = catalog.start_date, catalog.end_date
		bins_N, _ = catalog.bin_by_time_interval(start_date, end_date,
								time_delta=time_delta, include_incomplete=False,
								Mmin=Mmin, Mrelation=Mrelation)
		histogram_datasets.append(bins_N)

		nmax = max(nmax, bins_N.max())

		## Theoretical Poisson distribution
		n = np.arange(nmax + 1)
		tau = catalog_num_days / float(num_events)
		if verbose:
			print(catalog.name)
			print("Number of events in catalog: %d" % num_events)
			print("Number of days in catalog: %s" % catalog_num_days)
			print("Number of %d-day intervals: %d" % (interval, num_intervals))
			print("Average return period for M>=%s: %d days" % (Mmin, tau))
		poisson_probs = PT.get_prob_n(n, tau)
		poisson_n = poisson_probs * num_intervals
		if poisson_n.max() < 1:
			print('Warning: Poisson max. intervals < 1: consider longer interval!')
		xy_datasets.append((n, poisson_n))
		catalog_taus.append(tau)

	## Plot
	xmin = kwargs.pop('xmin', -0.5)
	xmax = kwargs.pop('xmax', nmax + 0.5)
	xlabel = kwargs.pop('xlabel', "Number of events per interval")
	ylabel = kwargs.pop('ylabel', "Number of intervals")
	if title is None:
		title = (r"Poisson test for $M\geq%.1f$ (t=%d d., nt=%d)"
				% (Mmin, interval, num_intervals))

	## Histogram of number of intervals having n events
	histogram_labels = ['%s (catalog)' % label for label in labels]
	bins = np.arange(nmax + 2) - 0.5
	ax = plot_histogram(histogram_datasets, bins=bins,
						labels=histogram_labels, colors=bar_colors,
						align='mid', stacked=False,
						fig_filespec='wait', skip_frame=True, ax=ax, **kwargs)

	if line_colors is None:
		line_colors = bar_colors
	plot_xy(xy_datasets, linewidths=[3], colors=['w'], labels=['_nolegend_'],
			skip_frame=True, fig_filespec='wait', ax=ax)
	xy_labels = [r'%s (Poisson, $\tau=%.1f$ d.)' % (label, tau)
				for (label, tau) in zip(labels, catalog_taus)]
	return plot_xy(xy_datasets, linewidths=[2],
						labels=xy_labels, colors=line_colors,
						xlabel=xlabel, ylabel=ylabel, xmin=xmin, xmax=xmax,
						title=title,
						fig_filespec=fig_filespec, dpi=dpi, ax=ax, **kwargs)


def plot_catalogs_map(catalogs, labels=[],
					symbols=[], edge_colors=[], fill_colors=[], edge_widths=[],
					symbol_size=9, symbol_size_inc=4,
					Mtype="MW", Mrelation={},
					circle=None, region=None, projection="merc", resolution="i",
					dlon=1., dlat=1.,
					source_model=None, sm_color='k', sm_line_style='-',
					sm_line_width=2, sm_label_size=11, sm_label_colname="ShortName",
					sites=[], site_symbol='o', site_color='b', site_size=10, site_legend="",
					title=None, legend_location=0, fig_filespec=None, fig_width=0, dpi=300):
	"""
	Plot multiple catalogs on a map

	:param catalogs:
		List containing instances of :class:`EQCatalog`
	:param symbols:
		List containing earthquake symbols for each catalog
		(matplotlib marker specifications)
	:param edge_colors:
		List containing symbol edge colors for each catalog
		(matplotlib color specifications)
		(default: [])
	:param fill_colors:
		List containing symbol fill colors for each catalog
		(matplotlib color specifications)
		(default: [])
	:param edge_widths:
		List containing symbol edge width for each catalog
		(default: [])
	:param labels:
		List containing plot labels, one for each catalog (default: [])
	:param symbol_size:
		Int or Float, symbol size in points (default: 9)
	:param symbol_size_inc:
		Int or Float, symbol size increment per magnitude relative to M=3
		(default: 4)
	:param Mtype:
		String, magnitude type for magnitude scaling (default: "MW")
	:param Mrelation:
		{str: str} dict, mapping name of magnitude conversion relation
		to magnitude type ("MW", "MS" or "ML")
		(default: {})
	:param circle:
		((lon, lat), float, string), respectively defining center, radius (in
		km) and color of circle to plot
	:param region:
		(w, e, s, n) tuple specifying rectangular region to plot in
		geographic coordinates (default: None)
	:param projection:
		String, map projection. See Basemap documentation
		(default: "merc")
	:param resolution:
		String, map resolution (coastlines and country borders):
		'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high), 'f' (full)
		(default: 'i')
	:param dlon:
		Float, meridian interval in degrees (default: 1.)
	:param dlat:
		Float, parallel interval in degrees (default: 1.)
	:param source_model:
		String, name of source model to overlay on the plot
		(default: None)
	:param sm_color:
		matplotlib color specification to plot source model
		(default: 'k')
	:param sm_line_style:
		String, line style to plot source model (default: '-')
	:param sm_line_width:
		Int, line width to plot source model (default: 2)
	:param sm_label_size:
		Int, font size of source labels. If 0 or None, no labels will
		be plotted (default: 11)
	:param sm_label_colname:
		Str, column name of GIS table to use as label (default: "ShortName")
	:param sites:
		List of (lon, lat) tuples or instance of :class:`PSHASite`
	:param site_symbol:
		matplotlib marker specifications for site symbols (default: 'o')
	:param site_color:
		matplotlib color specification for site symbols (default: 'b')
	:param site_size:
		Int, size to be used for site symbols (default: 10)
	:param site_legend:
		String, common text referring to all sites to be placed in legend
		(default: "")
	:param title:
		String, plot title (default: None)
	:param legend_location:
		String or Int: location of legend (matplotlib location code):
			"best" 	0
			"upper right" 	1
			"upper left" 	2
			"lower left" 	3
			"lower right" 	4
			"right" 	5
			"center left" 	6
			"center right" 	7
			"lower center" 	8
			"upper center" 	9
			"center" 	10
		(default: 0)
	:param fig_filespec:
		String, full path of image to be saved.
		If None (default), map is displayed on screen.
	:param fig_width:
		Float, figure width in cm, used to recompute :param:`dpi` with
		respect to default figure width (default: 0)
	:param dpi:
		Int, image resolution in dots per inch (default: 300)
	"""
	from mpl_toolkits.basemap import Basemap

	## Symbols, colors, and labels
	if not symbols:
		symbols = ["o"]
	if not edge_colors:
		edge_colors = ("r", "g", "b", "c", "m", "k")
	if not fill_colors:
		fill_colors = ["None"]
	if not edge_widths:
		edge_widths = [1]
	if not labels:
		labels = [None]

	## Determine map extent and center
	if not region:
		if catalogs[0].region:
			region = list(catalogs[0].region)
		else:
			region = list(catalogs[0].get_region())
			lon_range = region[1] - region[0]
			lat_range = region[3] - region[2]
			region[0] -= lon_range / 5.
			region[1] += lon_range / 5.
			region[2] -= lat_range / 5.
			region[3] += lat_range / 5.
	else:
		region = list(region)
	lon_0 = (region[0] + region[1]) / 2.
	lat_0 = (region[2] + region[3]) / 2.

	## Base map
	map = Basemap(projection=projection, resolution=resolution, llcrnrlon=region[0], llcrnrlat=region[2], urcrnrlon=region[1], urcrnrlat=region[3], lon_0=lon_0, lat_0=lat_0)
	map.drawcoastlines()
	map.drawcountries()

	## Meridians and parallels
	if dlon:
		first_meridian = np.ceil(region[0] / dlon) * dlon
		last_meridian = np.floor(region[1] / dlon) * dlon + dlon
		meridians = np.arange(first_meridian, last_meridian, dlon)
		map.drawmeridians(meridians, labels=[0,1,0,1])
	if dlat:
		first_parallel = np.ceil(region[2] / dlat) * dlat
		last_parallel = np.floor(region[3] / dlat) * dlat + dlat
		parallels = np.arange(first_parallel, last_parallel, dlat)
		map.drawparallels(parallels, labels=[0,1,0,1])

	## Source model
	if source_model:
		from .rob.source_models import (rob_source_models_dict, read_source_model)
		try:
			rob_source_models_dict[source_model_name]["gis_filespec"]
		except:
			source_model_name = os.path.splitext(os.path.split(source_model)[1])[0]
		else:
			source_model_name = source_model
		model_data = read_source_model(source_model)
		for i, zone_data in enumerate(model_data.values()):
			geom = zone_data['obj']
			lines = []
			if geom.GetGeometryName() == "LINESTRING":
				## In some versions of ogr, GetPoints method does not exist
				#points = linear_ring.GetPoints()
				points = [geom.GetPoint(i) for i in range(geom.GetPointCount())]
				lines.append(points)
				centroid = None
			elif geom.GetGeometryName() == "POLYGON":
				centroid = geom.Centroid()
				for linear_ring in geom:
					points = [linear_ring.GetPoint(i) for i in range(linear_ring.GetPointCount())]
					lines.append(points)
			for j, line in enumerate(lines):
				lons, lats, _ = zip(*line)
				x, y = map(lons, lats)
				if i == 0 and j == 0:
					label = source_model_name
				else:
					label = "_nolegend_"
				map.plot(x, y, ls=sm_line_style, lw=sm_line_width, color=sm_color, label=label)

				if centroid and sm_label_size:
					x, y = map(centroid.GetX(), centroid.GetY())
					if isinstance(sm_label_colname, basestring):
						zone_label = zone_data.get("sm_label_colname", "")
					else:
						zone_label = " / ".join([str(zone_data[colname]) for colname in sm_label_colname])
					pylab.text(x, y, zone_label, color=sm_color, fontsize=sm_label_size, fontweight='bold', ha='center', va='center')

	## Catalogs
	for i, catalog in enumerate(catalogs):
		if len(catalog):
			symbol = symbols[i%len(symbols)]
			edge_color = edge_colors[i%len(edge_colors)]
			if edge_color is None:
				edge_color = "None"
			fill_color = fill_colors[i%len(fill_colors)]
			if fill_color is None:
				fill_color = "None"
			edge_width = edge_widths[i%len(edge_widths)]
			label = labels[i%len(labels)]
			if label is None:
				label = catalog.name

			## Earthquake symbol size varying with magnitude
			if not symbol_size_inc:
				symbol_sizes = symbol_size ** 2
			else:
				magnitudes = catalog.get_magnitudes(Mtype, Mrelation)
				symbol_sizes = symbol_size + (magnitudes - 3.0) * symbol_size_inc
				symbol_sizes = symbol_sizes ** 2
				if symbol_sizes.min() <= 0:
					print("Warning: negative or zero symbol size encountered")
				#print(symbol_sizes.min(), symbol_sizes.max())

			## Earthquake epicenters
			if len(catalog.eq_list) > 0:
				lons, lats = catalog.get_longitudes(), catalog.get_latitudes()
				x, y = map(lons, lats)
				map.scatter(x, y, s=symbol_sizes, marker=symbol, edgecolors=edge_color, facecolors=fill_color, linewidth=edge_width, label=label)

	## Sites
	for i, site in enumerate(sites):
		try:
			lon, lat = site.longitude, site.latitude
			name = site.name
		except:
			lon, lat, name = site[:3]
		x, y = map(lon, lat)
		if i == 0:
			label = site_legend
		else:
			label = None
		map.plot(x, y, site_symbol, markerfacecolor=site_color, markeredgecolor='k', markersize=site_size, label=label)

	## Circle
	if circle:
		from openquake.hazardlib.geo.geodetic import point_at
		center, radius, color = circle
		x, y = [], []
		for azimuth in range(0, 360):
			lon, lat = point_at(center[0], center[1], azimuth, radius)
			x.append(lon)
			y.append(lat)
		x.append(x[0])
		y.append(y[0])
		x, y = map(x,y)
		map.plot(x, y, c=color)

	## Map border and title
	map.drawmapboundary()
	if title:
		pylab.title(title)
	plt.legend(loc=legend_location)

	#plt.tight_layout()
	if fig_filespec:
		default_figsize = pylab.rcParams['figure.figsize']
		#default_dpi = pylab.rcParams['figure.dpi']
		if fig_width:
			fig_width /= 2.54
			dpi = dpi * (fig_width / default_figsize[0])
		pylab.savefig(fig_filespec, dpi=dpi)
		pylab.clf()
	else:
		pylab.show()
