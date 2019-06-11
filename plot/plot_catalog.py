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

import eqcatalog.time_functions_np as tf
from .plot_generic import plot_xy


def plot_magnitude_time(catalogs, Mtype, Mrelation, rel_time_unit=None,
						Mrange=(None, None), start_date=None, end_date=None,
						markers=['o'], marker_sizes=[8], colors=[], fill_colors=[],
						labels=[],
						completeness=None, completeness_color='r',
						lang='en', **kwargs):
	"""
	:param catalogs:
		list with instances of :class:`EQCatalog`
	:param Mtype:
		str, magnitude type
	:param Mrelation:
		OrderedDict or str, magnitude relations to use for conversion
		to :param:`Mtype`
		(default: {})
	:param rel_time_unit:
		str, relative time unit
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
		list of line colors to cycle over for each catalog
		(default: [], will use default colors for :param:`style_sheet`)
	:param fill_colors:
		list of fill colors to cycle over for each catalog
		(default: [], will not apply fill color)
	:param labels:
		list of labels to cycle over for each catalog
		(default: [], will not label catalogs)
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

	See :func:`eqcatalog.plot.plot_generic.plot_xy` for additional
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
		kwargs['xmin'] = 0
		kwargs['xmax'] = tf.timespan(start_date, end_date, rel_time_unit)
	else:
		kwargs['xmin'] = tf.to_py_datetime(start_date)
		kwargs['xmax'] = tf.to_py_datetime(end_date)

	min_mag, max_mag = Mrange
	if min_mag is not None:
		kwargs['ymin'] = min_mag
	if max_mag is not None:
		kwargs['ymax'] = max_mag

	## Catalogs
	datasets = []
	for i, catalog in enumerate(catalogs):
		mags = catalog.get_magnitudes(Mtype, Mrelation)
		if rel_time_unit:
			dates = tf.timespan(start_date, catalog.get_datetimes(), rel_time_unit)
			idxs = (dates >= 0)
			dates = dates[idxs]
			mags = mags[idxs]
		else:
			dates = tf.to_py_datetime(catalog.get_datetimes())
		datasets.append((dates, mags))
		if isinstance(labels, list) and len(labels) <= i:
			labels.append(catalog.name)

	kwargs['marker_edge_colors'] = colors or list('bgrcmyk')
	kwargs['colors'] = ['None'] * len(datasets)
	kwargs['marker_fill_colors'] = fill_colors
	kwargs['fill_colors'] = []
	kwargs['markers'] = markers
	kwargs['marker_sizes'] = marker_sizes

	## Completeness
	if completeness:
		clabel = {"en": "Completeness magnitude",
				"nl": "Compleetheidsmagnitude",
				"fr": u"Magnitude de complétude",
				"nlfr": u"Compleetheid / Complétude"}[lang]
		x, y = completeness.min_dates, completeness.min_mags
		#x = np.concatenate([x[:1], np.repeat(x[1:], 2), [tf.to_fractional_year(end_date)]])
		x = np.concatenate([x[:1], np.repeat(x[1:], 2), [tf.to_py_datetime(end_date)]])
		if rel_time_unit:
			x = tf.timespan(start_date, x, rel_time_unit)
		y = np.repeat(y, 2)
		datasets.append((x, y))

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
	if not 'xlabel' in kwargs:
		kwargs['xlabel'] = {"en": "Time",
							"nl": "Tijd",
							"fr": "Temps",
							"nlfr": "Tijd / Temps"}[lang]
		if rel_time_unit:
			time_unit_str = {'Y': 'years', 'W': 'weeks', 'D': 'days',
							'h': 'hours', 'm': 'minutes', 's': 'seconds'}[rel_time_unit]
			kwargs['xlabel'] += ' (%s)' % time_unit_str
	kwargs['ylabel'] = kwargs.get('ylabel', "Magnitude ($M_%s$)" % Mtype[1])

	## Default tick intervals
	kwargs['xtick_interval'] = kwargs.get('xtick_interval', (AutoDateLocator(),
											matplotlib.ticker.AutoMinorLocator()))
	kwargs['ytick_interval'] = kwargs.get('ytick_interval', (1, 0.1))

	## Convert xticks to datetimes if necessary
	if 'xticks' in kwargs:
		xticks = []
		for xt in kwargs['xticks']:
			if isinstance(xt, int):
				xt = str(xt)
			xt = tf.as_np_datetime(xt)
			xticks.append(tf.to_py_datetime(xt))
		kwargs['xticks'] = xticks
		## If xticks are datetimes, we need to explicitly set a date formatter
		## for the labels
		kwargs['xticklabels'] = kwargs.get('xticklabels', DateFormatter('%Y'))

	kwargs['xgrid'] = kwargs.get('xgrid', 1)
	kwargs['ygrid'] = kwargs.get('ygrid', 1)

	return plot_xy(datasets, **kwargs)

plot_magnitude_time.__doc__ += plot_xy.__doc__.split("\n", 4)[4]


def plot_cumulated_moment(catalogs, start_date=None, end_date=None, time_unit='Y',
						Mrelation={}, M0max=None, **kwargs):
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
	:param time_unit:
		str, time unit to use for X axis, one of 'Y', 'D', 'h', 'm', 's'
		(default: 'Y')
	:param Mrelation:
		OrderedDict or str, magnitude relations to use for conversion
		to seismic moment
		(default: {})
	:param M0max:
		float, maximum moment value in Y axis
		(default: None)
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
		if time_unit == 'Y':
			dates = catalog.get_fractional_years()
		else:
			dates = tf.timespan(start_date, catalog.get_datetimes(), time_unit)

		unbinned_M0 = catalog.get_M0(Mrelation=Mrelation)
		M0_cumul = np.cumsum(unbinned_M0)

		## Construct arrays with duplicate points in order to plot horizontal
		## lines between subsequent points
		M0_cumul2 = np.concatenate([[0.], np.repeat(M0_cumul, 2)[:-1]])
		dates2 = np.repeat(dates, 2)

		datasets.append((dates2, M0_cumul2))

	kwargs['marker_intervals'] = [2]

	time_unit_str = {'Y': 'years', 'W': 'weeks', 'D': 'days', 'h': 'hours',
					'm': 'minutes', 's': 'seconds'}[time_unit]
	kwargs['xlabel'] = "Time (%s)" % time_unit_str
	kwargs['ylabel'] = "Seismic Moment (N.m)"

	if time_unit == 'Y':
		kwargs['xmin'] = tf.to_fractional_year(start_date)
		kwargs['xmax'] = tf.to_fractional_year(end_date)
	else:
		kwargs['xmin'] = 0
		kwargs['xmax'] = tf.timespan(start_date, end_date, time_unit)

	kwargs['ymax'] = M0max

	kwargs['marker_edge_colors'] = kwargs.get('colors', [])[:]
	kwargs['marker_fill_colors'] = ['None']

	return plot_xy(datasets, **kwargs)

plot_cumulated_moment.__doc__ += plot_xy.__doc__.split("\n", 4)[4]
