"""
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int


import os

import numpy as np
import pylab
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, MaxNLocator

import eqcatalog.time_functions_np as tf


MPL_FONT_SIZES = ['xx-small', 'x-small', 'small', 'medium',
				'large', 'x-large', 'xx-large']


def plot_xy(datasets, xscaling='lin', yscaling='lin',
			colors=[], linewidths=[], linestyles=[], labels=[],
			markers=[], marker_sizes=[], marker_intervals=[],
			marker_edge_colors=[], marker_fill_colors=[], marker_edge_widths=[],
			xmin=None, xmax=None, ymin=None, ymax=None,
			xlabel='', ylabel='', title='', grid=False,
			label_font_size='medium', title_font_size='large',
			legend_font_size='medium', legend_location=0,
			fig_filespec=None, dpi=300, ax=None):
	"""
	Generic function to plot (X, Y) data sets

	:param datasets:
		list with (x, y) array tuples
	"""
	# TODO: xticks, xtick_labels, yticks, ytick_labels, marker_edge_widths...
	from itertools import cycle
	from matplotlib.font_manager import FontProperties

	if ax is None:
		#ax = pylab.axes()
		fig, ax = pylab.subplots()
	else:
		fig_filespec = "hold"

	## markers, colors, linewidhts, linestyles, labels, etc.
	if not colors:
		colors = ["r", "g", "b", "c", "m", "k"]
	if not linewidths:
		linewidths = [1]
	if not linestyles:
		linestyles = ['-']
	if not markers:
		markers = ['']
	if not marker_sizes:
		marker_sizes = [6]
	if not marker_intervals:
		marker_intervals = [None]
	if not marker_edge_colors:
		marker_edge_colors = ['k']
	if not marker_fill_colors:
		marker_fill_colors = colors[:]
	if not marker_edge_widths:
		marker_edge_widths = [1]
	if not labels:
		labels = ['_nolegend_']

	colors = cycle(colors)
	linewidths = cycle(linewidths)
	linestyles = cycle(linestyles)
	markers = cycle(markers)
	marker_sizes = cycle(marker_sizes)
	marker_intervals = cycle(marker_intervals)
	marker_edge_colors = cycle(marker_edge_colors)
	marker_fill_colors = cycle(marker_fill_colors)
	marker_edge_widths = cycle(marker_edge_widths)
	labels = cycle(labels)

	#if xscaling == 'lin':
	#	if yscaling == 'lin':
	#		plotfunc = getattr(ax, 'plot')
	#	elif yscaling == 'log':
	#		plotfunc = getattr(ax, 'semilogy')
	#elif xscaling == 'log':
	#	if yscaling == 'lin':
	#		plotfunc = getattr(ax, 'semilogx')
	#	elif yscaling == 'log':
	#		plotfunc = getattr(ax, 'loglog')

	for (x, y) in datasets:
		color = colors.next()
		linewidth = linewidths.next()
		linestyle = linestyles.next()
		marker = markers.next()
		marker_size = marker_sizes.next()
		marker_interval = marker_intervals.next()
		marker_edge_color = marker_edge_colors.next()
		marker_fill_color = marker_fill_colors.next()
		marker_edge_width = marker_edge_widths.next()
		label = labels.next()

		ax.plot(x, y, marker, color=color, ls=linestyle, lw=linewidth,
				ms=marker_size, mec=marker_edge_color, mfc=marker_fill_color,
				mew=marker_edge_width, markevery=marker_interval, label=label)

	if grid:
		ax.grid('on')

	legend_font_size = legend_font_size or label_font_size
	legend_font = FontProperties(size=legend_font_size)
	ax.legend(loc=legend_location, prop=legend_font)
	ax.set_xlabel(xlabel, fontsize=title_font_size)
	ax.set_ylabel(ylabel, fontsize=title_font_size)
	ax.set_title(title, fontsize=title_font_size)

	xscaling = {'lin': 'linear', 'log': 'log'}[xscaling]
	ax.set_xscale(xscaling)
	yscaling = {'lin': 'linear', 'log': 'log'}[yscaling]
	ax.set_yscale(yscaling)

	_xmin, _xmax = ax.get_xlim()
	xmin = xmin or _xmin
	xmax = xmax or _xmax
	ax.set_xlim(xmin, xmax)

	_ymin, _ymax = ax.get_ylim()
	ymin = ymin or _ymin
	ymax = ymax or _ymax
	ax.set_ylim(ymin, ymax)

	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size(label_font_size)

	if fig_filespec == "hold":
		return
	elif fig_filespec:
		#default_figsize = pylab.rcParams['figure.figsize']
		#if fig_width:
		#	fig_width /= 2.54
		#	dpi = dpi * (fig_width / default_figsize[0])
		pylab.savefig(fig_filespec, dpi=dpi)
		pylab.clf()
	else:
		pylab.show()


def plot_cumulated_moment(catalogs, colors=[], linewidths=[], linestyles=[], markers=[],
						marker_sizes=[], marker_edge_colors=[], marker_fill_colors=[],
						labels=[], start_date=None, end_date=None, time_unit='Y',
						Mrelation="default", M0max=None, grid=True, title='',
						label_font_size='medium', title_font_size='large',
						legend_font_size='medium', legend_location=2,
						fig_filespec=None, dpi=300, ax=None):
	"""
	:param time_unit:
		'Y', 'D', 'h', 'm', 's'
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
	marker_intervals = [2]
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

	time_unit_str = {'Y': 'years', 'D': 'days', 'h': 'hours',
					'm': 'minutes', 's': 'seconds'}[time_unit]
	xlabel = "Time (%s)" % time_unit_str
	ylabel = "Seismic Moment (N.m)"

	if time_unit == 'Y':
		xmin = tf.to_fractional_year(start_date)
		xmax = tf.to_fractional_year(end_date)
	else:
		xmin = 0
		xmax = tf.timespan(start_date, end_date, time_unit)

	ymax = M0max

	marker_edge_colors = colors[:]
	marker_fill_colors = ['None']

	return plot_xy(datasets, colors=colors, linewidths=linewidths,
			linestyles=linestyles, labels=labels, markers=markers,
			marker_sizes=marker_sizes, marker_intervals=marker_intervals,
			marker_edge_colors=marker_edge_colors, marker_fill_colors=marker_fill_colors,
			xmin=xmin, xmax=xmax, ymax=ymax, legend_location=legend_location,
			xlabel=xlabel, ylabel=ylabel, title=title, grid=grid,
			label_font_size=label_font_size, title_font_size=title_font_size,
			fig_filespec=fig_filespec, dpi=dpi, ax=ax)
