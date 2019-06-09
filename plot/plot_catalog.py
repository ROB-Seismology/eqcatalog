"""
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int


import os

import numpy as np
import pylab
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.font_manager import FontProperties

import eqcatalog.time_functions_np as tf


MPL_FONT_SIZES = ['xx-small', 'x-small', 'small', 'medium',
				'large', 'x-large', 'xx-large']


def plot_xy(datasets,
			colors=[], fill_colors=[], linewidths=[1], linestyles=['-'], labels=[],
			markers=[], marker_sizes=[6], marker_intervals=[],
			marker_edge_colors=['k'], marker_fill_colors=[], marker_edge_widths=[1],
			xscaling='lin', yscaling='lin',
			xmin=None, xmax=None, ymin=None, ymax=None,
			xlabel='', ylabel='', ax_label_fontsize='large',
			xticks=None, xticklabels=None, yticks=None, yticklabels=None,
			tick_label_fontsize='medium',
			title='', title_fontsize='large',
			legend_location=0, legend_fontsize='medium',
			grid=False, style_sheet='classic',
			fig_filespec=None, dpi=300, ax=None):
	"""
	Generic function to plot (X, Y) data sets (lines, symbols and/or polygons)

	:param datasets:
		list with (x, y) array tuples
	:param colors:
		list of line colors to cycle over for each dataset
		(default: [], will use default colors for :param:`style_sheet`)
	:param fill_colors:
		list of fill colors to cycle over for each dataset
		(default: [], will not apply fill color)
	:param linewidths:
		list of line widths to cycle over for each dataset
		(default: [1])
	:param linestyles:
		list of line styles to cycle over for each dataset
		(default: ['-'])
	:param labels:
		list of labels to cycle over for each dataset
		(default: [], will not label curves)
	:param markers:
		list of marker symbols to cycle over for each dataset
		(default: [], will not draw markers)
	:param marker_sizes:
		list of marker sizes to cycle over for each dataset
		(default: [6])
	:param marker_intervals:
		(default: [], will draw marker for each datapoint)
	:param marker_edge_colors:
		list of marker line colors to cycle over for each dataset
		(default: ['k'])
	:param marker_fill_colors:
		list of marker fill colors to cycle over for each dataset
		(default: [], will use colors defined in :param:`colors`)
	:param marker_edge_widths:
		list of marker line widths to cycle over for each dataset
		(default: [1])
	:param xscaling:
		str, scaling to use for X axis ('lin' or 'log')
		(default: 'lin')
	:param yscaling:
		str, scaling to use for Y axis ('lin' or 'log')
		(default: 'lin')
	:param xmin:
		float, start value for X axis
		(default: None, let matplotlib decide)
	:param xmax:
		float, end value for X axis
		(default: None, let matplotlib decide)
	:param ymin:
		float, start value for Y axis
		(default: None, let matplotlib decide)
	:param ymax:
		float, end value for Y axis
		(default: None, let matplotlib decide)
	:param xlabel:
		str, label for X axis
		(default: '')
	:param ylabel:
		str, label for Y axis
		(default: '')
	:param ax_label_fontsize:
		int or str, font size to use for axis labels
		(default: 'large')
	:param xticks:
		list or array, X axis tick positions
		(default: None, let matplotlib decide)
	:param xticklabels:
		list of labels corresponding to X axis ticks
		(default: None, let matplotlib decide)
	:param yticks:
		list or array, Y axis tick positions
		(default: None, let matplotlib decide)
	:param yticklabels:
		list of labels corresponding to Y axis ticks
		(default: None, let matplotlib decide)
	:param tick_label_fontsize:
		int or str, font size to use for axis tick labels
		(default: 'medium')
	:param title:
		str, plot title
		(default: '')
	:param title_fontsize:
		str, font size to use for plot title
		(default: 'large')
	:param legend_location:
		int or str, matplotlib spec where to plot legend
		(default: 0 = auto-detect optimal position)
	:param legend_fontsize:
		int or str, font size to use for legend labels
		If not specified, will use the value of :param:`tick_label_fontsize`
		(default: 'medium')
	:param grid:
		bool, whether or not to show grid lines
		(default: False)
	:param style_sheet:
		str, matplotlib style sheet to apply to plot
		See matplotlib.style.available for availabel style sheets
		(default: 'classic')
	:param fig_filespec:
		str, full path to output file
		If None, will plot on screen
		If 'hold', plotting is deferred
		(default: None)
	:param dpi:
		int, resolution of plot,
		only applies if :param:`fig_filespec` is set to output file
		(default: 300)
	:param ax:
		matplotlib Axes instance, in which plot will be drawn
		If specified, :param:`fig_filespec` will be overridden with 'hold'
		(default: None, will generate new Axes instance)

	:return:
		matplotlib Axes instance if :param:`fig_filespec` is set to
		'hold', else None
	"""
	from itertools import cycle

	pylab.style.use(style_sheet)

	if ax is None:
		#ax = pylab.axes()
		fig, ax = pylab.subplots()
	else:
		fig_filespec = "hold"

	## markers, colors, linewidhts, linestyles, labels, etc.
	if not colors:
		#colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
		#colors = 'bgrcmyk'
		colors = pylab.rcParams['axes.prop_cycle'].by_key()['color']
	if not fill_colors:
		fill_colors = [None]
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
	fill_colors = cycle(fill_colors)
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
		fill_color = fill_colors.next()
		linewidth = linewidths.next()
		linestyle = linestyles.next()
		marker = markers.next()
		marker_size = marker_sizes.next()
		marker_interval = marker_intervals.next()
		marker_edge_color = marker_edge_colors.next()
		marker_fill_color = marker_fill_colors.next()
		marker_edge_width = marker_edge_widths.next()
		label = labels.next()

		if fill_color:
			ax.fill(x, y, facecolor=fill_color, edgecolor=color, lw=linewidth,
				ls=linestyle, label=label)
			if marker:
				ax.plot(x, y, marker, lw=0, ms=marker_size, mec=marker_edge_color,
				mfc=marker_fill_color, mew=marker_edge_width, markevery=marker_interval,
				label='_nolegend_')
		else:
			ax.plot(x, y, marker, color=color, ls=linestyle, lw=linewidth,
				ms=marker_size, mec=marker_edge_color, mfc=marker_fill_color,
				mew=marker_edge_width, markevery=marker_interval, label=label)

	if grid:
		ax.grid('on')

	## Legend and title
	legend_fontsize = legend_fontsize or tick_label_fontsize
	legend_font = FontProperties(size=legend_fontsize)
	ax.legend(loc=legend_location, prop=legend_font)
	ax.set_title(title, fontsize=title_fontsize)

	## Axis labels
	ax.set_xlabel(xlabel, fontsize=ax_label_fontsize)
	ax.set_ylabel(ylabel, fontsize=ax_label_fontsize)

	## Axis scaling
	xscaling = {'lin': 'linear', 'log': 'log'}[xscaling]
	ax.set_xscale(xscaling)
	yscaling = {'lin': 'linear', 'log': 'log'}[yscaling]
	ax.set_yscale(yscaling)

	## Axis limits
	_xmin, _xmax = ax.get_xlim()
	xmin = xmin or _xmin
	xmax = xmax or _xmax
	ax.set_xlim(xmin, xmax)

	_ymin, _ymax = ax.get_ylim()
	ymin = ymin or _ymin
	ymax = ymax or _ymax
	ax.set_ylim(ymin, ymax)

	## Ticks and tick labels
	if xticks is not None:
		ax.set_xticks(xticks)
	if xticklabels:
		ax.set_xticklabels(xticklabels)
	if yticks is not None:
		ax.set_yticks(yticks)
	if yticklabels:
		ax.set_yticklabels(yticklabels)

	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size(tick_label_fontsize)

	if fig_filespec == "hold":
		return ax
	elif fig_filespec:
		#default_figsize = pylab.rcParams['figure.figsize']
		#if fig_width:
		#	fig_width /= 2.54
		#	dpi = dpi * (fig_width / default_figsize[0])
		pylab.savefig(fig_filespec, dpi=dpi)
		pylab.clf()
	else:
		pylab.show()

	pylab.style.use('default')


def plot_cumulated_moment(catalogs, start_date=None, end_date=None, time_unit='Y',
						Mrelation="default", M0max=None, **kwargs):
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
	kwargs['marker_intervals'] = [2]
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
