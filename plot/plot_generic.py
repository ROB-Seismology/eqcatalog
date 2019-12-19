"""
Generic plot functions based on matplotlib
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str

import datetime

import numpy as np
import pylab
import matplotlib
import matplotlib.ticker
import matplotlib.dates as mpl_dates
from matplotlib.font_manager import FontProperties


__all__ = ['plot_xy', 'plot_density', 'plot_histogram', 'plot_grid',
			'plot_ax_frame']


MPL_FONT_SIZES = ['xx-small', 'x-small', 'small', 'medium',
				'large', 'x-large', 'xx-large']

MPL_INTERVAL_DICT = {'Y': 0, 'M': 1, 'W': 2, 'D': 3, 'h': 4, 'm': 5, 's': 6}

MPL_DATE_LOCATOR_DICT = {'Y': mpl_dates.YearLocator,
						'M': mpl_dates.MonthLocator,
						'd': mpl_dates.WeekdayLocator,
						'D': mpl_dates.DayLocator,
						'h': mpl_dates.HourLocator,
						'm': mpl_dates.MinuteLocator,
						's': mpl_dates.SecondLocator}


def _create_date_locator(tick_interval):
	"""
	Create matplotlib date locator from tick interval specification

	:param tick_interval:
		- 0 (= no ticks)
		- None (= automatic ticks)
		- string XXY, with XX interval and Y time unit:
			'Y', 'M', 'D', 'd', 'h', 'm', 's'
			(year|month|day|weekday|hour|minute|second)

	:return:
		matplotlib date locator object
	"""
	if tick_interval == 0:
			date_loc = matplotlib.ticker.NullLocator()
	elif tick_interval is None:
		date_loc = mpl_dates.AutoDateLocator(interval_multiples=True)
	else:
		if isinstance(tick_interval, basestring):
			val, tick_unit = int(tick_interval[:-1]), tick_interval[-1:]
		else:
			val = tick_interval
			tick_unit = 'Y'
		#tu_key = MPL_INTERVAL_DICT[tick_unit]
		#for key in range(tu_key):
		#	date_loc.intervald[key] = []
		#date_loc.intervald[tu_key] = [val]
		loc_kwargs = {}
		loc_kwargs[{'Y': 'base'}.get(tick_unit, 'interval')] = val
		date_loc = MPL_DATE_LOCATOR_DICT[tick_unit](**loc_kwargs)

	return date_loc


def plot_xy(datasets,
			colors=[], fill_colors=[], linewidths=[1], linestyles=['-'], labels=[],
			markers=[], marker_sizes=[6], marker_intervals=[],
			marker_edge_colors=['k'], marker_fill_colors=[], marker_edge_widths=[1],
			marker_labels=[], marker_label_fontsize='small',
			xscaling='lin', yscaling='lin',
			xmin=None, xmax=None, ymin=None, ymax=None,
			xlabel='', ylabel='', ax_label_fontsize='large',
			xticks=None, xticklabels=None, xtick_interval=None, xtick_rotation=0,
			xtick_direction='', xtick_side='',
			yticks=None, yticklabels=None, ytick_interval=None, ytick_rotation=0,
			ytick_direction='', ytick_side='',
			tick_label_fontsize='medium', tick_params={},
			title='', title_fontsize='large',
			xgrid=0, ygrid=0,
			hlines=[], hline_args={}, vlines=[], vline_args={},
			legend_location=0, legend_fontsize='medium',
			style_sheet='classic', border_width=0.2,
			fig_filespec=None, figsize=None, dpi=300, ax=None):
	"""
	Generic function to plot (X, Y) data sets (lines, symbols and/or polygons)

	:param datasets:
		list with (x, y) array tuples (either values or datetimes)
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
		Prepend '-' to invert orientation of X axis
		(default: 'lin')
	:param yscaling:
		str, scaling to use for Y axis ('lin' or 'log')
		Prepend '-' to invert orientation of Y axis
		(default: 'lin')
	:param xmin:
		float, start value for X axis
		Note that, if X values of :param:`datasets` are datetimes,
		this should be datetime also
		(default: None, let matplotlib decide)
	:param xmax:
		float, end value for X axis
		Note that, if X values of :param:`datasets` are datetimes,
		this should be datetime also
		(default: None, let matplotlib decide)
	:param ymin:
		float, start value for Y axis
		Note that, if Y values of :param:`datasets` are datetimes,
		this should be datetime also
		(default: None, let matplotlib decide)
	:param ymax:
		float, end value for Y axis
		Note that, if Y values of :param:`datasets` are datetimes,
		this should be datetime also
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
		Note that, if X values of :param:`datasets` are datetimes,
		these should be datetimes also
		(default: None, let matplotlib decide)
	:param xticklabels:
		X axis tick labels, either:
		- None (= automatic labels)
		- list of labels corresponding to :param:`xticks`
		- matplotlib Formatter object
		- format string (for dates or scalars)
		- '' or [] (= no tick labels)
		(default: None, let matplotlib decide)
	:param xtick_interval:
		X axis tick interval specification
		single value (major ticks only) or tuple (major/minor ticks) of:
		- matplotlib Locator object
		- None (= automatic ticks)
		- 0 (= no ticks)
		- int (= integer tick interval)
		- str (= tick interval for dates, where last char is in YMDdhms
			(year|month|day|weekday|hour|minute|second)
		(default: None)
	:param xtick_rotation:
		float, rotation angle for X axis tick labels
		(default: 0)
	:param xtick_direction:
		str, X axis tick direction: 'in', 'out' or 'both'
		(default: '')
	:param xtick_side:
		str, on which side of the plot X ticks should be drawn:
		'bottom', 'top' or 'both'
		(default: '')
	:param yticks:
		list or array, Y axis tick positions
		Note that, if Y values of :param:`datasets` are datetimes,
		these should be datetimes also
		(default: None, let matplotlib decide)
	:param yticklabels:
		Y axis tick labels
		See :param:`xticklabels` for options
	:param ytick_interval:
		Y axis tick interval specification
		see :param:`xtick_interval` for options
	:param ytick_rotation:
		float, rotation angle for Y axis tick labels
		(default: 0)
	:param ytick_direction:
		str, Y axis tick direction: 'in', 'out' or 'both'
		(default: '')
	:param ytick_side:
		str, on which side of the plot Y ticks should be drawn:
		'left', 'right' or 'both'
		(default: '')
	:param tick_label_fontsize:
		int or str, font size to use for axis tick labels
		(default: 'medium')
	:param tick_params:
		dict, containing keyword arguments for :func:`ax.tick_params`,
		that will be applied to both the X and Y axes
		(default: {})
	:param title:
		str, plot title
		(default: '')
	:param title_fontsize:
		str, font size to use for plot title
		(default: 'large')
	:param xgrid:
		int, 0/1/2/3 = draw no/major/minor/major+minor X grid lines
		(default: 0)
	:param ygrid:
		int, 0/1/2/3 = draw no/major/minor/major+minor Y grid lines
		(default: 0)
	:param hlines:
		[y, xmin, xmax] list of arrays (of same length) or scalars
		If xmin or xmax are None, limits of X axis will be used
		(default: [])
	:param hline_args:
		dict, containing keyword arguments understood by :func:`pylab.hlines`
		(e.g., 'colors', 'linestyles', 'linewidth', 'label')
		(default: {})
	:param vlines:
		[x, ymin, ymax] list of arrays (of same length) or scalars
		If ymin or ymax are None, limints of Y axis will be used
		(default: [])
	:param vline_args:
		dict, containing keyword arguments understood by :func:`pylab.vlines`
		(e.g., 'colors', 'linestyles', 'linewidth', 'label')
		(default: {})
	:param legend_location:
		int or str, location of legend (matplotlib location code):
			"best" 	0
			"upper right" 	1
			"upper left" 	2
			"lower left" 	3
			"lower right" 	4
			"right" 		5
			"center left" 	6
			"center right" 	7
			"lower center" 	8
			"upper center" 	9
			"center" 		10
		(default: 0)
	:param legend_fontsize:
		int or str, font size to use for legend labels
		If not specified, will use the value of :param:`tick_label_fontsize`
		(default: 'medium')
	:param style_sheet:
		str, matplotlib style sheet to apply to plot
		See matplotlib.style.available for availabel style sheets
		(default: 'classic')
	:param border_width:
		float, width of border around plot frame in cm
		If None, white space will not be removed
		(default: 0.2)
	:param fig_filespec:
		str, full path to output file
		If None, will plot on screen
		If 'wait', plotting is deferred
		(default: None)
	:param figsize:
		(width, height) tuple of floats, plot size in inches,
		only applies if :param:`ax` is None
		(default: None)
	:param dpi:
		int, resolution of plot,
		only applies if :param:`fig_filespec` is set to output file
		(default: 300)
	:param ax:
		matplotlib Axes instance, in which plot will be drawn
		If specified, :param:`fig_filespec` will be overridden with 'wait'
		(default: None, will generate new Axes instance)

	:return:
		matplotlib Axes instance if :param:`fig_filespec` is set to
		'wait', else None
	"""
	frame_args = {key: val for (key, val) in locals().items()
				if not key in ['datasets', 'colors', 'fill_colors', 'linewidths',
							'linestyles', 'labels', 'markers', 'marker_sizes',
							'marker_intervals', 'marker_edge_colors',
							'marker_fill_colors', 'marker_edge_widths',
							'marker_labels', 'marker_label_fontsize',
							'legend_location', 'legend_fontsize', 'style_sheet',
							'border_width', 'fig_filespec', 'figsize', 'dpi', 'ax']}

	from itertools import cycle

	pylab.style.use(style_sheet)

	fig = None
	if ax is None:
		#ax = pylab.axes()
		fig, ax = pylab.subplots(figsize=figsize)

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
	unique_labels = set(labels)

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
		assert len(x) == len(y)
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

		if isinstance(x[0], datetime.datetime):
			## Doesn't seem to be necessary
			#x = pylab.date2num(x)
			x_is_date = True
		else:
			x_is_date = False
		if isinstance(y[0], datetime.datetime):
			#y = pylab.date2num(y)
			y_is_date = True
		else:
			y_is_date = False

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
			#ax.scatter(x, y, s=symbol_size, edgecolors=edge_color, label=label,
			#	marker=symbol, facecolors=fill_color, linewidth=edge_width)

		for i, lbl in enumerate(marker_labels):
			ax.annotate(lbl, (x[i], y[i]), fontsize=marker_label_fontsize,
							clip_on=True)

	## Frame
	plot_ax_frame(ax, x_is_date=x_is_date, y_is_date=y_is_date, **frame_args)

	## Legend
	legend_fontsize = legend_fontsize or tick_label_fontsize
	legend_font = FontProperties(size=legend_fontsize)
	## Avoid warning if there are no labeled curves
	if len(unique_labels.difference(set(['_nolegend_']))):
		ax.legend(loc=legend_location, prop=legend_font)

	#if fig and tight_layout:
	#	fig.tight_layout(pad=0)

	## Output
	if fig_filespec == "wait":
		return ax
	elif fig_filespec:
		kwargs = {}
		if border_width is not None:
			kwargs = dict(bbox_inches="tight", pad_inches=border_width/2.54)
		fig.savefig(fig_filespec, dpi=dpi, **kwargs)
		pylab.clf()
	else:
		pylab.show()
		return ax

	## Restore default style if we get here
	pylab.style.use('default')


def plot_density(x, y, grid_size, density_type='hist2d', min_cnt=None, max_cnt=None,
			bins=None, cmap='plasma', cbar_args={}, cbar_label='N',
			xscaling='lin', yscaling='lin',
			xmin=None, xmax=None, ymin=None, ymax=None,
			xlabel='', ylabel='', ax_label_fontsize='large',
			xticks=None, xticklabels=None, xtick_interval=None, xtick_rotation=0,
			xtick_direction='', xtick_side='',
			yticks=None, yticklabels=None, ytick_interval=None, ytick_rotation=0,
			tick_label_fontsize='medium', tick_params={},
			ytick_direction='', ytick_side='',
			xgrid=0, ygrid=0,
			hlines=[], hline_args={}, vlines=[], vline_args={},
			title='', title_fontsize='large',
			style_sheet='classic', border_width=0.2,
			fig_filespec=None, figsize=None, dpi=300, ax=None):
	"""
	Plot XY data as density (number of data points per grid cell)

	:param x:
		1-D array, X data
	:param y:
		1-D array, Y data
	:param grid_size:
		int or (int, int) tuple, the number of grid cells in the X/Y
		direction
	:param density_type:
		str, type of density plot: 'hist2d', 'hexbin' or 'kde'
		(default: 'hist2d')
	:param min_cnt:
		int, minimum density to plot
		(default: None)
	:param max_cnt:
		int, maximum density to plot
		(default: None)
	:param bins:
		None, 'log' or list or array with bin edges, density bins
		(default: None)
	:param cmap:
		str or matplotlib Colormap object, colormap
		(default: 'plasma')
	:param cbar_args:
		dict, arguments to pass to :func:`matplotlib.colorbar`
	:param cbar_label:
		str, colorbar label
		(default: 'N')

	See :func:`plot_xy` for additional keyword arguments
	"""
	frame_args = {key: val for (key, val) in locals().items()
				if not key in ['x', 'y', 'grid_size', 'density_type',
							'min_cnt', 'max_cnt', 'cmap', 'bins', 'cbar_args',
							'cbar_label', 'style_sheet', 'border_width',
							'fig_filespec', 'figsize', 'dpi', 'ax']}

	pylab.style.use(style_sheet)

	if ax is None:
		#ax = pylab.axes()
		fig, ax = pylab.subplots(figsize=figsize)
	else:
		fig = None
		fig_filespec = "wait"

	## Density plot
	if isinstance(grid_size, int):
		grid_size = (grid_size, grid_size)

	if cmap is None:
		cmap = pylab.rcParams['image.cmap']
	if not isinstance(cmap, matplotlib.colors.Colormap):
		cmap = pylab.cm.get_cmap(cmap)
	cmap.set_bad((1,1,1,0))
	cmap.set_under((1,1,1,0))

	nan_idxs = np.isnan(x) | np.isnan(y)
	x, y = x[~nan_idxs], y[~nan_idxs]

	_xmin = xmin if xmin is not None else x.min()
	_xmax = xmax if xmax is not None else x.max()
	_ymin = ymin if ymin is not None else y.min()
	_ymax = ymax if ymax is not None else y.max()

	if density_type == 'hist2d':
		range = [[_xmin, _xmax], [_ymin, _ymax]]
		if bins is None:
			#norm = None
			norm = matplotlib.colors.Normalize(vmin=min_cnt, vmax=max_cnt)
		elif bins == 'log':
			norm = matplotlib.colors.LogNorm()
		else:
			from mapping.layeredbasemap.cm.norm import PiecewiseLinearNorm
			norm = PiecewiseLinearNorm(bins)
		_, _, _, sm = ax.hist2d(x, y, bins=grid_size, range=range, cmap=cmap,
								cmin=min_cnt, cmax=max_cnt, norm=norm)

	elif density_type == 'hexbin':
		extent = (_xmin, _xmax, _ymin, _ymax)
		sm = ax.hexbin(x, y, gridsize=grid_size, cmap=cmap, bins=bins,
						mincnt=min_cnt, extent=extent)

	elif density_type == 'kde':
		from scipy.stats import kde
		k = kde.gaussian_kde([x, y])
		xi, yi = np.mgrid[_xmin: _xmax: grid_size[0]*1j, _ymin: _ymax: grid_size[1]*1j]
		zi = k(np.vstack([xi.flatten(), yi.flatten()]))
		## Un-normalize density
		zi *= (float(len(x)) / np.sum(zi))
		extent = (_xmin, _xmax, _ymin, _ymax)
		if bins is None:
			#norm = None
			norm = matplotlib.colors.Normalize(vmin=min_cnt, vmax=max_cnt)
		elif bins == 'log':
			norm = matplotlib.colors.LogNorm()
		else:
			from mapping.layeredbasemap.cm.norm import PiecewiseLinearNorm
			norm = PiecewiseLinearNorm(bins)
		sm = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=cmap, norm=norm)
		ax.axis(extent)

	## Frame
	if isinstance(x[0], datetime.datetime):
		x_is_date = True
	else:
		x_is_date = False
	if isinstance(y[0], datetime.datetime):
		y_is_date = True
	else:
		y_is_date = False
	plot_ax_frame(ax, x_is_date=x_is_date, y_is_date=y_is_date, **frame_args)

	## Colorbar
	cbar = pylab.colorbar(sm, ax=ax, **cbar_args)
	cbar.set_label(cbar_label)

	## Output
	if fig_filespec == "wait":
		return ax
	elif fig_filespec:
		kwargs = {}
		if border_width is not None:
			kwargs = dict(bbox_inches="tight", pad_inches=border_width/2.54)
		fig.savefig(fig_filespec, dpi=dpi, **kwargs)
		pylab.clf()
	else:
		pylab.show()
		return ax

	## Restore default style if we get here
	pylab.style.use('default')


def plot_histogram(datasets, bins, data_is_binned=False,
				histogram_type='bar', cumulative=False, stacked=True, normed=False,
				orientation='vertical', align='mid', bar_width=None, baseline=None,
				colors=[], labels=[],
				line_color='k', line_width=0.5,
				xscaling='lin', yscaling='lin',
				xmin=None, xmax=None, ymin=None, ymax=None,
				xlabel='', ylabel='', ax_label_fontsize='large',
				xticks=None, xticklabels=None, xtick_interval=None, xtick_rotation=0,
				xtick_direction='', xtick_side='',
				yticks=None, yticklabels=None, ytick_interval=None, ytick_rotation=0,
				ytick_direction='', ytick_side='',
				tick_label_fontsize='medium', tick_params={},
				title='', title_fontsize='large',
				xgrid=0, ygrid=0,
				hlines=[], hline_args={}, vlines=[], vline_args={},
				legend_location=0, legend_fontsize='medium',
				style_sheet='classic', border_width=0.2,
				fig_filespec=None, figsize=None, dpi=300, ax=None):
	"""
	Plot histograms

	:param datasets:
		list of 1-D arrays
	:param bins:
		int (number of bins) or list or array (bin edges)
	:param data_is_binned:
		bool, whether or not data in :param:`datasets` is already binned
		Note that, if this is True, :param:`bins` must correspond to
		the bin edges!
		(default: False)
	"""
	frame_args = {key: val for (key, val) in locals().items()
				if not key in ['datasets', 'bins', 'data_is_binned', 'histogram_type',
							'cumulative', 'stacked', 'normed', 'orientation',
							'align', 'bar_width', 'baseline', 'colors', 'labels',
							'line_color', 'line_width',
							'legend_location', 'legend_fontsize', 'style_sheet',
							'border_width', 'fig_filespec', 'figsize', 'dpi', 'ax']}

	from itertools import cycle

	pylab.style.use(style_sheet)

	if ax is None:
		fig, ax = pylab.subplots(figsize=figsize)
	else:
		fig = None
		fig_filespec = "wait"

	## markers, colors, linewidhts, linestyles, labels, etc.
	if not colors:
		#colors = 'bgrcmyk'
		colors = pylab.rcParams['axes.prop_cycle'].by_key()['color']
	if not labels:
		labels = ['%d' % i for i in range(len(datasets))]
	unique_labels = set(labels)

	colors = cycle(colors)
	labels = cycle(labels)

	colors = [colors.next() for i in range(len(datasets))]
	labels = [labels.next() for i in range(len(datasets))]

	## Histogram
	if orientation == 'vertical' and 'log' in yscaling:
		log = True
	elif orientation == 'horizontal' and 'log' in xscaling:
		log = True
	else:
		log = False

	if data_is_binned:
		#The weights are the y-values of the input binned data
		weights = datasets
		#The dataset values are the bin centres
		bins = np.asarray(bins)
		datasets = [((bins[1:] + bins[:-1]) / 2.) for i in range(len(datasets))]
	else:
		weights = None

	ax.hist(datasets, bins, normed=normed, cumulative=cumulative,
			histtype=histogram_type, align=align, orientation=orientation,
			rwidth=bar_width, color=colors, label=labels, stacked=stacked,
			edgecolor=line_color, linewidth=line_width, bottom=baseline,
			log=log, weights=weights)

	## Frame
	plot_ax_frame(ax, **frame_args)

	## Legend
	legend_fontsize = legend_fontsize or tick_label_fontsize
	legend_font = FontProperties(size=legend_fontsize)
	## Avoid warning if there are no labeled curves
	if len(unique_labels.difference(set(['_nolegend_']))):
		ax.legend(loc=legend_location, prop=legend_font)

	## Output
	if fig_filespec == "wait":
		return ax
	elif fig_filespec:
		kwargs = {}
		if border_width is not None:
			kwargs = dict(bbox_inches="tight", pad_inches=border_width/2.54)
		pylab.savefig(fig_filespec, dpi=dpi, **kwargs)
		pylab.clf()
	else:
		pylab.show()
		return ax

	## Restore default style if we get here
	pylab.style.use('default')


def grid_center_to_edge_coordinates(Xc, Yc):
	"""
	Transform grid (or mesh) center coordinates to edge coordinates

	:param Xc:
		2D array (num_lats x num_lons), X center coordinates
	:param Yc:
		2D array (num_lats x num_lons), Y center coordinates

	:return:
		(Xe, Ye)
		2D arrays (num_lats+1 x num_lons+1), X and Y edge coordinates
	"""
	assert Xc.shape == Yc.shape

	## Output dimension
	nx, ny = Xc.shape[1] + 1, Xc.shape[0] + 1

	## First pass: compute edge coordinates along respective axes
	_Xe, _Ye = np.zeros((ny-1, nx)), np.zeros((ny, nx-1))
	dxx, dyy = np.diff(Xc, axis=1), np.diff(Yc, axis=0)
	_Xe[:,1:-1] = Xc[:,:-1] + dxx / 2.
	_Xe[:,:1] = Xc[:,:1] - dxx[:,:1] / 2.
	_Xe[:,-1:] = Xc[:,-1:] + dxx[:,-1:] / 2.
	_Ye[1:-1] = Yc[:-1] + dyy / 2.
	_Ye[:1] = Yc[:1] - dyy[:1] / 2.
	_Ye[-1:] = Yc[-1:] + dyy[-1:] / 2.

	## Second pass: compute edge coordinates along opposite axes
	Xe, Ye = np.zeros((ny, nx)), np.zeros((ny, nx))
	dxy, dyx = np.diff(_Xe, axis=0), np.diff(_Ye, axis=1)
	Xe[1:-1] = _Xe[:-1] + dxy / 2.
	Xe[:1] = _Xe[:1] - dxy[:1] / 2.
	Xe[-1:] = _Xe[-1:] + dxy[-1:] / 2.
	Ye[:,1:-1] = _Ye[:,:-1] + dyx / 2.
	Ye[:,:1] = _Ye[:,:1] + dyx[:,:1] / 2.
	Ye[:,-1:] = _Ye[:,-1:] + dyx[:,-1:] / 2.

	return (Xe, Ye)


def grid_edge_to_center_coordinates(Xe, Ye):
	"""
	Transform grid (or mesh) edge coordinates to center coordinates

	:param Xe:
		2D array (num_lats x num_lons), X edge coordinates
	:param Ye:
		2D array (num_lats x num_lons), Y edge coordinates

	:return:
		(Xc, Yc)
		2D arrays (num_lats-1 x num_lons-1), X and Y center coordinates
	"""
	## Output dimension
	nx, ny = Xe.shape[1] - 1, Xe.shape[0] - 1

	## First pass: compute center coordinates along respective axes
	_Xc, _Yc = np.zeros((ny+1, nx)), np.zeros((ny, nx+1))
	dxx, dyy = np.diff(Xe, axis=1), np.diff(Ye, axis=0)
	_Xc = Xe[:,:-1] + dxx / 2.
	_Yc = Ye[:-1] + dyy / 2.

	## Second pass: compute center coordinates along opposite axes
	dxy, dyx = np.diff(_Xc, axis=0), np.diff(_Yc, axis=1)
	Xc = _Xc[:-1] + dxy / 2.
	Yc = _Yc[:,:-1] + dyx / 2.

	return (Xc, Yc)


def plot_grid(data, X=None, Y=None,
			cmap='jet', norm=None, vmin=None, vmax=None,
			color_gradient='cont', shading=False, smoothed=False,
			colorbar=True, cax=None, cax_size=0.1, cax_padding=0.1, cax_shrink=1.,
			cbar_length=1., cbar_aspect=20, cbar_location='bottom center',
			cbar_spacing='uniform', cbar_ticks=None, cbar_label_format=None,
			cbar_title='', cbar_extend='neither', cbar_lines=False,
			contour_lines=None, contour_color='k', contour_width=0.5,
			contour_style='-', contour_labels=None, alpha=1,
			xscaling='lin', yscaling='lin',
			xmin=None, xmax=None, ymin=None, ymax=None,
			xlabel='', ylabel='', ax_label_fontsize='large',
			xticks=None, xticklabels=None, xtick_interval=None, xtick_rotation=0,
			xtick_direction='', xtick_side='',
			yticks=None, yticklabels=None, ytick_interval=None, ytick_rotation=0,
			ytick_direction='', ytick_side='',
			tick_label_fontsize='medium', tick_params={},
			title='', title_fontsize='large',
			xgrid=0, ygrid=0,
			hlines=[], hline_args={}, vlines=[], vline_args={},
			style_sheet='classic', border_width=0.2,
			fig_filespec=None, figsize=None, dpi=300, ax=None):
	"""
	Plot raster or mesh data

	:param data:
		2D array, gridded data
	:param X/Y:
		[x/ymin, x/ymax] or 1D array or 2D array or None, X/Y coodinates
		dimension may be either the same as data (= center coordinates)
		or 1 larger (= edge coordinates)
		(default: None, will just plot rectangular grid)
	:param cmap:
		str or instance of :class:`matplotlib.colors.Colormap, color palette
		(default: 'jet')
	:param norm:
		instance of :class:`matplotlib.colors.Normalize`, defining how
		to scale data values to the [0 - 1] interval, and hence to colors
		(default: None, uses default linear scaling)
	:param vmin:
	:param vmax:
		float, min/max data values to be mapped to 0/1, respectively,
		overriding vmin/vmax values of :param:`norm` if specified
		(default: None)
	:param color_gradient:
		str, if colors should be 'cont[inuous]' or 'disc[rete]'.
		Note that this mainly depends on the normalization, and can
		only be honoured in certain cases
		(default: 'cont')
	:param shading;
		bool, whether or not Gouraud shading should be applied to each
		cell or quad. Only applies if :param:`smoothed` is False
		(default: False)
	:param smoothed:
		bool, whether or not grid cells should be smoothed; this is
		accomplished using matplotlib's contourf function instead of
		pcolor(mesh)
		(default: False)
	:param colorbar:
		bool, whether or not to plot color bar
		(default: True)
	:param cax:
		matplotlib Axes instance to be used for the colorbar
		(default: None, will steal place from parent Axes instance
		given in :param:`ax`)
	:param cax_size:
		float, fraction of original Axes to use for colorbar Axes.
		Ignored if :param:`cax` is not None.
		(default: 0.10)
	:param cax_padding:
		float, fraction between colorbar and original Axes
		Ignored if :param:`cax` is not None.
		(default: 0.10)
	:param cax_shrink:
		float, fraction by which to shrink cax
		Ignored if :param:`cax` is not None.
		(default: 1.)
	:param cbar_length:
		float, length of colorbar as fraction of Axes width or height
		Ignored if :param:`cax` is not None.
		(default: 1.)
	:param cbar_aspect:
		float, aspect ratio (long/short dimension) of colorbar
		Ignored if :param:`cax` is not None.
		(default: 20)
	:param cbar_location:
		str, location (side of parent axes) and alignment of colorbar,
		location: 'left' / 'right' (vertical), 'top' / 'bottom' (horizontal)
		alignment: 'center' or 'left' / 'right' (if orientation is horizontal)
		or 'top' / 'bottom' (if orientation is vertical)
		Ignored if :param:`cax` is not None.
		(default: 'bottom center')
	:param cbar_spacing:
		str, either 'uniform' (each discrete color gets the same space)
		or 'proportional' (space proportional to represented data interval)
		(default: 'uniform')
	:param cbar_ticks:
		list or array, tick positions for colorbar
		(default: None, will position ticks automatically)
	:param cbar_label_format:
		str or instance of :class:`matplotlib.ticker.Formatter`, format for
		colorbar tick labels (e.g., '%.2f')
		(default: None)
	:param cbar_title:
		str, title for colorbar
		(default: '')
	:param cbar_extend:
		str, if and how colorbar should be extended with triangular
		ends for out-of-range values, one of 'neither', 'both',
		'min' or 'max'
		(default: 'neither')
	:param cbar_lines:
		bool, whether or not lines should be drawn at color boundaries
		in colorbar
		(default: False)
	:param contour_lines:
		int, list or array, values of contour lines to be drawn on top of grid:
		- 'None' or 0 = no contours
		- N (int): number of contours
		- list or array specifying contour values
		(default: None)
	:param contour_color:
		matplotlib color specification (or list), color to use for contour
		lines (default: 'k')
	:param contour_width:
		float or list, line width(s) of contour lines
		(default: 0.5)
	:param contour_style:
		str or list, line style(s) of contour lines: '-', '--', ':' or '-:'
		(default: '-')
	:param contour_labels:
		list, labels for contour lines. If None, use the contour
		line values; if empty list, no labels will be drawn
		(default: None)
	:param alpha:
		float in the range 0 - 1, grid opacity
		(default: 1)

	See :func:`plot_xy` for additional keyword arguments
	"""
	frame_args = {key: val for (key, val) in locals().items()
				if not key in ['data', 'X', 'Y', 'cmap', 'norm', 'vmin', 'vmax',
							'color_gradient', 'shading', 'smoothed',
							'colorbar', 'cax', 'cax_size', 'cax_padding',
							'cax_shrink', 'cbar_length', 'cbar_aspect',
							'cbar_location', 'cbar_spacing',
							'cbar_ticks', 'cbar_label_format', 'cbar_title',
							'cbar_extend', 'cbar_lines', 'contour_lines',
							'contour_color', 'contour_width', 'contour_style',
							'contour_labels', 'alpha', 'style_sheet', 'border_width',
							'fig_filespec', 'figsize', 'dpi', 'ax', 'kwargs']}

	from mpl_toolkits.axes_grid1.inset_locator import inset_axes
	from matplotlib.colors import BoundaryNorm
	from matplotlib.colorbar import make_axes
	from mapping.layeredbasemap.cm.norm import (PiecewiseLinearNorm,
												PiecewiseConstantNorm)

	pylab.style.use(style_sheet)

	fig = None
	if ax is None:
		fig, ax = pylab.subplots(figsize=figsize)

	## Determine if we need center or edge coordinates or both
	need_center_coordinates = False
	need_edge_coordinates = False
	if smoothed or shading or contour_lines not in (None, 0):
		need_center_coordinates = True
	if not smoothed:
		need_edge_coordinates = True

	## Construct X/Y arrays
	if X is not None and Y is not None:
		if len(X) == len(Y) == 2:
			## X/Y specified as x/ymin / x/ymax
			nx, ny = data.shape[1], data.shape[0]
			if need_edge_coordinates:
				nx, ny = nx + 1, ny + 1
			X = np.linspace(X[0], X[1], nx)
			Y = np.linspace(Y[0], Y[1], ny)
		if len(X.shape) == len(Y.shape) == 1:
			## X/Y are 1D arrays
			X, Y = np.meshgrid(X, Y)

		if X.shape == data.shape:
			## Center coordinates
			Xc, Yc = X, Y
			if need_edge_coordinates:
				print("Transforming center to edge coordinates!")
				Xe, Ye = grid_center_to_edge_coordinates(Xc, Yc)
			else:
				Xe, Ye = None, None
		elif X.shape[0] == data.shape[0] + 1:
			## Edge coordinates
			Xe, Ye = X, Y
			if need_center_coordinates:
				print("Transforming edge to center coordinates!")
				Xc, Yc = grid_edge_to_center_coordinates(Xe, Ye)
			else:
				Xc, Yc = None, None
		else:
			raise Exception('Dimensions of data and coordinates do not match!')

	## Mask NaN values
	data = np.ma.masked_array(data, mask=np.isnan(data))

	## Try to convert to piecewise constant norm or limit the number of colors
	## in the color palette if color_gradient is 'discontinuous'
	if color_gradient[:4] == 'disc':
		if isinstance(norm, PiecewiseLinearNorm):
			norm = norm.to_piecewise_constant_norm()
		elif not isinstance(norm, (PiecewiseConstantNorm, BoundaryNorm)):
			print('Warning: need constant norm to plot discrete colors')
			## Alternatively, we can try limiting the number of colors in the palette
			if not isinstance(cmap, matplotlib.colors.Colormap):
				cmap = matplotlib.cm.get_cmap(cmap, 10)

	## Plot grid
	cs = None
	common_kwargs = {'cmap': cmap, 'norm': norm, 'vmin': vmin, 'vmax': vmax,
					'alpha': alpha}

	if smoothed:
		## data must have same size as X and Y for contourf
		if color_gradient[:4] == 'disc':
			if X is None and Y is None:
				cs = ax.contourf(data, **common_kwargs)
			else:
				cs = ax.contourf(Xc, Yc, data, **common_kwargs)
		else:
			if X is None and Y is None:
				cs = ax.contourf(data, 256, **common_kwargs)
			else:
				cs = ax.contourf(Xc, Yc, data, 256, **common_kwargs)

	else:
		## both pcolor and pcolormesh need edge coordinates,
		## except if shading == 'gouraud'
		if X is None and Y is None:
			shading = {True: 'gouraud', False: 'flat'}[shading]
			cs = ax.pcolormesh(data, shading=shading, **common_kwargs)
			# or use imshow?

		else:
			if shading:
				cs = ax.pcolormesh(Xc, Yc, data, shading='gouraud', **common_kwargs)
			else:
				cs = ax.pcolormesh(Xe, Ye, data, shading='flat', **common_kwargs)

	## Contour lines
	if contour_lines:
		# X and Y must have same shape as data !
		cl = ax.contour(Xc, Yc, data, contour_lines, colors=contour_color,
					linewidths=contour_width, linestyles=contour_style)

		## Contour labels:
		if contour_labels is None:
			contour_labels = contour_lines
		if contour_labels:
			clabels = ax.clabel(cl, contour_labels, colors=contour_color, inline=True,
								fontsize=tick_label_fontsize, fmt=cbar_label_format)
		# TODO: white background for contour labels
		#bbox_args = label_style.to_kwargs()['bbox']
		#[txt.set_bbox(bbox_args) for txt in clabels]

	## Frame
	plot_ax_frame(ax, x_is_date=False, y_is_date=False, **frame_args)

	## Color bar
	if colorbar:
		cbar_align = 'center'
		if ' ' in cbar_location:
			cbar_location, cbar_align = cbar_location.split()

		if cbar_location in ("top", "bottom"):
			cbar_orientation = "horizontal"
		else:
			cbar_orientation = "vertical"

		if cax is None:
			#ax_pos = ax.get_position()
			#fig = ax.get_figure()
			#cax = fig.add_axes([ax_pos.x1+0.01, ax_pos.y0, 0.02, ax_pos.height])

			#from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
			#divider = make_axes_locatable(ax)
			#if cbar_orientation == 'vertical':
			#	size = axes_size.AxesY(ax, aspect=1./cbar_aspect)
			#else:
			#	size = axes_size.AxesX(ax, aspect=1./cbar_aspect)
			#pad = axes_size.Fraction(cbar_padding, size)
			#cax = divider.append_axes(cbar_location, size=size, pad=pad)

			cbar_aspect *= cbar_length
			cax, _ = make_axes(ax, location=cbar_location, fraction=cax_size,
							aspect=cbar_aspect, shrink=cax_shrink, pad=cax_padding)

			cax_pos = cax.get_position()
			left, bottom = cax_pos.x0, cax_pos.y0
			width, height = cax_pos.width, cax_pos.height
			#print(left, bottom, width, height)
			if cbar_orientation == 'vertical':
				unshrinked_height = height / cax_shrink
				center = bottom + height / 2
				height *= cbar_length
				if cbar_align == 'center':
					bottom = center - height / 2
				elif cbar_align == 'top':
					top = center + unshrinked_height / 2
					bottom = top - height
				elif cbar_align == 'bottom':
					bottom = center - unshrinked_height / 2
			elif cbar_orientation == 'horizontal':
				unshrinked_width = width / cax_shrink
				center = left + width / 2
				width *= cbar_length
				if cbar_align == 'center':
					left = center - width / 2
				elif cbar_align == 'right':
					right = center + unshrinked_width / 2
					left = right - width
				elif cbar_align == 'left':
					left = center - unshrinked_width / 2
			cax.set_position((left, bottom, width, height))

		elif isinstance(cax, tuple):
			## Test
			anchor = cax
			cax = None

		elif cax == 'inside':
			if cbar_orientation == 'horizontal':
				width = cbar_length
				height = width / cbar_aspect
				loc = cbar_location + ' ' + cbar_align
			else:
				height = cbar_length
				width = height / cbar_aspect
				loc = cbar_align + ' ' + cbar_location
			width = '%.0f%%' % (width * 100)
			height = '%.0f%%' % (height * 100)
			print(width, height)
			loc = loc.replace('top', 'upper').replace('bottom', 'lower')
			loc = {'upper right': 1,
					'upper left': 2,
					'lower left': 3,
					'lower right': 4,
					'right': 5,
					'center left': 6,
					'center right': 7,
					'lower center': 8,
					'upper center': 9,
					'center': 10}[loc]
			cax = inset_axes(ax, width=width, height=height, loc=loc,
							borderpad=cax_padding)

		if cax:
			cbar = pylab.colorbar(cs, cax=cax, orientation=cbar_orientation,
							spacing=cbar_spacing, ticks=cbar_ticks,
							format=cbar_label_format, extend=cbar_extend,
							drawedges=cbar_lines)

		if cbar_orientation == 'horizontal':
			cbar.set_label(cbar_title, size=ax_label_fontsize)
		else:
			cbar.ax.set_title(cbar_title, size=ax_label_fontsize)
		cbar.ax.tick_params(labelsize=tick_label_fontsize)

		# TODO: boundaries / values, cf. layeredbasemap ?


	## Output
	if fig_filespec == "wait":
		return ax
	elif fig_filespec:
		kwargs = {}
		if border_width is not None:
			kwargs = dict(bbox_inches="tight", pad_inches=border_width/2.54)
		fig.savefig(fig_filespec, dpi=dpi, **kwargs)
		pylab.clf()
	else:
		pylab.show()
		return ax

	## Restore default style if we get here
	pylab.style.use('default')


def plot_ax_frame(ax, x_is_date=False, y_is_date=False,
				xscaling='lin', yscaling='lin',
				xmin=None, xmax=None, ymin=None, ymax=None,
				xlabel='', ylabel='', ax_label_fontsize='large',
				xticks=None, xticklabels=None, xtick_interval=None, xtick_rotation=0,
				xtick_direction='', xtick_side='',
				yticks=None, yticklabels=None, ytick_interval=None, ytick_rotation=0,
				ytick_direction='', ytick_side='',
				tick_label_fontsize='medium', tick_params={},
				title='', title_fontsize='large',
				xgrid=0, ygrid=0,
				hlines=[], hline_args={}, vlines=[], vline_args={}):
	"""
	Plot ax frame

	:param ax:
		matplotlib Axes instance, in which frame will be drawn
	:param x_is_date:
		bool, whether or not X axis contains datetimes
		(default: False)
	:para y_is_date:
		bool, whether or not Y axis contains datetimes
		(default: False)

	:param xscaling:
	:param yscaling:
	:param xmin:
	:param xmax:
	:param ymin:
	:param ymax:
	:param xlabel:
	:param ylabel:
	:param ax_label_fontsize:
	:param xticks:
	:param xticklabels:
	:param xtick_interval:
	:param xtick_rotation:
	:param xtick_direction:
	:param xtick_side:
	:param yticks:
	:param yticklabels:
	:param ytick_interval:
	:param ytick_rotation:
	:param ytick_direction:
	:param ytick_side:
	:param tick_label_fontsize:
	:param tick_params:
	:param title:
	:param title_fontsize:
	:param xgrid:
	:param ygrid:
	:param hlines:
	:param hline_args:
	:param vlines:
	:param vline_args:
		see :func:`plot_xy`

	:return:
		None
	"""
	## Axis scaling
	if xscaling[0] == '-':
		xscaling = xscaling[1:]
		ax.invert_xaxis()
	xscaling = {'lin': 'linear', 'log': 'log'}[xscaling[:3]]
	ax.set_xscale(xscaling)
	if yscaling[0] == '-':
		yscaling = yscaling[1:]
		ax.invert_yaxis()
	yscaling = {'lin': 'linear', 'log': 'log'}[yscaling[:3]]
	ax.set_yscale(yscaling)

	## Axis labels
	if xlabel:
		ax.set_xlabel(xlabel, fontsize=ax_label_fontsize)
	if ylabel:
		ax.set_ylabel(ylabel, fontsize=ax_label_fontsize)

	## Axis limits
	_xmin, _xmax = ax.get_xlim()
	xmin = _xmin if xmin is None else xmin
	xmax = _xmax if xmax is None else xmax
	ax.set_xlim(xmin, xmax)

	_ymin, _ymax = ax.get_ylim()
	ymin = _ymin if ymin is None else ymin
	ymax = _ymax if ymax is None else ymax
	ax.set_ylim(ymin, ymax)

	## Horizontal / vertical lines
	if hlines:
		y, xmin, xmax = hlines
		_xmin, _xmax = ax.get_xlim()
		xmin = _xmin if xmin is None else xmin
		xmax = _xmax if xmax is None else xmax
		ax.hlines(y, xmin, xmax, **hline_args)

	if vlines:
		x, ymin, ymax = vlines
		_ymin, _ymax = ax.get_ylim()
		ymin = _ymin if ymin is None else ymin
		ymax = _ymax if ymax is None else ymax
		ax.vlines(x, ymin, ymax, **vline_args)

	## X ticks
	if xticks is not None:
		ax.set_xticks(xticks)
	#elif xtick_interval is not None:
	else:
		if isinstance(xtick_interval, tuple) and len(xtick_interval) == 2:
			major_tick_interval, minor_tick_interval = xtick_interval
		else:
			major_tick_interval, minor_tick_interval = xtick_interval, None

		if isinstance(major_tick_interval, matplotlib.ticker.Locator):
			major_loc = major_tick_interval
		elif x_is_date:
			major_loc = _create_date_locator(major_tick_interval)
		elif major_tick_interval:
			major_loc = matplotlib.ticker.MultipleLocator(major_tick_interval)
		elif major_tick_interval is None:
			if xscaling[:3] == 'log':
				major_loc = matplotlib.ticker.LogLocator()
			else:
				major_loc = matplotlib.ticker.AutoLocator()
		else:
			major_loc = matplotlib.ticker.NullLocator()
		ax.xaxis.set_major_locator(major_loc)
		if isinstance(major_loc, mpl_dates.DateLocator):
			if xticklabels is None:
				ax.xaxis.set_major_formatter(mpl_dates.AutoDateFormatter(locator=major_loc))

		if isinstance(minor_tick_interval, matplotlib.ticker.Locator):
			minor_loc = minor_tick_interval
		elif x_is_date:
			minor_loc = _create_date_locator(minor_tick_interval)
		elif minor_tick_interval:
			minor_loc = matplotlib.ticker.MultipleLocator(minor_tick_interval)
		elif minor_tick_interval is None:
			minor_loc = matplotlib.ticker.AutoLocator()
		else:
			minor_loc = matplotlib.ticker.NullLocator()
		ax.xaxis.set_minor_locator(minor_loc)
		## Note: no formatter for minor ticks, as we don't print them

	## X ticklabels
	if xscaling[:3] == 'log' and xticklabels is None:
		## Do not use log notation for small exponents
		if xmin > 1E-4 and xmax < 1E+4:
			xticklabels = matplotlib.ticker.FormatStrFormatter('%g')
	if isinstance(xticklabels, matplotlib.ticker.Formatter):
		ax.xaxis.set_major_formatter(xticklabels)
	elif isinstance(xticklabels, basestring):
		if xticklabels == '':
			major_formatter = matplotlib.ticker.NullFormatter()
		elif x_is_date:
			major_formatter = mpl_dates.DateFormatter(xticklabels)
		else:
			major_formatter = matplotlib.ticker.FormatStrFormatter(xticklabels)
		ax.xaxis.set_major_formatter(major_formatter)
	elif xticklabels is not None:
		ax.set_xticklabels(xticklabels)

	## Y ticks
	if yticks is not None:
		ax.set_yticks(yticks)
	#if ytick_interval is not None:
	else:
		if isinstance(ytick_interval, tuple) and len(ytick_interval) == 2:
			major_tick_interval, minor_tick_interval = ytick_interval
		else:
			major_tick_interval, minor_tick_interval = ytick_interval, None

		if isinstance(major_tick_interval, matplotlib.ticker.Locator):
			major_loc = major_tick_interval
		elif y_is_date:
			major_loc = _create_date_locator(major_tick_interval)
		elif major_tick_interval:
			major_loc = matplotlib.ticker.MultipleLocator(major_tick_interval)
		elif major_tick_interval is None:
			if yscaling[:3] == 'log':
				major_loc = matplotlib.ticker.LogLocator()
			else:
				major_loc = matplotlib.ticker.AutoLocator()
		else:
			major_loc = matplotlib.ticker.NullLocator()
		ax.yaxis.set_major_locator(major_loc)
		if isinstance(major_loc, mpl_dates.DateLocator):
			if yticklabels is None:
				ax.yaxis.set_major_formatter(mpl_dates.AutoDateFormatter(locator=major_loc))

		if isinstance(minor_tick_interval, matplotlib.ticker.Locator):
			minor_loc = minor_tick_interval
		elif y_is_date:
			minor_loc = _create_date_locator(minor_tick_interval)
		elif minor_tick_interval:
			minor_loc = matplotlib.ticker.MultipleLocator(minor_tick_interval)
		elif minor_tick_interval is None:
			minor_loc = matplotlib.ticker.AutoMinorLocator()
		else:
			minor_loc = matplotlib.ticker.NullLocator()
		ax.yaxis.set_minor_locator(minor_loc)
		## Note: no formatter for minor ticks, as we don't print them

	## Y tick labels
	if yscaling[:3] == 'log' and yticklabels is None:
		## Do not use log notation for small exponents
		if ymin > 1E-4 and ymax < 1E+4:
			yticklabels = matplotlib.ticker.FormatStrFormatter('%g')
	if isinstance(yticklabels, matplotlib.ticker.Formatter):
		ax.yaxis.set_major_formatter(yticklabels)
	elif isinstance(yticklabels, basestring):
		if yticklabels == '':
			major_formatter = matplotlib.ticker.NullFormatter()
		elif y_is_date:
			major_formatter = mpl_dates.DateFormatter(yticklabels)
		else:
			major_formatter = matplotlib.ticker.FormatStrFormatter(yticklabels)
		ax.yaxis.set_major_formatter(major_formatter)
	elif yticklabels is not None:
		ax.set_yticklabels(yticklabels)

	## Tick label size and rotation
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size(tick_label_fontsize)

	if xtick_rotation:
		for label in ax.get_xticklabels():
			label.set_horizontalalignment('right')
			label.set_rotation(xtick_rotation)

	if ytick_rotation:
		for label in ax.get_yticklabels():
			label.set_horizontalalignment('right')
			label.set_rotation(ytick_rotation)

	## Tick aspect
	if tick_params:
		ax.tick_params(axis='both', **tick_params)

	if xtick_direction:
		ax.tick_params(axis='x', direction=xtick_direction)

	if xtick_side:
		side_kwargs = {}
		if xtick_side in ('top', 'both'):
			side_kwargs['top'] = True
		if xtick_side in ('bottom', 'both'):
			side_kwargs['bottom'] = True
		ax.tick_params(axis='x', **side_kwargs)

	if ytick_direction:
		ax.tick_params(axis='y', direction=ytick_direction)

	if ytick_side:
		side_kwargs = {}
		if ytick_side in ('left', 'both'):
			side_kwargs['left'] = True
		if ytick_side in ('right', 'both'):
			side_kwargs['right'] = True
		ax.tick_params(axis='y', **side_kwargs)

	## Grid
	if xgrid:
		which = {1: 'major', 2: 'minor', 3: 'both'}[xgrid]
		ax.grid(True, which=which, axis='x')
	if ygrid:
		which = {1: 'major', 2: 'minor', 3: 'both'}[ygrid]
		ax.grid(True, which=which, axis='y')

	## Title
	if title:
		ax.set_title(title, fontsize=title_fontsize)
