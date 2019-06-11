"""
Generic plot functions based on matplotlib
"""

from __future__ import absolute_import, division, print_function, unicode_literals

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str

import datetime

import pylab
import matplotlib
import matplotlib.ticker
import matplotlib.dates as mpl_dates
from matplotlib.font_manager import FontProperties


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
			xscaling='lin', yscaling='lin',
			xmin=None, xmax=None, ymin=None, ymax=None,
			xlabel='', ylabel='', ax_label_fontsize='large',
			xticks=None, xticklabels=None, xtick_interval=None, xtick_rotation=0,
			yticks=None, yticklabels=None, ytick_interval=None, ytick_rotation=0,
			tick_label_fontsize='medium',
			title='', title_fontsize='large',
			legend_location=0, legend_fontsize='medium',
			xgrid=0, ygrid=0, style_sheet='classic',
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
		(default: 'lin')
	:param yscaling:
		str, scaling to use for Y axis ('lin' or 'log')
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
	:param xgrid:
		int, 0/1/2/3 = draw no/major/minor/major+minor X grid lines
		(default: 0)
	:param ygrid:
		int, 0/1/2/3 = draw no/major/minor/major+minor Y grid lines
		(default: 0)
	:param style_sheet:
		str, matplotlib style sheet to apply to plot
		See matplotlib.style.available for availabel style sheets
		(default: 'classic')
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
	from itertools import cycle

	pylab.style.use(style_sheet)

	if ax is None:
		#ax = pylab.axes()
		fig, ax = pylab.subplots(figsize=figsize)
	else:
		fig_filespec = "wait"

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

	## Axis labels
	if xlabel:
		ax.set_xlabel(xlabel, fontsize=ax_label_fontsize)
	if ylabel:
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
	elif xtick_interval is not None:
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
			major_loc = matplotlib.ticker.AutoLocator()
		else:
			major_loc = matplotlib.ticker.NullLocator()
		ax.xaxis.set_major_locator(major_loc)
		if isinstance(major_loc, mpl_dates.DateLocator):
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

	if yticks is not None:
		ax.set_yticks(yticks)
	if ytick_interval is not None:
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
			major_loc = matplotlib.ticker.AutoLocator()
		else:
			major_loc = matplotlib.ticker.NullLocator()
		ax.yaxis.set_major_locator(major_loc)
		if isinstance(major_loc, mpl_dates.DateLocator):
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

	## Grid
	if xgrid:
		which = {1: 'major', 2: 'minor', 3: 'both'}[xgrid]
		ax.grid(True, which=which, axis='x')
	if ygrid:
		which = {1: 'major', 2: 'minor', 3: 'both'}[ygrid]
		ax.grid(True, which=which, axis='y')

	## Legend and title
	legend_fontsize = legend_fontsize or tick_label_fontsize
	legend_font = FontProperties(size=legend_fontsize)
	## Avoid warning if there are no labeled curves
	if len(unique_labels.difference(set(['_nolegend_']))):
		ax.legend(loc=legend_location, prop=legend_font)
	if title:
		ax.set_title(title, fontsize=title_fontsize)

	if fig_filespec == "wait":
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
		return ax

	pylab.style.use('default')
