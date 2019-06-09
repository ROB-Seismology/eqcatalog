"""
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int


import os

import numpy as np

import eqcatalog.time_functions_np as tf
from .plot_generic import plot_xy


def plot_cumulated_moment(catalogs, start_date=None, end_date=None, time_unit='Y',
						Mrelation="default", M0max=None, **kwargs):
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
		(default: "default")
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

plot_cumulated_moment.__doc__ += plot_xy.__doc__.split("\n", 4)[4]
