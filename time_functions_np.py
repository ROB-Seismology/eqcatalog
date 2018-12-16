"""
Useful time functions based on numpy datetime64
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import datetime
import numpy as np


AVG_SECS_PER_YEAR = 31556925.9747
SECS_PER_MINUTE = 60
SECS_PER_HOUR = SECS_PER_MINUTE * 60
SECS_PER_DAY = SECS_PER_HOUR * 24
SECS_PER_WEEK = SECS_PER_DAY * 7

EPOCH = np.datetime64(0, 's')
ONE_SECOND = np.timedelta64(1, 's')


def as_np_datetime(dt, unit='s'):
	"""
	Convert to numpy datetime64

	:param dt:
		str (ISO 8601)
		or instance of :class:`datetime.datetime`
		or instance of :class:`datetime.date`
		or list with strings, instances of :class:`datetime.datetime`
			or instances of :class:`datetime.date`
		or instance of :class:`np.datetime64`
		or array of type datetime64

	:return:
		instance of :class:`np.datetime64` or array of type datetime64
	"""
	if isinstance(dt, (np.datetime64, np.ndarray)):
		return dt.astype('M8[%s]' % unit)
	elif isinstance(dt, (datetime.datetime, datetime.date, str)):
		return np.datetime64(dt, unit)
	elif isinstance(dt, list):
		return np.array(dt, dtype='M8[%s]' % unit)


def as_np_date(dt):
	"""
	Truncate datetime to date

	:param dt:
		datetime spec understood by :func:`as_np_datetime`

	:return:
		instance of :class:`np.datetime64` or array of type datetime64,
		date
	"""
	return as_np_datetime(dt, 'D')
	#if np.isscalar(dt):
	#	return np.datetime64(str(dt)[:10])
	#else:
	#	return np.array([str(el)[:10] for el in dt], dtype='M8[D]')


def to_py_datetime(dt64):
	"""
	Convert numpy datetime64 to datetime.datetime.
	Note that this will fail for datetimes where year > 9999.

	:param dt64:
		instance of :class:`numpy.datetime64` or array of instances

	:return:
		instance of :class:`datetime.datetime`
	"""
	#if isinstance(dt, datetime.datetime):
	#	return dt
	#elif isinstance(dt, (np.datetime64, np.ndarray)):
	return dt64.astype(object)


def to_py_time(dt64):
	"""
	Convert numpy datetime64 to datetime.time.
	This does not fail for datetimes where year > 9999.

	:param dt64:
		instance of :class:`numpy.datetime64` or array of instances

	:return:
		instance of :class:`datetime.time`
	"""
	start_of_day = as_np_datetime(dt64, 'D')
	secs = timespan(start_of_day, dt64, 'us') / 1E+6
	## Note: np.divmod only from numpy 1.15 onwards...
	hours = np.floor(secs / SECS_PER_HOUR).astype('int')
	secs = np.mod(secs, SECS_PER_HOUR)
	minutes = np.floor(secs / SECS_PER_MINUTE).astype('int')
	secs = np.mod(secs, SECS_PER_MINUTE)
	fracs, secs = np.modf(secs)
	secs = secs.astype('int')
	microsecs = (fracs * 1E+6).astype('int')
	if np.isscalar(dt):
		return datetime.time(hours, minutes, secs, microsecs)
	else:
		return [datetime.time(h, m, s, us)
				for (h, m, s, us) in zip(hours, minutes, secs, microsecs)]


def to_py_date(dt64):
	"""
	Convert numpy datetime64 to datetime.date.
	Note that this will fail for datetimes where year > 9999.

	:param dt64:
		instance of :class:`numpy.datetime64` or array of instances

	:return:
		instance of :class:`datetime.date`
	"""
	if np.isscalar(dt64):
		return to_py_datetime(dt64).date()
	else:
		return [to_py_datetime(item).date() for item in dt64]


def to_year(dt):
	"""
	Extract year from datetime object

	:param dt:
		datetime spec understood by :func:`as_np_datetime`

	:return:
		int or int array, year
	"""
	return as_np_datetime(dt, 'Y').astype('int64') + 1970


def to_fractional_year(dt):
	"""
	Compute fractional year of date

	:param dt:
		datetime spec understood by :func:`as_np_datetime`

	:return:
		float or float array, fractional year
	"""
	dt = as_np_datetime(dt, 's')
	years = to_year(dt)
	if np.isscalar(dt):
		num_days_per_year = days_in_year(years)
		start_of_year = np.datetime64('%s-01-01' % years, dtype='M8[D]')
	else:
		num_days_per_year = np.array([days_in_year(yr) for yr in years])
		start_of_year = np.array(['%s-01-01' % yr for yr in years], dtype='M8[D]')
	elapsed_days = timespan(start_of_year, dt, 'D')
	fraction = (elapsed_days / num_days_per_year)
	return years + fraction


def to_ymd_tuple(dt):
	"""
	Extract year, month and day from datetime object

	:param dt:
		datetime spec understood by :func:`as_np_datetime`

	:return:
		(year, month, day) tuple of ints or int arrays
	"""
	years = to_year(dt)
	months = as_np_datetime(dt, 'M').astype('int64') % 12 + 1
	days = (as_np_datetime(dt, 'D') - as_np_datetime(dt, 'M')).astype('int64') + 1
	return (years, months, days)


def to_time_tuple(dt):
	"""
	Convert datetime object to (year, month, day, hour, minute, second)
	tuple

	:param dt:
		datetime spec understood by :func:`as_np_datetime`

	:return:
		(year, month, day, hour, minute, second) tuple of ints
		or list of tuples
	"""
	if isinstance(dt, (datetime.datetime)):
		return (dt.year, dt.month, dt.day, dt.hour, dt.minute,
				dt.second + dt.microsecond / 1E+6)
	else:
		dt = as_np_datetime(dt, 's')
		if np.isscalar(dt):
			date, time = str(dt).split('T')
			#time, tz_shift = time.split('+')
			year, month, day = map(int, date.split('-'))
			hour, minute, second = time.split(':')
			hour, minute = int(hour), int(minute)
			second = float(second)
			return (year, month, day, hour, minute, second)
		else:
			time_tuples = []
			for item in dt:
				date, time = str(item).split('T')
				year, month, day = map(int, date.split('-'))
				hour, minute, second = time.split(':')
				hour, minute = int(hour), int(minute)
				second = float(second)
				time_tuples.append((year, month, day, hour, minute, second))
			return time_tuples


def timespan(start_dt, end_dt, unit='Y'):
	"""
	Return total time span in given unit between start date and end date.

	:param start_date:
		datetime spec understood by :func:`as_np_datetime`, start date
	:param end_date:
		datetime spec understood by :func:`as_np_datetime`, end date
	:param unit:
		str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
		(year|week|day|hour|minute|second|millisecond|microsecond)
		(default: 'Y')

	:return:
		float, time span in fractions of :param:`unit`
	"""
	assert unit in ['Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us']
	if unit == 'us':
		resolution = 'ps'
	elif unit == 'ms':
		resolution == 'us'
	elif unit == 's':
		resolution = 'ms'
	else:
		resolution = 's'
	start_dt = as_np_datetime(start_dt, resolution)
	end_dt = as_np_datetime(end_dt, resolution)
	time_delta = (end_dt - start_dt)
	return fractional_time_delta(time_delta, unit=unit)


def fractional_time_delta(td64, unit):
	"""
	Convert numpy timedelta64 to a fraction of the specified unit

	:param td64:
		instance of :class:`numpy.timdelta64`
	:param unit:
		str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
		(year|week|day|hour|minute|second|millisecond|microsecond)

	:return:
		float, time span in fractions of :param:`unit`
	"""
	if unit == 'us':
		resolution = 'ps'
	elif unit == 'ms':
		resolution == 'us'
	elif unit == 's':
		resolution = 'ms'
	else:
		resolution = 's'
	td64 = td64.astype('m8[%s]' % resolution).astype('float64')
	divisor = {'Y': AVG_SECS_PER_YEAR,
				'W': SECS_PER_WEEK,
				'D': SECS_PER_DAY,
				'h': SECS_PER_HOUR,
				'm': SECS_PER_MINUTE,
				's': 1.,
				'ms': 1.,
				'us': 1.}[unit]
	return (td64 / divisor)


def days_in_year(year):
	"""
	Return number of days in given year

	:param year:
		int, year

	:return:
		int, number of days
	"""
	start_of_year = np.datetime64('%s-01-01' % year, 'D')
	start_of_next_year = np.datetime64('%s-01-01' % (year + 1), 'D')
	num_days = (start_of_next_year - start_of_year).astype('m8[D]').astype('int')
	return num_days


def seconds_since_epoch(dt):
	"""
	Return time in seconds since epoch

	:param dt:
		datetime spec understood by :func:`as_np_datetime`

	:return:
		int or int array, time in seconds
	"""
	dt = as_np_datetime(dt, 's')
	time_delta = (dt - EPOCH) / ONE_SECOND
	return time_delta.astype('int64')
