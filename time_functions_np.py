"""
Useful time functions based on numpy datetime64
"""

from __future__ import absolute_import, division, print_function, unicode_literals

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str


import datetime
import numpy as np

from distutils.version import LooseVersion
assert LooseVersion(np.__version__) >= LooseVersion('1.7.0')
if LooseVersion(np.__version__) <= LooseVersion('1.11.0'):
	print("Warning: this version of numpy applies timezone offset in datetime64")
	print("Please upgrade to version 1.11.0 or higher")


AVG_SECS_PER_YEAR = 31556925.9747
SECS_PER_MINUTE = 60
SECS_PER_HOUR = SECS_PER_MINUTE * 60
SECS_PER_DAY = SECS_PER_HOUR * 24
SECS_PER_WEEK = SECS_PER_DAY * 7

EPOCH = np.datetime64(0, 's')
#ONE_SECOND = np.timedelta64(1, 's')


def is_np_datetime(dt):
	"""
	Check if argument is a numpy datetime64 object

	:return:
		bool
	"""
	if isinstance(dt, np.datetime64):
		return True
	elif isinstance(dt, np.ndarray) and np.issubdtype(dt.dtype, np.datetime64):
		return True
	else:
		return False


def is_np_timedelta(td):
	"""
	Check if argument is a numpy timedelta64 object

	:return:
		bool
	"""
	if isinstance(td, np.timedelta64):
		return True
	elif isinstance(td, np.ndarray) and np.issubdtype(td.dtype, np.timedelta64):
		return True
	else:
		return False


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
	:param unit:
		str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
		(year|week|day|hour|minute|second|millisecond|microsecond)
		(default: 's')

	:return:
		instance of :class:`np.datetime64` or array of type datetime64
	"""
	if is_np_datetime(dt):
		return dt.astype('M8[%s]' % unit)
	elif isinstance(dt, (datetime.datetime, datetime.date, basestring)):
		return np.datetime64(dt, unit)
	elif isinstance(dt, list):
		return np.array(dt, dtype='M8[%s]' % unit)


def as_np_timedelta(td):
	"""
	Convert to numpy timedelta64

	:param td:
		instance of :class:`datetime.timedelta` or list with such instances
		or instance of :class:`np.timedelta64` or array of type timedelta64

	:return:
		instance of :class:`np.timedelta64` or array of type timedelta64
	"""
	if is_np_timedelta(td):
		return td
	elif isinstance(td, datetime.timedelta):
		return np.timedelta64(td)
	elif isinstance(td, list):
		return np.array(td, dtype=np.timedelta64)


def get_datetime_unit(dt64):
	"""
	Extract time unit from numpy datetime64 or timedelta64 object

	:param dt64:
		instance of :class:`np.datetime64` or :class:`np.timedelta64`
		or array of such types

	:return:
		str, time unit
	"""
	assert is_np_datetime(dt64) or is_np_timedelta(dt64)
	return str(dt64.dtype).split('[')[1].split(']')[0]


def time_tuple_to_np_datetime(year, month=0, day=0, hour=0, minute=0, second=0,
							microsecond=0):
	"""
	Construct numpy datetime from time tuple

	:param year:
		int, year
	:param month:
		int, month
		(default: 0)
	:param day:
		int, day
		(default: 0)
	:param hour:
		int, hour
		(default: 0)
	:param minute:
		int, minute
		(default: 0)
	:param second:
		int or float, second
		(default: 0)
	:param microsecond:
		int, microsecond
		(default: 0)

	:return:
		instance of :class:`np.datetime64`
	"""
	dt_string = '%d' % year
	if month:
		dt_string += '-%02d' % month
		if day:
			dt_string += "-%02d" % day
			if hour:
				dt_string += 'T%02d' % hour
				if minute:
					dt_string += ':%02d' % minute
					if second:
						if microsecond:
							second += microsecond * 1E-6
						dt_string += ':%09.6f' % second
	return np.datetime64(dt_string)


def utcnow(unit='s'):
	"""
	Return numpy datetime representing current UTC date and time

	:return:
		instance of :class:`np.datetime64`
	"""
	return np.datetime64('now', unit)


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


def to_py_datetime(dt64):
	"""
	Convert numpy datetime64 to datetime.datetime.
	Note that this will fail for datetimes where year > 9999.

	:param dt64:
		instance of :class:`numpy.datetime64` or array of such instances

	:return:
		instance of :class:`datetime.datetime` or list of such instances
	"""
	#if isinstance(dt, datetime.datetime):
	#	return dt
	#elif is_np_datetime(dt):
	assert is_np_datetime(dt64)
	#return dt64.astype(object)
	return dt64.tolist()


def to_py_time(dt64):
	"""
	Convert numpy datetime64 to datetime.time.
	This does not fail for datetimes where year > 9999.

	:param dt64:
		instance of :class:`numpy.datetime64` or array of instances

	:return:
		instance of :class:`datetime.time`
	"""
	assert is_np_datetime(dt64)

	## Try to convert to datetime.datetime first
	py_dt = to_py_datetime(dt64)
	if isinstance(py_dt, datetime.datetime):
		return py_dt.time()
	elif isinstance(py_dt, list) and isinstance(py_dt[0], datetime.datetime):
		return [item.time() for item in py_dt]
	else:
		## Conversion failed, extract time manually
		start_of_day = as_np_datetime(dt64, 'D')
		secs = timespan(start_of_day, dt64, 's')
		## Note: np.divmod only from numpy 1.15 onwards...
		hours = np.floor(secs / SECS_PER_HOUR).astype('int')
		secs = np.mod(secs, SECS_PER_HOUR)
		minutes = np.floor(secs / SECS_PER_MINUTE).astype('int')
		secs = np.mod(secs, SECS_PER_MINUTE)
		fracs, secs = np.modf(secs)
		secs = secs.astype('int')
		microsecs = (fracs * 1E+6).astype('int')
		if np.isscalar(dt64):
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
	assert is_np_datetime(dt64)

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
		Note: to compute the (fractional) time span, :param:`start_dt` and
		:param:`end_dt` are first converted to seconds if unit is Y, W, D, h or m,
		or to the next higher resolution for the other units. Care should be taken
		to make sure that the dates remain within the allowed range, which is:
		- unit='s' : [2.9e11 BC, 2.9e11 AD]
		- unit='ms': [ 2.9e8 BC, 2.9e8 AD]
		- unit='us': [290301 BC, 294241 AD]
		- unit='ns': [ 1678 AD, 2262 AD]
		As a result, for most earthquake catalogs, :param:`unit` should not
		be more precise than 'ms', and for very long synthetic earthquake catalogs,
		perhaps not more precise than 's'!

	:return:
		float, time span in fractions of :param:`unit`
	"""
	assert unit in ['Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us']
	if unit == 'us':
		resolution = 'ns'
	elif unit == 'ms':
		resolution = 'us'
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
		resolution = 'ns'
	elif unit == 'ms':
		resolution = 'us'
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
				's': 1E+3,
				'ms': 1E+3,
				'us': 1E+3}[unit]
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
		float or float array, time in seconds
	"""
	#dt = as_np_datetime(dt, 's')
	#time_delta = (dt - EPOCH) / ONE_SECOND
	#return time_delta.astype('int64')

	return timespan(EPOCH, dt, 's')


def py_time_to_seconds(py_time):
	"""
	Convert datetime.time to seconds

	:param py_time:
		instance of :class:`datetime.time`

	:return:
		float, number of seconds
	"""
	assert isinstance(py_time, datetime.time)
	secs = py_time.hour * 3600 + py_time.minute * 60 + py_time.second
	secs += (py_time.microsecond * 1E-6)
	return secs


def py_time_to_fractional_hours(py_time):
	"""
	Convert datetime.time to fractional hours

	:param py_time:
		instance of :class:`datetime.time`

	:return:
		float, fractional hours
	"""
	assert isinstance(py_time, datetime.time)
	return py_time.hour + py_time.minute/60.0 + py_time.second/3600.0


def py_time_to_np_timedelta(py_time):
	"""
	Convert datetime.time to numpy timedelta64

	:param py_time:
		instance of :class:`datetime.time`

	:return:
		instance of :class:`np.timedelta64`
	"""
	secs = py_time_to_seconds(py_time)
	microsecs = int(round(secs * 1E+6))
	return np.timedelta64(microsecs, 'us')


def combine_np_date_and_py_time(dt64, py_time, unit='s'):
	"""
	Combine date from numpy datetime and time from datetime.time object

	:param dt64:
		instance of :class:`np.datetime64`
	:param py_time:
		instance of :class:`datetime.time`
	:param unit:
		str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
		(year|week|day|hour|minute|second|millisecond|microsecond)
		(default: 's')

	:return:
		instance of :class:`np.datetime64`
	"""
	dt64 = as_np_datetime(dt64, 'us')
	td = py_time_to_np_timedelta(py_time)
	return as_np_datetime(dt64 + td, unit=unit)
