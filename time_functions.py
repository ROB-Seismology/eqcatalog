"""
Useful time functions
"""

import mx.DateTime as mxDateTime


avg_secs_per_year = 31557600


def timespan(start_date, end_date):
	"""
	Return total time span in years between start date and end date.

	:param start_date:
		datetime.date(time) or mxDateTime.DateTime object
	:param end_date:
		datetime.date(time) or mxDateTime.DateTime object

	:return:
		time span in fractional years
	"""
	num_intervening_years = end_date.year - start_date.year
	if num_intervening_years == 0:
		year = end_date.year
		timespan = (end_date - start_date).days * 1. / (mxDateTime.Date(year+1,1,1) - mxDateTime.Date(year,1,1)).days
	else:
		end_of_last_year = mxDateTime.Date(end_date.year-1,12,31)
		end_of_this_year = mxDateTime.Date(end_date.year,12,31)
		timespan = (end_date - end_of_last_year).days * 1./ (end_of_this_year - end_of_last_year).days
		start_of_this_year = mxDateTime.Date(start_date.year,1,1)
		start_of_next_year = mxDateTime.Date(start_date.year+1,1,1)
		timespan += (start_of_next_year - start_date).days * 1. / (start_of_next_year - start_of_this_year).days
		timespan += (num_intervening_years - 1)
	return timespan


def since_epoch(date):
	"""
	Return time in seconds since epoch

	:param date:
		datetime.date(time) or mxDateTime.DateTime object

	:return:
		int, time in seconds
	"""
	# returns seconds since epoch
	epoch = mxDateTime.Date(1970, 1, 1)
	diff = epoch - date
	return diff.days * 24. * 3600. + diff.seconds
	## The line below only works for dates after the epoch
	#return time.mktime(date.timetuple())


def fractional_year(date):
	"""
	Compute fractional year of date

	:param date:
		datetime.date(time) or mxDateTime.DateTime object

	:return:
		Float, fractional year
	"""
	year = date.year
	startOfThisYear = mxDateTime.DateFrom(year=year, month=1, day=1)
	startOfNextYear = mxDateTime.DateFrom(year=year+1, month=1, day=1)

	yearElapsed = since_epoch(date) - since_epoch(startOfThisYear)
	yearDuration = since_epoch(startOfNextYear) - since_epoch(startOfThisYear)
	fraction = yearElapsed/yearDuration

	return date.year + fraction
