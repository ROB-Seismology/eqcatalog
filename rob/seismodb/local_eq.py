 #-*- coding: iso-Latin-1 -*-

"""
seismodb
Python module to retrieve information from the ROB seismology database
======================================================================
Author: Kris Vanneste, Royal Observatory of Belgium.
Date: Apr 2008.

Required modules:
	Third-party:
		MySQLdb
	ROB:
		db_secrets (python file containing host, user, password and database name)
"""


from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str


## Import standard python modules
import datetime

## Import third-party modules
import numpy as np



__all__ = ["query_local_eq_catalog", "query_local_eq_catalog_by_id",
			"get_id_from_seiscomp_id", "get_last_earthID"]



def query_local_eq_catalog(region=None, start_date=None, end_date=None,
						Mmin=None, Mmax=None, min_depth=None, max_depth=None,
						id_earth=None, sort_key="date", sort_order="asc",
						event_type="ke", has_open_enquiry=None, null_value=np.nan,
						skip_null=True, verbose=False, errf=None):
	"""
	Query ROB catalog of local earthquakes.

	Notes:
	Magnitude used for selection is based on MW first, then MS, then ML.
	NULL values in the database are converted to NaN values (magnitudes) or zeros.
	Only real earthquakes are extracted (is_true = 1).

	:param region:
		(w, e, s, n) tuple specifying rectangular region of interest in
		geographic coordinates (default: None)
	:param start_date:
		Int, year
		or str, date(time) string
		or datetime.date, datetime.datetime or np.datetime64 object
		specifying start of time window of interest
		(default: None)
	:param end_date:
		Int, year
		or str, date(time) string
		or datetime.date, datetime.datetime or np.datetime64 object
		specifying end of time window of interest
		(default: None)
	:param Mmin:
		Float, minimum magnitude to extract (default: None)
	:param Mmax:
		Float, maximum magnitude to extract (default: None)
	:param min_depth:
		Float, minimum depth in km to extract (default: None)
	:param max_depth:
		Float, maximum depth in km to extract (default: None)
	:param id_earth:
		Int or List, ID(s) of event to extract (default: None)
	:param sort_key":
		String, property name to sort results with: "date" (= "time")
		or "mag" (= "size"). May also be None to prevent sorting.
		(default: "date")
	:param sort_order:
		String, sort order, either "asc" or "desc" (default: "asc")
	:param event_type:
		str, event type (one of "all", "cb", "ex", "ke", "ki", "km",
		"kr", "kx", "qb", "sb", "se", "si", "sm", "sr", "sx" or "uk")
		or comma-separated str
		(default: "ke" = known earthquakes)
	:param null_value:
		float, value to use for NULL values (except magnitude)
		(default: np.nan)
	:param skip_null:
		bool, whether or not to skip records with NULL location and magnitude
		(default: True)
	:param verbose:
		Bool, if True the query string will be echoed to standard output
	:param errf:
		File object, where to print errors

	:return:
		instance of :class:`EQCatalog`
		Note that catalog will have default Mrelations and completeness
		attached!
	"""
	from ..eqrecord import (ROBLocalEarthquake, DEFAULT_MRELATIONS)
	from ..completeness import DEFAULT_COMPLETENESS
	from ...eqcatalog import EQCatalog
	from ...time import (as_np_datetime, combine_np_date_and_py_time)
	from .base import query_seismodb_table

	## Convert input arguments, if necessary
	if isinstance(id_earth, (int, np.integer, basestring)):
		id_earth = [id_earth]

	if not id_earth:
		## Do not impose start or end date if id_earth is given
		if isinstance(start_date, int):
			start_date = '%d-01-01' % start_date
		elif start_date is None:
			#start_date = '1350-01-01'
			start_date = '800-01-01'
		start_date = as_np_datetime(start_date)

		if isinstance(end_date, int):
			end_date = '%d-12-31T23:59:59.999999' % end_date
		elif end_date is None:
			end_date = np.datetime64('now')
		end_date = as_np_datetime(end_date)
		## If start_date and end_date are the same,
		## set time of end date to end of the day
		if start_date and end_date and end_date - start_date == np.timedelta64(0):
			end_time = datetime.time(23, 59, 59, 999999)
			end_date = combine_np_date_and_py_time(end_date, end_time)

	## Construct SQL query
	table_clause = "earthquakes"
	column_clause = [
		'id_earth',
		'DATE_FORMAT(date, "%Y-%m-%d") as date',
		'TIME_FORMAT(time, "%H:%i:%s") as time',
		'tm_hund',
		'longitude',
		'latitude',
		'depth',
		'ML',
		'MS',
		'MW',
		'MWH',
		'IF(MW, MW, IF(MWH, MWH, IF(MS, MS, ML))) as M',
		'intensity_max',
		'macro_radius',
		'errh',
		'errz',
		'errt',
		'errM',
		'earthquakes.name',
		'type',
		'providing_institute.code as agency']

	where_clause = ""
	having_clause = ""
	if not (id_earth is None or id_earth in ([], '')):
		where_clause += 'id_earth in (%s)' % ",".join([str(item) for item in id_earth])
	else:
		where_clause += 'is_true = 1'
		if event_type != "all":
			if event_type in ('NULL', None):
				where_clause += ' and type IS NULL'
			elif ',' in event_type:
				## Multiple event types
				event_types = ','.join(['"%s"' % et for et in event_type.split(',')])
				where_clause += ' AND type IN (%s)' % event_types
			else:
				where_clause += ' AND type = "%s"' % event_type
		if region:
			w, e, s, n = region
			where_clause += ' AND longitude BETWEEN %f and %f' % (w, e)
			where_clause += ' AND latitude BETWEEN %f and %f' % (s, n)

		#where_clause += (' and date Between "%s" and "%s"'
		where_clause += (' AND TIMESTAMP(date, time) BETWEEN "%s" and "%s"'
						% (start_date, end_date))
		if min_depth:
			where_clause += ' AND depth >= %f' % min_depth
		if max_depth:
			where_clause += ' AND depth <= %f' % max_depth

		if has_open_enquiry is not None:
			where_clause += ' AND web_status = %d' % int(has_open_enquiry)

		if not(Mmax is None and Mmin is None):
			if Mmin is None:
				Mmin = 0.0
			if not Mmax:
				Mmax = 10.0
			having_clause += 'HAVING M BETWEEN %f and %f' % (Mmin, Mmax)

	if sort_key:
		if sort_order.lower()[:3] == "asc":
			sort_order = "asc"
		else:
			sort_order = "desc"
		order_clause = ""
		if sort_key.lower() in ("date", "time"):
			order_clause += 'date %s, time %s' % (sort_order, sort_order)
		elif sort_key.lower() in ("size", "mag"):
			order_clause += 'M %s' % sort_order
	else:
		order_clause = ""

	join_clause = [('LEFT',
					'providing_institute',
					'id_providingInst = providing_institute.id')]

	if errf !=None:
		errf.write("Querying KSB-ORB local earthquake catalog:\n")

	## Fetch records
	eq_list = []
	num_skipped = 0
	## Note: id_earth variable overwritten in loop!
	query_for_id = bool(id_earth)
	for rec in query_seismodb_table(table_clause, column_clause=column_clause,
					where_clause=where_clause, having_clause=having_clause,
					order_clause=order_clause, join_clause=join_clause,
					verbose=verbose, errf=errf):
		id_earth, name = rec["id_earth"], rec["name"] or ''
		date, time = rec["date"], rec["time"]
		lon, lat = rec["longitude"], rec["latitude"]
		depth = rec["depth"]
		ML, MS, MW, M = rec["ML"], rec["MS"], rec["MW"], rec["M"]
		## "Historical MW"
		# TODO: may be necessary to add MWHp ('1/4', '1/2', '3/4') in the future
		MWH = rec["MWH"]
		MW = MWH or MW
		intensity_max, macro_radius = rec["intensity_max"], rec["macro_radius"]
		errh, errz, errt, errM = rec["errh"], rec["errz"], rec["errt"], rec["errM"]
		etype = rec["type"]  ## Avoid conflict with event_type parameter!

		## Skip records without lon, lat, depth and magnitude
		if not query_for_id:
			if skip_null and (lon == lat == depth == M == None):
				num_skipped += 1
				continue

		year, month, day = [int(s) for s in date.split("-")]
		## Take into account historical earthquakes where part of date is missing
		if month and day:
			date = datetime.date(year, month, day)
		elif month:
			date = datetime.date(year, month, 1)
		else:
			date = datetime.date(year, 1, 1)
		hour, minutes, seconds = [int(s) for s in time.split(":")]
		microseconds = 0
		if rec["tm_hund"]:
			microseconds = int(rec["tm_hund"]) * 10000
		time = datetime.time(hour, minutes, seconds, microseconds)

		lon = null_value if lon is None else lon
		lat = null_value if lat is None else lat
		depth = null_value if depth is None else depth
		ML = np.nan if ML is None else ML
		MS = np.nan if MS is None else MS
		MW = np.nan if MW is None else MW
		mb = np.nan
		intensity_max = null_value if intensity_max is None else intensity_max
		macro_radius = null_value if macro_radius is None else macro_radius
		errh = null_value if errh is None else errh
		errz = null_value if errz is None else errz
		errt = null_value if errt is None else errt
		errM = null_value if errM is None else errM

		"""
		if convert_NULL:
			if lon == None:
				lon = 0.0
			if lat == None:
				lat = 0.0
			if depth == None:
				depth = 0.0
			if ML == None:
				ML = np.nan
			if MS == None:
				MS = np.nan
			if MW == None:
				MW = np.nan
			if intensity_max == None:
				intensity_max = 0
			if macro_radius == None:
				macro_radius = 0
			if errh == None:
				errh = 0.
			if errz == None:
				errz = 0.
			if errt == None:
				errt = 0.
			if errM == None:
				errM = 0.

			mb = np.nan
			"""

		agency = rec["agency"]
		if agency[:3] == 'UCC':
			agency = 'ROB'

		eq = ROBLocalEarthquake(id_earth, date, time, lon, lat, depth, {},
							ML, MS, MW, mb, name, intensity_max, macro_radius,
							errh, errz, errt, errM, agency=agency, event_type=etype)
		eq_list.append(eq)

	if verbose and num_skipped:
		print('Skipped %d records with NULL location and magnitude'
				% num_skipped)

	name = "ROB Catalog %s - %s" % (start_date, end_date)
	start_date, end_date = np.datetime64(start_date), np.datetime64(end_date)
	return EQCatalog(eq_list, start_date, end_date, region=region, name=name,
					default_Mrelations=DEFAULT_MRELATIONS,
					default_completeness=DEFAULT_COMPLETENESS)


def query_local_eq_catalog_by_id(id_earth, sort_key=None, sort_order="asc",
								verbose=False, errf=None):
	"""
	Query local earthquake catalog using earthquake IDs

	:param id_earth:
		- int or list, ID(s) of event to extract
		- str: hashed ID or SeisComP ID
		(default: None)
	:param sort_key:
	:param sort_order:
	:param verbose:
	:param errf:
		see :func:`query_local_eq_catalog`

	:return:
		instance of :class:`EQCatalog`
	"""
	if not id_earth in (None, []):
		if isinstance(id_earth, basestring):
			if id_earth[:2] == 'be' and id_earth[2:6].isdigit():
				id_earth = get_id_from_seiscomp_id(id_earth)
			else:
				from .hash import hash2id
				id_earth = hash2id(id_earth)

		return query_local_eq_catalog(id_earth=id_earth, sort_key=sort_key,
								sort_order=sort_order, verbose=verbose, errf=errf)


def get_id_from_seiscomp_id(id_seiscomp):
	"""
	Fetch ROB earthquake ID corresponding to SeisComP ID from database
	(based on the 'comment' field)

	:param id_seiscomp:
		str, SeisComP ID (e.g. 'be2020yqrg')

	:return:
		int, id_earth
	"""
	from .base import query_seismodb_table

	table_clause = 'earthquakes'
	column_clause = ['id_earth']
	where_clause = "comment LIKE 'SC3:%s'" % id_seiscomp

	try:
		[rec] = query_seismodb_table(table_clause, column_clause=column_clause,
										 where_clause=where_clause)
	except:
		id_earth = None
	else:
		id_earth = int(rec['id_earth'])

	return id_earth


def get_last_earthID():
	"""
	Return ID of most recently added earthquake in database

	:return:
		int, earthquake ID
	"""
	from .base import query_seismodb_table_generic

	query = ('SELECT id_earth FROM earthquakes WHERE type="ke" AND is_true = 1 '
			'ORDER BY id_earth DESC LIMIT 0 , 1')
	id_earth = query_seismodb_table_generic(query)[0]['id_earth']

	return id_earth
