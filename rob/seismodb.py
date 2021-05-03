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

## Import ROB modules
from db.simpledb import (build_sql_query, query_mysql_db_generic)



__all__ = ["query_seismodb_table_generic", "query_seismodb_table",
			"query_local_eq_catalog", "query_local_eq_catalog_by_id",
			"get_id_from_seiscomp_id",
			"get_last_earthID", "query_focal_mechanisms",
			"query_online_macro_catalog", "query_traditional_macro_catalog",
			"query_official_macro_catalog", "query_historical_macro_catalog",
			"query_traditional_macro_catalog_aggregated",
			"query_official_macro_catalog_aggregated",
			"query_historical_macro_catalog_aggregated",
			"query_historical_texts",
			"query_online_macro_catalog_aggregated",
			"get_num_online_macro_enquiries",
			"get_num_official_enquiries", "get_num_traditional_mdps",
			"get_earthquakes_with_traditional_enquiries",
			"get_earthquakes_with_online_enquiries",
			"query_stations", "get_station_coordinates",
			"get_station_catalog", "query_phase_picks",
			"zip2ID", "get_communes", "get_subcommunes", "get_subcommune_ids"]


# Note: median is not supported by MySQL
AGG_FUNC_DICT = {"average": "AVG", "mean": "AVG",
				"minimum": "MIN", "maximum": "MAX"}


def query_seismodb_table_generic(query, verbose=False, print_table=False, errf=None):
	"""
	Query MySQL table using generic clause, returning each record as a dict

	:param query:
		str, SQL query
	:param verbose:
		bool, whether or not to print the query
		(default: False)
	:param print_table:
		bool, whether or not to print results of query in a table
		rather than returning records
		(default: False)
	:param errf:
		file object, where to print errors

	:return:
		generator object, yielding a dictionary for each record
	"""
	## Database information
	from secrets.seismodb import host, user, passwd, database
	try:
		from secrets.seismodb import port
	except:
		port = 3306

	## Avoid Warning: Row XXX was cut by GROUP_CONCAT()
	## For 32bit systems, the maximum value is 4294967295
	if "GROUP_CONCAT" in query.upper():
		query0 = 'SET SESSION group_concat_max_len=4294967295'
		query_mysql_db_generic(database, host, user, passwd, query0, port=port,
								verbose=verbose, errf=errf)

	return query_mysql_db_generic(database, host, user, passwd, query, port=port,
							verbose=verbose, print_table=print_table, errf=errf)


def query_seismodb_table(table_clause, column_clause="*", join_clause="",
						where_clause="", having_clause="", order_clause="",
						group_clause="", verbose=False, print_table=False,
						errf=None):
	"""
	Query seismodb table using clause components

	:param table_clause:
		str or list of strings, name(s) of database table(s)
	:param column_clause:
		str or list of strings, column clause or list of columns (default: "*")
	:param join_clause:
		str or list of (join_type, table_name, condition) tuples,
		join clause (default: "")
	:param where_clause:
		str, where clause (default: "")
	:param having_clause:
		str, having clause (default: "")
	:param order_clause:
		str, order clause (default: "")
	:param group_clause:
		str, group clause (default: "")
	:param verbose:
	:param print_table:
	:param errf:
		see :func:`query_seismodb_table_generic`

	:return:
		generator object, yielding a dictionary for each record
	"""
	query = build_sql_query(table_clause, column_clause, join_clause, where_clause,
							having_clause, order_clause, group_clause)

	return query_seismodb_table_generic(query, verbose=verbose,
										print_table=print_table, errf=errf)


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
	from .eqrecord import (ROBLocalEarthquake, DEFAULT_MRELATIONS)
	from .completeness import DEFAULT_COMPLETENESS
	from ..eqcatalog import EQCatalog
	from ..time import (as_np_datetime, combine_np_date_and_py_time)

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


def query_focal_mechanisms(region=None, start_date=None, end_date=None,
							Mmin=None, Mmax=None, id_earth=None, sort_key="Mag",
							sort_order="asc", verbose=False, errf=None):
	"""
	Query ROB focal mechanism database

	:param region:
		(w, e, s, n) tuple specifying rectangular region of interest in
		geographic coordinates (default: None)
	:param start_date:
		Int or date or datetime object specifying start of time window of interest
		If integer, start_date is interpreted as start year
		(default: None)
	:param end_date:
		Int or date or datetime object specifying end of time window of interest
		If integer, end_date is interpreted as end year
		(default: None)
	:param Mmin:
		Float, minimum magnitude to extract (default: None)
	:param Mmax:
		Float, maximum magnitude to extract (default: None)
	:param id_earth:
		Int or List, ID(s) of event to extract (default: None)
	:param sort_key":
		String, property name to sort results with: "date" (= "time")
		or "mag" (= "size") (default: "date")
	:param sort_order:
		String, sort order, either "asc" or "desc" (default: "asc")
	:param verbose:
		Bool, if True the query string will be echoed to standard output
	:param errf:
		File object, where to print errors

	:return:
		list with instances of :class:`FocMecRecord`
	"""
	from ..eqrecord import FocMecRecord

	if isinstance(id_earth, (int, basestring)):
		id_earth = [id_earth]

	## Construct SQL query
	table_clause = ['focal_mechanisms', 'earthquakes']

	column_clause = [
		'earthquakes.id_earth',
		'DATE_FORMAT(earthquakes.date, "%Y-%m-%d") as date',
		'TIME_FORMAT(earthquakes.time, "%H:%i:%s") as time',
		'earthquakes.longitude',
		'earthquakes.latitude',
		'earthquakes.depth',
		'earthquakes.ML',
		'earthquakes.MS',
		'earthquakes.MW',
		'IF(MW, MW, IF(MS, MS, ML)) as M',
		'strike',
		'dip',
		'slip',
		'earthquakes.intensity_max',
		'earthquakes.macro_radius',
		'earthquakes.name']

	where_clause = 'focal_mechanisms.id_earth = earthquakes.id_earth'
	where_clause += ' and earthquakes.type = "ke"'
	where_clause += ' and earthquakes.is_true = 1'
	if id_earth:
		where_clause += (' and earthquakes.id_earth in (%s)'
						% ",".join([str(item) for item in id_earth]))
	if region:
		w, e, s, n = region
		where_clause += ' and earthquakes.longitude Between %f and %f' % (w, e)
		where_clause += ' and earthquakes.latitude Between %f and %f' % (s, n)
	if start_date or end_date:
		if not end_date:
			end_date = datetime.datetime.now()
		if not start_date:
			start_date = datetime.datetime(100, 1, 1)
		where_clause += (' and earthquakes.date Between "%s" and "%s"'
						% (start_date.isoformat(), end_date.isoformat()))

	having_clause = ""
	if Mmax or Mmin != None:
		if Mmin:
			Mmin = 0.0
		if not Mmax:
			Mmax = 8.0
		having_clause += ' HAVING M Between %f and %f' % (Mmin, Mmax)

	if sort_order.lower()[:3] == "asc":
		sort_order = "asc"
	else:
		sort_order = "desc"
	order_clause = ""
	if sort_key.lower() in ("date", "time"):
		order_clause += 'date %s, time %s' % (sort_order, sort_order)
	elif sort_key.lower() in ("size", "mag"):
		order_clause += 'M %s' % sort_order

	if errf !=None:
		errf.write("Querying KSB-ORB focal mechanism catalog:\n")

	## Fetch records
	focmec_list = []
	for rec in query_seismodb_table(table_clause, column_clause=column_clause,
					where_clause=where_clause, having_clause=having_clause,
					order_clause=order_clause, verbose=verbose, errf=errf):

		id_earth, name = rec["id_earth"], rec["name"]
		date, time = rec["date"], rec["time"]
		lon, lat = rec["longitude"], rec["latitude"]
		depth = rec["depth"]
		ML, MS, MW, M = rec["ML"], rec["MS"], rec["MW"], rec["M"]
		intensity_max, macro_radius = rec["intensity_max"], rec["macro_radius"]
		strike, dip, rake = rec["strike"], rec["dip"], rec["slip"]

		if name == lon == lat == depth == M == None:
			continue
		year, month, day = [int(s) for s in date.split("-")]
		date = datetime.date(year, month, day)
		hour, minutes, seconds = [int(s) for s in time.split(":")]
		time = datetime.time(hour, minutes, seconds)

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

		eq = FocMecRecord(id_earth, date, time, lon, lat, depth, {}, ML, MS, MW,
						strike, dip, rake, name, intensity_max, macro_radius)
		focmec_list.append(eq)

	return focmec_list


def query_traditional_macro_catalog(id_earth, id_com=None, data_type='',
								group_by_main_commune=False, min_fiability=80,
								verbose=False, errf=None):
	"""
	Query ROB traditional macroseismic catalog
	This includes both "official" communal inquiries and historical
	data.
	Currently, this catalog contains only 1 record per commune
	for each earthquake. Consequently, official and historical data
	are mutually exclusive.

	:param id_earth:
		int, earthquake ID
	:param id_com:
		int, commune ID
		or list of ints
		(default: None)
		Note that :param:`id_earth` must not be None if :param:`id_com`
		is a list or None
	:param data_type:
		str, type of macroseismic data: '', 'official' or 'historical'
		(default: '')
	:param group_by_main_commune:
		bool, whether or not the value of :param:`id_com` corresponds
		to main village
		(default: False)
	:param min_fiability:
		float, minimum fiability of enquiry
		(default: 80)
	:param verbose:
		Bool, if True the query string will be echoed to standard output
	:param errf:
		File object, where to print errors

	:return:
		instance of :class:`eqcatalog.macro.MDPCollection`
	"""
	from ..macro import MDP, MDPCollection

	assert not (id_earth is None and
				(id_com is None or isinstance(id_com, (list, np.ndarray))))

	## Construct SQL query
	table_clause = ['macro_detail']

	column_clause = [
		'id_macro_detail',
		'macro_detail.id_earth',
		'fiability',
		'comment',
		'macro_detail.id_com',
		'communes.id_main',
		'IF(com_zip.latitude, com_zip.latitude, communes.latitude) as latitude',
		'IF(com_zip.longitude, com_zip.longitude, communes.longitude) as longitude']

	if data_type == 'official':
		column_clause += [
			'macro_official_inquires.id_macro_off',
			'macro_official_inquires.id_source',
			'macro_official_inquires.source']

	## Replace intensity values of 13 with NULL
	Imin_clause = 'IF (intensity_min < 13, intensity_min, NULL) AS "Imin"'
	column_clause.append(Imin_clause)
	Imax_clause = 'IF (intensity_max < 13, intensity_max, NULL) AS "Imax"'
	column_clause.append(Imax_clause)

	where_clause = 'macro_detail.fiability >= %d' % min_fiability
	if id_earth is not None:
		where_clause += ' AND macro_detail.id_earth = %d' % id_earth
	if id_com is not None:
		if group_by_main_commune:
			where_clause += ' AND communes.id_main'
		else:
			where_clause += ' AND macro_detail.id_com'
		if isinstance(id_com, int):
			where_clause += ' = %d' % id_com
		elif isinstance(id_com, (list, np.ndarray)):
			id_com_str = ','.join(['%s' % id for id in id_com])
			where_clause += ' IN (%s)' % id_com_str
	if data_type == 'historical':
		where_clause += (' AND CONCAT(id_earth, macro_detail.id_com) NOT IN'
						' (SELECT CONCAT(id_earth, id_com)'
						' FROM macro_official_inquires)')

	join_clause = [('LEFT JOIN', 'communes', 'macro_detail.id_com = communes.id'),
						('LEFT JOIN',
						_get_com_zip_table_clause(['BE']),
						'communes.id = com_zip.id_com')]
						# AND communes.code_p = com_zip.zip
	if data_type == 'official':
		join_clause.append(('RIGHT JOIN', 'macro_official_inquires',
			'macro_detail.id_com = macro_official_inquires.id_com'
			' AND macro_detail.id_earth = macro_official_inquires.id_earth'))

	if errf !=None:
		errf.write("Querying KSB-ORB traditional macroseismic catalog:\n")

	## Fetch records
	macro_recs = query_seismodb_table(table_clause, column_clause=column_clause,
				join_clause=join_clause, where_clause=where_clause,
				verbose=verbose, errf=errf)

	## Construct MDP Collection
	mdp_list = []
	imt = 'EMS98'
	for rec in macro_recs:
		## Convert None Imin/Imax values to np.nan
		rec['Imin'] = rec['Imin'] or np.nan
		rec['Imax'] = rec['Imax'] or np.nan

		mdp = MDP(rec.pop('id_macro_detail'), rec.pop('id_earth'), rec.pop('Imin'),
				rec.pop('Imax'), imt, rec.pop('longitude'), rec.pop('latitude'),
				data_type or 'traditional', rec.pop('id_com'), rec.pop('id_main'),
				rec.pop('fiability'), **rec)
		mdp_list.append(mdp)

	return MDPCollection(mdp_list)


def query_official_macro_catalog(id_earth, id_com=None, group_by_main_commune=False,
								min_fiability=80, verbose=False, errf=None):
	"""
	Query ROB catalog of official communal macroseismic enquiries
	This is a wrapper for :func:`query_traditional_macro_catalog`
	"""
	kwargs = locals().copy()
	return query_traditional_macro_catalog(data_type='official', **kwargs)


def query_historical_macro_catalog(id_earth, id_com=None, group_by_main_commune=False,
									min_fiability=80, verbose=False, errf=None):
	"""
	Query ROB catalog of historical macroseismic data
	This is a wrapper for :func:`query_traditional_macro_catalog`
	"""
	kwargs = locals().copy()
	return query_traditional_macro_catalog(data_type='historical', **kwargs)


def query_traditional_macro_catalog_aggregated(id_earth, id_com=None, data_type='',
					Imin_or_max='max', group_by_main_commune=False, agg_method="mean",
					min_fiability=80, verbose=False, errf=None):
	"""
	Query ROB traditional macroseismic catalog
	This includes both "official" communal inquiries and historical
	data.
	Currently, this catalog contains only 1 record per commune
	for each earthquake. Consequently, official and historical data
	are mutually exclusive.

	:param id_earth:
		int, earthquake ID
	:param id_com:
		int, commune ID
		or list of ints
		(default: None)
		Note that :param:`id_earth` must not be None if :param:`id_com`
		is a list or None
	:param data_type:
		str, type of macroseismic data: '', 'official' or 'historical'
		(default: '')
	:param Imin_or_max:
		str, one of 'min', 'mean' or 'max' to select between
		intensity_min and intensity_max values in database
		(default: 'max')
	:param group_by_main_commune:
		bool, whether or not to aggregate the results by main village
		(default: False)
	:param agg_method:
		str, aggregation function to use if :param:`group_by_main_commune`
		is True, one of "minimum", "maximum" or "average"
		(default: "average")
	:param min_fiability:
		float, minimum fiability of enquiry
		(default: 80)
	:param verbose:
		Bool, if True the query string will be echoed to standard output
	:param errf:
		File object, where to print errors

	:return:
		instance of :class:`AggregatedMacroInfoCollection`
	"""
	from ..macro.macro_info import AggregatedMacroInfo, AggregatedMacroInfoCollection

	assert not (id_earth is None and
				(id_com is None or isinstance(id_com, (list, np.ndarray))))

	if id_earth == 18280223:
		# TODO: check how this can be integrated in seismodb
		from secrets.eqmapperdb import (host as _host, user as _user,
									passwd as _passwd, database as _database)

		query = 'SELECT id_com_main as "id_com",'
		query += ' Longitude as "longitude", Latitude as "latitude",'
		query += ' id as "id_db",'
		if Imin_or_max == 'max':
			query += ' Imax'
		elif Imin_or_max == 'min':
			query += ' Imin'
		elif Imin_or_max == 'mean':
			query += ' (Imin + Imax) / 2.'
		query += ' as "Intensity"'
		query += ' from Macro18280223 Where country = "BE"'

		if errf !=None:
			errf.write("Querying Macro18280223 table on eqmapperdb:\n")

		macro_recs = query_mysql_db_generic(_database, _host, _user, _passwd, query)

	else:
		## Construct SQL query
		table_clause = ['macro_detail']

		if group_by_main_commune:
			column_clause = [
				'communes.id_main AS id_com',
				'main_communes.longitude',
				'main_communes.latitude',
				'COUNT(*) as num_replies']
		else:
			column_clause = [
				'macro_detail.id_com',
				'communes.longitude',
				'communes.latitude']

		## Replace intensity values of 13 with NULL
		Imin_clause = 'IF (intensity_min < 13, intensity_min, NULL)'
		Imax_clause = 'IF (intensity_max < 13, intensity_max, NULL)'

		if group_by_main_commune:
			agg_function = AGG_FUNC_DICT.get(agg_method.lower())
			if Imin_or_max == 'min':
				intensity_col = '%s(%s) AS "Intensity"'
				intensity_col %= (agg_function, Imin_clause)
			elif Imin_or_max == 'max':
				intensity_col = '%s(%s) AS "Intensity"'
				intensity_col %= (agg_function, Imax_clause)
			elif Imin_or_max == 'mean':
				intensity_col = '%s((%s + %s)/2.) AS "Intensity"'
				intensity_col %= (agg_function, Imin_clause, Imax_clause)
			column_clause.append(intensity_col)
			group_clause = 'communes.id_main'
			column_clause.append('GROUP_CONCAT(id_macro_detail SEPARATOR ",") AS id_db')
		else:
			agg_method = None
			if Imin_or_max == 'min':
				intensity_col = '%s AS "Intensity"' % Imin_clause
			elif Imin_or_max == 'max':
				intensity_col = '%s AS "Intensity"' % Imax_clause
			elif Imin_or_max == 'mean':
				intensity_col = '(%s + %s)/2. AS "Intensity"'
				intensity_col %= (Imin_clause, Imax_clause)
			column_clause.append(intensity_col)
			group_clause = ""
			column_clause.append('id_macro_detail AS id_db')

		where_clause = 'macro_detail.fiability >= %d' % min_fiability
		if id_earth is not None:
			where_clause += ' AND macro_detail.id_earth = %d' % id_earth
		if id_com is not None:
			if group_by_main_commune:
				where_clause += ' AND communes.id_main'
			else:
				where_clause += ' AND macro_detail.id_com'
			if isinstance(id_com, int):
				where_clause += ' = %d' % id_com
			elif isinstance(id_com, (list, np.ndarray)):
				id_com_str = ','.join(['%s' % id for id in id_com])
				where_clause += ' IN (%s)' % id_com_str
		if data_type == 'historical':
			where_clause += (' AND CONCAT(id_earth, id_com) NOT IN'
							' (SELECT CONCAT(id_earth, id_com)'
							' FROM macro_official_inquires)')

		join_clause = [('LEFT JOIN', 'communes', 'macro_detail.id_com = communes.id')]
		if data_type == 'official':
			join_clause.append(('RIGHT JOIN', 'macro_official_inquires',
				'macro_detail.id_com = macro_official_inquires.id_com'
				' AND macro_detail.id_earth = macro_official_inquires.id_earth'))
		if group_by_main_commune:
			join_clause.append(('JOIN', 'communes AS main_communes',
							'communes.id_main = main_communes.id'))

		#having_clause = 'Intensity >= %d' % min_val

		if errf !=None:
			errf.write("Querying KSB-ORB traditional macroseismic catalog:\n")

		macro_recs = query_seismodb_table(table_clause, column_clause=column_clause,
					join_clause=join_clause, where_clause=where_clause,
					#having_clause=having_clause, order_clause=order_clause,
					group_clause=group_clause, verbose=verbose, errf=errf)

	## Fetch records
	macro_infos = []
	agg_type = {False: 'id_com', True: 'id_main'}[group_by_main_commune]
	imt = 'EMS98'
	for rec in macro_recs:
		id_com = rec['id_com']
		I = float(rec['Intensity'])
		lon, lat = rec['longitude'], rec['latitude']
		num_replies = rec.get('num_replies', 1)
		if isinstance(rec['id_db'], (int, basestring)):
			db_ids = [rec['id_db']]
		else:
			db_ids = list(map(int, rec['id_db'].split(',')))
		if data_type == '':
			enq_type = 'traditional'
		else:
			enq_type = data_type
		macro_info = AggregatedMacroInfo(id_earth, id_com, I, imt, agg_type,
										enq_type, num_replies=num_replies,
										lon=lon, lat=lat, db_ids=db_ids)
		macro_infos.append(macro_info)

	proc_info = dict(agg_method=agg_method, min_fiability=min_fiability,
					Imin_or_max=Imin_or_max)
	macro_info_coll = AggregatedMacroInfoCollection(macro_infos, agg_type, enq_type,
													proc_info=proc_info)

	return macro_info_coll


def query_official_macro_catalog_aggregated(id_earth, id_com=None, Imin_or_max='max',
					group_by_main_commune=False, agg_method="mean",
					min_fiability=80, verbose=False, errf=None):
	"""
	Query ROB catalog of official communal macroseismic enquiries
	This is a wrapper for :func:`query_traditional_macro_catalog_aggregated`
	"""
	kwargs = locals().copy()
	return query_traditional_macro_catalog_aggregated(data_type='official', **kwargs)


def query_historical_macro_catalog_aggregated(id_earth, id_com=None, Imin_or_max='max',
					group_by_main_commune=False, agg_method="mean",
					min_fiability=80, verbose=False, errf=None):
	"""
	Query ROB catalog of historical macroseismic data
	This is a wrapper for :func:`query_traditional_macro_catalog_aggregated`
	"""
	kwargs = locals().copy()
	return query_traditional_macro_catalog_aggregated(data_type='historical', **kwargs)


def query_online_macro_catalog_aggregated(id_earth, min_replies=3, query_info="cii",
					min_fiability=80, group_by_main_commune=False,
					filter_floors=(0, 4), agg_method="mean", verbose=False,
					errf=None):
	"""
	Query ROB internet macroseismic catalog (= online inquiries)

	:param id_earth:
		int, earthquake ID
	:param min_replies:
		int, minimum number of replies
		(default: 3)
	:param query_info:
		str, either "cii", "cdi" or "mi"
		(default: "cii")
	:param min_fiability:
		float, minimum fiability of enquiry
		(default: 80)
	:param group_by_main_commune:
		bool, whether or not to aggregate the results by main village
		(default: False)
	:param filter_floors:
		(min_floor, max_floor) tuple, floors outside this range
		(basement floors and upper floors) are filtered out
		(default: (0, 4))
	:param agg_method:
		str, aggregation function to use, one of "minimum", "maximum" or
		"average"/"mean". If :param:`group_by_main_commune` is False,
		aggregation applies to the enquiries within a given (sub)commune
		(default: "mean")
	:param verbose:
		Bool, if True the query string will be echoed to standard output
	:param errf:
		File object, where to print errors

	:return:
		instance of :class:`AggregatedMacroInfoCollection`
	"""
	from ..macro.macro_info import AggregatedMacroInfo, AggregatedMacroInfoCollection

	## Construct SQL query
	table_clause = ['web_analyse']

	## Hack to include enquiries where ZIP code is given but not matched in web_analyse
	join_clause = [('JOIN', 'web_input', 'web_analyse.id_web = web_input.id_web'),
				('LEFT JOIN', 'communes comm1',
					'web_analyse.id_com != 0 AND web_analyse.id_com = comm1.id'),
				('LEFT JOIN', 'communes comm2',
					'web_analyse.id_com = 0 AND web_input.zip = comm2.code_p '
					'AND web_input.country = comm2.country AND comm2.id = comm2.id_main')]

	if group_by_main_commune:
		column_clause = ['COALESCE(comm1.id_main, comm2.id_main) AS id_comm']
	else:
		column_clause = ['COALESCE(comm1.id, comm2.id) AS id_comm']

	column_clause += [
		'COUNT(*) as "num_replies"',
		'COALESCE(comm1.longitude, comm2.longitude) AS lon',
		'COALESCE(comm1.latitude, comm2.latitude) AS lat',
		'GROUP_CONCAT(web_input.id_web SEPARATOR ",") AS id_web']

	group_clause = "id_comm"

	if agg_method.lower() == "average":
		agg_method = "mean"
	agg_function = AGG_FUNC_DICT.get(agg_method.lower(), "AVG")
	column_clause.append('%s(web_analyse.%s) as "Intensity"'
						% (agg_function, query_info.upper()))

	where_clause = 'web_analyse.id_earth = %d' % id_earth
	where_clause += ' AND web_analyse.m_fiability >= %.1f' % float(min_fiability)
	where_clause += ' AND web_analyse.deleted = false'
	if filter_floors:
		where_clause += ' AND (web_input.floor IS NULL'
		where_clause += ' OR web_input.floor BETWEEN %d AND %d)' % filter_floors

	having_clause = 'num_replies >= %d' % min_replies
	#if query_info.lower() in ("cii", "cdi", "mi"):
	#	having_clause += ' and Intensity >= %d' % min_val

	## Not really useful, as we return a dict...
	order_clause = 'num_replies DESC'

	if errf !=None:
		errf.write("Querying KSB-ORB web macroseismic catalog:\n")

	## Fetch records
	macro_infos = []
	imt = query_info.upper()
	agg_type = {False: 'id_com', True: 'id_main'}[group_by_main_commune]
	for rec in query_seismodb_table(table_clause, column_clause=column_clause,
					join_clause=join_clause, where_clause=where_clause,
					#having_clause=having_clause, order_clause=order_clause,
					group_clause=group_clause, verbose=verbose, errf=errf):
		id_com = rec['id_comm']
		I = rec['Intensity']
		lon, lat = rec['lon'], rec['lat']
		num_replies = rec['num_replies']
		web_ids = list(map(int, rec['id_web'].split(',')))
		macro_info = AggregatedMacroInfo(id_earth, id_com, I, imt, agg_type,
										'internet', num_replies=num_replies,
										lon=lon, lat=lat, db_ids=web_ids)
		macro_infos.append(macro_info)

	proc_info = dict(min_replies=min_replies, min_fiability=min_fiability,
					agg_method=agg_method, filter_floors=filter_floors)

	macro_info_coll = AggregatedMacroInfoCollection(macro_infos, agg_type,
												'internet', proc_info=proc_info)

	return macro_info_coll


def query_online_macro_catalog(id_earth=None, id_com=None, zip_code=None,
						min_fiability=80, min_location_quality=6,
						web_ids=[], verbose=False, errf=None):
	"""
	Query internet enquiries.

	:param id_earth:
		int, ROB earthquake ID: enquiries assigned to given earthquake
		or 'all': enquiries assigned to any event type
		or event type string (e.g., 'ke,ki'): enquiries assigned to
		specific event type
		or date string, datetime.date or np.datetime64: enquiries from
		a particular date, but not assigned to any earthquake
		(default: None = all unassigned enquiries)
	:param id_com:
		int or list of ints, ROB commune ID
		(default: None)
	:param zip_code:
		int or list of ints, ZIP code
		(default: None)
	:param min_fiability:
		float, minimum fiability of enquiry
		(default: 80)
	:param min_location_quality:
		int, minimum quality of location to read from web_location table
		(default: 6)
	:param web_ids:
		list of IDs of individual questionnaires
		(default: [])
	:param verbose:
		bool, if True the query string will be echoed to standard output
		(default: False)
	:param errf:
		File object, where to print errors
		(default: None)

	:return:
		instance of :class:`MacroseismicEnquiryEnsemble`
	"""
	from ..macro import ROBDYFIEnsemble

	table_clause = ['web_input']

	column_clause = ['web_input.*', 'web_analyse.*',
					'web_location.longitude', 'web_location.latitude',
					'web_location.quality AS location_quality']

	join_clause = [('JOIN', 'web_analyse', 'web_input.id_web=web_analyse.id_web'),
					('LEFT JOIN', 'web_location', 'web_input.id_web=web_location.id_web'
								' AND web_location.quality >= %d' % min_location_quality)]

	if isinstance(id_earth, (int, basestring)) and not '-' in str(id_earth):
		where_clause = 'web_analyse.m_fiability >= %.1f' % float(min_fiability)
		where_clause += ' AND web_analyse.deleted = false'

		## Only fetch enquiries assigned to an earthquake
		if id_earth == "all":
			where_clause = 'web_analyse.id_earth > 0'
		elif isinstance(id_earth, basestring):
			event_types = ','.join(['"%s"' % et for et in id_earth.split(',')])
			join_clause.append(('JOIN', 'earthquakes',
					'web_analyse.id_earth = earthquakes.id_earth'
					' AND earthquakes.type in (%s)' % event_types))
		else:
			where_clause += ' AND web_analyse.id_earth = %d' % id_earth
		if id_com is not None:
			if isinstance(id_com, int):
				where_clause += ' AND web_analyse.id_com=%d' % id_com
			elif isinstance(id_com, (list, np.ndarray)):
				id_com_str = ','.join(['%s' % id for id in id_com])
				where_clause += ' AND web_analyse.id_com IN (%s)' % id_com_str
		elif zip_code:
			if isinstance(zip_code, (int, basestring)):
				where_clause += ' AND web_input.zip="%s"' % zip_code
			elif isinstance(zip_code, (list, np.ndarray)):
				zip_code_str = ','.join(['"%s"' % zip for zip in zip_code])
				where_clause += ' AND web_input.zip IN (%s)' % zip_code_str

	elif web_ids:
		where_clause = 'web_input.id_web in (%s)' % ','.join(['%d' % ID for ID in web_ids])
	else:
		## Fetch unassigned enquiries
		where_clause = 'web_analyse.id_earth = 0'
		if isinstance(id_earth, (basestring, datetime.date, np.datetime64)):
			## Interpret id_earth as a date
			from ..time import to_ymd_tuple
			year, month, day = to_ymd_tuple(id_earth)
			where_clause += ' AND web_input.time_year = %d' % year
			where_clause += ' AND web_input.time_month = %d' % month
			where_clause += ' AND web_input.time_day = %d' % day

	if errf !=None:
		errf.write("Querying KSB-ORB web macroseismic enquiries:\n")

	## Fetch records
	recs = query_seismodb_table(table_clause, column_clause=column_clause,
						join_clause=join_clause, where_clause=where_clause,
						verbose=verbose, errf=errf)

	return ROBDYFIEnsemble(id_earth, recs)


def get_num_online_macro_enquiries(id_earth, min_fiability=80, verbose=False):
	"""
	Count number of online macroseismic enquiries for a particular event.

	:param id_earth:
		int or list of ints, ID(s) of event in ROB catalog
	:param min_fiability:
		float, minimum fiability of enquiries
		(default: 80)
	:param verbose:
		bool, whether or not to print SQL query
		(default: False)

	:return:
		list of ints, number of enquiries for each earthquake
	"""
	## Convert input arguments, if necessary
	if isinstance(id_earth, (int, basestring)):
		id_earth = [id_earth]

	table_clause = ['earthquakes']
	## Note: apply constraints on web_analyse to the join clause
	## in order to get 0 if there are no entries in web_analyse for id_earth
	join_where_clause = 'web_analyse.id_earth=earthquakes.id_earth'
	if min_fiability:
		join_where_clause += ' AND web_analyse.m_fiability >= %.1f' % float(min_fiability)
	join_where_clause += ' AND web_analyse.deleted = false'
	join_clause = [('LEFT JOIN', 'web_analyse', join_where_clause)]

	column_clause = ['Count(web_analyse.id_earth) as num_enquiries']
	where_clause = 'earthquakes.id_earth in (%s)'
	where_clause %= ",".join([str(item) for item in id_earth])
	group_clause = 'earthquakes.id_earth'

	db_recs = query_seismodb_table(table_clause, column_clause, join_clause=join_clause,
								where_clause=where_clause, group_clause=group_clause,
								verbose=verbose)
	num_enquiries = [rec['num_enquiries'] for rec in db_recs]
	return num_enquiries


def get_num_official_enquiries(id_earth, verbose=False):
	"""
	Count number of official macroseismic enquiries for a particular
	earthquake. Note that this may not correspond to the number of MDPs
	(= records in 'macro_detail' table = evaluated enquiries?)

	:param id_earth:
		int or list of ints, ID(s) of event in ROB catalog
	:param verbose:
		bool, whether or not to print SQL query
		(default: False)

	:return:
		list of ints, number of enquiries for each earthquake
	"""
	## Convert input arguments, if necessary
	if isinstance(id_earth, (int, basestring)):
		id_earth = [id_earth]

	table_clause = 'macro_official_inquires'
	column_clause = ['Count(*) as num_enquiries']
	where_clause = 'id_earth in (%s)'
	where_clause %= ",".join([str(item) for item in id_earth])
	group_clause = 'id_earth'

	db_recs = query_seismodb_table(table_clause, column_clause,
								where_clause=where_clause, group_clause=group_clause,
								verbose=verbose)
	num_enquiries = [rec['num_enquiries'] for rec in db_recs]
	return num_enquiries


def get_num_traditional_mdps(id_earth, data_type='', min_fiability=80,
									verbose=False):
	"""
	Count number of traditional macroseismic data points for a particular
	earthquake

	:param id_earth:
		int or list of ints, ID(s) of event in ROB catalog
	:param data_type:
		str, type of macroseismic data: '', 'official' or 'historical'
		(default: '')
	:param min_fiability:
		float, minimum fiability of enquiry
		(default: 80)
	:param verbose:
		bool, whether or not to print SQL query
		(default: False)

	:return:
		list of ints, number of enquiries for each earthquake
	"""
	## Convert input arguments, if necessary
	if isinstance(id_earth, (int, basestring)):
		id_earth = [id_earth]

	## Construct SQL query
	table_clause = ['macro_detail']

	column_clause = ['Count(*) as num_enquiries']

	where_clause = 'macro_detail.id_earth in (%s)'
	where_clause %= ",".join([str(item) for item in id_earth])
	if min_fiability:
		where_clause += ' AND macro_detail.fiability >= %d' % min_fiability

	if data_type == 'historical':
		where_clause += (' AND CONCAT(macro_detail.id_earth, macro_detail.id_com)'
						' NOT IN (SELECT CONCAT(id_earth, id_com)'
						' FROM macro_official_inquires)')

	join_clause = []
	if data_type == 'official':
		join_clause.append(('RIGHT JOIN', 'macro_official_inquires',
			'macro_detail.id_com = macro_official_inquires.id_com'
			' AND macro_detail.id_earth = macro_official_inquires.id_earth'))

	group_clause = 'macro_detail.id_earth'

	db_recs = query_seismodb_table(table_clause, column_clause,
								where_clause=where_clause, join_clause=join_clause,
								group_clause=group_clause, verbose=verbose)
	num_enquiries = [rec['num_enquiries'] for rec in db_recs]
	return num_enquiries


def get_earthquakes_with_traditional_enquiries():
	"""

	Create catalog of earthquakes that have traditional (official or
	historical) enquiries associated with them.

	:return:
		instance of :class:`EQCatalog`
	"""
	table_clause = 'macro_official_inquires'
	column_clause = 'id_earth'
	group_clause = 'id_earth'
	db_recs = query_seismodb_table(table_clause, column_clause,
									group_clause=group_clause)
	eq_ids = [rec['id_earth'] for rec in db_recs]
	return query_local_eq_catalog_by_id(eq_ids)


def get_earthquakes_with_online_enquiries():
	"""
	Create catalog of earthquakes that have online enquiries
	associated with them.

	:return:
		instance of :class:`EQCatalog`
	"""
	table_clause = 'web_analyse'
	column_clause = 'id_earth'
	group_clause = 'id_earth'
	db_recs = query_seismodb_table(table_clause, column_clause,
									group_clause=group_clause)
	eq_ids = [rec['id_earth'] for rec in db_recs]
	return query_local_eq_catalog_by_id(eq_ids)


def query_historical_texts(id_earth, id_com, include_doubtful=False, verbose=False):
	"""
	Query historical database for texts corresponding to paritucular
	earthquake and commune

	:param id_earth:
		int, earthquake ID
	:param id_com:
		int, commune ID
	:param include_doubtful:
		bool, whether or not to include doubtful records
	:param verbose:
		bool, whether or not to print SQL query

	:return:
		list of dicts
	"""
	table_clause = 'historical_text'

	column_clause = ['id_commune AS id_com',
					'id_earth',
					'historical_text.id_text',
					'historical_text.id_source',
					'historical_text.id_historian',
					'historical_tp.doubtful',
					'historical_text.origin',
					'historical_text.text',
					'historical_text.translation',
					'historical_text.remark',
					'historical_place.id_place',
					'historical_place.name',
					'historical_source.name',
					'historical_source.date',
					'historical_source.edition',
					'historical_source.redaction_place',
					'historical_source.remark',
					'id_bibliography']

	join_clause = [('LEFT JOIN', 'historical_tp',
					'historical_text.id_text=historical_tp.id_text'),
					('JOIN', 'historical_place',
					'historical_tp.id_place=historical_place.id_place'),
					('JOIN', 'historical_source',
					'historical_text.id_source = historical_source.id_source')]

	where_clause = 'id_earth = %d AND id_commune = %d'
	where_clause %= (id_earth, id_com)
	if not include_doubtful:
		where_clause += ' AND doubtful = 0'

	return query_seismodb_table(table_clause, column_clause=column_clause,
								join_clause=join_clause, where_clause=where_clause,
								verbose=verbose)


def query_stations(network='UCC', activity_date_time=None, verbose=False):
	"""
	Query the ROB station database

	:param network:
		str, seismic network code
		(default: 'UCC')
	:param activity_date_time:
		datetime object, time of activity
		(default: None)
	:param verbose:
		bool, if True the query string will be echoed to standard output
		(default: False)

	:return:
		generator object, yielding a dictionary for each record
	"""
	table_clause = ["station", "station_place", "station_sismometer",
					"station_sismometer_type"]

	where_clause = "station.id_place=station_place.id_place"
	where_clause += " AND station.id_sismometer=station_sismometer.id_sismometer"
	where_clause += " AND station_sismometer.id_type=station_sismometer_type.id_type"
	if network:
		where_clause += " AND network='%s'" % network
	if activity_date_time:
		where_clause += " AND station.begin <= '%s'" % activity_date_time.isoformat()
		where_clause += (" AND (station.end >= '%s' OR station.end IS NULL)"
						% activity_date_time.isoformat())

	column_clause = ["id_station",
					"station_place.code",
					"code_sup",
					"station_place.international_code",
					"station_place.name",
					"station_place.country",
					"station.type",
					"network",
					"kind",
					"project",
					"station.begin",
					"station.end",
					"station_place.longitude",
					"station_place.latitude",
					"station_place.altitude",
					"station_sismometer_type.code as instrument_code",
					"station_sismometer_type.type as instrument_type",
					"station_sismometer_type.description",
					"station_sismometer_type.component_z",
					"station_sismometer_type.component_ns",
					"station_sismometer_type.component_ew"]

	return query_seismodb_table(table_clause, column_clause=column_clause,
				where_clause=where_clause, verbose=verbose)


def get_station_coordinates(station_codes, include_z=False, verbose=False):
	"""
	Quick and dirty extraction of station coordinates from database

	:param station_codes:
		str or list of strings, station code(s) (either 3 or 4 characters)
	:param verbose:
		bool, if True the query string will be echoed to standard output
		(default: False)

	:return:
		(lons, lats, [altitudes]) tuple or
		(lon, lat, [altitude]) tuple if :param:`station_codes` is not a list
		- lon, lat: geographic coordinates in degrees
		- altitude: altitude in m
	"""
	is_list = True
	if not isinstance(station_codes, list):
		station_codes = [station_codes]
		is_list = False

	table_clause = 'station'
	code_lengths = set([len(code) for code in station_codes])
	if sorted(code_lengths)[-1] > 3:
		column_clause = ["CONCAT(station_place.code, station.code_sup) AS station_code"]
	else:
		column_clause = ["station_place.code AS station_code"]
	column_clause += ["IFNULL(station.longitude, station_place.longitude) AS longitude",
					"IFNULL(station.latitude, station_place.latitude) AS latitude"]
	if include_z:
		column_clause.append("IFNULL(station.altitude, station_place.altitude) AS altitude")
	having_clause = 'station_code in (%s)'
	having_clause %= ','.join(['"%s"' % code for code in station_codes])
	join_clause = [('LEFT JOIN', 'station_place',
					'station.id_place = station_place.id_place')]
	recs = query_seismodb_table(table_clause, column_clause=column_clause,
								having_clause=having_clause, join_clause=join_clause,
								verbose=verbose)

	## Note that order may not be preserved in query result
	recs = {rec['station_code']: rec for rec in recs}
	lons = [recs[code]['longitude'] for code in station_codes]
	lats = [recs[code]['latitude'] for code in station_codes]
	if include_z:
		altitudes = [recs[code]['altitude'] for code in station_codes]
	if is_list:
		if include_z:
			return (lons, lats, altitudes)
		else:
			return (lons, lats)
	else:
		if include_z:
			return (lons[0], lats[0], altitudes[0])
		else:
			return (lons[0], lats[0])


def get_station_catalog(station_code, ignore_code_sup=False, verbose=False):
	"""
	Fetch catalog of earthquakes that have been measured using a
	particular station

	:param station_code:
		str, 3- or 4-character station code
		(default: None)
	:param ignore_code_sup:
		bool, whether or not to ignore supplemantary station code
		(4th char)
		(default: False)
	:param verbose:
		bool, if True the query string will be echoed to standard output
		(default: False)

	:return:
		instance of :class:`EQCatalog`
	"""
	table_clause = 'mesure_t'
	column_clause = 'id_earth'
	join_clause = [('LEFT JOIN', 'stations_network',
					'mesure_t.id_eq = stations_network.id_station')]
	if ignore_code_sup:
		where_clause = 'code = "%s"' % station_code[:3]
	else:
		where_clause = 'CONCAT(code, code_sup) = "%s"' % station_code
	recs = query_seismodb_table(table_clause, column_clause=column_clause,
							where_clause=where_clause, join_clause=join_clause,
							verbose=verbose)
	eq_ids = [r['id_earth'] for r in recs]
	catalog = query_local_eq_catalog_by_id(eq_ids, verbose=verbose)
	catalog.name = 'Catalog measured at station %s' % station_code
	return catalog


def query_phase_picks(id_earth, station_code=None, network=None, verbose=False):
	"""
	Query phase picks from database

	:param id_earth:
		int or str, earthquake ID
	:param station_code:
		str, 3- or 4-character station code
		(default: None)
	:param network:
		str, network code
		(default: None)
	:param verbose:
		bool, if True the query string will be echoed to standard output
		(default: False)

	:return:
		generator object, yielding a dictionary for each record
	"""
	table_clause = "mesure_t"
	where_clause = "id_earth = %s" % id_earth
	column_clause = ["CONCAT(station_place.code, station.code_sup) AS station_code"]
	column_clause += ['mesure_t.id_mesure_t', 'id_earth', 'comp', 'movement',
					'TIMESTAMP(date, time) AS datetime', 'hund',
					'include_in_loc', 'periode', 'amplitude', 'distance',
					'magnitude', 'mag_type', 'lookup_phase.name', 'network']
	if station_code:
		station_code, code_sup = station_code[:3], station_code[3:]
		where_clause += ' AND station_place.code = "%s"' % station_code
		if code_sup:
			where_clause += ' AND station.code_sup = "%s"' % code_sup
	if network:
		where_clause += ' AND network = "%s"' % network

	join_clause = [('LEFT JOIN', 'mesure_a',
					'mesure_t.id_mesure_t = mesure_a.id_mesure_t'),
					('LEFT JOIN', 'lookup_phase',
					'mesure_t.id_phase = lookup_phase.id_phase'),
					('LEFT JOIN', 'station',
					'mesure_t.id_eq = station.id_station'),
					('LEFT JOIN', 'station_place',
					'station.id_place = station_place.id_place')]

	return query_seismodb_table(table_clause, column_clause=column_clause,
								where_clause=where_clause, join_clause=join_clause,
								verbose=verbose)


def get_last_earthID():
	"""
	Return ID of most recently added earthquake in database

	:return:
		int, earthquake ID
	"""
	query = ('SELECT id_earth FROM earthquakes WHERE type="ke" AND is_true = 1 '
			'ORDER BY id_earth DESC LIMIT 0 , 1')
	id_earth = query_seismodb_table_generic(query)[0]['id_earth']
	return id_earth


def zip2ID(zip_code):
	"""
	Look up ID corresponding to ZIP code in database

	:return:
		int, commune ID
	"""
	query = 'SELECT id FROM communes WHERE code_p = %d' % zip_code
	id_com = query_seismodb_table_generic(query)[0]['id']
	return id_com


def get_communes_legacy(country='BE', main_communes=False, id_com=None):
	"""
	Fetch communes from database

	:param country:
		2-char string, country code
		(default: 'BE')
	:param main_communes:
		bool, whether or not to get main communes only
		(default: False)
	:param id_com:
		int, str or list: commune ID(s),
		mutually exclusive with :param:`main_communes`
		(default: None)

	:return:
		list of dicts
	"""
	if isinstance(id_com, (int, np.integer, basestring)):
		id_com = [id_com]

	table_clause = 'communes'
	where_clause = []
	if country:
		where_clause += ['country = "%s"' % country]
	if not (id_com is None or id_com in ([], '')):
		where_clause += ['id in (%s)' % ",".join([str(item) for item in id_com])]
	elif main_communes:
		where_clause += ['id = id_main']
	return query_seismodb_table(table_clause, where_clause=where_clause)


def get_subcommunes_legacy(id_main):
	"""
	Return subcommune records for particular main commune

	:param id_main:
		int, main commune ID

	:return:
		list of dicts
	"""
	table_clause = 'communes'
	where_clause = 'id_main = %d' %  id_main
	return query_seismodb_table(table_clause, where_clause=where_clause)


def _get_com_zip_table_clause(countries=[], average_multiple_zips=False):
	"""
	Construct SQL query to extract information (id_com, zip, name, lon, lat)
	from com_zip_* database tables

	:param countries:
		list of strings, country codes
		(default: [])
	:param average_multiple_zips:
		bool, whether to average the lon, lat coordinates if multiple ZIPs are
		associated with 1 id_com, or else return thoe of the first entry
		(default: False)

	:return:
		str, SQL query
	"""
	if len(countries) == 0:
		countries = ['BE', 'DE', 'FR', 'LU', 'GB', 'NL']

	if average_multiple_zips:
		lon_col = 'AVG(longitude) AS longitude'
		lat_col = 'AVG(latitude) AS latitude'
	else:
		lon_col, lat_col = 'longitude', 'latitude'

	sql = []
	if 'BE' in countries:
		sql.append('SELECT id_com, zip, city, %s, %s FROM com_zip_BE_fr GROUP BY id_com'
						% (lon_col, lat_col))
	if 'DE' in countries:
		sql.append('SELECT id_com, zip, commune AS city, %s, %s FROM com_zip_DE GROUP BY id_com'
						% (lon_col, lat_col))
	if 'FR' in countries:
		sql.append('SELECT id_com, zip, commune AS city, %s, %s FROM com_zip_FR GROUP BY id_com'
						% (lon_col, lat_col))
	if 'LU' in countries:
		sql.append('SELECT id_com, zip, city, %s, %s FROM com_zip_LU_fr GROUP BY id_com'
						% (lon_col, lat_col))
	if 'GB' in countries:
		sql.append('SELECT id_com, zip, city, %s, %s FROM com_zip_GB GROUP BY id_com'
						% (lon_col, lat_col))
	if 'NL' in countries:
		sql.append('SELECT id_com, zip, city, %s, %s FROM com_zip_NL GROUP BY id_com'
						% (lon_col, lat_col))

	sql = ' UNION '.join(sql)

	sql = '(%s) AS com_zip' % sql

	return sql


def get_communes(country='BE', main_communes=False, id_com=None, verbose=False):
	"""
	More powerful version of :func:`get_communes_legacy` to fetch communes from
	the database, but overriding the lon/lat coordinates with those from the
	com_zip_XX tables

	:param country:
		2-char string, country code
		(default: 'BE')
	:param main_communes:
		bool, whether or not to get main communes only
		(default: False)
	:param id_com:
		int, str or list: commune ID(s),
		mutually exclusive with :param:`main_communes`
		(default: None)
	:param verbose:
		bool, whether or not to print the SQL query
		(default: False)

	:return:
		list of dicts
	"""
	if isinstance(id_com, (int, np.integer, basestring)):
		id_com = [id_com]

	table_clause = 'communes'
	column_clause = ['communes.id', 'id_main', 'code_p', 'id_province', 'border_ref',
						'IF(com_zip.latitude, com_zip.latitude, communes.latitude) as latitude',
						'IF(com_zip.longitude, com_zip.longitude, communes.longitude) as longitude',
						'language', 'country', 'name']
	where_clause = []
	if country:
		where_clause += ['country = "%s"' % country]
		com_zip_table = 'com_zip_%s' % country
		countries = [country]
	else:
		countries = []
	join_clause = [('LEFT JOIN',
						_get_com_zip_table_clause(countries, average_multiple_zips=False),
						'communes.id = com_zip.id_com')]
						#  AND communes.code_p = com_zip.zip
	if not (id_com is None or id_com in ([], '')):
		where_clause += ['communes.id in (%s)' % ",".join([str(item) for item in id_com])]
	elif main_communes:
		where_clause += ['communes.id = id_main']

	return query_seismodb_table(table_clause, column_clause,
									where_clause=where_clause, join_clause=join_clause,
									verbose=verbose)


def get_subcommune_ids(id_main, country='BE', verbose=False):
	"""
	Return subcommune IDs for particular main commune

	:param id_main:
		int, main commune ID
	:param country:
		2-char string, country code
		(default: 'BE')
	:param verbose:
		bool, whether or not to print the SQL query
		(default: False)

	:return:
		list of ints
	"""
	column_clause = ['id']
	table_clause = 'communes'
	where_clause = ['id_main = %d' % id_main]
	if country:
		where_clause.append('country = "%s"' % country)
	recs = query_seismodb_table(table_clause, where_clause=where_clause,
										verbose=verbose)
	return [rec['id'] for rec in recs]


def get_subcommunes(id_main, country='BE', verbose=False):
	"""
	Return subcommune records for particular main commune

	:param id_main:
	:param country:
	:param verbose
		see :func:`get_subcommune_ids`

	:return:
		list of dicts
	"""
	if not country:
		table_clause = 'communes'
		column_clause = ['country']
		where_clause = 'id = %d' %  id_main
		try:
			[rec] = query_seismodb_table(table_clause, where_clause=where_clause,
												column_clause=column_clause,
												verbose=verbose)
		except:
			raise
		else:
			country = rec['country']

	subcommune_ids = get_subcommune_ids(id_main, country=country,
												verbose=verbose)

	if subcommune_ids:
		return get_communes(country=country, id_com=subcommune_ids,
								verbose=verbose)
	else:
		return []


# TODO: historical texts
"""
SELECT	c2.id_com,
       c2.commune_name,
       COUNT(distinct ht.id_text) AS cpt,
       GROUP_CONCAT(distinct ht.id_text) AS id_text
FROM (
   SELECT 	c.id 			AS id_com,
           c.name 			AS commune_name,
           c.country,
           c.id_main 		AS cmain
     FROM 	macro_detail 	AS md,
           communes 		AS c
    WHERE 	md.id_com 	= c.id
      AND 	md.id_earth = 89
) AS c2
LEFT JOIN 	communes 			AS c3	ON (c2.id_com 	= c3.id 		OR c2.id_com 	= c3.id_main)
LEFT JOIN 	historical_place	AS hp 	ON c3.id 		= hp.id_commune
LEFT JOIN 	historical_tp 		AS htp 	ON hp.id_place 	= htp.id_place
LEFT JOIN 	historical_text 	AS ht 	ON htp.id_text 	= ht.id_text
   WHERE 	c3.id IS NOT NULL
     AND 	ht.id_earth 	= 89
     AND 	ht.validation 	= 1
     AND 	ht.public_data 	= 1
GROUP BY 	c2.id_com
ORDER BY 	c2.country , c2.commune_name
"""



if __name__ == "__main__":
	start_date = datetime.date(1983, 1,1)
	end_date = datetime.date(2007, 12, 31)
	catalogue_length = (end_date.year - start_date.year) * 1.0
	catalogue = query_local_eq_catalog(start_date=start_date, end_date=end_date)
	print("%d events" % len(catalogue))
