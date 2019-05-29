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
			"get_last_earthID", "query_focal_mechanisms",
			"query_online_macro_catalog", "query_traditional_macro_catalog",
			"query_official_macro_catalog", "query_historical_macro_catalog",
			"query_traditional_macro_catalog_aggregated",
			"query_official_macro_catalog_aggregated",
			"query_historical_macro_catalog_aggregated",
			"query_historical_texts",
			"query_online_macro_catalog_aggregated",
			"get_num_online_macro_enquiries",
			"get_num_official_enquiries",
			"get_earthquakes_with_official_enquiries",
			"get_earthquakes_with_online_enquiries",
			"query_stations", "get_station_coordinates",
			"query_phase_picks",
			"zip2ID", "get_communes", "get_subcommunes"]


# Note: median is not supported by MySQL
AGG_FUNC_DICT = {"average": "AVG", "mean": "AVG",
				"minimum": "MIN", "maximum": "MAX"}


def query_seismodb_table_generic(query, verbose=False, print_table=False, errf=None):
	"""
	Query MySQL table using clause components, returning each record as a dict

	:param query:
		str, SQL query
	:param verbose:
		bool, whether or not to print the query (default: False)
	:param print_table:
		rather than returning records
		(default: False)
	:param errf:
		file object, where to print errors

	:return:
		generator object, yielding a dictionary for each record
	"""
	## Database information
	from seismodb_secrets import host, user, passwd, database
	try:
		from seismodb_secrets import port
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
						event_type="ke", null_value=np.nan, verbose=False, errf=None):
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
		or "mag" (= "size") (default: "date")
	:param sort_order:
		String, sort order, either "asc" or "desc" (default: "asc")
	:param event_type:
		str, event type (one of "all", "cb", "ex", "ke", "ki", "km",
		"kr", "kx", "qb", "sb", "se", "si", "sm", "sr", "sx" or "uk")
		(default: "ke" = known earthquakes)
	:param null_value:
		float, value to use for NULL values (except magnitude)
		(default: np.nan)
	:param verbose:
		Bool, if True the query string will be echoed to standard output
	:param errf:
		File object, where to print errors

	:return:
		instance of :class:`EQCatalog`
	"""
	from .eqrecord import (ROBLocalEarthquake, DEFAULT_MRELATIONS)
	from ..eqcatalog import EQCatalog
	from .. import time_functions_np as tf

	## Convert input arguments, if necessary
	if isinstance(id_earth, (int, basestring)):
		id_earth = [id_earth]

	if not id_earth:
		## Do not impose start or end date if id_earth is given
		if isinstance(start_date, int):
			start_date = '%d-01-01' % start_date
		elif start_date is None:
			start_date = '1350-01-01'
		start_date = tf.as_np_datetime(start_date)

		if isinstance(end_date, int):
			end_date = '%d-12-31T23:59:59.999999' % end_date
		elif end_date is None:
			end_date = np.datetime64('now')
		end_date = tf.as_np_datetime(end_date)
		## If start_date and end_date are the same,
		## set time of end date to end of the day
		if start_date and end_date and end_date - start_date == np.timedelta64(0):
			end_time = datetime.time(23, 59, 59, 999999)
			end_date = tf.combine_np_date_and_py_time(end_date, end_time)

	## Construct SQL query
	table_clause = "earthquakes"
	column_clause = [
		'id_earth',
		'DATE_FORMAT(date, "%Y-%m-%d") as date',
		'TIME_FORMAT(time, "%H:%i:%s") as time',
		'longitude',
		'latitude',
		'depth',
		'ML',
		'MS',
		'MW',
		'MWH',
		'IF(MWH, MWH, IF(MW, MW, IF(MS, MS, ML))) as M',
		'intensity_max',
		'macro_radius',
		'errh',
		'errz',
		'errt',
		'errM',
		'name',
		'type']

	where_clause = ""
	having_clause = ""
	if not (id_earth is None or id_earth in ([], '')):
		where_clause += 'id_earth in (%s)' % ",".join([str(item) for item in id_earth])
	else:
		where_clause += 'is_true = 1'
		if event_type != "all":
			where_clause += ' and type = "%s"' % event_type
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

		if Mmax or Mmin != None:
			if Mmin is None:
				Mmin = 0.0
			if not Mmax:
				Mmax = 10.0
			having_clause += 'HAVING M BETWEEN %f and %f' % (Mmin, Mmax)

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
		errf.write("Querying KSB-ORB local earthquake catalog:\n")

	## Fetch records
	eq_list = []
	for rec in query_seismodb_table(table_clause, column_clause=column_clause,
					where_clause=where_clause, having_clause=having_clause,
					order_clause=order_clause, verbose=verbose, errf=errf):
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

		## Skip records without lon,lat, depth and magnitude
		if lon == lat == depth == M == None:
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
		time = datetime.time(hour, minutes, seconds)

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

		eq = ROBLocalEarthquake(id_earth, date, time, lon, lat, depth, {},
							ML, MS, MW, mb, name, intensity_max, macro_radius,
							errh, errz, errt, errM, agency="ROB", event_type=etype)
		eq_list.append(eq)

	name = "ROB Catalog %s - %s" % (start_date, end_date)
	start_date, end_date = np.datetime64(start_date), np.datetime64(end_date)
	return EQCatalog(eq_list, start_date, end_date, region=region, name=name,
					default_Mrelations=DEFAULT_MRELATIONS)


def query_local_eq_catalog_by_id(id_earth, verbose=False, errf=None):
	return query_local_eq_catalog(id_earth=id_earth, verbose=verbose, errf=errf)


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
		bool, whether or not to group the results by main village
		(default: False)
	:param min_fiability:
		float, minimum fiability of enquiry
		(default: 80)
	:param verbose:
		Bool, if True the query string will be echoed to standard output
	:param errf:
		File object, where to print errors

	:return:
		dict, mapping commune IDs (if :param:`id_earth` is not None) or
		earthquake IDs (if :param:`id_earth` is None) to lists of
		database records (1 or more depending on :param:`group_by_main_commune`)
	"""
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
		'communes.longitude',
		'communes.latitude']

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
		where_clause += (' AND CONCAT(id_earth, id_com) NOT IN'
						' (SELECT CONCAT(id_earth, id_com)'
						' FROM macro_official_inquires)')

	join_clause = [('LEFT JOIN', 'communes', 'macro_detail.id_com = communes.id')]
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

	## Construct result dictionary
	macro_rec_dict = {}
	for rec in macro_recs:
		## Convert None Imin/Imax values to np.nan
		rec['Imin'] = rec['Imin'] or np.nan
		rec['Imax'] = rec['Imax'] or np.nan

		if id_earth is not None:
			if group_by_main_commune:
				key = rec['id_main']
			else:
				key = rec['id_com']
		else:
			key = rec['id_earth']

		if not key in macro_rec_dict:
			macro_rec_dict[key] = [rec]
		else:
			macro_rec_dict[key].append(rec)

	return macro_rec_dict


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
					min_or_max='max', group_by_main_commune=False, agg_method="mean",
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
	:param min_or_max:
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
		instance of :class:`MacroInfoCollection`
	"""
	from ..macro.macro_info import MacroseismicInfo, MacroInfoCollection

	assert not (id_earth is None and
				(id_com is None or isinstance(id_com, (list, np.ndarray))))

	if id_earth == 18280223:
		# TODO: check how this can be integrated in seismodb
		from eqmapperdb_secrets import (host as _host, user as _user,
									passwd as _passwd, database as _database)

		query = 'SELECT id_com_main as "id_com",'
		query += ' Longitude as "longitude", Latitude as "latitude",'
		query += ' id as "id_db",'
		if min_or_max == 'max':
			query += ' Imax'
		elif min_or_max == 'min':
			query += ' Imin'
		elif min_or_max == 'mean':
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
			if min_or_max == 'min':
				intensity_col = '%s(%s) AS "Intensity"'
				intensity_col %= (agg_function, Imin_clause)
			elif min_or_max == 'max':
				intensity_col = '%s(%s) AS "Intensity"'
				intensity_col %= (agg_function, Imax_clause)
			elif min_or_max == 'mean':
				intensity_col = '%s((%s + %s)/2.) AS "Intensity"'
				intensity_col %= (agg_function, Imin_clause, Imax_clause)
			column_clause.append(intensity_col)
			group_clause = 'communes.id_main'
			column_clause.append('GROUP_CONCAT(id_macro_detail SEPARATOR ",") AS id_db')
		else:
			agg_method = None
			if min_or_max == 'min':
				intensity_col = '%s AS "Intensity"' % Imin_clause
			elif min_or_max == 'max':
				intensity_col = '%s AS "Intensity"' % Imax_clause
			elif min_or_max == 'mean':
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
		macro_info = MacroseismicInfo(id_earth, id_com, I, agg_type,
									enq_type, num_replies=num_replies,
									lon=lon, lat=lat, db_ids=db_ids)
		macro_infos.append(macro_info)

	proc_info = dict(agg_method=agg_method, min_fiability=min_fiability,
					min_or_max=min_or_max)
	macro_info_coll = MacroInfoCollection(macro_infos, agg_type, enq_type,
										proc_info=proc_info)

	return macro_info_coll


def query_official_macro_catalog_aggregated(id_earth, id_com=None, min_or_max='max',
					group_by_main_commune=False, agg_method="mean",
					min_fiability=80, verbose=False, errf=None):
	"""
	Query ROB catalog of official communal macroseismic enquiries
	This is a wrapper for :func:`query_traditional_macro_catalog`
	"""
	kwargs = locals().copy()
	return query_traditional_macro_catalog_aggregated(data_type='official', **kwargs)


def query_historical_macro_catalog_aggregated(id_earth, id_com=None, min_or_max='max',
					group_by_main_commune=False, agg_method="mean",
					min_fiability=80, verbose=False, errf=None):
	"""
	Query ROB catalog of historical macroseismic data
	This is a wrapper for :func:`query_traditional_macro_catalog`
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
		instance of :class:`MacroInfoCollection`
	"""
	from ..macro.macro_info import MacroseismicInfo, MacroInfoCollection

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
		macro_info = MacroseismicInfo(id_earth, id_com, I, agg_type,
									'internet', num_replies=num_replies,
									lon=lon, lat=lat, db_ids=web_ids)
		macro_infos.append(macro_info)

	proc_info = dict(min_replies=min_replies, min_fiability=min_fiability,
					agg_method=agg_method, filter_floors=filter_floors)

	macro_info_coll = MacroInfoCollection(macro_infos, agg_type, 'internet',
										proc_info=proc_info)

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
	from ..macro.dyfi import MacroseismicEnquiryEnsemble

	table_clause = ['web_input']

	column_clause = ['web_input.*', 'web_analyse.*',
					'web_location.longitude', 'web_location.latitude',
					'web_location.quality AS location_quality']

	join_clause = [('JOIN', 'web_analyse', 'web_input.id_web=web_analyse.id_web'),
					('LEFT JOIN', 'web_location', 'web_input.id_web=web_location.id_web'
								' AND web_location.quality >= %d' % min_location_quality)]

	if isinstance(id_earth, (int, basestring)):
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
			if isinstance(zip_code, int):
				where_clause += ' AND web_input.zip=%d' % zip_code
			elif isinstance(zip_code, (list, np.ndarray)):
				zip_code_str = ','.join(['%s' % zip for zip in zip_code])
				where_clause += ' AND web_input.zip IN (%s)' % zip_code_str

	elif web_ids:
		where_clause = 'web_input.id_web in (%s)' % ','.join(['%d' % ID for ID in web_ids])
	else:
		## Fetch unassigned enquiries
		where_clause = 'web_analyse.id_earth = 0'
		if isinstance(id_earth, (str, datetime.date, np.datetime64)):
			## Interpret id_earth as a date
			from ..time_functions_np import to_ymd_tuple
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

	return MacroseismicEnquiryEnsemble(id_earth, recs)


def get_num_online_macro_enquiries(id_earth, min_fiability=80):
	"""
	Count number of online macroseismic enquiries for a particular event.

	:param id_earth:
		int or list of ints, ID(s) of event in ROB catalog
	:param min_fiability:
		float, minimum fiability of enquiries
		(default: 80)

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
	join_where_clause += ' AND web_analyse.m_fiability >= %.1f' % float(min_fiability)
	join_where_clause += ' AND web_analyse.deleted = false'
	join_clause = [('LEFT JOIN', 'web_analyse', join_where_clause)]

	column_clause = ['Count(web_analyse.id_earth) as num_enquiries']
	where_clause = 'earthquakes.id_earth in (%s)'
	where_clause %= ",".join([str(item) for item in id_earth])
	group_clause = 'earthquakes.id_earth'

	db_recs = query_seismodb_table(table_clause, column_clause, join_clause=join_clause,
								where_clause=where_clause, group_clause=group_clause)
	num_enquiries = [rec['num_enquiries'] for rec in db_recs]
	return num_enquiries


def get_num_official_enquiries(id_earth):
	"""
	Count number of official macroseismic enquiries for a particular
	earthquake

	:param id_earth:
		int or list of ints, ID(s) of event in ROB catalog

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
								where_clause=where_clause, group_clause=group_clause)
	num_enquiries = [rec['num_enquiries'] for rec in db_recs]
	return num_enquiries


def get_earthquakes_with_official_enquiries():
	"""
	Create catalog of earthquakes that have official enquiries
	associated with them.

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


def get_station_coordinates(station_codes):
	"""
	Quick and dirty extraction of station coordinates from database

	:param station_codes:
		list of strings, station codes (note that these are place codes,
		not instrument codes)

	:return:
		(lons, lats) tuple
	"""
	table_clause = "station_place"
	where_clause = 'code in (%s)'
	where_clause %= ','.join(['"%s"' % code for code in station_codes])
	recs = query_seismodb_table(table_clause, where_clause=where_clause)
	## Note that order may not be preserved in query result
	recs = {rec['code']: rec for rec in recs}
	lons = [recs[code]['longitude'] for code in station_codes]
	lats = [recs[code]['latitude'] for code in station_codes]
	return (lons, lats)


def query_phase_picks(id_earth, station_code=None, verbose=False):
	"""
	Query phase picks from database

	:param id_earth:
		int or str, earthquake ID
	:param station_code:
		str, station code
		(default: None)
	:param verbose:
		bool, if True the query string will be echoed to standard output
		(default: False)

	:return:
		generator object, yielding a dictionary for each record
	"""
	table_clause = "mesure_t"
	where_clause = "id_earth = %s" % id_earth
	if station_code:
		station_code, code_sup = station_code[:3], station_code[3:]
		where_clause += ' AND stations_network.code = "%s"' % station_code
		if code_sup:
			where_clause += ' AND stations_network.code_sup = "%s"' % code_sup
	join_clause = [('LEFT JOIN', 'mesure_a',
					'mesure_t.id_mesure_t = mesure_a.id_mesure_t'),
					('LEFT JOIN', 'lookup_phase',
					'mesure_t.id_phase = lookup_phase.id_phase'),
					('LEFT JOIN', 'stations_network',
					'mesure_t.id_eq = stations_network.id_station')]

	return query_seismodb_table(table_clause, where_clause=where_clause,
								join_clause=join_clause, verbose=verbose)


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


def get_communes(country='BE', main_communes=False):
	"""
	Fetch communes from database

	:param country:
		2-char string, country code
		(default: 'BE')
	:param main_communes:
		bool, whether or not to get main communes only

	:return:
		list of dicts
	"""
	table_clause = 'communes'
	where_clause = 'country = "%s"' % country
	if main_communes:
		where_clause += ' AND id = id_main'
	return query_seismodb_table(table_clause, where_clause=where_clause)


def get_subcommunes(id_main):
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



if __name__ == "__main__":
	start_date = datetime.date(1983, 1,1)
	end_date = datetime.date(2007, 12, 31)
	catalogue_length = (end_date.year - start_date.year) * 1.0
	catalogue = query_local_eq_catalog(start_date=start_date, end_date=end_date)
	print("%d events" % len(catalogue))
