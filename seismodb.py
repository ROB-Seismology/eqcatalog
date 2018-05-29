# -*- coding: iso-Latin-1 -*-

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


## Import standard python modules
import datetime

## Import third-party modules
import numpy as np

## Import ROB modules
from db.simpledb import (build_sql_query, query_mysql_db_generic)

#import eqrecord
#reload(eqrecord)
from eqrecord import LocalEarthquake, MacroseismicRecord, FocMecRecord

#import eqcatalog
#reload(eqcatalog)
from eqcatalog import EQCatalog



__all__ = ["LocalEarthquake", "MacroseismicRecord", "FocMecRecord",
			"query_ROB_LocalEQCatalog", "query_ROB_LocalEQCatalogByID",
			"query_ROB_FocalMechanisms", "query_ROB_Official_MacroCatalog",
			"query_ROB_Web_MacroCatalog", "get_last_earthID", "query_ROB_Web_enquiries"]


def query_seismodb_table_generic(query, verbose=False, errf=None):
	"""
	Query MySQL table using clause components, returning each record as a dict

	:param query:
		str, SQL query
	:param verbose:
		bool, whether or not to print the query (default: False)
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

	return query_mysql_db_generic(database, host, user, passwd, query, port=port,
									verbose=verbose, errf=errf)


def query_seismodb_table(table_clause, column_clause="*", where_clause="",
						having_clause="", order_clause="", group_clause="",
						verbose=False, errf=None):
	"""
	Query seismodb table using clause components

	:param table_clause:
		str or list of strings, name(s) of database table(s)
	:param column_clause:
		str or list of strings, column clause or list of columns (default: "*")
	:param where_clause:
		str, where clause (default: "")
	:param having_clause:
		str, having clause (default: "")
	:param order_clause:
		str, order clause (default: "")
	:param group_clause:
		str, group clause (default: "")
	:param verbose:
		bool, whether or not to print the query (default: False)
	:param errf:
		file object, where to print errors

	:return:
		generator object, yielding a dictionary for each record
	"""
	query = build_sql_query(table_clause, column_clause, where_clause,
							having_clause, order_clause, group_clause)

	return query_seismodb_table_generic(query, verbose=verbose, errf=errf)


def query_ROB_LocalEQCatalog(region=None, start_date=None, end_date=None,
						Mmin=None, Mmax=None, min_depth=None, max_depth=None,
						id_earth=None, sort_key="date", sort_order="asc",
						event_type="ke", convert_NULL=True, verbose=False, errf=None):
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
	:param convert_NULL:
		Bool, whether or not to convert NULL values to zero values
		(default: True)
	:param verbose:
		Bool, if True the query string will be echoed to standard output
	:param errf:
		File object, where to print errors

	:return:
		instance of :class:`EQCatalog`
	"""
	## Convert input arguments, if necessary
	if isinstance(id_earth, (int, long)):
		id_earth = [id_earth]

	if isinstance(start_date, int):
		start_date = datetime.date(start_date, 1, 1)
	elif isinstance(start_date, datetime.datetime):
		start_date = start_date.date()
	if not start_date:
		start_date = datetime.date(1350, 1, 1)

	if isinstance(end_date, int):
		end_date = datetime.date(end_date, 12, 31)
	elif isinstance(end_date, datetime.datetime):
		end_date = end_date.date()
	if not end_date:
		end_date = datetime.datetime.now().date()

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
		'IF(MW, MW, IF(MS, MS, ML)) as M',
		'intensity_max',
		'macro_radius',
		'errh',
		'errz',
		'errt',
		'errM',
		'name',
		'type']

	where_clause = ""
	if id_earth:
		where_clause += 'id_earth in (%s)' % ",".join([str(item) for item in id_earth])
	else:
		where_clause += 'is_true = 1'
		if event_type != "all":
			where_clause += ' and type = "%s"' % event_type
	if region:
		w, e, s, n = region
		where_clause += ' and longitude Between %f and %f' % (w, e)
		where_clause += ' and latitude Between %f and %f' % (s, n)
	if start_date or end_date:
		if not end_date:
			end_date = datetime.datetime.now()
		if not start_date:
			start_date = datetime.datetime(100, 1, 1)
		where_clause += ' and date Between "%s" and "%s"' % (start_date.isoformat(), end_date.isoformat())
	if min_depth:
		where_clause += ' and depth >= %f' % min_depth
	if max_depth:
		where_clause += ' and depth <= %f' % max_depth

	having_clause = ""
	if Mmax or Mmin != None:
		if Mmin is None:
			Mmin = 0.0
		if not Mmax:
			Mmax = 10.0
		having_clause += 'HAVING M Between %f and %f' % (Mmin, Mmax)

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
		id_earth, name = rec["id_earth"], rec["name"]
		date, time = rec["date"], rec["time"]
		lon, lat = rec["longitude"], rec["latitude"]
		depth = rec["depth"]
		ML, MS, MW, M = rec["ML"], rec["MS"], rec["MW"], rec["M"]
		intensity_max, macro_radius = rec["intensity_max"], rec["macro_radius"]
		errh, errz, errt, errM = rec["errh"], rec["errz"], rec["errt"], rec["errM"]
		etype = rec["type"]  ## Avoid conflict with event_type parameter!

		if name == lon == lat == depth == M == None:
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

		eq = LocalEarthquake(id_earth, date, time, lon, lat, depth, {}, ML, MS, MW, mb, name, intensity_max, macro_radius, errh, errz, errt, errM, event_type=etype)
		eq_list.append(eq)

	name = "ROB Catalog %s - %s" % (start_date.isoformat(), end_date.isoformat())
	return EQCatalog(eq_list, start_date, end_date, region=region, name=name)


def query_ROB_LocalEQCatalogByID(id_earth, verbose=False, errf=None):
	return query_ROB_LocalEQCatalog(id_earth=id_earth, verbose=verbose, errf=errf)


def query_ROB_FocalMechanisms(region=None, start_date=None, end_date=None,
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
	if type(id_earth) in (type(1), type(1L)):
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
		where_clause += ' and earthquakes.id_earth in (%s)' % ",".join([str(item) for item in id_earth])
	if region:
		w, e, s, n = region
		where_clause += ' and earthquakes.longitude Between %f and %f' % (w, e)
		where_clause += ' and earthquakes.latitude Between %f and %f' % (s, n)
	if start_date or end_date:
		if not end_date:
			end_date = datetime.datetime.now()
		if not start_date:
			start_date = datetime.datetime(100, 1, 1)
		where_clause += ' and earthquakes.date Between "%s" and "%s"' % (start_date.isoformat(), end_date.isoformat())

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

		eq = FocMecRecord(id_earth, date, time, lon, lat, depth, {}, ML, MS, MW, strike, dip, rake, name, intensity_max, macro_radius)
		focmec_list.append(eq)

	return focmec_list


def query_ROB_Official_MacroCatalog(id_earth, Imax=True, min_val=1,
					group_by_main_village=False, agg_function="maximum",
					verbose=False, errf=None):
	"""
	Query ROB "official" macroseismic catalog (= commune inquiries)

	:param id_earth:
		int, earthquake ID
	:param Imax:
		bool, if True, intensity_max is returned, else intensity_min
		(default: True)
	:param min_val:
		float, minimum intensity to return
		(default: 1)
	:param group_by_main_village:
		bool, whether or not to aggregate the results by main village
		(default: False)
	:param agg_function:
		str, aggregation function to use if :param:`group_by_main_village`
		is True, one of "minimum", "maximum" or "average"
		(default: "maximum")
	:param verbose:
		Bool, if True the query string will be echoed to standard output
	:param errf:
		File object, where to print errors

	:return:
		dict mapping commune IDs to instances of :class:`MacroseismicRecord`
	"""

	if id_earth == 18280223:
		# TODO: check how this can be integrated in seismodb
		import eqmapperdb_secrets
		db = MySQLdb.connect(host=eqmapperdb_secrets.host, user=eqmapperdb_secrets.user, passwd=eqmapperdb_secrets.passwd, db=eqmapperdb_secrets.database)
		c = db.cursor()
		query = 'SELECT id_com_main,'
		if Imax:
			query += ' Imax'
		else:
			query += ' Imin'
		query += ' from Macro18280223 Where country = "BE"'

	else:
		## Construct SQL query
		table_clause = ['macro_detail', 'communes']
		column_clause = [
			'macro_detail.id_com',
			'communes.longitude',
			'communes.latitude']

		if group_by_main_village:
			agg_function = {"average": "AVG", "minimum": "MIN", "maximum": "MAX"}.get(agg_function.lower(), "MAX")
			if Imax:
				column_clause.append('%s(macro_detail.intensity_max) as "Intensity"' % agg_function)
			else:
				column_clause.append('%s(macro_detail.intensity_min) as "Intensity"' % agg_function)
			group_clause = 'communes.id_main'
		else:
			if Imax:
				column_clause.append('macro_detail.intensity_max as "Intensity"')
			else:
				column_clause.append('macro_detail.intensity_min as "Intensity"')
			group_clause = ""

		where_clause = 'macro_detail.id_earth = %d' % id_earth
		where_clause += ' and macro_detail.fiability != 0'
		where_clause += ' and macro_detail.id_com = communes.id'

		having_clause = 'Intensity >= %d' % min_val
		order_clause = ''

	if errf !=None:
		errf.write("Querying KSB-ORB official macroseismic catalog:\n")

	## Fetch records
	macro_info = {}
	for rec in query_seismodb_table(table_clause, column_clause=column_clause,
					where_clause=where_clause, having_clause=having_clause,
					order_clause=order_clause, group_clause=group_clause,
					verbose=verbose, errf=errf):
		id_com = rec['id_com']
		I = rec['Intensity']
		lon, lat = rec['longitude'], rec['latitude']
		macro_info[id_com] = MacroseismicRecord(id_earth, id_com, I, num_replies=1, lon=lon, lat=lat)

	return macro_info

	"""
	from seismodb_secrets import host, user, passwd, database
	port = 3306
	lonlat = True

	db = MySQLdb.connect(host=host, user=user, passwd=passwd, db=database, port=port)
	c = db.cursor()
	if group_by_main_village == False:
		query = 'SELECT macro_detail.id_com,'
		if Imax:
			query += ' macro_detail.intensity_max as "Intensity"'
		else:
			query += ' macro_detail.intensity_min as "Intensity"'
		if lonlat:
			query += ' , communes.longitude, communes.latitude'
		query += ' from macro_detail'
		if lonlat:
			query += ', communes'
		query +=' Where macro_detail.id_earth = %d' % id_earth
		query += ' and macro_detail.fiability != 0'
		if lonlat:
			query += ' and macro_detail.id_com = communes.id'
		query += ' HAVING Intensity >= %d' % min_val
		query += ' Order By Intensity asc'
	else:
		agg_function = {"average": "AVG", "minimum": "MIN", "maximum": "MAX"}[agg_function.lower()]
		query = 'SELECT communes.id_main,'
		if Imax:
			query += ' %s(macro_detail.intensity_max) as "Intensity"' % agg_function
		else:
			query += ' %s(macro_detail.intensity_min) as "Intensity"' % agg_function
		if lonlat:
			query += ' , communes.longitude, communes.latitude'
		query += ' from macro_detail, communes'
		query += ' Where macro_detail.id_earth = %d' % id_earth
		query += ' and macro_detail.fiability != 0'
		query += ' and macro_detail.id_com = communes.id'
		query += ' GROUP BY communes.id_main'
		query += ' HAVING Intensity >= %d' % min_val
		query += ' Order By Intensity asc'

	if verbose:
		print query
	c.execute(query)

	macro_info = {}
	if lonlat:
		for rec in c.fetchall():
			id_com, I, lon, lat = rec
			try:
				macro_info[id_com] = (float(lon), float(lat), I)
			except TypeError:
				if errf != None:
					errf.write("Record without geographic coordinates!")
					errf.flush()
	else:
		for rec in c.fetchall():
			id_com, I = rec
			macro_info[id_com] = I

	return macro_info
	"""


def query_ROB_Web_MacroCatalog(id_earth, min_replies=3, query_info="cii",
					min_val=1, min_fiability=20.0, group_by_main_village=False,
					agg_function="average", verbose=False, errf=None):
	"""
	Query ROB web macroseismic catalog (= online inquiries)

	:param id_earth:
		int, earthquake ID
	:param min_replies:
		int, minimum number of replies
		(default: 3)
	:param query_info:
		str, either "cii" or "manual_intensity"
		(default: "cii")
	:param min_val:
		float, minimum intensity to return
		(default: 1)
	:param min_fiability:
		float, minimum fiability of enquiry
		(default: 20.)
	:param group_by_main_village:
		bool, whether or not to aggregate the results by main village
		(default: False)
	:param agg_function:
		str, aggregation function to use, one of "minimum", "maximum" or
		"average". If :param:`group_by_main_village` is False, aggregation
		applies to the enquiries within a given (sub)commune.
		(default: "average")
	:param verbose:
		Bool, if True the query string will be echoed to standard output
	:param errf:
		File object, where to print errors

	:return:
		dict mapping commune IDs to instances of :class:`MacroseismicRecord`
	"""
	## Construct SQL query
	table_clause = ['web_analyse', 'communes']
	column_clause = [
		'web_analyse.id_com',
		'COUNT(*) as "Num_Replies"',
		'communes.longitude',
		'communes.latitude']

	agg_function = {"average": "AVG", "minimum": "MIN", "maximum": "MAX"}.get(agg_function.lower(), "AVG")
	if query_info.lower() == "manual_intensity":
		column_clause.append('%s(web_analyse.MI) as "Intensity"' % agg_function)
	elif query_info.lower() == "cii":
		column_clause.append('%s(web_analyse.CII) as "Intensity"' % agg_function)

	if group_by_main_village:
		group_clause = 'communes.id_main'
	else:
		group_clause = "web_analyse.id_com"

	where_clause = 'web_analyse.id_earth = %d' % id_earth
	where_clause += ' and web_analyse.m_fiability > %.1f' % float(min_fiability)
	where_clause += ' and web_analyse.deleted = false'
	where_clause += ' and web_analyse.id_com = communes.id'

	having_clause = 'Num_Replies >= %d' % min_replies
	if query_info.lower() in ("cii", "manual_intensity"):
		having_clause += ' and Intensity >= %d' % min_val

	order_clause = ''

	if errf !=None:
		errf.write("Querying KSB-ORB web macroseismic catalog:\n")

	## Fetch records
	macro_info = {}
	for rec in query_seismodb_table(table_clause, column_clause=column_clause,
					where_clause=where_clause, having_clause=having_clause,
					order_clause=order_clause, group_clause=group_clause,
					verbose=verbose, errf=errf):
		id_com = rec['id_com']
		I = rec['Intensity']
		lon, lat = rec['longitude'], rec['latitude']
		macro_info[id_com] = MacroseismicRecord(id_earth, id_com, I, num_replies=1, lon=lon, lat=lat)

	return macro_info

	"""
	macro_info = []

	db = MySQLdb.connect(host=host, user=user, passwd=passwd, db=database, port=port)
	c = db.cursor()
	if group_by_main_village == False:
		query = 'SELECT web_analyse.id_com, COUNT(*) as "Num_Replies"'
		if query_info == "manual_intensity":
			query += ', AVG(web_analyse.MI) as "Intensity"'
		else:
			query += ', AVG(web_analyse.CII) as "Intensity"'
		if lonlat:
			query += ' , communes.longitude, communes.latitude'
		query += ' from web_analyse'
		if lonlat:
			query += ', communes'
		query += ' Where web_analyse.id_earth = %d' % id_earth
		query += ' and web_analyse.m_fiability > %.1f' % float(min_fiability)
		query += ' and web_analyse.deleted = false'
		if lonlat:
			query += ' and web_analyse.id_com = communes.id'
		query += ' GROUP BY web_analyse.id_com'
		query += ' HAVING Num_Replies >= %d' % min_replies
		if query_info.lower() in ("cii", "manual_intensity"):
			query += ' and Intensity >= %d' % min_val
		if sort_key.lower() in ("intensity", "cii", "manual_intensity"):
			query += ' Order By Intensity %s, Num_Replies desc' % sort_order
		elif sort_key.lower() == "num_replies":
			query += ' Order By Num_Replies %s, Intensity asc' % sort_order
		elif sort_key.lower() == "id_com":
			query += ' Order By web_analyse.id_com %s' % sort_order
	else:
		#agg_function = {"average": "AVG", "minimum": "MIN", "maximum": "MAX"}[agg_function.lower()]
		query = 'SELECT communes.id_main, COUNT(*) as "Num_Replies"'
		if query_info == "manual_intensity":
			query += ', AVG(web_analyse.MI) as "Intensity"'
		else:
			query += ', AVG(web_analyse.CII) as "Intensity"'
		if lonlat:
			query += ' , communes.longitude, communes.latitude'
		query += ' from web_analyse, communes'
		query += ' Where web_analyse.id_earth = %d' % id_earth
		query += ' and web_analyse.m_fiability >= %.1f' % float(min_fiability)
		query += ' and web_analyse.deleted = false'
		query += ' and web_analyse.id_com = communes.id'
		query += ' GROUP BY communes.id_main'
		query += ' HAVING Num_Replies >= %d' % min_replies
		if query_info.lower() in ("cii", "manual_intensity"):
			query += ' and Intensity >= %d' % min_val
		if sort_key.lower() in ("intensity", "cii", "manual_intensity"):
			query += ' Order By Intensity %s, Num_Replies desc' % sort_order
		elif sort_key.lower() == "num_replies":
			query += ' Order By Num_Replies %s, Intensity asc' % sort_order
		elif sort_key.lower() == "id_com":
			query += ' Order By web_analyse.id_com %s' % sort_order

	if errf !=None:
		errf.write("%s\n" % query)
		errf.flush()
	elif verbose:
		print query

	c.execute(query)
	if lonlat:
		for rec in c.fetchall():
			id_com, num_replies, I, lon, lat = rec
			try:
				lon, lat = float(lon), float(lat)
			except TypeError:
				lon, lat = 0.0, 0.0
				if errf != None:
					errf.write("Record without geographic coordinates!")
					errf.flush()
			I = int(round(I))

			macro_info.append(MacroseismicRecord(id_com, I, num_replies, lon, lat))

	else:
		for rec in c.fetchall():
			id_com, num_replies, I = rec

			macro_info.append(MacroseismicRecord(id_com, I, num_replies))

	return macro_info
	"""


def query_ROB_Web_enquiries(id_earth, id_com=None, zip_code=None, min_fiability=20,
							verbose=False, errf=None):
	"""
	:param min_fiability:
		float, minimum fiability of enquiry
		(default: 20.)
	"""
	table_clause = ['web_input', 'web_analyse']

	where_clause = 'web_analyse.id_earth = %d' % id_earth
	where_clause += ' AND web_input.id_web=web_analyse.id_web'
	where_clause += ' AND web_analyse.m_fiability > %.1f' % float(min_fiability)
	where_clause += ' AND web_analyse.deleted = false'
	if id_com is not None:
		where_clause += ' AND web_analyse.id_com=%d' % id_com
	elif zip_code:
		where_clause += ' AND web_input.zip=%d' % zip_code

	if errf !=None:
		errf.write("Querying KSB-ORB web macroseismic enquiries:\n")

	## Fetch records
	return query_seismodb_table(table_clause, where_clause=where_clause,
								verbose=verbose, errf=errf)


def query_ROB_Stations(network='UCC', activity_date_time=None, verbose=False):
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

	:return:
		generator object, yielding a dictionary for each record
	"""
	table_clause = ["station", "station_place", "station_sismometer", "station_sismometer_type"]

	where_clause = "station.id_place=station_place.id_place"
	where_clause += " AND station.id_sismometer=station_sismometer.id_sismometer"
	where_clause += " AND station_sismometer.id_type=station_sismometer_type.id_type"
	if network:
		where_clause += " AND network='%s'" % network
	if activity_date_time:
		where_clause += " AND station.begin <= '%s'" % activity_date_time.isoformat()
		where_clause += " AND (station.end >= '%s' OR station.end IS NULL)" % activity_date_time.isoformat()

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


def get_last_earthID():
	"""
	Return ID of most recently added earthquake in database

	:return:
		int, earthquake ID
	"""
	query = 'SELECT id_earth FROM earthquakes WHERE type="ke" and is_true = 1 ORDER BY id_earth DESC LIMIT 0 , 1'
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



if __name__ == "__main__":
	start_date = datetime.date(1983, 1,1)
	end_date = datetime.date(2007, 12, 31)
	catalogue_length = (end_date.year - start_date.year) * 1.0
	catalogue = query_ROB_LocalEQCatalog(start_date=start_date, end_date=end_date)
	print "%d events" % len(catalogue)
