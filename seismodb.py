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
import MySQLdb
import MySQLdb.cursors

## Import ROB modules
import eqrecord
reload(eqrecord)
from eqrecord import LocalEarthquake, MacroseismicRecord, FocMecRecord

import eqcatalog
reload(eqcatalog)
from eqcatalog import EQCatalog

## Database information
from seismodb_secrets import host, user, passwd, database
try:
	from seismodb_secrets import port
except:
	port = 3306


__all__ = ["LocalEarthquake", "MacroseismicRecord", "FocMecRecord", "query_ROB_LocalEQCatalog", "query_ROB_LocalEQCatalogByID", "query_ROB_FocalMechanisms", "query_ROB_Official_MacroCatalog", "query_ROB_Web_MacroCatalog", "get_last_earthID"]


def query_ROB_LocalEQCatalog(region=None, start_date=None, end_date=None, Mmin=None, Mmax=None, min_depth=None, max_depth=None, id_earth=None, sort_key="date", sort_order="asc", convert_NULL=True, verbose=False, errf=None):
	"""
	Query ROB catalog of local earthquakes.

	Notes:
	Magnitude used for selection is based on MW first, then MS, then ML.
	NULL values in the database are converted to 0.0 (this may change in the future)
	Only real earthquakes are extracted (type = "ke" and is_true = 1).

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
	db = MySQLdb.connect(host=host, user=user, passwd=passwd, db=database, port=port, cursorclass=MySQLdb.cursors.DictCursor, use_unicode=True)
	c = db.cursor()
	query = 'SELECT id_earth, DATE_FORMAT(date, "%Y-%m-%d") as date, TIME_FORMAT(time, "%H:%i:%s") as time,'
	query += ' longitude, latitude, depth, ML, MS, MW,'
	query += ' IF(MW, MW, IF(MS, MS, ML)) as M,'
	query += ' intensity_max, macro_radius, errh, errz, errt, errM,'
	query += ' name from earthquakes'
	query += ' Where '
	if id_earth:
		query += ' id_earth in (%s)' % ",".join([str(item) for item in id_earth])
	else:
		query += ' type = "ke"'
		query += ' and is_true = 1'
	if region:
		w, e, s, n = region
		query += ' and longitude Between %f and %f' % (w, e)
		query += ' and latitude Between %f and %f' % (s, n)
	if start_date or end_date:
		if not end_date:
			end_date = datetime.datetime.now()
		if not start_date:
			start_date = datetime.datetime(100, 1, 1)
		query += ' and date Between "%s" and "%s"' % (start_date.isoformat(), end_date.isoformat())
	if min_depth:
		query += ' and depth >= %f' % min_depth
	if max_depth:
		query += ' and depth <= %f' % max_depth
	if Mmin or Mmax:
		if not Mmin:
			Mmin = 0.0
		if not Mmax:
			Mmax = 8.0
		#query += ' and ML Between %f and %f' % (Mmin, Mmax)
		query += ' HAVING M Between %f and %f' % (Mmin, Mmax)
	if sort_order.lower()[:3] == "asc":
		sort_order = "asc"
	else:
		sort_order = "desc"
	if sort_key.lower() in ("date", "time"):
		query += ' Order By date %s, time %s' % (sort_order, sort_order)
	elif sort_key.lower() in ("size", "mag"):
		query += ' Order By M %s' % sort_order

	if errf !=None:
		errf.write("%s\n" % query)
		errf.flush()
	elif verbose:
		print query

	## Fetch records
	c.execute(query)
	eq_list = []
	for rec in c.fetchall():
		id_earth, name = rec["id_earth"], rec["name"]
		date, time = rec["date"], rec["time"]
		lon, lat = rec["longitude"], rec["latitude"]
		depth = rec["depth"]
		ML, MS, MW, M = rec["ML"], rec["MS"], rec["MW"], rec["M"]
		intensity_max, macro_radius = rec["intensity_max"], rec["macro_radius"]
		errh, errz, errt, errM = rec["errh"], rec["errz"], rec["errt"], rec["errM"]

		if name == lon == lat == depth == ML == None:
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
				ML = 0.0
			if MS == None:
				MS = 0.0
			if MW == None:
				MW = 0.0
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

		eq = LocalEarthquake(id_earth, date, time, lon, lat, depth, ML, MS, MW, name, intensity_max, macro_radius, errh, errz, errt, errM)
		eq_list.append(eq)

	name = "ROB Catalog %s - %s" % (start_date.isoformat(), end_date.isoformat())
	return EQCatalog(eq_list, start_date, end_date, region=region, name=name)


def query_ROB_LocalEQCatalogByID(id_earth, verbose=False, errf=None):
	db = MySQLdb.connect(host=host, user=user, passwd=passwd, db=database, port=port)
	c = db.cursor()
	query = 'SELECT id_earth, DATE_FORMAT(date, "%Y-%m-%d"), TIME_FORMAT(time, "%H:%i:%s"),'
	query += ' longitude, latitude, depth, ML, MS, MW,'
	query += ' IF(MW, MW, IF(MS, MS, ML)) as M,'
	query += ' intensity_max, macro_radius,'
	query += ' name from earthquakes'
	query += ' Where '
	query += ' id_earth = %d' % id_earth

	if errf !=None:
		errf.write("%s\n" % query)
		errf.flush()
	elif verbose:
		print query

	c.execute(query)
	recs = c.fetchall()
	id_earth, date, time, lon, lat, depth, ML, MS, MW, M, intensity_max, macro_radius, name = recs[0]
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
		ML = 0.0
	if MS == None:
		MS = 0.0
	if MW == None:
		MW = 0.0
		if intensity_max == None:
			intensity_max = 0
		if macro_radius == None:
			macro_radius = 0
	eq = LocalEarthquake(id_earth, date, time, lon, lat, depth, ML, MS, MW, name, intensity_max, macro_radius)

	return eq


def query_ROB_FocalMechanisms(region=None, start_date=None, end_date=None, Mmin=None, Mmax=None, id_earth=None , sort_key="Mag", sort_order="asc", verbose=False, errf=None):
	global MT
	import eqgeology.FocMec.MomentTensor as MT

	catalogue = []
	if type(id_earth) in (type(1), type(1L)):
		id_earth = [id_earth]

	db = MySQLdb.connect(host=host, user=user, passwd=passwd, db=database, port=port)
	c = db.cursor()

	query = 'SELECT earthquakes.id_earth, DATE_FORMAT(earthquakes.date, "%Y-%m-%d") as date, TIME_FORMAT(earthquakes.time, "%H:%i:%s") as time,'
	query += ' earthquakes.longitude, earthquakes.latitude, earthquakes.depth, earthquakes.ML as ML, earthquakes.MS as MS, earthquakes.MW as MW,'
	query += ' IF(MW, MW, IF(MS, MS, ML)) as M,'
	query += ' strike, dip, slip,'
	query += ' earthquakes.intensity_max, earthquakes.macro_radius,'
	query += ' earthquakes.name'
	query += ' from focal_mechanisms, earthquakes'
	query += ' Where'
	query += ' focal_mechanisms.id_earth = earthquakes.id_earth'
	query += ' and earthquakes.type = "ke"'
	query += ' and earthquakes.is_true = 1'
	if id_earth:
		query += ' and earthquakes.id_earth in (%s)' % ",".join([str(item) for item in id_earth])
	if region:
		w, e, s, n = region
		query += ' and earthquakes.longitude Between %f and %f' % (w, e)
		query += ' and earthquakes.latitude Between %f and %f' % (s, n)
	if start_date or end_date:
		if not end_date:
			end_date = datetime.datetime.now()
		if not start_date:
			start_date = datetime.datetime(100, 1, 1)
		query += ' and earthquakes.date Between "%s" and "%s"' % (start_date.isoformat(), end_date.isoformat())
	if Mmin or Mmax:
		if not Mmin:
			Mmin = 0.0
		if not Mmax:
			Mmax = 8.0
		#query += ' and ML Between %f and %f' % (Mmin, Mmax)
		query += ' HAVING M Between %f and %f' % (Mmin, Mmax)
	if sort_order.lower()[:3] == "asc":
		sort_order = "asc"
	else:
		sort_order = "desc"
	if sort_key.lower() in ("date", "time"):
		query += ' Order By date %s, time %s' % (sort_order, sort_order)
	elif sort_key.lower() in ("size", "mag"):
		query += ' Order By M %s' % sort_order

	if errf !=None:
		errf.write("Querying KSB-ORB focal mechanism catalog:\n")
		errf.write("  %s\n" % query)
		errf.flush()
	elif verbose:
		print query

	c.execute(query)
	for rec in c.fetchall():
		id_earth, date, time, lon, lat, depth, ML, MS, MW, M, strike, dip, rake, intensity_max, macro_radius, name = rec
		if name == lon == lat == depth == ML == None:
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
			ML = 0.0
		if MS == None:
			MS = 0.0
		if MW == None:
			MW = 0.0
		if intensity_max == None:
			intensity_max = 0
		if macro_radius == None:
			macro_radius = 0
		eq = FocMecRecord(id_earth, date, time, lon, lat, depth, ML, MS, MW, strike, dip, rake, name, intensity_max, macro_radius)
		catalogue.append(eq)

	return catalogue


def query_ROB_Official_MacroCatalog(id_earth, Imax=True, min_val=1, group_by_main_village=False, agg_function="maximum", lonlat=False, verbose=False, errf=None):
	import eqmapperdb_secrets
	macro_info = {}

	if id_earth == 18280223:
		db = MySQLdb.connect(host=eqmapperdb_secrets.host, user=eqmapperdb_secrets.user, passwd=eqmapperdb_secrets.passwd, db=eqmapperdb_secrets.database)
		c = db.cursor()
		query = 'SELECT id_com_main,'
		if Imax:
			query += ' Imax'
		else:
			query += ' Imin'
		query += ' from Macro18280223 Where country = "BE"'

	else:
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

	if errf !=None:
		errf.write("%s\n" % query)
		errf.flush()
	elif verbose:
		print query

	c.execute(query)
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
	Example how to aggregate by main village
	SELECT id_com, AVG(intensity_max) ... GROUP BY id_com
	"""


def query_ROB_Web_MacroCatalog(id_earth, min_replies=3, query_info="cii", min_val=1, min_fiability=10.0, group_by_main_village=False, agg_function="", lonlat=False, sort_key="intensity", sort_order="asc", verbose=False, errf=None):
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


def get_last_earthID():
	db = MySQLdb.connect(host=host, user=user, passwd=passwd, db=database)
	c = db.cursor()
	query = 'SELECT id_earth FROM earthquakes WHERE type="ke" and is_true = 1 ORDER BY id_earth DESC LIMIT 0 , 1'
	c.execute(query)
	rec = c.fetchone()
	id_earth = rec[0]
	return id_earth


def zip2ID(zip_code):
	db = MySQLdb.connect(host=host, user=user, passwd=passwd, db=database)
	c = db.cursor()
	query = 'SELECT id FROM communes WHERE code_p = %d' % zip_code
	c.execute(query)
	rec = c.fetchone()
	id_com = rec[0]
	return id_com



if __name__ == "__main__":
	start_date = datetime.date(1983, 1,1)
	end_date = datetime.date(2007, 12, 31)
	catalogue_length = (end_date.year - start_date.year) * 1.0
	catalogue = query_ROB_LocalEQCatalog(start_date=start_date, end_date=end_date)
	print "%d events" % len(catalogue)
	for eq in catalogue:
		print eq.getMW()
