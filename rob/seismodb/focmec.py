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



__all__ = ["query_focal_mechanisms"]


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
	from ...eqrecord import FocMecRecord
	from .base import query_seismodb_table

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
