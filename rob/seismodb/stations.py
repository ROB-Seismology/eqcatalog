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



__all__ = ["query_stations", "get_station_coordinates", "get_station_catalog"]



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
	from .base import query_seismodb_table

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
	from .base import query_seismodb_table

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
	from .base import query_seismodb_table
	from .local_eq import query_local_eq_catalog_by_id

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
