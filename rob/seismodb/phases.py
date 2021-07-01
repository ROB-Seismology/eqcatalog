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


__all__ = ["query_phase_picks"]


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
	from .base import query_seismodb_table

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
