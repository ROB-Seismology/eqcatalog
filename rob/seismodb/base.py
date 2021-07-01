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


## Import ROB modules
from db.simpledb import (build_sql_query, query_mysql_db_generic)



__all__ = ["query_seismodb_table_generic", "query_seismodb_table"]



def query_seismodb_table_generic(query, verbose=False, print_table=False, errf=None):
	"""
	Query seismodb table using generic clause, returning each record as a dict

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

	from db.simpledb.mysql import OperationalError

	## Test if server is online; if not, fall back to xseisalert
	query0 = 'SELECT 1'
	try:
		query_mysql_db_generic(database, host, user, passwd, query0, port=port,
									verbose=verbose, errf=errf)
	except OperationalError:
		from secrets.seismodb import (alt_host as host, alt_user as user,
											alt_passwd as passwd)
		query0 = "SET sql_mode=(SELECT REPLACE(@@sql_mode, 'ONLY_FULL_GROUP_BY', ''))"
		query_mysql_db_generic(database, host, user, passwd, query0, port=port,
									verbose=verbose, errf=errf)

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
