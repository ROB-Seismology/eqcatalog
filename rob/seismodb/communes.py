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


## Import third-party modules
import numpy as np



__all__ = ["zip2ID", "get_communes", "get_subcommunes", "get_subcommune_ids"]


def zip2ID(zip_code, country='BE'):
	"""
	Look up ID corresponding to ZIP code in database

	:return:
		int, commune ID
	"""
	from .base import query_seismodb_table_generic

	if country in ('DE', 'FR', 'LU', 'NL', 'GB'):
		table_name = 'com_zip_%s' % country
		if country == 'LU':
			table_name += '_fr'
		id_column_name = 'id_com'
		zip_column_name = 'zip'
	else:
		table_name = 'communes'
		id_column_name = 'id'
		zip_column_name = 'code_p'

	query = 'SELECT %s FROM %s WHERE %s = %d'
	query %= (id_column_name, table_name, zip_column_name, zip_code)
	if table_name == 'communes':
		query += ' AND country = "%s"' % country

	id_com = query_seismodb_table_generic(query)[0][id_column_name]

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
	from .base import query_seismodb_table

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
	from .base import query_seismodb_table

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
	from .base import query_seismodb_table

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
	from .base import query_seismodb_table

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
	from .base import query_seismodb_table

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
