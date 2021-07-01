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



__all__ = ["query_online_macro_catalog", "query_traditional_macro_catalog",
			"query_official_macro_catalog", "query_historical_macro_catalog",
			"query_traditional_macro_catalog_aggregated",
			"query_official_macro_catalog_aggregated",
			"query_historical_macro_catalog_aggregated",
			"query_online_macro_catalog_aggregated",
			"get_num_online_macro_enquiries",
			"get_num_official_enquiries", "get_num_traditional_mdps",
			"get_earthquakes_with_traditional_enquiries",
			"get_earthquakes_with_online_enquiries"]


# Note: median is not supported by MySQL
AGG_FUNC_DICT = {"average": "AVG", "mean": "AVG",
				"minimum": "MIN", "maximum": "MAX"}


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
	from ...macro import MDP, MDPCollection
	from .base import query_seismodb_table
	from .commune import _get_com_zip_table_clause

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
	from ...macro.macro_info import AggregatedMacroInfo, AggregatedMacroInfoCollection
	from .base import (query_mysql_db_generic, query_seismodb_table)

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
	from ...macro.macro_info import AggregatedMacroInfo, AggregatedMacroInfoCollection
	from .base import query_seismodb_table

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
	from ...macro import ROBDYFIEnsemble
	from .base import query_seismodb_table

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
	from .base import query_seismodb_table

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
	from .base import query_seismodb_table

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
	from .base import query_seismodb_table

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
	from .base import query_seismodb_table
	from .local_eq import query_local_eq_catalog_by_id

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
	from .base import query_seismodb_table
	from .local_eq import query_local_eq_catalog_by_id

	table_clause = 'web_analyse'
	column_clause = 'id_earth'
	group_clause = 'id_earth'
	db_recs = query_seismodb_table(table_clause, column_clause,
									group_clause=group_clause)
	eq_ids = [rec['id_earth'] for rec in db_recs]
	return query_local_eq_catalog_by_id(eq_ids)
