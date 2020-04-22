"""
Provides interface to Harvard CMT catalog:
- download from the internet
- store in spatialite database
- partial interface with eqcatalog
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
if sys.version[0] == '3':
	## Python 3
	basestring = str
	from urllib.request import urlopen, HTTPError
else:
	# Fall back to Python 2's urllib2
	from urllib2 import urlopen, HTTPError


import os
import datetime

import numpy as np

import db.simpledb as simpledb

from .moment import (moment_to_mag, mag_to_moment)


__all__ = ['HarvardCMTCatalog', 'HarvardCMTRecord',
			'get_harvard_cmt_catalog', 'update_harvard_cmt_catalog']


BASE_URL = "http://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog"


HarvardCMTColDef = [
	dict(name='ID', type='STRING', notnull=1, pk=1),
	dict(name='ref_catalog', type='STRING'),
	dict(name='hypo_date_time', type='TIMESTAMP'),
	dict(name='hypo_lon', type='REAL'),
	dict(name='hypo_lat', type='REAL'),
	dict(name='hypo_depth', type='REAL'),
	dict(name='ref_mb', type='REAL'),
	dict(name='ref_MS', type='REAL'),
	dict(name='location', type='STRING'),
	dict(name='source_type', type='STRING'),
	dict(name='mrf_type', type='STRING'),
	dict(name='mrf_duration', type='REAL'),
	dict(name='centroid_reltime', type='REAL'),
	dict(name='centroid_reltime_sigma', type='REAL'),
	dict(name='centroid_lon', type='REAL'),
	dict(name='centroid_lon_sigma', type='REAL'),
	dict(name='centroid_lat', type='REAL'),
	dict(name='centroid_lat_sigma', type='REAL'),
	dict(name='centroid_depth', type='REAL'),
	dict(name='centroid_depth_sigma', type='REAL'),
	dict(name='centroid_depth_type', type='STRING'),
	dict(name='exp', type='INTEGER'),
	dict(name='Mrr', type='REAL'),
	dict(name='Mrr_sigma', type='REAL'),
	dict(name='Mtt', type='REAL'),
	dict(name='Mtt_sigma', type='REAL'),
	dict(name='Mpp', type='REAL'),
	dict(name='Mpp_sigma', type='REAL'),
	dict(name='Mrt', type='REAL'),
	dict(name='Mrt_sigma', type='REAL'),
	dict(name='Mrp', type='REAL'),
	dict(name='Mrp_sigma', type='REAL'),
	dict(name='Mtp', type='REAL'),
	dict(name='Mtp_sigma', type='REAL'),
	dict(name='eva1', type='REAL'),
	dict(name='pl1', type='INTEGER'),
	dict(name='az1', type='INTEGER'),
	dict(name='eva2', type='REAL'),
	dict(name='pl2', type='INTEGER'),
	dict(name='az2', type='INTEGER'),
	dict(name='eva3', type='REAL'),
	dict(name='pl3', type='INTEGER'),
	dict(name='az3', type='INTEGER'),
	dict(name='moment', type='REAL'),
	dict(name='strike1', type='INTEGER'),
	dict(name='dip1', type='INTEGER'),
	dict(name='rake1', type='INTEGER'),
	dict(name='strike2', type='INTEGER'),
	dict(name='dip2', type='INTEGER'),
	dict(name='rake2', type='INTEGER')]


def get_coltype(colname):
	for col_def in HarvardCMTColDef:
		if col_def['name'] == colname:
			return col_def['type']

def cnv_coltype(colname, value):
	func = {'INTEGER': int,
			'REAL': float,
			'STRING': str,
			'TIMESTAMP': lambda x:x}[get_coltype(colname)]
	return func(value)


class HarvardCMTRecord:
	def __init__(self,
		ID,
		ref_catalog,
		hypo_date_time,
		hypo_lon,
		hypo_lat,
		hypo_depth,
		ref_mb,
		ref_MS,
		location,
		source_type,
		mrf_type,
		mrf_duration,
		centroid_reltime,
		centroid_reltime_sigma,
		centroid_lon,
		centroid_lon_sigma,
		centroid_lat,
		centroid_lat_sigma,
		centroid_depth,
		centroid_depth_sigma,
		centroid_depth_type,
		exp,
		Mrr,
		Mrr_sigma,
		Mtt,
		Mtt_sigma,
		Mpp,
		Mpp_sigma,
		Mrt,
		Mrt_sigma,
		Mrp,
		Mrp_sigma,
		Mtp,
		Mtp_sigma,
		eva1,
		pl1,
		az1,
		eva2,
		pl2,
		az2,
		eva3,
		pl3,
		az3,
		moment,
		strike1,
		dip1,
		rake1,
		strike2,
		dip2,
		rake2):

		self.ID = cnv_coltype('ID', ID)
		self.ref_catalog = cnv_coltype('ref_catalog', ref_catalog)
		self.hypo_date_time = hypo_date_time
		self.hypo_lon = cnv_coltype('hypo_lon', hypo_lon)
		self.hypo_lat = cnv_coltype('hypo_lat', hypo_lat)
		self.hypo_depth = cnv_coltype('hypo_depth', hypo_depth)
		self.ref_mb = cnv_coltype('ref_mb', ref_mb)
		self.ref_MS = cnv_coltype('ref_MS', ref_MS)
		self.location = cnv_coltype('location', location)
		self.source_type = cnv_coltype('source_type', source_type)
		self.mrf_type = cnv_coltype('mrf_type', mrf_type)
		self.mrf_duration = cnv_coltype('mrf_duration', mrf_duration)
		self.centroid_reltime = cnv_coltype('centroid_reltime', centroid_reltime)
		self.centroid_reltime_sigma = cnv_coltype('centroid_reltime_sigma',
												centroid_reltime_sigma)
		self.centroid_lon = cnv_coltype('centroid_lon', centroid_lon)
		self.centroid_lon_sigma = cnv_coltype('centroid_lon_sigma',
											centroid_lon_sigma)
		self.centroid_lat = cnv_coltype('centroid_lat', centroid_lat)
		self.centroid_lat_sigma = cnv_coltype('centroid_lat_sigma',
											centroid_lat_sigma)
		self.centroid_depth = cnv_coltype('centroid_depth', centroid_depth)
		self.centroid_depth_sigma = cnv_coltype('centroid_depth_sigma',
												centroid_depth_sigma)
		self.centroid_depth_type = cnv_coltype('centroid_depth_type',
											centroid_depth_type)
		self.exp = cnv_coltype('exp', exp)
		self.Mrr = cnv_coltype('Mrr', Mrr)
		self.Mrr_sigma = cnv_coltype('Mrr_sigma', Mrr_sigma)
		self.Mtt = cnv_coltype('Mtt', Mtt)
		self.Mtt_sigma = cnv_coltype('Mtt_sigma', Mtt_sigma)
		self.Mpp = cnv_coltype('Mpp', Mpp)
		self.Mpp_sigma = cnv_coltype('Mpp_sigma', Mpp_sigma)
		self.Mrt = cnv_coltype('Mrt', Mrt)
		self.Mrt_sigma = cnv_coltype('Mrt_sigma', Mrt_sigma)
		self.Mrp = cnv_coltype('Mrp', Mrp)
		self.Mrp_sigma = cnv_coltype('Mrp_sigma', Mrp_sigma)
		self.Mtp = cnv_coltype('Mtp', Mtp)
		self.Mtp_sigma = cnv_coltype('Mtp_sigma', Mtp_sigma)
		self.eva1 = cnv_coltype('eva1', eva1)
		self.pl1 = cnv_coltype('pl1', pl1)
		self.az1 = cnv_coltype('az1', az1)
		self.eva2 = cnv_coltype('eva2', eva2)
		self.pl2 = cnv_coltype('pl2', pl2)
		self.az2 = cnv_coltype('az2', az2)
		self.eva3 = cnv_coltype('eva3', eva3)
		self.pl3 = cnv_coltype('pl3', pl3)
		self.az3 = cnv_coltype('az3', az3)
		self.moment = cnv_coltype('moment', moment)
		self.strike1 = cnv_coltype('strike1', strike1)
		self.dip1 = cnv_coltype('dip1', dip1)
		self.rake1 = cnv_coltype('rake1', rake1)
		self.strike2 = cnv_coltype('strike2', strike2)
		self.dip2 = cnv_coltype('dip2', dip2)
		self.rake2 = cnv_coltype('rake2', rake2)

	# TODO: get absolute centroid_date_time

	@property
	def hypo_date(self):
		return self.hypo_date_time.date()

	@property
	def hypo_time(self):
		return self.hypo_date_time.time()

	@property
	def MW(self):
		return moment_to_mag(self.get_moment(), unit='dyn.cm')

	def get_moment(self, unit='dyn.cm'):
		"""
		Get seismic moment, combining 'moment' and 'exp' values in DB

		:param unit:
			str, moment unit, either 'dyn.cm' or 'N.m'
			(default: 'dyn.cm')

		:return:
			float, seismic moment
		"""
		moment = self.moment * 10**self.exp
		if unit == 'N.m':
			moment *= 1E-7
		return moment

	@classmethod
	def from_ndk_record(cls, ndk_record):
		"""
		:param ndk_record:
			str or list of strings corresponding to record lines
		"""
		if isinstance(ndk_record, basestring):
			ndk_record = ndk_record.strip().splitlines()
		for i, line in enumerate(ndk_record):
			line = line.strip('\t')
			if i == 0:
				ref_catalog = line[:4].strip()
				(hypo_date, hypo_time, hypo_lat, hypo_lon, hypo_depth, ref_mb,
					ref_MS) = line[4:56].split()
				year, month, day = map(int, hypo_date.split('/'))
				date = datetime.date(year, month, day)
				hour, minute, second = hypo_time.split(':')
				idx = second.index('.')
				second, microsecond = second[:idx], second[idx:]
				microsecond = float(microsecond) * 1E6
				if second == '60':
					second = 59
					microsecond = 999999
				time = datetime.time(int(hour), int(minute), int(second), int(microsecond))
				hypo_date_time = datetime.datetime.combine(date, time)
				location = line[56:].strip()
			elif i == 1:
				ID = line[:16].strip()
				source_type = line[62:68].strip()
				mrf_type, mrf_duration = line[69:].split()
			elif i == 2:
				centroid_time = line[14:19].strip()
				(centroid_time_sigma, centroid_lat, centroid_lat_sigma, centroid_lon,
					centroid_lon_sigma, centroid_depth, centroid_depth_sigma,
					centroid_depth_type) = line[19:63].split()
			elif i == 3:
				(exp, Mrr, Mrr_sigma, Mtt, Mtt_sigma, Mpp, Mpp_sigma, Mrt,
					Mrt_sigma, Mrp, Mrp_sigma, Mtp, Mtp_sigma) = line.split()
			elif i == 4:
				(eva1, pl1, az1, eva2, pl2, az2, eva3, pl3, az3, moment,
					strike1, dip1, rake1, strike2, dip2, rake2) = line[3:].split()

		return cls(ID, ref_catalog, hypo_date_time, hypo_lon, hypo_lat, hypo_depth,
				ref_mb, ref_MS, location, source_type, mrf_type, mrf_duration,
				centroid_time, centroid_time_sigma, centroid_lon, centroid_lon_sigma,
				centroid_lat, centroid_lat_sigma, centroid_depth, centroid_depth_sigma,
				centroid_depth_type, exp, Mrr, Mrr_sigma, Mtt, Mtt_sigma, Mpp, Mpp_sigma,
				Mrt, Mrt_sigma, Mrp, Mrp_sigma, Mtp, Mtp_sigma, eva1, pl1, az1,
				eva2, pl2, az2, eva3, pl3, az3, moment, strike1, dip1, rake1, strike2, dip2, rake2)

	@classmethod
	def from_sql_record(cls, sql_rec):
		return cls(**sql_rec)

	def to_dict(self):
		return self.__dict__

	def to_mt(self):
		pass

	def to_focmec(self):
		pass

	def to_local_eq(self):
		"""
		Convert to earthquake record understood by eqcatalog

		:return:
			instance of :class:`eqcatalog.LocalEarthquake`
		"""
		from .eqcatalog import LocalEarthquake

		ID = self.ID
		date = self.hypo_date
		time = self.hypo_time
		lon = self.hypo_lon
		lat = self.hypo_lat
		depth = self.hypo_depth
		mag = {'MW': self.MW, 'MS': self.ref_MS, 'mb': self.ref_mb}
		name = self.location
		agency = self.ref_catalog
		return LocalEarthquake(ID, date, time, lon, lat, depth, mag=mag, name=name,
								agency=agency)


class HarvardCMTCatalog:
	"""
	Class representing Harvard CMT Catalog

	:param db_filespec:
		str, full path to Sqlite/Spatialite database
		or ':memory:'
	"""
	def __init__(self, db_filespec):
		self.db_filespec = db_filespec
		self.db = simpledb.SQLiteDB(db_filespec)
		self.table_name = 'harvard_cmt'

	def __len__(self):
		return self.db.get_num_rows(self.table_name)

	def __del__(self):
		self.db.close()

	def __getitem__(self, idx):
		"""
		:param idx:
			int, record index

		:return:
			instance of :class:`HarvardCMTRecord`
		"""
		if idx < 0:
			idx = len(self) + idx
		query = 'SELECT * FROM %s WHERE rowid = %d'
		query %= (self.table_name, idx + 1)
		try:
			[result] = list(self.query_generic(query))
		except ValueError:
			pass
		else:
			return result

	@staticmethod
	def parse_ndk_file(ndk_filespec_or_url):
		"""
		Parse NDK file or URL

		:param ndk_filespec_or_url:
			str, full path or url to NDK file

		:return:
			list with instances of :class:`HarvardCMTRecord`
		"""
		cmt_records = []
		if ndk_filespec_or_url[:4] == "http":
			try:
				ndk = urlopen(ndk_filespec_or_url)
			except HTTPError:
				return []
		else:
			ndk = open(ndk_filespec_or_url)

		for i, line in enumerate(ndk.readlines()):
			if i % 5 == 0:
				if i != 0:
					cmt_rec = HarvardCMTRecord.from_ndk_record(ndk_record)
					cmt_records.append(cmt_rec)
				ndk_record = []
			ndk_record.append(line)

		## Append last record
		cmt_rec = HarvardCMTRecord.from_ndk_record(ndk_record)
		cmt_records.append(cmt_rec)
		ndk.close()
		return cmt_records

	def clear_db(self):
		"""
		Clear database
		"""
		if self.table_name in self.db.list_tables():
			self.db.drop_geo_table(self.table_name)
		if not self.table_name in self.db.list_tables():
			self.db.create_table(self.table_name, HarvardCMTColDef)

	def import_records(self, cmt_records, clear_db=False):
		"""
		Import CMT records in the database

		:param cmt_records:
			list with instances of :class:`HarvardCMTRecord`
		:param clear_db:
			bool, whether or not to clear the database first
			(default: False)
		"""
		if clear_db:
			self.clear_db()
		try:
			self.db.add_records(self.table_name, [rec.to_dict() for rec in cmt_records])
		except:
			print('You may need to discard spatial geometries first!')
			raise

	def import_ndk(self, ndk_filespecs, start_date=datetime.date(1900, 1, 1),
					clear_db=False, verbose=False):
		"""
		Import CMT records from one or more NDK files in the database

		:param ndk_filespecs:
			list of strings, full paths to NDK files
		:param start_date:
			instance of :class:`datetime.date`, start date for importing
			records (older records will be skipped)
			(default: datetime.date(1900, 1, 1))
		:param clear_db:
			bool, whether or not to clear the database first
			(default: False)

		:return:
			int, number of records imported
		"""
		if clear_db:
			self.clear_db()
		num_recs = 0
		for ndk_filespec in ndk_filespecs:
			recs = self.parse_ndk_file(ndk_filespec)
			if start_date:
				recs = [rec for rec in recs if rec.hypo_date >= start_date]
			if verbose:
				filename = os.path.split(ndk_filespec)[-1]
				num_recs += len(recs)
				print('%s: %d records' % (filename, len(recs)))
			if recs:
				self.import_records(recs, clear_db=False)
				num_recs += len(recs)
			else:
				break

		return num_recs

	@staticmethod
	def download_ndk_file(url, overwrite=False):
		"""
		Download NDK file from given URL

		:param url:
			string, URL pointing to NDK file
		:param overwrite:
			bool, whether or not an existing NDK file with the same
			name will be overwritten

		:return:
			str, full path to NDK file on local filesystem
		"""
		filename = url.split('/')[-1]
		filespec = os.path.join(ROOT_FOLDER, "NDK", filename)
		if not os.path.exists(filespec) or overwrite:
			try:
				ndk = urlopen(url)
			except HTTPError:
				return None
			else:
				f = open(filespec, "w")
				f.write(ndk.read())
				f.close()
		return filespec

	@staticmethod
	def get_ndk_url(year, month):
		"""
		Construct URL to NDK file for a particular year and month

		:param year:
			int, year
		:param month:
			int, month (zero-based!)

		:return:
			str, URL
		"""
		MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
					"jul", "aug", "sep", "oct", "nov", "dec"]

		url = "NEW_MONTHLY/%d/%s%d.ndk" % (year, MONTHS[month], year-2000)
		url = BASE_URL + '/' + url

		return url

	def reload(self, clear_db=False, start_date=None,
				include_quick_cmt_solutions=False,
				create_spatial_geometries=True, verbose=False):
		"""
		Read entire Harvard CMT catalog from NDK files, automatically
		downloaded from the internet if necessary

		:param clear_db:
			bool, whether or not to clear the database first
			(default: False)
		:param start_date:
			int (year) or instance of :class:`datetime.date`
			start date from which onwards to import records
			(only applies if :param:`clear_db` is False)
			(default: None)
		:param include_quick_cmt_solutions:
			bool, whether or not to include Quick CMT solutions
			(default: True)
		:param create_spatial_geometries:
			bool, whether or not to create spatial geometries
			(default: True)
		"""
		today = datetime.date.today()
		current_year, current_month = today.year, today.month
		## Harvard CMT catalog is always 4 months behind
		if current_month > 4:
			current_month -= 4
		else:
			current_year -= 1
			current_month += (12 - 4)

		ndk_urls = [BASE_URL + '/' + "jan76_dec13.ndk"]
		months = ["jan", "feb", "mar", "apr", "may", "jun",
					"jul", "aug", "sep", "oct", "nov", "dec"]
		for year in range(2014, current_year+1):
			if year == current_year:
				end_month = current_month
			else:
				end_month = 12
			for m in range(end_month):
				url = self.get_ndk_url(year, m)
				ndk_urls.append(url)

		## Download NDK files if necessary, then import them
		ndk_files = []
		for ndk_url in ndk_urls:
			ndk_filespec = self.download_ndk_file(ndk_url)
			if ndk_filespec:
				ndk_files.append(ndk_filespec)
			else:
				filename = ndk_url.split('/')[-1]
				print("Warning: NDK file %s missing!" % filename)
				#basename = os.path.splitext(ndk_url.split('/')[-1])[0]
				#m, yr = basename[:3], int(basename[3:])
				#m = months.index(m) + 1
				#yr += 2000
				#start_date = datetime.date(yr, m, 1)

		if clear_db:
			self.clear_db()

		elif start_date:
			## Remove more recent records
			if isinstance(start_date, int):
				## year
				start_date = datetime.date(start_date, 1, 1)
			else:
				## Always start from beginning of month
				start_date = datetime.date(start_date.year, start_date.month, 1)
			where_clause = "julianday(hypo_date_time) >= julianday('%s')"
			where_clause %= start_date
			self.db.delete_records(self.table_name, where_clause)

		self.import_ndk(ndk_files, start_date=start_date, clear_db=False,
						verbose=verbose)

		## Add Quick CMTs
		if include_quick_cmt_solutions:
			self.import_quick_cmt_solutions(verbose=verbose)

		## Create SpatiaLite geometries
		if create_spatial_geometries:
			self.create_spatialite_geometries()

	def import_quick_cmt_solutions(self, verbose=False):
		"""
		Import Quick CMT solutions
		"""
		ndk_url = BASE_URL + '/NEW_QUICK/qcmt.ndk'
		start_date = self.get_end_datetime().date() + datetime.timedelta(1)
		self.import_ndk([ndk_url], start_date=start_date, verbose=verbose)

	def create_spatialite_geometries(self):
		"""
		Create spatialite geometries (points defined by hypo_lon and
		hypo_lat columns)
		"""
		if self.db.HAS_SPATIALITE:
			if not 'spatial_ref_sys' in self.db.list_tables():
				self.db.init_spatialite()
			self.db.add_geometry_column(self.table_name, 'geom')
			self.db.create_points_from_columns(self.table_name, 'hypo_lon',
												'hypo_lat')

	def drop_spatialite_geometries(self):
		"""
		Drop spatialite geometries, keeping database intact
		"""
		if 'geom' in self.db.list_table_columns(self.table_name):
			self.db.discard_geometry_column(self.table_name, 'geom')

	def query_generic(self, query, verbose=False):
		"""
		Perform generic query on the database

		:param query:
			string, SQL query
		:param verbose:
			whether or not to print the query
			(default: False)

		:return:
			generator yielding instances of class:`HarvardCMTRecord`
		"""
		for rec in self.db.query_generic(query, verbose=verbose):
			rec = rec.to_dict()
			rec.pop('geom', None)
			yield HarvardCMTRecord.from_sql_record(rec)

	def query(self, column_clause='*', join_clause='',
				where_clause='', having_clause='', order_clause='',
				group_clause='', verbose=False):
		"""
		Query the main harvard_cmt table using separate clauses

		:param column_clause:
			str or list of strings, column clause or list of columns
			(default: "*")
		:param join_clause:
			str or list of (join_type, table_name, condition) tuples,
			join clause
			(default: "")
		:param where_clause:
			str, where clause
			(default: "")
		:param having_clause:
			str, having clause
			(default: "")
		:param order_clause:
			str, order clause
			(default: "")
		:param group_clause:
			str, group clause
			(default: "")

		:return:
			generator yielding instances of class:`HarvardCMTRecord`
		"""
		table_clause = self.table_name

		for rec in self.db.query(table_clause, column_clause=column_clause,
								join_clause=join_clause, where_clause=where_clause,
								having_clause=having_clause, order_clause=order_clause,
								group_clause=group_clause, verbose=verbose):
			rec = rec.to_dict()
			rec.pop('geom', None)
			yield HarvardCMTRecord.from_sql_record(rec)

	def get_records(self, region=None, start_date=None, end_date=None,
					Mmin=None, Mmax=None, min_depth=None, max_depth=None,
					ref_catalog='', verbose=False):
		"""
		Fetch records corresponding to given criteria

		:param region:
			(west, east, south, north) defining limits of rectangular
			geographic region
			(default: None)
		:param start_date:
			instance of :class:`datetime.date`, start date
			(default: None)
		:param end_date:
			instance of :class:`datetime.date`, end date
			(default: None)
		:param Mmin:
			float, minimum magnitude
			(default: None)
		:param Mmax:
			float, maximum magnitude
			(default: None)
		:param min_depth:
			float, minimum depth
			(default: None)
		:param max_depth:
			float, maximum depth
			(default: None)
		:param verbose:
			bool, whether or not to print query
			(default: False)

		:return:
			generator yielding instances of class:`HarvardCMTRecord`
		"""
		where_clauses = []
		if region:
			clause = "(hypo_lon BETWEEN %f AND %f) AND (hypo_lat BETWEEN %f AND %f)"
			clause %= region
			where_clauses.append(clause)
		if start_date:
			if isinstance(start_date, int):
				## year
				start_date = datetime.date(start_date, 1, 1)
			clause = "julianday(hypo_date_time) >= julianday('%s')" % start_date
			where_clauses.append(clause)
		if end_date:
			if isinstance(end_date, int):
				end_date = datetime.date(end_date, 12, 31)
			clause = "julianday(hypo_date_time) <= julianday('%s')" % end_date
			where_clauses.append(clause)
		if min_depth:
			clause = "hypo_depth >= %f" % min_depth
			where_clauses.append(clause)
		if max_depth:
			clause = "hypo_depth <= %f" % max_depth
			where_clauses.append(clause)
		if Mmin:
			min_moment = mag_to_moment(Mmin, unit='dyn.cm')
			clause = "(moment * POWER(10, exp)) >= %E" % min_moment
			where_clauses.append(clause)
		if Mmax:
			max_moment = mag_to_moment(Mmax, unit='dyn.cm')
			clause = "(moment * POWER(10, exp)) <= %E" % max_moment
			where_clauses.append(clause)
		if ref_catalog:
			clause = "ref_catalog = '%s'" % ref_catalog
			where_clauses.append(clause)

		where_clause = " AND ".join(where_clauses)

		return self.query(where_clause=where_clause, verbose=verbose)

	def subselect(self, db_filespec=':memory:', region=None, start_date=None,
					end_date=None, Mmin=None, Mmax=None, min_depth=None,
					max_depth=None, ref_catalog='', verbose=False):
		"""
		Similar to :meth:`get_records`, but returns a new CMT catalog,
		stored in memory

		:param db_filespec:
			str, full path to output database file
			(default: ':memory:')

		:return:
			instance of :class:`HarvardCMTCatalog` (without spatial
			tables and columns)
		"""
		hcmt = HarvardCMTCatalog(':memory:')
		hcmt.clear_db()
		hcmt.import_records(self.get_records(region=region, start_date=start_date,
							end_date=end_date, Mmin=Mmin, min_depth=min_depth,
							max_depth=max_depth, ref_catalog=ref_catalog,
							verbose=verbose))
		return hcmt

	def copy(self, db_filespec=':memory:'):
		"""
		Copy CMT catalog to another database file or to memory

		:param db_filespec:
			see :meth:`subselect`

		:return:
			instance of :class:`HarvardCMTCatalog` (without spatial
			tables and columns)
		"""
		return self.subselect(db_filespec=db_filespec, verbose=False)

	def get_end_datetime(self):
		"""
		Determine datetime of most recent record in the database

		:return:
			instance of :class:`datetime.datetime`
		"""
		query = "SELECT * FROM '%s' ORDER BY hypo_date_time DESC LIMIT 1"
		query %= self.table_name
		[rec] = list(self.query_generic(query))
		return rec.hypo_date_time

	def get_total_moment(self, unit='dyn.cm'):
		"""
		Compute total seismic moment in catalog

		:return:
			float,
		"""
		query = 'SELECT SUM(moment * POWER(10,exp)) AS "total_moment" from %s'
		query %= self.table_name
		[result] = list(self.db.query_generic(query))
		total_moment = result['total_moment']
		if unit == 'N.m':
			total_moment *= 1E-7

		return total_moment

	def to_eq_catalog(self):
		"""
		Convert to generic earthquake catalog

		:return:
			instance of :class:`EQCatalog`
		"""
		from .eqcatalog import EQCatalog

		eq_list = []
		for rec in self.get_records():
			eq_list.append(rec.to_local_eq())
		return EQCatalog(eq_list, name="Harvard CMT",
						start_date=datetime.date(1976, 1, 1))


def get_harvard_cmt_catalog():
	"""
	Get Harvard CMT catalog from seismogis

	:return:
		instance of :class:`HarvardCMTCatalog`
	"""
	from .rob import get_dataset_file_on_seismogis

	GCMT_DB_FILE = get_dataset_file_on_seismogis('Harvard_CMT', 'Harvard_CMT')

	if GCMT_DB_FILE:
		hcmt_cat = HarvardCMTCatalog(GCMT_DB_FILE)
		return hcmt_cat


def update_harvard_cmt_catalog():
	"""
	Update local database containing Harvard CMT catalog
	"""
	harvard_cmt = get_harvard_cmt_catalog()
	harvard_cmt.reload(clear_db=True, verbose=True)



if __name__ == "__main__":
	## Update catalog
	update_harvard_cmt_catalog()
	exit()

	ndk_records = """
	PDE  2005/01/01 01:20:05.4  13.78  -88.78 193.1 5.0 0.0 EL SALVADOR
	C200501010120A   B:  4    4  40 S: 27   33  50 M:  0    0   0 CMT: 1 TRIHD:  0.6
	CENTROID:     -0.3 0.9  13.76 0.06  -89.08 0.09 162.8 12.5 FREE S-20050322125201
	23  0.838 0.201 -0.005 0.231 -0.833 0.270  1.050 0.121 -0.369 0.161  0.044 0.240
	V10   1.581 56  12  -0.537 23 140  -1.044 24 241   1.312   9 29  142 133 72   66
	"""

	"""
	PDE  2005/01/01 01:42:24.9   7.29   93.92  30.0 5.1 0.0 NICOBAR ISLANDS, INDIA R
	C200501010142A   B: 17   27  40 S: 41   58  50 M:  0    0   0 CMT: 1 TRIHD:  0.7
	CENTROID:     -1.1 0.8   7.24 0.04   93.96 0.04  12.0  0.0 BDY  S-20050322125628
	23 -1.310 0.212  2.320 0.166 -1.010 0.241  0.013 0.535 -2.570 0.668  1.780 0.151
	V10   3.376 16 149   0.611 43  44  -3.987 43 254   3.681 282 48  -23  28 73 -136
	"""
	print(ndk_records.splitlines()[1:6])
	rec = HarvardCMTRecord.from_ndk_record(ndk_records)
	print(rec.ID)
	print(dir(rec))
	print(rec.__dict__)

	"""
	db_filespec = "C:\\Users\\kris\\Documents\\Python\\notebooks\\Vienna\\HarvardCMT.sqlite"
	cmt_catalog = HarvardCMTCatalog(db_filespec)
	print(len(cmt_catalog))
	exit()
	#cmt_catalog.read_from_web(clear_db=True)
	#recs = list(cmt_catalog.get_records())
	#print(recs[0].hypo_date_time.isoformat())

	eq_catalog = cmt_catalog.to_eq_catalog()
	iets = eq_catalog.get_inter_event_times()
	print(iets.mean(), iets.std())
	import pylab
	pylab.hist(iets, bins=np.linspace(0,3,50))
	pylab.show()
	exit()

	from eqcatalog import Completeness
	completeness = Completeness([1975], [4.], Mtype="MW")
	mfd = eq_catalog.get_incremental_MFD(4, 9, 0.1, completeness=completeness)
	mfd.plot()
	exit()
	"""

	cmt_catalog = HarvardCMTCatalog(':memory:')
	ndk_filespec = "C:\\Users\\kris\\Downloads\\jan76_dec13.ndk"
	ndk_url = BASE_URL + "/jan76_dec13.ndk"
	cmt_records = cmt_catalog.parse_ndk_file(ndk_filespec)
	print(len(cmt_records))
	rec = cmt_records[-1]
	print(rec.Mrr, rec.Mtt, rec.Mpp)
	print(rec.hypo_lon, rec.centroid_lon)

	db = cmt_catalog.import_ndk([ndk_filespec])
	rec = cmt_catalog.get_records().next()
	print(rec.location)
