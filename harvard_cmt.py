import datetime
from collections import OrderedDict

import numpy as np

import db.simpledb as simpledb


HarvardCMTColDef = OrderedDict([
	('ID', 'STRING'),
	('ref_catalog', 'STRING'),
	('hypo_date_time', 'TIMESTAMP'),
	('hypo_lon', 'REAL'),
	('hypo_lat', 'REAL'),
	('hypo_depth', 'REAL'),
	('ref_mb', 'REAL'),
	('ref_MS', 'REAL'),
	('location', 'STRING'),
	('source_type', 'STRING'),
	('mrf_type', 'STRING'),
	('mrf_duration', 'REAL'),
	('centroid_reltime', 'REAL'),
	('centroid_reltime_sigma', 'REAL'),
	('centroid_lon', 'REAL'),
	('centroid_lon_sigma', 'REAL'),
	('centroid_lat', 'REAL'),
	('centroid_lat_sigma', 'REAL'),
	('centroid_depth', 'REAL'),
	('centroid_depth_sigma', 'REAL'),
	('centroid_depth_type', 'STRING'),
	('exp', 'INTEGER'),
	('Mrr', 'REAL'),
	('Mrr_sigma', 'REAL'),
	('Mtt', 'REAL'),
	('Mtt_sigma', 'REAL'),
	('Mpp', 'REAL'),
	('Mpp_sigma', 'REAL'),
	('Mrt', 'REAL'),
	('Mrt_sigma', 'REAL'),
	('Mrp', 'REAL'),
	('Mrp_sigma', 'REAL'),
	('Mtp', 'REAL'),
	('Mtp_sigma', 'REAL'),
	('eva1', 'REAL'),
	('pl1', 'INTEGER'),
	('az1', 'INTEGER'),
	('eva2', 'REAL'),
	('pl2', 'INTEGER'),
	('az2', 'INTEGER'),
	('eva3', 'REAL'),
	('pl3', 'INTEGER'),
	('az3', 'INTEGER'),
	('moment', 'REAL'),
	('strike1', 'INTEGER'),
	('dip1', 'INTEGER'),
	('rake1', 'INTEGER'),
	('strike2', 'INTEGER'),
	('dip2', 'INTEGER'),
	('rake2', 'INTEGER')
])

def cnv_coltype(colname, value):
	func = {'INTEGER': int, 'REAL': float, 'STRING': str, 'TIMESTAMP': lambda x:x}[HarvardCMTColDef[colname]]
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
		self.centroid_reltime_sigma = cnv_coltype('centroid_reltime_sigma', centroid_reltime_sigma)
		self.centroid_lon = cnv_coltype('centroid_lon', centroid_lon)
		self.centroid_lon_sigma = cnv_coltype('centroid_lon_sigma', centroid_lon_sigma)
		self.centroid_lat = cnv_coltype('centroid_lat', centroid_lat)
		self.centroid_lat_sigma = cnv_coltype('centroid_lat_sigma', centroid_lat_sigma)
		self.centroid_depth = cnv_coltype('centroid_depth', centroid_depth)
		self.centroid_depth_sigma = cnv_coltype('centroid_depth_sigma', centroid_depth_sigma)
		self.centroid_depth_type = cnv_coltype('centroid_depth_type', centroid_depth_type)
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

	@property
	def hypo_date(self):
		return self.hypo_date_time.date()

	@property
	def hypo_time(self):
		return self.hypo_date_time.time()

	@property
	def MW(self):
		return (2./3) * np.log10(self.moment * 10**self.exp) - 10.7

	@classmethod
	def from_ndk_record(cls, ndk_record):
		"""
		:param ndk_record:
			str or list of strings corresponding to record lines
		"""
		if isinstance(ndk_record, (str, unicode)):
			ndk_record = ndk_record.strip().splitlines()
		for i, line in enumerate(ndk_record):
			line = line.strip('\t')
			if i == 0:
				ref_catalog = line[:4].strip()
				hypo_date, hypo_time, hypo_lat, hypo_lon, hypo_depth, ref_mb, ref_MS = line[4:56].split()
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
				centroid_time_sigma, centroid_lat, centroid_lat_sigma, centroid_lon, centroid_lon_sigma, centroid_depth, centroid_depth_sigma, centroid_depth_type = line[19:63].split()
			elif i == 3:
				exp, Mrr, Mrr_sigma, Mtt, Mtt_sigma, Mpp, Mpp_sigma, Mrt, Mrt_sigma, Mrp, Mrp_sigma, Mtp, Mtp_sigma = line.split()
			elif i == 4:
				eva1, pl1, az1, eva2, pl2, az2, eva3, pl3, az3, moment, strike1, dip1, rake1, strike2, dip2, rake2 = line[3:].split()

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
		from eqcatalog import LocalEarthquake

		ID = self.ID
		date = self.hypo_date
		time = self.hypo_time
		lon = self.hypo_lon
		lat = self.hypo_lat
		depth = self.hypo_depth
		mag = {'MW': self.MW, 'MS': self.ref_MS, 'mb': self.ref_mb}
		name = self.location
		return LocalEarthquake(ID, date, time, lon, lat, depth, mag=mag, name=name)


class HarvardCMTCatalog:
	def __init__(self, db_filespec):
		self.db_filespec = db_filespec
		self.db = simpledb.SQLiteDB(db_filespec)
		self.table_name = 'harvard_cmt'

	def __len__(self):
		return self.db.get_num_rows(self.table_name)

	@classmethod
	def parse_ndk_file(cls, ndk_filespec_or_url):
		"""
		Parse NDK file or URL

		:param ndk_filespec_or_url:
			str, full path or url to NDK file

		:return:
			list with instances of :class:`HarvardCMTRecord`
		"""
		cmt_records = []
		if ndk_filespec_or_url[:4] == "http":
			import urllib
			ndk = urllib.urlopen(ndk_filespec_or_url)
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

	def import_ndk(self, ndk_filespecs, start_date=datetime.date(1900, 1, 1), clear_db=False):
		if clear_db:
			self.db.drop_table(self.table_name)
		if not self.table_name in self.db.list_tables():
			self.db.create_table(self.table_name, HarvardCMTColDef, 'ID')
		for ndk_filespec in ndk_filespecs:
			print ndk_filespec
			recs = self.parse_ndk_file(ndk_filespec)
			self.db.add_records(self.table_name, [rec.to_dict() for rec in recs if rec.hypo_date >= start_date])

	def read_from_web(self, clear_db=False):
		base_url = "http://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog"
		today = datetime.date.today()
		current_year, current_month = today.year, today.month
		if current_month > 4:
			current_month -= 4
		else:
			current_year -= 1
			current_month += (12 - 4)

		ndk_urls = ["jan76_dec13.ndk"]
		months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
		for year in range(2014, current_year+1):
			if year == current_year:
				end_month = current_month
			else:
				end_month = 12
			for m in range(end_month):
				url = "NEW_MONTHLY/%d/%s%d.ndk" % (year, months[m], year-2000)
				ndk_urls.append(url)

		ndk_urls = [base_url + '/' + url for url in ndk_urls]
		self.import_ndk(ndk_urls, clear_db=clear_db)

		## Add Quick CMTs
		ndk_url = base_url + '/NEW_QUICK/qcmt.ndk'
		if current_month == 12:
			start_month = 1
			start_year = current_year + 1
		else:
			start_month = current_month + 1
			start_year = current_year
		start_date = datetime.date(start_year, start_month, 1)
		self.import_ndk([ndk_url], start_date=start_date)

	def get_records(self):
		for rec in self.db.query(self.table_name):
			yield HarvardCMTRecord.from_sql_record(rec)

	def to_eq_catalog(self):
		from eqcatalog import EQCatalog

		eq_list = []
		for rec in self.get_records():
			eq_list.append(rec.to_local_eq())
		return EQCatalog(eq_list, name="Harvard CMT", start_date=datetime.date(1976, 1, 1))



if __name__ == "__main__":
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
	print ndk_records.splitlines()[1:6]
	rec = HarvardCMTRecord.from_ndk_record(ndk_records)
	print rec.ID
	print dir(rec)
	print rec.__dict__

	"""
	db_filespec = r"C:\Users\kris\Documents\Python\notebooks\Vienna\HarvardCMT.sqlite"
	cmt_catalog = HarvardCMTCatalog(db_filespec)
	print len(cmt_catalog)
	exit()
	#cmt_catalog.read_from_web(clear_db=True)
	#recs = list(cmt_catalog.get_records())
	#print recs[0].hypo_date_time.isoformat()

	eq_catalog = cmt_catalog.to_eq_catalog()
	iets = eq_catalog.get_inter_event_times()
	print iets.mean(), iets.std()
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
	ndk_filespec = r"C:\Users\kris\Downloads\jan76_dec13.ndk"
	ndk_url = "http://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec13.ndk"
	cmt_records = cmt_catalog.parse_ndk_file(ndk_filespec)
	print len(cmt_records)
	rec = cmt_records[-1]
	print rec.Mrr, rec.Mtt, rec.Mpp
	print rec.hypo_lon, rec.centroid_lon

	db = cmt_catalog.import_ndk([ndk_filespec])
	rec = cmt_catalog.get_records().next()
	print rec.location