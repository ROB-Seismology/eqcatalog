"""
Read earthquake catalogs from various sources
"""

from __future__ import absolute_import, division, print_function #, unicode_literals

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str


## Import standard python modules
import os
import datetime

## Import thirdparty modules
import numpy as np

## Import ROB modules
import db.simpledb as simpledb

## Import submodules
from ..time import time_tuple_to_np_datetime
from ..eqrecord import LocalEarthquake
from ..eqcatalog import EQCatalog
from ..rob import get_dataset_file_on_seismogis, GIS_ROOT


# TODO: use seismo-gis

__all__ = ["read_named_catalog", "read_catalog_sql", "read_catalog_csv",
			"read_catalog_gis"]


def read_named_catalog(catalog_name, fix_zero_days_and_months=False, null_value=np.nan,
						verbose=True):
	"""
	Read a known catalog (corresponding files should be in standard location)

	:param catalog_name:
		str, name of catalog ("ROB", HARVARD_CMT", "SHEEC", "CENEC", "ISC-GEM",
		"CEUS-SCR", "BGS", "EMEC")
	:param fix_zero_days_and_months:
		bool, if True, zero days and months are replaced with ones
		(default: False)
	:param null_value:
		float, value to use for NULL values (except magnitude)
		(default: np.nan)
	:param verbose:
		bool, whether or not to print information while reading
		GIS table (default: True)

	:return:
		instance of :class:`EQCatalog`
	"""
	if catalog_name.upper() == "ROB":
		from ..rob import query_local_eq_catalog
		region = (0., 8., 49., 52.)
		start_date = datetime.date(1350, 1, 1)
		return query_local_eq_catalog(region=region, start_date=start_date,
									null_value=null_value, verbose=verbose)

	elif catalog_name.upper() in ("HARVARD_CMT", "HARVARD CMT"):
		from ..harvard_cmt import get_harvard_cmt_catalog

		hcmt_cat = get_harvard_cmt_catalog()
		sql_file = hcmt_cat.db_filespec
		#return hcmt_cat.to_eq_catalog()
		sqldb = simpledb.SQLiteDB(sql_file)
		table_name = 'harvard_cmt'
		if not sqldb.HAS_SPATIALITE:
			sqldb.connection.create_function("LOG10", 1, np.log10)
		query = ('SELECT *, ((2./3) * (exp + LOG10(moment)) - 10.73) '
				'as "MW" FROM harvard_cmt')
		column_map =  {'ID': 'ID', 'datetime': 'hypo_date_time',
					'lon': 'hypo_lon', 'lat': 'hypo_lat', 'depth': 'hypo_depth',
					'MS': 'ref_MS', 'MW': 'MW', 'name': 'location',
					'agency': 'Harvard CMT'}
		return read_catalog_sql(sqldb, table_name, query=query, column_map=column_map,
								null_value=null_value, verbose=verbose)

	elif catalog_name.upper() == "EMEC":
		csv_file = get_dataset_file_on_seismogis('EMEC', 'EMEC')

		if csv_file:
			column_map = {'year': 0, 'month': 1, 'day': 2, 'hour': 3, 'minute': 4,
						'lat': 5, 'lon': 6, 'depth': 7, 'intensity_max': 8,
						'Mag': 9, 'Mtype': 10, 'MW': 11, 'errM': 12, 'agency': 13}
			return read_catalog_csv(csv_file, column_map, has_header=False,
								ID_prefix='EMEC', delimiter=';', null_value=null_value)

	elif catalog_name.upper() == "BGS":
		csv_file = get_dataset_file_on_seismogis('BGS_Seismology',
												'BGS earthquake catalog.csv')

		if csv_file:
			column_map = {'date': 'yyyy-mm-dd', 'time': 'hh:mm:ss.ss',
						'intensity_max': 'intensity', 'name': 'locality'}
			return read_catalog_csv(csv_file, column_map, has_header=True,
								ID_prefix='BGS', null_value=null_value,
								ignore_chars= '>+F')

	else:
		date_sep = '/'
		if catalog_name.upper() == "SHEEC":
			gis_filespec = os.path.join(GIS_ROOT, "SHARE", "SHEEC", "Ver3.3", "SHAREver3.3.shp")
			column_map = {'lon': 'Lon', 'lat': 'Lat',
						'year': 'Year', 'month': 'Mo', 'day': 'Da',
						'hour': 'Ho', 'minute': 'Mi', 'second': 'Se',
						'MW': 'Mw', 'depth': 'H', 'ID': 'event_id',
						'errh': 'LatUnc', 'errz': 'HUnc', 'errM': 'MwUnc'}
			#convert_zero_magnitudes = True
		elif catalog_name.upper() == "CENEC":
			gis_filespec = os.path.join(GIS_ROOT, "Seismology", "Earthquake Catalogs",
										"CENEC", "CENEC 2008.TAB")
			column_map = {'lon': 'lon', 'lat': 'lat',
						'year': 'year', 'month': 'month', 'day': 'day',
						'hour': 'hour', 'minute': 'minute',
						'Mag': 'Morig', 'Mtype': 'Mtype', 'MW': 'Mw',
						'depth': 'depth', 'intensity_max': 'Imax',
						'agency': 'ref'}
			#convert_zero_magnitudes = True
		elif catalog_name.upper() == "ISC-GEM":
			gis_filespec = get_dataset_file_on_seismogis('ISC-GEM',
														'isc-gem-cat')
			column_map = {'ID': 'eventid', 'lon': 'lon', 'lat': 'lat',
						'date': 'date', 'time': 'time',
						'MW': 'mw', 'depth': 'depth',
						'errz': 'unc', 'errM': 'unc_2', 'agency': 'ISC-GEM'}
			#convert_zero_magnitudes = True
		elif catalog_name.upper() == "CEUS-SCR":
			gis_filespec = get_dataset_file_on_seismogis('CEUS_SSC_SCR',
														'CEUS_SCR_Catalog_2012')
			column_map = {'lon': 'Longitude', 'lat': 'Latitude',
						'year': 'Year', 'month': 'Month', 'day': 'Day',
						'hour': 'Hour', 'minute': 'Minute', 'second': 'Second',
						'MW': 'E_M_', 'errM': 'sigma_M', 'zone': 'DN'}
			#convert_zero_magnitudes = True
		else:
			raise Exception("Catalog not recognized: %s" % catalog_name)

		if not os.path.exists(gis_filespec):
			raise Exception("Catalog file not found: %s" % gis_filespec)
		ID_prefix = catalog_name + "-"
		#eqc = read_catalogGIS(gis_filespec, column_map, fix_zero_days_and_months=fix_zero_days_and_months,
		#					convert_zero_magnitudes=convert_zero_magnitudes, ID_prefix=ID_prefix, verbose=verbose)
		eqc = read_catalog_gis(gis_filespec, column_map, date_sep=date_sep,
					ID_prefix=ID_prefix, null_value=null_value, verbose=verbose)
		eqc.name = catalog_name
		return eqc


def read_catalog_sql(sql_db, tab_name, query='', column_map={}, ID_prefix='',
					date_sep='-', time_sep=':', date_order='YMD',
					null_value=np.nan, ignore_errors=False, verbose=True):
	"""
	Read catalog from SQL database

	:param sqldb:
		instance of :class:`db.simpledb.SQLDB`
	:param tab_name:
		str, name of database table containing catalog
		Will be ignored if :param:`query` is not empty (but is still
		useful as it will be used as catalog name)
	:param query:
		str, generic query string
		(default: '')
	:param column_map:
		dict, mapping property names of :class:`LocalEarthquake` to
		database column names.
		(default: {})
	:param ID_prefix:
		str, prefix to add to earthquake IDs
		(default: '')
	:param date_sep:
		str, character separating date elements
		(default: '-')
	:param time_sep:
		str, character separating time elements
		(default: ':'
	:param date_order:
		str, order of year (Y), month (M), day (D) in date string
		(default: 'YMD')
	:param null_value:
		float, value to use for NULL values (except magnitude)
		(default: np.nan)
	:param ignore_errors:
		bool, whether or not records that cannot be parsed should be
		silently ignored
		(default: False, will raise exception)
	:param verbose:
		bool, whether or not to print information while reading file
		(default: True)

	:return:
		instance of :class:`EQCatalog`
	"""
	if verbose:
		print("Reading catalog from SQL database table %s" % tab_name)

	if not query:
		query = 'SELECT * FROM %s' % tab_name

	eq_list = []
	num_skipped = 0
	for r, rec in enumerate(sql_db.query_generic(query, verbose=verbose)):
		rec = rec.to_dict()
		## If no ID is present, use record number
		ID_key = column_map.get('ID', 'ID')
		rec[ID_key] = ID_prefix + str(rec.get(ID_key, r))

		try:
			eq = LocalEarthquake.from_dict_rec(rec, column_map=column_map,
				date_sep=date_sep, time_sep=time_sep, date_order=date_order,
				null_value=null_value)
		except:
			if not ignore_errors:
				raise
			else:
				num_skipped += 1
		else:
			eq_list.append(eq)

	catalog = EQCatalog(eq_list, name=tab_name)
	if verbose and num_skipped:
		print("  Skipped %d records" % num_skipped)
	return(catalog)


def read_catalog_csv(csv_filespec, column_map={}, has_header=None, ID_prefix='',
					date_sep='-', time_sep=':', date_order='YMD',
					comment_char='#', ignore_chars=[], ignore_errors=False,
					null_value=np.nan, verbose=False, **fmtparams):
	"""
	Read earthquake catalog from CSV file with columns defining
	earthquake properties: ID, datetime or (date or (year, month, day))
	and (time or (hours, minutes, seconds)), lon, lat, depth, name,
	zone, (Mtype and Mag) or (ML and/or MS and/or MW), intensity_max,
	macro_radius, errh, errz, errt, errM.

	Property names can be specified in a (single) header line.
	If no header line is present, :param:`column_map` should map
	standard earthquake property names to column numbers.
	If header line is present, but property names do not correspond
	to standard names, a column map should provide mapping between
	standard property names and column names in header.

	Most properties are optional.
	For the date, at least the year must be given. If month or day
	are not given or zero, they will be silently set to 1
	Empty magnitudes are set to NaN,
	all other empty properties are set to 0 or ''.

	:param csv_filespec:
		str, full path to CSV file containing earthquake records
	:param column_map:
		dict, mapping property names of :class:`LocalEarthquake` to
		column names in header or to column numbers (zero-based) if no
		header is present.
		(default: {})
	:param has_header:
		bool, whether or not header line with column names is present.
		If None, presence of a header will be auto-detected
		(default: None)
	:param ID_prefix:
		str, prefix to add to earthquake IDs
		(default: '')
	:param date_sep:
		str, character separating date elements
		(default: '-')
	:param time_sep:
		str, character separating time elements
		(default: ':')
	:param date_order:
		str, order of year (Y), month (M), day (D) in date string
		(default: 'YMD')
	:param comment_char:
		char, character used to denote comments. All text following
		this character will be ignored
		(default: '#')
	:param ignore_chars:
		string containing characters or list containing strings that may
		sometimes be present in a column and should be ignored
		(e.g., '*')
		(default: [])
	:param ignore_errors:
		bool, whether or not records that cannot be parsed should be
		silently ignored
		(default: False, will raise exception)
	:param null_value:
		float, value to use for NULL values (except magnitude)
		(default: np.nan)
	:param verbose:
		bool, whether or not to print information while reading file
		(default: True)
	:param **fmtparams:
		kwargs for csv reader (e.g. "delimiter", "quotechar",
		"doublequote", "escapechar")

	:return:
		instance of :class:`EQCatalog`
	"""
	import csv

	## python CSV module has no mechanism to skip comments
	def decomment(csv_fp, comment_char):
		for row in csv_fp:
			raw = row.split(comment_char)[0].strip()
			if raw:
				yield raw

	if verbose:
		print("Reading CSV earthquake catalog from %s" % csv_filespec)

	with open(csv_filespec, "r") as fp:
		## Auto-detect header containing column names
		if has_header is None:
			sniffer = csv.Sniffer()
			has_header = sniffer.has_header(fp.read(1024))
			fp.seek(0)

		if not has_header:
			## If there is no header, column_map should map standard
			## LocalEarthquake property names to integer column numbers
			assert column_map and isinstance(list(column_map.values())[0], int)
			cm2 = {val:key for key,val in column_map.items()}
			fieldnames = [cm2.get(k) for k in range(0, max(cm2.keys())+1)]
			column_map = {}
		else:
			fieldnames = None

		eq_list = []
		num_skipped = 0
		reader = csv.DictReader(decomment(fp, comment_char),
								fieldnames=fieldnames, **fmtparams)
		for r, row in enumerate(reader):
			## If column_map is still empty, infer it from keys of 1st row
			if r == 0 and not column_map:
				for col_name in row.keys():
					col_name = col_name.strip()
					if col_name in ('ID', 'ML', 'MS', 'MW', 'Mtype'):
						column_map[col_name] = col_name
					else:
						if col_name is not None:
							column_map[col_name.lower()] = col_name

			## Remove unmapped columns
			if None in row:
				del row[None]

			## Remove ignore_chars from record values
			for ic in ignore_chars:
				for key, val in row.items():
					if val and not key.isalpha():
						row[key] = val.replace(ic, '')

			## Strip leading/traling white space
			## Note: list(row.items()) because altering of keys during loop
			## not allowed in PY3
			for key, val in list(row.items()):
				stripped_key = key.strip()
				if stripped_key != key:
					del row[key]
					key = stripped_key
					if val:
						row[key] = val.strip()

			## If no ID is present, use record number
			ID_key = column_map.get('ID', 'ID')
			row[ID_key] = ID_prefix + str(row.get(ID_key, r))

			try:
				eq = LocalEarthquake.from_dict_rec(row, column_map=column_map,
					date_sep=date_sep, time_sep=time_sep, date_order=date_order,
					null_value=null_value)
			except:
				if not ignore_errors:
					print("Error in record #%d" % r)
					print(row)
					raise
				else:
					num_skipped += 1
			else:
				eq_list.append(eq)

	name = os.path.split(csv_filespec)[-1]
	catalog = EQCatalog(eq_list, name=name)
	if verbose and num_skipped:
		print("  Skipped %d records" % num_skipped)
	return(catalog)


def read_catalog_gis(gis_filespec, column_map={}, ID_prefix='',
					date_sep='-', time_sep=':', date_order='YMD',
					null_value=np.nan, ignore_chars=[], ignore_errors=False,
					verbose=False):
	"""
	Read catalog from GIS file

	:param gis_filespec:
		str, full path to GIS file containing earthquake records
	:param column_map:
		dict, mapping property names of :class:`LocalEarthquake` to
		GIS record attributes.
		If 'lon' or 'lat' are not specified, they will be derived from
		the geographic object.
		(default: {})
	:param ID_prefix:
		str, prefix to add to earthquake IDs
		(default: '')
	:param date_sep:
		str, character separating date elements
		(default: '-')
	:param time_sep:
		str, character separating time elements
		(default: ':'
	:param date_order:
		str, order of year (Y), month (M), day (D) in date string
		(default: 'YMD')
	:param null_value:
		float, value to use for NULL values (except magnitude)
		(default: np.nan)
	:param ignore_chars:
		string containing characters or list containing strings that may
		sometimes be present in a column and should be ignored
		(e.g., '*')
		(default: [])
	:param ignore_errors:
		bool, whether or not records that cannot be parsed should be
		silently ignored
		(default: False, will raise exception)
	:param verbose:
		bool, whether or not to print information while reading file
		(default: True)

	:return:
		instance of :class:`EQCatalog`
	"""
	from mapping.geotools.read_gis import read_gis_file

	if verbose:
		print("Reading GIS earthquake catalog from %s" % gis_filespec)

	eq_list = []
	num_skipped = 0
	data = read_gis_file(gis_filespec, verbose=verbose)
	for r, rec in enumerate(data):
		## Remove ignore_chars from record values
		for ic in ignore_chars:
			for key, val in rec.items():
				if isinstance(val, basestring):
					rec[key] = val.replace(ic, '')

		## If no ID is present, use record number
		ID_key = column_map.get('ID', 'ID')
		rec[ID_key] = ID_prefix + str(rec.get(ID_key, r))

		## Get lon/lat from object if not present in GIS attributes
		lon_key = column_map.get('lon', 'lon')
		if not lon_key in rec:
			rec[lon_key] = rec["obj"].GetX()

		lat_key = column_map.get('lat', 'lat')
		if not lat_key in rec:
			rec[lat_key] = rec["obj"].GetY()

		try:
			eq = LocalEarthquake.from_dict_rec(rec, column_map=column_map,
				date_sep=date_sep, time_sep=time_sep, date_order=date_order,
				null_value=null_value)
		except:
			if not ignore_errors:
				raise
			else:
				num_skipped += 1
		else:
			eq_list.append(eq)

	name = os.path.split(gis_filespec)[-1]
	catalog = EQCatalog(eq_list, name=name)
	if verbose and num_skipped:
		print("  Skipped %d records" % num_skipped)
	return(catalog)



## The following functions are obsolete, and have been replaced by
## read_catalog_csv and read_catalog_gis respectively

def read_catalogTXT(filespec, column_map={"id": 0, "date": 1, "time": 2, "name": 3, "lon": 4, "lat": 5, "depth": 6, "ML": 7, "MS": 8, "MW": 9},
					header=True, date_sep='-', time_sep=':', date_order='YMD', convert_zero_magnitudes=False, ignore_warnings=False, ID_prefix="",
					ignore_chars=[], **fmtparams):
	"""
	Read ROB local earthquake catalog from csv file.

	:param filespec:
		String, defining filespec of a text file with columns defining attributes
		id, date (or year, month and day), time (or hours, minutes and seconds),
		lon, lat, depth, ML, MS and MW. All are optional.
	:param column_map:
		Dictionary, mapping attributes to number of column (starting from 0).
		(default: {"id": 0, "date": 1, "time": 2, "name": 3, "lon": 4, "lat": 5, "depth": 6, "ML": 7, "MS": 8, "MW": 9})
	:param header:
		bool, if one-line header is present
		or int, number of header lines
		(default: True).
	:param date_sep:
		str, character separating date elements
		(default: '-')
	:param time_sep:
		str, character separating time elements
	:param date_order:
		str, order of year (Y), month (M), day (D) in date string
		(default: 'YMD')
	:param convert_zero_magnitudes:
		bool, whether or not to convert zero magnitudes to NaN values
		(default: False)
	:param ignore_warnings:
		bool, whether or not to print warnings when fields cannot be parsed
		(default: False, will print warnings)
	:param ID_prefix:
		str, prefix to add to earthquake IDs
		(default: "")
	:param ignore_chars:
		list containing characters or strings that may sometimes be
		present in a column and should be ignored
		(e.g., '*')
		(default: [])
	:param **fmtparams:
		kwargs for csv reader (e.g. "delimiter" and "quotechar")

	:returns:
		instance of :class:`EQCatalog`
	"""
	import csv
	#from .time import parse_isoformat_datetime

	date_order = date_order.upper()
	earthquakes = []
	with open(filespec, "r") as f:
		lines = csv.reader(f, **fmtparams)
		for i, line in enumerate(lines):
			if i < header:
				continue
			for j in range(len(line)):
				for ic in ignore_chars:
					line[j] = line[j].replace(ic, '')
			if hasattr(column_map, "id"):
				ID = int(line[column_map["id"]])
			else:
				ID = i - header + 1
			if ID_prefix:
				ID = ID_prefix + str(ID)

			if "datetime" in column_map:
				dt = np.datetime64(line[column_map["datetime"]])
				date = dt.date()
				time = dt.time()
			else:
				if "date" in column_map:
					date = line[column_map["date"]]
					date_elements = date.split(date_sep)
					year = int(date_elements[date_order.index('Y')])
					month = int(date_elements[date_order.index('M')])
					day = int(date_elements[date_order.index('D')])
				else:
					if "year" in column_map:
						try:
							year = int(line[column_map["year"]])
						except ValueError:
							## Skip record if year is invalid
							if not ignore_warnings:
								print("Invalid year in line %d: %s" % (i, line[column_map["year"]]))
							continue
					else:
						year = 1
					if "month" in column_map:
						try:
							month = max(1, int(line[column_map["month"]]))
						except:
							if not ignore_warnings:
								print("Invalid month in line %d: %s. Set to 1." % (i, line[column_map["month"]]))
							month = 1
					else:
						month = 1
					if "day" in column_map:
						try:
							day = max(1, int(line[column_map["day"]]))
						except:
							if not ignore_warnings:
								print("Invalid day in line %d: %s. Set to 1." % (i, line[column_map["day"]]))
							day = 1
					else:
						day = 1
				try:
					date = tf.time_tuple_to_np_datetime(year, month, day)
				except:
					print(line)
				if "time" in column_map:
					time = line[column_map["time"]]
					time_elements = time.split(time_sep)
					try:
						hour = int(time_elements[0])
					except (IndexError, ValueError):
						hour = 0
					try:
						minute = int(time_elements[1])
					except (IndexError, ValueError):
						minute = 0
					try:
						second = float(time_elements[2])
					except (IndexError, ValueError):
						second = 0.
				else:
					if "hour" in column_map:
						try:
							hour = int(line[column_map["hour"]])
						except:
							hour = 0
					else:
						hour = 0
					if "minute" in column_map:
						try:
							minute = int(line[column_map["minute"]])
						except:
							minute = 0
					else:
						minute = 0
					if "second" in column_map:
						try:
							second = int(line[column_map["second"]])
						except:
							second = 0
					else:
						second = 0
				time = datetime.time(hour, minute, second)

			if "lon" in column_map:
				lon = float(line[column_map["lon"]])
			else:
				lon = 0.
			if "lat" in column_map:
				lat = float(line[column_map["lat"]])
			else:
				lat = 0.
			if "depth" in column_map:
				try:
					depth = float(line[column_map["depth"]])
				except:
					depth = 0
			else:
				depth = 0.

			mag = {}
			if "Mtype" in column_map:
				Mtype = line[column_map["Mtype"]]
				if "Mag" in column_map:
					try:
						M = float(line[column_map["Mag"]])
					except ValueError:
						if not ignore_warnings:
							print("Invalid Mag in line %d: %s" % (i, line[column_map["Mag"]]))
						M = np.nan
					if convert_zero_magnitudes:
						mag = {Mtype: M or np.nan}
					else:
						mag = {Mtype: M}

			if "ML" in column_map:
				try:
					ML = float(line[column_map["ML"]])
				except ValueError:
					if not ignore_warnings:
						print("Invalid ML in line %d: %s" % (i, line[column_map["ML"]]))
					ML = np.nan
			else:
				ML = np.nan
			if convert_zero_magnitudes:
				ML = ML or np.nan
			if "MS" in column_map:
				try:
					MS = float(line[column_map["MS"]])
				except ValueError:
					if not ignore_warnings:
						print("Invalid MS in line %d: %s" % (i, line[column_map["MS"]]))
					MS = np.nan
			else:
				MS = np.nan
			if convert_zero_magnitudes:
				MS = MS or np.nan
			if "MW" in column_map:
				try:
					MW = float(line[column_map["MW"]])
				except ValueError:
					if not ignore_warnings:
						print("Invalid MW in line %d: %s" % (i, line[column_map["MW"]]))
					MW = np.nan
			else:
				MW = np.nan
			if convert_zero_magnitudes:
				MW = MW or np.nan
			if "name" in column_map:
				name = line[column_map["name"]]
			else:
				name = ""
			if "intensity_max" in column_map:
				try:
					intensity_max = float(line[column_map["intensity_max"]])
				except:
					intensity_max = 0.
			else:
				intensity_max = 0.
			if "macro_radius" in column_map:
				try:
					macro_radius = float(line[column_map["macro_radius"]])
				except:
					macro_radius = 0.
			else:
				macro_radius = 0.
			if "zone" in column_map:
				zone =  line[column_map["zone"]]
			else:
				zone = ""

			earthquakes.append(LocalEarthquake(ID, date, time, lon, lat, depth,
							mag, ML, MS, MW, name=name, intensity_max=intensity_max,
							macro_radius=macro_radius, zone=zone))
	catalog = EQCatalog(earthquakes)
	return catalog


def read_catalogGIS(gis_filespec, column_map, fix_zero_days_and_months=False,
					convert_zero_magnitudes=False, ID_prefix="", verbose=True):
	"""
	Read catalog from GIS file

	:param gis_filespec:
		Str, full path to GIS file containing catalog
	:param column_map:
		dict, mapping properties ('date', 'year', 'month', 'day', 'time',
			'hour', 'minute', 'second', 'lon', 'lat', 'depth', 'MW', 'MS', 'ML',
			'name', 'intensity_max', 'macro_radius', 'errh', 'errz', 'errt', 'errM', 'zone')
			to column names in the GIS file.
			If 'lon' or 'lat' are not specified, they will be derived from
			the geographic object.
	:param fix_zero_days_and_months:
		bool, if True, zero days and months are replaced with ones
		(default: False)
	:param convert_zero_magnitudes:
		bool, whether or not to convert zero magnitudes to NaN values
		(default: False)
	:param ID_prefix:
		str, prefix to add to earthquake IDs
		(default: "")
	:param verbose:
		Boolean, whether or not to print information while reading
		GIS table (default: True)

	:return:
		instance of :class:`EQCatalog`
	"""
	from mapping.geotools.read_gis import read_gis_file

	data = read_gis_file(gis_filespec, verbose=verbose)
	eq_list = []
	skipped = 0
	for i, rec in enumerate(data):
		if 'ID' in column_map:
			ID = rec[column_map['ID']]
		else:
			ID = i
		if ID_prefix:
			ID = ID_prefix + str(ID)

		if 'date' in column_map:
			date = rec[column_map['date']]
			if date:
				year, month, day = [int(s) for s in date.split('/')]
			else:
				year, month, day = 0, 0, 0
		else:
			if 'year' in column_map:
				year = rec[column_map['year']]
			if 'month' in column_map:
				month = rec[column_map['month']]
				if month == 0 and fix_zero_days_and_months:
					month = 1
			else:
				month = 1
			if 'day' in column_map:
				day = rec[column_map['day']]
				if day == 0 and fix_zero_days_and_months:
					day = 1
			else:
				day = 1
		try:
			date = tf.time_tuple_to_np_datetime(year, month, day)
		except:
			print(year, month, day)
			date = None

		if 'time' in column_map:
			time = rec[column_map['time']]
			hour, minute, second = [int(s) for s in time.split(':')]
		else:
			if 'hour' in column_map:
				hour = rec[column_map['hour']]
			else:
				hour = 0
			if 'minute' in column_map:
				minute = rec[column_map['minute']]
			else:
				minute = 0
			if 'second' in column_map:
				second = rec[column_map['second']]
			else:
				second = 0
			second = int(round(second))
			second = min(second, 59)
		try:
			time = datetime.time(hour, minute, second)
		except:
			print(hour, minute, second)
			time = None

		if 'lon' in column_map:
			lon = rec[column_map['lon']]
		else:
			lon = rec["obj"].GetX()

		if 'lat' in column_map:
			lat = rec[column_map['lat']]
		else:
			lat = rec["obj"].GetY()

		if 'depth' in column_map:
			depth = rec[column_map['depth']]
		else:
			depth = 0

		mag = {}
		if 'ML' in column_map:
			ML = rec[column_map['ML']]
			if convert_zero_magnitudes:
				ML = ML or np.nan
			mag['ML'] = ML

		if 'MS' in column_map:
			MS = rec[column_map['MS']]
			if convert_zero_magnitudes:
				MS = MS or np.nan
			mag['MS'] = MS

		if 'MW' in column_map:
			MW = rec[column_map['MW']]
			if convert_zero_magnitudes:
				MW = MW or np.nan
			mag['MW'] = MW

		if 'name' in column_map:
			name = rec[column_map['name']]
		else:
			name = ""

		if 'intensity_max' in column_map:
			intensity_max = rec[column_map['intensity_max']]
		else:
			intensity_max = None

		if 'macro_radius' in column_map:
			macro_radius = rec[column_map['macro_radius']]
		else:
			macro_radius = None

		if 'errh' in column_map:
			errh = rec[column_map['errh']]
		else:
			errh = 0.

		if 'errz' in column_map:
			errz = rec[column_map['errz']]
		else:
			errz = 0.

		if 'errt' in column_map:
			errt = rec[column_map['errt']]
		else:
			errt = 0.

		if 'errM' in column_map:
			errM = rec[column_map['errM']]
		else:
			errM = 0.

		if 'zone' in column_map:
			zone = rec[column_map['zone']]
		else:
			zone = ""

		#print(ID, date, time, lon, lat, depth, ML, MS, MW)
		try:
			eq = LocalEarthquake(ID, date, time, lon, lat, depth, mag, name=name,
							intensity_max=intensity_max, macro_radius=macro_radius,
							errh=errh, errz=errz, errt=errt, errM=errM, zone=zone)
		except:
			skipped += 1
		else:
			if date:
				eq_list.append(eq)
			else:
				skipped += 1

	name = os.path.split(gis_filespec)[-1]
	eqc = EQCatalog(eq_list, name=name)
	if verbose:
		print("Skipped %d records" % skipped)
	return eqc
