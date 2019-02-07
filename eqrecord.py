# -*- coding: iso-Latin-1 -*-
"""
Classes corresponding to records in database
"""

from __future__ import absolute_import, division, print_function, unicode_literals

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str


## Import standard python modules
import datetime
import time
import json
from collections import OrderedDict

## Third-party modules
import numpy as np

## Import ROB modules
import mapping.geotools.geodetic as geodetic

## Import package submodules
from . import msc
from . import time_functions_np as tf


__all__ = ["LocalEarthquake", "FocMecRecord"]

# TODO: allow nan values instead of zeros


default_Mrelations = {
	'ML': {},
	'MS': {"ML": "Ambraseys1985"},
	'MW': OrderedDict([("MS", "Geller1976"), ("ML", "Ahorner1983")])
}


class LocalEarthquake:
	"""
	Class representing a local earthquake retrieved from the
	earthquakes database table.

	Provides methods to convert magnitudes, and compute distances.

	:param ID:
		str or int, ID of earthquake in earthquakes table
	:param date:
		instance of :class:`np.datetime64` or :class:`datetime.date`
		or ISO-8601 formatted string, date when the earthquake occurred
	:param time:
		instance of :class:`datetime.time`, time when earthquake occurred
	:param lon:
		Float, longitude of epicenter in decimal degrees
	:param lat:
		Float, latitude of epicenter in decimal degrees
	:param depth:
		Float, hypocentral depth in km
	:param mag:
		dict, mapping magnitude types (str) to magnitude values (floats)
	:param ML:
		Float, local magnitude
		(default: np.nan)
	:param MS:
		Float, surface-wave magnitude
		(default: np.nan)
	:param MW:
		Float, moment magnitude
		(default: np.nan)
	:param mb:
		Float, body-wave magnitude
		(default: np.nan)
	:param name:
		String, name of location
		(default: "")
	:param intensity_max:
		Int, maximum intensity
		(default: None)
	:param macro_radius:
		Float, macroseismic radius (default: None)
	:param errh:
		Float, uncertainty on epicentral location, in km
		(default: 0)
	:param errz:
		Float, uncertainty on hypocentral depth, in km
		(default: 0)
	:param errt:
		Float, uncertainty on origin time, in s
		(default: 0)
	:param errM:
		Float, uncertainty on magnitude
		(default: 0)
	:param zone:
		Str, seismotectonic zone the earthquake belongs to
		(default: "")
	:param agency:
		str, agency reporting the event
		(default: "")
	:param event_type:
		str, type of event
		(default: "ke" = known earthquake)
	"""
	def __init__(self,
			ID,
			date,
			time,
			lon,
			lat,
			depth,
			mag,
			ML=np.nan,
			MS=np.nan,
			MW=np.nan,
			mb=np.nan,
			name="",
			intensity_max=None,
			macro_radius=None,
			errh=0.,
			errz=0.,
			errt=0.,
			errM=0.,
			zone="",
			agency="",
			event_type="ke"):
		self.ID = ID
		try:
			date = tf.as_np_date(date)
		except:
			raise TypeError("date not of correct type")
		try:
			self.datetime = tf.combine_np_date_and_py_time(date, time, unit='ms')
		except:
			raise TypeError("time not of correct type")

		self.lon = lon
		self.lat = lat
		self.depth = depth
		#self.mag = mag
		if not mag:
			self.mag = {}
		elif isinstance(mag, dict):
			self.mag = mag
		else:
			raise Exception("mag must be a dictionary!")

		if not ML in (None, np.nan):
			self.mag['ML'] = ML
		if not MS in (None, np.nan):
			self.mag['MS'] = MS
		if not MW in (None, np.nan):
			self.mag['MW'] = MW
		if not mb in (None, np.nan):
			self.mag['mb'] = mb

		"""
		try:
			self.lon = float(lon)
		except TypeError:
			self.lon = lon
		try:
			self.lat = float(lat)
		except TypeError:
			self.lat = lat
		try:
			self.depth = float(depth)
		except TypeError:
			self.depth = depth
		try:
			self.ML = float(ML)
		except TypeError:
			self.ML = ML
		try:
			self.MS = float(MS)
		except TypeError:
			self.MS = MS
		try:
			self.MW = float(MW)
		except TypeError:
			self.MW = MW
		"""
		self.name = name
		self.intensity_max = intensity_max
		self.macro_radius = macro_radius
		self.errh = errh
		self.errz = errz
		self.errt = errt
		# TODO: errM should be dict as well
		self.errM = errM

		self.zone = u'' + zone
		self.agency = u'' + agency
		self.event_type = event_type

	def __eq__(self, eq):
		if isinstance(eq, self.__class__) and self.ID == eq.ID:
			return True
		else:
			return False

	@classmethod
	def from_json(cls, s):
		"""
		Generate instance of :class:`LocalEarthquake` from a json string

		:param s:
			String, json format
		"""
		dct = json.loads(s)
		if len(dct) == 1:
			class_name = dct.keys()[0]
			#if class_name == LocalEarthquake.__class__.__name__:
			if class_name == "__LocalEarthquake__":
				return cls.from_dict(dct[class_name])
		#return LocalEarthquake.__init__(self, **json.loads(s))

	@classmethod
	def from_dict(cls, dct):
		"""
		Generate instance of :class:`LocalEarthquake` from a dictionary

		:param dct:
			Dictionary
		"""
		if 'datetime' in dct:
			dt = tf.as_np_datetime(dct['datetime'])
			dct['date'] = tf.as_np_date(dt)
			dct['time'] = tf.to_py_time(dt)
			del dct['datetime']
		else:
			if 'time' in dct:
				dct['time'] = datetime.time(*dct['time'])
			if 'date' in dct:
				#dct['date'] = datetime.date(*dct['date'])
				dct['date'] = tf.time_tuple_to_np_datetime(*dct['date'])
		if not 'mag' in dct:
			dct['mag'] = {}
			for key in dct.keys():
				if key[0].upper() == 'M' and len(key) == 2:
					dct['mag'][key] = dct[key]
					del dct[key]
		return cls(**dct)

	@classmethod
	def from_dict_rec(cls, rec, column_map={}, date_sep='-', time_sep=':',
					date_order='YMD'):
		"""
		Construct instance of :class:`LocalEarthquake` from a dict-like
		record mapping earthquake property names to values. If keys
		do not correspond to standard names, a column map should provide
		mapping between standard property names and keys in record.

		:param rec:
			dict-like, earthquake record
		:param column_map:
			dict, mapping property names of :class:`LocalEarthquake` to
			column names in header or to column numbers (zero-based) if no
			header is present.
			(default: {})
		:param date_sep:
			str, character separating date elements
			(default: '-')
		:param time_sep:
			str, character separating time elements
			(default: ':')
		:param date_order:
			str, order of year (Y), month (M), day (D) in date string
			(default: 'YMD')

		:return:
			instance of :class:`LocalEarthquake`
		"""
		## If key is not in column_map, we assume it has the default name
		ID_key = column_map.get('ID', 'ID')
		ID = rec.get(ID_key)

		datetime_key = column_map.get('datetime', 'datetime')
		if datetime_key in rec:
			dt = tf.as_np_datetime(rec[datetime_key])
			date = tf.as_np_date(dt)
			time = tf.to_py_time(dt)

		else:
			date_key = column_map.get('date', 'date')
			date = rec.get(date_key)
			## Year must always be present
			## Silently convert month/day to 1 if it is zero
			if date:
				if isinstance(date, basestring):
					date_elements = date.split(date_sep)
					year = int(date_elements[date_order.index('Y')])
					try:
						month = max(1, int(date_elements[date_order.index('M')]))
					except IndexError:
						month = 1
					try:
						day = max(1, int(date_elements[date_order.index('D')]))
					except:
						day = 1
				elif isinstance(date, datetime.date):
					year, month, day = date.year, date.month, date.day
			else:
				year_key = column_map.get('year', 'year')
				year = int(rec[year_key])
				month_key = column_map.get('month', 'month')
				month = max(1, int(rec.get(month_key, 1) or 1))
				day_key = column_map.get('day', 'day')
				day = max(1, int(rec.get(day_key, 1) or 1))
			try:
				date = tf.time_tuple_to_np_datetime(year, month, day)
			except:
				print("Invalid date in rec %s: %s-%s-%s"
					% (ID, year, month, day))
				date = None

			time_key = column_map.get('time', 'time')
			time = rec.get(time_key)
			if time:
				if isinstance(time, basestring):
					time_elements = time.split(time_sep)
					try:
						hour = min(23, int(time_elements[0]))
					except (IndexError, ValueError):
						hour = 0
					try:
						minute = min(59, int(time_elements[1]))
					except (IndexError, ValueError):
						minute = 0
					try:
						second = min(59, float(time_elements[2]))
					except (IndexError, ValueError):
						second = 0.
				elif isinstance(time, datetime.time):
					hour, minute, second = time.hour, time.minute, time.second
					second += (time.microsecond * 1E-6)
			else:
				hour_key = column_map.get('hour', 'hour')
				hour = min(23, int(rec.get(hour_key, 0) or 0))
				minute_key = column_map.get('minute', 'minute')
				minute = min(59, int(rec.get(minute_key, 0) or 0))
				second_key = column_map.get('second', 'second')
				second = float(rec.get(second_key, 0) or 0)
			fraction, second = np.modf(second)
			second = min(59, int(second))
			microsecs = int(round(fraction * 1E+6))
			try:
				time = datetime.time(hour, minute, second, microsecs)
			except:
				print(ID, second)
				raise

		## Default value for lon, lat, depth is 0, but could also be nan
		lon_key = column_map.get('lon', 'lon')
		lon = float(rec.get(lon_key, 0) or 0)

		lat_key = column_map.get('lat', 'lat')
		lat = float(rec.get(lat_key, 0) or 0)

		depth_key = column_map.get('depth', 'depth')
		depth = float(rec.get(depth_key, 0) or 0)

		mag = {}

		## Magnitude specified as value and magnitude type
		Mtype_key = column_map.get('Mtype', 'Mtype')
		if Mtype_key in rec:
			Mtype = rec.get(Mtype_key)
			if Mtype:
				if len(Mtype) == 1:
					Mtype = 'M' + Mtype
				if Mtype[-1] == 'b':
					Mtype = Mtype.lower()
				else:
					Mtype = Mtype.upper()
				Mag_key = column_map.get('Mag', 'Mag')
				M = float(rec.get(Mag_key, np.nan))
				mag[Mtype] = M
		## Specific magnitude type column
		## takes precedence over value/magnitude type columns
		ML_key = column_map.get('ML', 'ML')
		#ML = float(rec.get(ML_key, np.nan) or np.nan)
		ML = float(rec.get(ML_key, mag.get('ML', np.nan)))
		mag['ML'] = ML
		MS_key = column_map.get('MS', 'MS')
		#MS = float(rec.get(MS_key, np.nan) or np.nan)
		MS = float(rec.get(MS_key, mag.get('MS', np.nan)))
		mag['MS'] = MS
		MW_key = column_map.get('MW', 'MW')
		#MW = float(rec.get(MW_key, np.nan) or np.nan)
		MW = float(rec.get(MW_key, mag.get('MW', np.nan)))
		mag['MW'] = MW
		mb_key = column_map.get('mb', 'mb')
		#mb = float(rec.get(mb_key, np.nan) or np.nan)
		mb = float(rec.get(mb_key, mag.get('mb', np.nan)))
		mag['mb'] = mb

		name_key = column_map.get('name', 'name')
		name = rec.get(name_key, "")

		intensity_max_key = column_map.get('intensity_max', 'intensity_max')
		intensity_max = rec.get(intensity_max_key, 0)
		if intensity_max:
			if isinstance(intensity_max, basestring):
				## Strip trailing + and - if present
				if intensity_max[-1] in ('+', '-'):
					intensity_max = intensity_max[:-1]
				## Take average if Imax is specified as range
				Imax_vals = list(map(float, intensity_max.split('-')))
				intensity_max = np.mean(Imax_vals)
		else:
			intensity_max = 0.

		macro_radius_key = column_map.get('macro_radius', 'macro_radius')
		macro_radius = float(rec.get(macro_radius_key, 0) or 0)

		errh_key = column_map.get('errh', 'errh')
		errh = float(rec.get(errh_key, 0) or 0)

		errz_key = column_map.get('errz', 'errz')
		errz = float(rec.get(errz_key, 0) or 0)

		errt_key = column_map.get('errt', 'errt')
		errt = float(rec.get(errt_key, 0) or 0)

		errM_key = column_map.get('errM', 'errM')
		errM = float(rec.get(errM_key, 0) or 0)

		zone_key = column_map.get('zone', 'zone')
		zone = rec.get(zone_key, "")

		agency_key = column_map.get('agency', 'agency')
		agency = rec.get(agency_key, agency_key if not agency_key == 'agency' else "")

		event_type_key = column_map.get('event_type', 'event_type')
		event_type = rec.get(event_type_key, 'ke')

		return cls(ID, date, time, lon, lat, depth, mag, name=name,
						intensity_max=intensity_max, macro_radius=macro_radius,
						errh=errh, errz=errz, errt=errt, errM=errM, zone=zone,
						agency=agency, event_type=event_type)

	def to_dict(self):
		"""
		Convert to a dictionary

		:return:
			instance of :class:`dict`
		"""
		return self.__dict__.copy()

	def dump_json(self):
		"""
		Generate json string
		"""
		def json_handler(obj):
			if isinstance(obj, (datetime.time, datetime.date)):
				return repr(obj)
			elif isinstance(obj, np.datetime64):
				return str(obj)
			else:
				return obj.__dict__

		key = '__%s__' % self.__class__.__name__
		dct = {key: self.__dict__}
		return json.dumps(dct, default=json_handler, encoding="latin1")

	@classmethod
	def from_HY4(self, hypdat, Mtype='ML', ID=0):
		"""
		Construct from HYPDAT structure used in HY4 catalog format
		used by SeismicEruption

		:param hypdat:
			instance of :class:`HYPDAT`
		:param Mtype:
			Str, magnitude type, either 'ML', 'MS' or 'MW' (default: 'ML')
		:param ID:
			Int, identifier
		"""
		lat = hypdat.latitude * 0.001 / 3600.0
		lon = hypdat.longitude * 0.001 / 3600.0
		year = int(hypdat.year)
		month = int(hypdat.month)
		day = int(hypdat.day)
		date = datetime.date(year, month, day)
		hour, minutes = divmod(hypdat.minutes, 60)
		seconds = hypdat.tseconds * 0.1
		time = datetime.time(hour, minutes, seconds)
		depth = hypdat.depth / 100000.0
		magnitude = hypdat.magnitude / 10.0
		ML = {True: magnitude, False: 0.}[Mtype == 'ML']
		MS = {True: magnitude, False: 0.}[Mtype == 'MS']
		MW = {True: magnitude, False: 0.}[Mtype == 'MW']

		return LocalEarthquake(ID, date, time, lon, lat, depth, ML, MS, MW)

	def to_HY4(self, Mtype='ML', Mrelation=None):
		"""
		Convert to HYPDAT structure used by SeismicEruption HY4 catalog format

		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "ML")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			instance of :class:`HYPDAT`
		"""
		from .io.HY4 import HYPDAT

		latitude = int(round(self.lat * 3600 / 0.001))
		longitude = int(round(self.lon * 3600 / 0.001))
		year, month, day, tm_hour, tm_min, tm_sec = tf.to_time_tuple(self.datetime)
		minutes = int(round(tm_hour * 60 + tm_min))
		tseconds = int(round(tm_sec / 0.1))
		depth = int(round(self.depth * 100000))
		M = self.get_M(Mtype, Mrelation)
		magnitude = int(round(M * 10))

		return HYPDAT(latitude, longitude, year, month, day, minutes, tseconds,
						depth, magnitude, 0, 0, 0)

	def to_hypo71(self, Mtype='MW', Mrelation="default"):
		"""
		Convert earthquake record to HYPO71-2000 format.

		:param Mtype:
		:param Mrelation:
			see :meth:`get_mag`

		:return:
			str
		"""
		year, month, day = map(int, str(self.date).split('-'))
		hr, minute, sec = str(self.time).split(':')
		hr, minute, sec = int(hr), int(minute), float(sec)
		lonmin, londeg = np.modf(np.abs(self.lon))
		lonmin, londeg = lonmin * 60, londeg
		ew = 'E' if self.lon > 0 else ' '
		latmin, latdeg = np.modf(np.abs(self.lat))
		latmin, latdeg = latmin * 60, latdeg
		ns = 'S' if self.lat < 0 else ' '
		mag = self.get_M(Mtype, Mrelation)
		hypo71 = ('%04d%02d%02d ' % (year, month, day),
				'%02d%02d%6.2f' % (hr, minute, sec),
				'%3.0f%c%5.2f' % (latdeg, ns, latmin),
				'%4.0f%c%5.2f' % (londeg, ew, lonmin),
				'%7.2f ' % self.depth,
				'%c%5.2f' % (Mtype[-1], mag),
				' ' * 17,
				'%5.1f%5.1f' % (self.errh, self.errz),
				' ' * 4,
				'%10s' % str(self.ID)[:10],
				' ' * 5)
		hypo71 = ''.join(hypo71)
		return hypo71

	def print_info(self):
		"""
		Print earthquake attributes in pretty table.
		"""
		try:
			from prettytable import PrettyTable
		except:
			has_prettytable = False
		else:
			has_prettytable = True

		col_names = ["Attribute", "Value"]
		if has_prettytable:
			tab = PrettyTable(col_names)
		else:
			tab = []
		for attrib in ['ID', 'name', 'date', 'time', 'lon', 'lat', 'depth',
						'ML', 'MS', 'MW', 'mb', 'intensity_max', 'macro_radius',
						'zone', 'errt', 'errh', 'errz', 'errM', 'zone',
						'agency', 'event_type']:
			val = getattr(self, attrib)
			if val and not val is np.nan:
				row = [attrib, val]
				if has_prettytable:
					tab.add_row(row)
				else:
					tab.append(row)

		if has_prettytable:
			print(tab)
		else:
			print('\t'.join(col_names))
			for row in tab:
				print('\t'.join(row))

	## date-time-related methods

	@property
	def date(self):
		return tf.as_np_date(self.datetime)

	@property
	def time(self):
		return tf.to_py_time(self.datetime)

	@property
	def year(self):
		return tf.to_year(self.datetime)

	def get_fractional_year(self):
		"""
		Compute fractional year of event

		:return:
			Float, fractional year
		"""
		#from .time_functions import fractional_year
		return tf.to_fractional_year(self.datetime)

	def get_fractional_hour(self):
		"""
		Compute fractional hour of event

		:return:
			Float, fractional hour
		"""
		return tf.py_time_to_fractional_hours(self.time)

	## Magnitude-related methods

	@property
	def ML(self):
		"""Local magnitude"""
		return self.get_mag('ML')

	@property
	def MS(self):
		"""Surface-wave magnitude"""
		return self.get_mag('MS')

	@property
	def MW(self):
		"""Moment magnitude"""
		return self.get_mag('MW')

	@property
	def mb(self):
		"""Body-wave magnitude"""
		return self.get_mag('mb')

	def has_mag(self, Mtype):
		"""
		Determine whether particular magnitude type is defined

		:param Mtype:
			str, magnitude type to check

		:return:
			bool
		"""
		if Mtype in self.mag and not np.isnan(self.mag[Mtype]):
			return True
		else:
			return False

	def get_Mtypes(self):
		"""
		Get defined magnitude types

		:return:
			list of strings
		"""
		return [Mtype for Mtype in self.mag.keys() if not np.isnan(self.mag[Mtype])]

	def get_mag(self, Mtype):
		"""
		Return particular magnitude without conversion.
		If magnitude type is not defined, NaN is returned.

		:param Mtype:
			str, magnitude type to return

		:return:
			float or np.nan, magnitude
		"""
		return self.mag.get(Mtype, np.nan)

	def set_mag(self, Mtype, mag):
		"""
		Set magnitude type to given value

		:param Mtype:
			str, magnitude type to set
		:param mag:
			float, magnitude value
		"""
		self.mag[Mtype] = mag

	def get_or_convert_mag(self, Mtype, Mrelation={}):
		"""
		Return particular magnitude. If the magnitude type is not defined,
		it will be converted from another magnitude type.

		:param Mtype:
			str, magnitude type to return
		:param Mrelation:
			{str: str} dict, mapping magnitude types to names of magnitude
			conversion relations
			Note: use an ordered dict if more than 1 magnitude type is
			specified, as the first magnitude type that is defined will
			be used for conversion.
			If specified as "default" or None, the default Mrelation
			for the given Mtype will be selected.
			(default: {})
		"""
		if self.has_mag(Mtype):
			return self.mag[Mtype]
		else:
			if Mrelation in (None, "default"):
				Mrelation = default_Mrelations.get(Mtype, {})
			return self.convert_mag(Mrelation)

	def convert_mag(self, Mrelation):
		"""
		Return magnitude based on conversion.

		:param Mrelation:
			{str: str} dict, mapping magnitude types to names of magnitude
			conversion relations
			Note: use an ordered dict if more than 1 magnitude type is
			specified, as the first magnitude type that is defined will
			be used for conversion
			(default: {})
		"""
		if len(Mrelation) > 1 and not isinstance(Mrelation, OrderedDict):
			print("Warning: Mrelation should be ordered dictionary!")
		for Mtype, msce_name in Mrelation.items():
			if self.has_mag(Mtype):
				msce = getattr(msc, msce_name)()
				return msce.get_mean(self.get_mag(Mtype))

		## If Mrelation is empty or none of the Mtype's match
		return np.nan

	def get_ML(self, Mrelation="default"):
		"""
		Return ML
		Not yet implemented!
		"""
		if Mrelation in (None, "default"):
			Mrelation = default_Mrelations['ML']

		return self.get_or_convert_mag('ML', Mrelation)

	def get_MS(self, Mrelation="default"):
		"""
		Return MS.
		If MS is None or zero, calculate it using the specified
		magnitude conversion relation

		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("ML" or "MW").
			The following relations are currently supported (see module msc):
			- ML -> MS:
				- "Ambraseys1985"
			- MW -> MS:
				None
			(default: {"ML": "Ambraseys1985"})

		:return:
			Float, surface-wave magnitude
		"""
		## Set default conversion relation
		if Mrelation in (None, "default"):
			Mrelation = default_Mrelations['MS']

		return self.get_or_convert_mag('MS', Mrelation)

	def get_MW(self, Mrelation="default"):
		"""
		Return MW.
		If MW is None or zero, calculate it using the specified
		magnitude conversion relation

		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MS" or "ML").
			The following relations are supported:
			- MS --> MW:
				- "AmbraseysFree1997"
				- "BungumEtAl2003NCEurope"
				- "BungumEtAl2003SEurope"
				- "Geller1976"
				- "ISC_GEM2013"
				- "OkalRomanowicz1994"
				- "Scordilis2006"
				- "Utsu2002"
			- ML --> MW:
				- "Ahorner1983"
				- "Goutbeek2008"
				- "GruenthalWahlstrom2003"
				- "GruenthalEtAl2009"
				- "ReamerHinzen2004L"
				- "ReamerHinzen2004Q"

			(default: {"MS": "Geller1976", "ML": "Ahorner1983"})
			Note that MS -> MW relations take precedence over ML -> MW relations

		:return:
			Float, moment magnitude
		"""
		## Set default conversion relation
		if Mrelation in (None, "default"):
			Mrelation = default_Mrelations['MW']

		return self.get_or_convert_mag('MW', Mrelation)

	def get_M(self, Mtype, Mrelation="default"):
		"""
		Wrapper for get_ML, get_MS, and get_MW functions

		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			Float, magnitude
		"""
		return getattr(self, "get_"+Mtype)(Mrelation=Mrelation)

	def get_M0(self, Mrelation="default"):
		"""
		Compute seismic moment.
		If MW is None or zero, it will be computed using the specified
		magntiude conversion relation.

		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type.
			See :meth:`get_MW`

		:return:
			Float, scalar seismic moment in N.m
		"""
		return 10**((self.get_MW(Mrelation=Mrelation) + 6.06) * 3.0 / 2.0)

	## Location-related methods

	def epicentral_distance(self, pt):
		"""
		Compute epicentral distance to point in km using haversine formula

		:param pt:
			(lon, lat) tuple or object having 'lon' and 'lat' attributes
			Note that lon, lat may also be arrays

		:return:
			Float, epicentral distance in km
		"""
		if hasattr(pt, 'lon') and hasattr(pt, 'lat'):
			pt = (pt.lon, pt.lat)
		return geodetic.spherical_distance(self.lon, self.lat, pt[0], pt[1]) / 1000.

	def hypocentral_distance(self, pt):
		"""
		Compute hypocentral distance to point in km using haversine formula,
		and taking into account depth

		:param pt:
			(lon, lat, [z]) tuple or other instance of :class:`LocalEarthquake`
			Note that lon, lat, z may also be arrays

		:return:
			Float, hypocentral distance in km
		"""
		d_epi = self.epicentral_distance(pt)
		if hasattr(pt, 'depth'):
			depth2 = pt.depth
		elif isinstance(pt, (tuple, list)) and len(pt) == 3:
			depth2 = pt[2]
		else:
			depth2 = 0.
		delta_depth = np.abs(self.depth - depth2)
		d_hypo = np.sqrt(d_epi**2 + delta_depth**2)
		return d_hypo

	def azimuth_to(self, pt):
		"""
		Compute bearing or azimuth from epicenter to another point.

		:param pt:
			(lon, lat) tuple or other instance of :class:`LocalEarthquake`
			Note that lon, lat may also be arrays

		:return:
			Float, azimuth in decimal degrees
		"""
		if isinstance(pt, LocalEarthquake):
			pt = (pt.lon, pt.lat)
		return geodetic.spherical_azimuth(self.lon, self.lat, pt[0], pt[1])

	def azimuth_from(self, pt):
		"""
		Compute azimuth from another point to epicenter.

		:param pt:
			(lon, lat) tuple or other instance of :class:`LocalEarthquake`
			Note that lon, lat may also be arrays

		:return:
			Float, azimuth in decimal degrees
		"""
		if isinstance(pt, LocalEarthquake):
			pt = (pt.lon, pt.lat)
		return geodetic.spherical_azimuth(pt[0], pt[1], self.lon, self.lat)

	def get_point_at(self, distance, azimuth):
		"""
		Compute coordinates of point at given distance and azimuth
		from epicenter.

		:param distance:
			Float or array, distance in km
		:param azimuth:
			Float or array, azimuth in decimal degrees
		Note that only one of :param:`distance` and :param:`azimuth` may be
		an array

		:return:
			...
		"""
		return geodetic.spherical_point_at(self.lon, self.lat, distance*1000, azimuth)



#class FocMecRecord(LocalEarthquake, MT.FaultGeometry):
class FocMecRecord(LocalEarthquake):
	"""
	Container class to hold information of records retrieved from the
	focal_mechanisms database table.
	"""
	def __init__(self, ID, date, time, lon, lat, depth, mag, ML, MS, MW,
				strike, dip, rake, name="", intensity_max=None, macro_radius=None):
		LocalEarthquake.__init__(self, ID, date, time, lon, lat, depth, mag,
								ML, MS, MW, name=name, intensity_max=intensity_max,
								macro_radius=macro_radius)
		#MT.FaultGeometry.__init__(self, strike, dip, rake)
		self.Mw = self.get_MW()
		self.strike = strike
		self.dip = dip
		self.rake = rake

	def get_focmec(self):
		from eqgeology.faultlib.tensorlib import FocalMechanism
		return FocalMechanism(self.strike, self.dip, self.rake, MW=self.get_MW())

	def get_mtensor(self):
		from eqgeology.faultlib.tensorlib import MomentTensor
		mt = MomentTensor()
		mt.from_sdr(self.strike, self.dip, self.rake, MW=self.get_MW())
		return mt

	#property(focmec, get_focmec)
