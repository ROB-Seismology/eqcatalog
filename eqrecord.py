# -*- coding: iso-Latin-1 -*-
"""
"""

## Import standard python modules
import datetime
import time
import json
import numpy as np


## Import ROB modules
import mapping.geo.geodetic as geodetic
import seismodb
import msc


__all__ = ["LocalEarthquake", "FocMecRecord", "MacroseismicRecord"]

# TODO: allow nan values instead of zeros


class LocalEarthquake:
	"""
	Class representing a local earthquake retrieved from the
	earthquakes database table.

	Provides methods to convert magnitudes, and compute distances.

	:param ID:
		Int, ID of earthquake in earthquakes table
	:param date:
		datetime.date object, date when the earthquake happened
	:param time:
		datetime.time object, time when the earthquake happened
	:param lon:
		Float, longitude of epicenter in decimal degrees
	:param lat:
		Float, latitude of epicenter in decimal degrees
	:param depth:
		Float, hypocentral depth in km
	:param ML:
		Float, local magnitude
	:param MS:
		Float, surface-wave magnitude
	:param MW:
		Float, moment magnitude
	:param name:
		String, name of location
	"""
	def __init__(self, ID, date, time, lon, lat, depth, ML, MS, MW, name="", intensity_max=None, macro_radius=None, errh=0., errz=0., errt=0., errM=0.):
		self.ID = ID
		self.datetime = datetime.datetime.combine(date, time)
		self.lon = lon
		self.lat = lat
		self.depth = depth
		self.ML = ML
		self.MS = MS
		self.MW = MW

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
			self.MS = float(MS)
		try:
			self.MW = float(MW)
		except TypeError:
			self.MW = float(MW)
		"""
		self.name = name
		self.intensity_max = intensity_max
		self.macro_radius = macro_radius
		self.errh = errh
		self.errz = errz
		self.errt = errt
		self.errM = errM

	@classmethod
	def from_json(self, s):
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
				return self.from_dict(dct[class_name])
		#return LocalEarthquake.__init__(self, **json.loads(s))

	@classmethod
	def from_dict(self, dct):
		"""
		Generate instance of :class:`LocalEarthquake` from a dictionary

		:param dct:
			Dictionary
		"""
		if 'time' in dct:
			dct['time'] = datetime.time(*dct['time'])
		if 'date' in dct:
			dct['date'] = datetime.date(*dct['date'])
		if 'datetime' in dct:
			dt = eval(dct['datetime'])
			dct['date'] = dt.date()
			dct['time'] = dt.time()
			del dct['datetime']
		return LocalEarthquake(**dct)

	def dump_json(self):
		"""
		Generate json string
		"""
		def json_handler(obj):
			if isinstance(obj, datetime.datetime):
				return repr(obj)
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
		from HY4 import HYPDAT

		latitude = int(round(self.lat * 3600 / 0.001))
		longitude = int(round(self.lon * 3600 / 0.001))
		year, month, day, tm_hour, tm_min, tm_sec = self.datetime.timetuple()[:6]
		minutes = int(round(tm_hour * 60 + tm_min))
		tseconds = int(round(tm_sec / 0.1))
		depth = int(round(self.depth * 100000))
		M = self.get_M(Mtype, Mrelation)
		magnitude = int(round(M * 10))

		return HYPDAT(latitude, longitude, year, month, day, minutes, tseconds, depth, magnitude, 0, 0, 0)

	@property
	def date(self):
		return self.datetime.date()

	@property
	def time(self):
		return self.datetime.time()

	def get_ML(self, Mrelation=None):
		"""
		Return ML
		Not yet implemented!
		"""
		return self.ML

	def get_MS(self, Mrelation={"ML": "ambraseys"}):
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
		if Mrelation is None:
			Mrelation={"ML": "Ambraseys1985"}

		if not self.MS:
			if self.ML and Mrelation.has_key("ML"):
				msce = getattr(msc, Mrelation["ML"])()
				return msce.get_mean(self.ML)
			# TODO: add relation for MW
			else:
				return 0.
		else:
			return self.MS

	def get_MW(self, Mrelation={"MS": "Geller1976", "ML": "ReamerHinzen2004Q"}):
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

			(default: {"MS": "Geller1976", "ML": "ReamerHinzen2004Q"})
			Note that MS -> MW relations take precedence over ML -> MW relations

		:return:
			Float, moment magnitude
		"""
		## Set default conversion relation
		if Mrelation is None:
			Mrelation={"MS": "Geller1976", "ML": "ReamerHinzen2004Q"}

		if not self.MW:
			if self.MS and Mrelation.has_key("MS"):
				msce = getattr(msc, Mrelation["MS"])()
				MW = msce.get_mean(self.MS)
			elif self.ML and Mrelation.has_key("ML"):
				msce = getattr(msc, Mrelation["ML"])()
				MW = msce.get_mean(self.ML)
			else:
				MW = 0.
		else:
			MW = self.MW
		return MW

	def get_M(self, Mtype, Mrelation=None):
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

	def get_M0(self, Mrelation=None):
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

	def get_fractional_year(self):
		"""
		Compute fractional year of event

		:return:
			Float, fractional year
		"""
		def sinceEpoch(date):
			# returns seconds since epoch
			epoch = datetime.datetime(1970, 1, 1)
			diff = epoch - date
			return diff.days * 24. * 3600. + diff.seconds
			## The line below only works for dates after the epoch
			#return time.mktime(date.timetuple())

		year = self.datetime.year
		startOfThisYear = datetime.datetime(year=year, month=1, day=1)
		startOfNextYear = datetime.datetime(year=year+1, month=1, day=1)

		yearElapsed = sinceEpoch(self.datetime) - sinceEpoch(startOfThisYear)
		yearDuration = sinceEpoch(startOfNextYear) - sinceEpoch(startOfThisYear)
		fraction = yearElapsed/yearDuration

		return self.datetime.year + fraction

	def get_fractional_hour(self):
		"""
		Compute fractional hour of event

		:return:
			Float, fractional hour
		"""
		return self.datetime.hour + self.datetime.minute/60.0 + self.datetime.second/3600.0

	def epicentral_distance(self, pt):
		"""
		Compute epicentral distance to point in km using haversine formula

		:param pt:
			(lon, lat) tuple or other instance of :class:`LocalEarthquake`

		:return:
			Float, epicentral distance in km
		"""
		if isinstance(pt, LocalEarthquake):
			pt = (pt.lon, pt.lat)
		return geodetic.distance((self.lon, self.lat), pt)

	def hypocentral_distance(self, pt):
		"""
		Compute hypocentral distance to point in km using haversine formula,
		and taking into account depth

		:param pt:
			(lon, lat) tuple or other instance of :class:`LocalEarthquake`

		:return:
			Float, hypocentral distance in km
		"""
		d_epi = self.epicentral_distance(pt)
		d_hypo = np.sqrt(d_epi**2 + self.depth**2)
		return d_hypo

	def azimuth_to(self, pt):
		"""
		Compute bearing or azimuth from epicenter to another point.

		:param pt:
			(lon, lat) tuple or other instance of :class:`LocalEarthquake`

		:return:
			Float, azimuth in decimal degrees
		"""
		if isinstance(pt, LocalEarthquake):
			pt = (pt.lon, pt.lat)
		return geodetic.bearing((self.lon, self.lat), pt)

	def azimuth_from(self, pt):
		"""
		Compute azimuth from another point to epicenter.

		:param pt:
			(lon, lat) tuple or other instance of :class:`LocalEarthquake`

		:return:
			Float, azimuth in decimal degrees
		"""
		if isinstance(pt, LocalEarthquake):
			pt = (pt.lon, pt.lat)
		return geodetic.bearing(pt, (self.lon, self.lat))

	def get_point_at(self, distance, azimuth):
		"""
		Compute coordinates of point at given distance and azimuth
		from epicenter.

		:param distance:
			Float, distance in km
		:param azimuth:
			Float, azimuth in decimal degrees
		"""
		return geodetic.get_point_at((self.lon, self.lat), distance, azimuth)

	def get_macroseismic_data_aggregated_web(self, min_replies=3, query_info="cii", min_val=1, min_fiability=10.0, group_by_main_village=False, agg_function="", sort_key="intensity", sort_order="asc", verbose=False):
		return seismodb.query_ROB_Web_MacroCatalog(self.ID, min_replies=min_replies, query_info=query_info, min_val=min_val, min_fiability=min_fiability, group_by_main_village=group_by_main_village, agg_function=agg_function, lonlat=True, sort_key=sort_key, sort_order=sort_order, verbose=verbose)

	def get_macroseismic_data_aggregated_official(self, Imax=True, min_val=1, group_by_main_village=False, agg_function="maximum", verbose=False):
		return seismodb.query_ROB_Official_MacroCatalog(self.ID, Imax=Imax, min_val=min_val, group_by_main_village=group_by_main_village, agg_function=agg_function, lonlat=True, verbose=verbose)

	def get_focal_mechanism(self, verbose=False):
		try:
			return seismodb.query_ROB_FocalMechanisms(id_earth=self.ID, verbose=verbose)[0]
		except IndexError:
			return None



class MacroseismicDataPoint:
	pass


class MacroseismicRecord:
	"""
	Container class to hold information of records retrieved from the macrocatalog database table.
	Currently has the following properties:
		id_com
		I
		lon
		lat
		num_replies
	"""
	def __init__(self, id_com, I, num_replies=1, lon=0, lat=0):
		self.id_com = id_com
		self.I = I
		self.num_replies = num_replies
		self.lon = lon
		self.lat = lat


#class FocMecRecord(LocalEarthquake, MT.FaultGeometry):
class FocMecRecord(LocalEarthquake):
	"""
	Container class to hold information of records retrieved from the focal_mechanisms database table.
	"""
	def __init__(self, ID, date, time, lon, lat, depth, ML, MS, MW, strike, dip, rake, name="", intensity_max=None, macro_radius=None):
		LocalEarthquake.__init__(self, ID, date, time, lon, lat, depth, ML, MS, MW, name=name, intensity_max=intensity_max, macro_radius=macro_radius)
		#MT.FaultGeometry.__init__(self, strike, dip, rake)
		self.Mw = self.get_MW()
		self.strike = strike
		self.dip = dip
		self.rake = rake

	def get_focmec(self):
		return MT.FaultGeometry(self.strike, self.dip, self.rake, Mw=self.get_MW())

	def get_mtensor(self):
		import eqgeology.FocMec.MomentTensor as MT
		mt = MT.MomentTensor()
		mt.fromsdr(self.strike, self.dip, self.rake, Mw=self.get_MW())
		return mt

	#property(focmec, get_focmec)


