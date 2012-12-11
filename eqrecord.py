# -*- coding: iso-Latin-1 -*-
"""
"""

## Import standard python modules
import datetime
import time
import math


## Import ROB modules
import mapping.geo.geodetic as geodetic


__all__ = ["LocalEarthquake", "FocMecRecord", "MacroseismicRecord"]


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
	def __init__(self, ID, date, time, lon, lat, depth, ML, MS, MW, name="", intensity_max=None, macro_radius=None):
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

	def get_ML(self, relation=None):
		"""
		Return ML
		Not yet implemented!
		"""
		return self.ML

	def get_MS(self, relation={"ML": "ambraseys"}):
		"""
		Return MS.
		If MS is None or zero, calculate it using the specified
		magnitude conversion relation

		:param relation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MS" or "ML").
			Only one relation is currently supported:
			- "ambraseys": relation with ML by Ambraseys (1985) for
				earthquakes in NW Europe (quoted in Leynaud et al., 2000).
			(default: {"ML": "ambraseys"})

		:return:
			Float, surface-wave magnitude
		"""
		if relation is None:
			relation={"ML": "ambraseys"}

		if not self.MS:
			if relation["ML"] == "ambraseys" and self.ML:
				return 0.09 + 0.93 * self.ML
			# TODO: add relation for MW
			else:
				return 0.
		else:
			return self.MS

	def get_MW(self, relation={"MS": "bungum", "ML": "hinzen"}):
		"""
		Return MW.
		If MW is None or zero, calculate it using the specified
		magnitude conversion relation

		:param relation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MS" or "ML"). Note that MS takes precedence
			over ML.
			The following relations are supported:
			- MS --> MW:
				- "bungum": calculate MW from MS using formulae by Bungum
					et al. (2003)
				- "geller": calculate MW from MS assuming a stress drop of
					50 bars (only valid for MS < 6.76);
			- ML --> MW:
				- "ahorner": caculate MW from ML using relation by Ahorner (1983)
				- "hinzen": calculate MW from ML using relation by Hinzen (2004)
				- "gruenthal": calculate MW from ML using chi-square maximum
					likelihood regression of Gruenthal and Wahlstrom (2003)
			(default: {"MS": "bungum", "ML": "hinzen"})

		:return:
			Float, moment magnitude
		"""
		if relation is None:
			relation={"MS": "bungum", "ML": "hinzen"}

		if not self.MW:
			if self.MS:
				## Conversion MS -> MW (Geller, 1976)
				if relation.lower() == "geller":
					log_Mo_dyncm = self.MS + 18.89
					MW = (2.0/3) * log_Mo_dyncm - 10.73
				## Conversion MS -> MW (Bungum  et al., 2003)
				elif relation.lower() == "bungum":
					if self.MS < 5.4:
						MW = 0.585 * self.MS + 2.422
					else:
						MW = 0.769 * self.MS + 1.280
			elif self.ML:
				## Relation with ML by Ahorner (1983)
				if relation.lower() == "ahorner":
					log_Mo_dyncm = 17.4 + 1.1 * self.ML
					MW = (2.0/3) * log_Mo_dyncm - 10.73
				## Relation with ML by Hinzen (2004)
				elif relation.lower() == "hinzen":
					log_Mo = 1.083 * self.ML + 10.215
					MW = (2.0/ 3) * log_Mo - 6.06
				## Relation with ML by Gruenthal & Wahlstrom (2003)
				elif relation.lower() == "gruenthal":
					MW = 0.67 + 0.56 * self.ML + 0.046 * self.ML**2
			else:
				MW = 0.
		else:
			MW = self.MW
		return MW

	def get_M(self, Mtype, relation=None):
		"""
		Wrapper for get_ML, get_MS, and get_MW functions

		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MS")
		:param relation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			Float, magnitude
		"""
		return getattr(self, "get_"+Mtype)(relation=relation)

	def get_M0(self, relation=None):
		"""
		Compute seismic moment.
		If MW is None or zero, it will be computed using the specified
		magntiude conversion relation.

		:param relation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type.
			See :meth:`get_MW`

		:return:
			Float, scalar seismic moment in N.m
		"""
		return 10**((self.get_MW(relation=relation) + 6.06) * 3.0 / 2.0)

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
		d_hypo = math.sqrt(d_epi**2 + self.depth**2)
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


