"""
"""

## Import standard python modules
import datetime
import math


## Import ROB modules
import mapping.geodetic as geodetic


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
		Not yet implemented
		"""
		pass

	def get_MS(self, relation=None):
		"""
		Return MS.
		If MS is None or zero, calculate it from ML using the specified
		magnitude conversion relation

		:param relation:
			String, name of magnitude conversion relation.
			Only one relation is currently supported:
			- "ambraseys": relation by Ambraseys (1985) for earthquakes
				in NW Europe (quoted in Leynaud et al., 2000).
			- None: use ambraseys
			(default: None)

		:return:
			Float, surface-wave magnitude
		"""
		if relation is None:
			relation = "ambraseys"
		if not self.MS:
			if relation == "ambraseys" and self.ML:
				return 0.09 + 0.93 * self.ML
			else:
				return 0.
		else:
			return self.MS

	def get_MW(self, relation=None):
		"""
		Return MW.
		If MW is None or zero, calculate it using the specified
		magnitude conversion relation

		:param relation:
			String, name of magnitude conversion relation.
			The following relations are supported:
			- "geller": calculate MW from MS assuming a stress drop of 50 bars
				(only valid for MS < 6.76);
				Note that MS is in most cases calculated in turn from ML,
				implying a double conversion
			- "ahorner": calculate MW from ML using relation by Ahorner (1983)
			- "hinzen": calculate MW from ML using relation by Hinzen (2004)
			- "grunthal": calculate MW from ML using chi-square maximum likelihood
				regression of Grünthal & Wahlström (2003)
			- "bungum": calculate MW from MS using formulae by Bungum et al. (2003)
			- None: use Geller relation if MS != 0, Hinzen relation otherwise
			(default: None)

		:return:
			Float, moment magnitude
		"""
		if not self.MW:
			if relation == None:
				if self.MS:
					relation = "geller"
				else:
					relation = "hinzen"
			## Conversion (ML ->) MS -> MW (Geller, 1976)
			if relation.lower() == "geller":
				log_Mo_dyncm = self.get_MS() + 18.89
				MW = (2.0/3) * log_Mo_dyncm - 10.73
			## Conversion (ML ->) MS -> MW (Bungum  et al., 2003)
			elif relation.lower() == "bungum":
				MS = self.get_MS()
				if MS < 5.4:
					MW = 0.585 * MS + 2.422
				else:
					MW = 0.769 * MS + 1.280
			## Relation with ML by Ahorner (1983)
			elif relation.lower() == "ahorner":
				log_Mo_dyncm = 17.4 + 1.1 * self.ML
				MW = (2.0/3) * log_Mo_dyncm - 10.73
			## Relation with ML by Hinzen (2004)
			elif relation.lower() == "hinzen":
				log_Mo = 1.083 * self.ML + 10.215
				MW = (2.0/ 3) * log_Mo - 6.06
			## Relation with ML by Grünthal & Wahlström (2003)
			elif relation.lower() in ("grunthal", "grünthal"):
				MW = 0.67 + 0.56 * self.ML + 0.046 * self.ML**2
		else:
			MW = self.MW
		return MW

	def get_M(self, Mtype, relation=None):
		"""
		Wrapper for get_ML, get_MS, and get_MW functions

		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MS")
		:param relation":
			String, magnitude conversion relation (default: None)

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
			String, name of magnitude conversion relation.
			See :meth:`get_MW`

		:return:
			Float, scalar seismic moment in N.m
		"""
		return 10**((self.get_MW(relation=relation) + 6.06) * 3.0 / 2.0)

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


