# -*- coding: iso-Latin-1 -*-

"""
MagFreq
Python module to calculate Frequency/Magnitude relations from the ROB catalog.
The catalog is read either from the online database, or from a running instance of MapInfo.
If read from MapInfo, it is also possible to split catalog according to a source zone model.
====================================================
Author: Kris Vanneste, Royal Observatory of Belgium.
Date: May 2008.

Required modules:
	Third-party:
		matplotlib / pylab
		numpy
		scipy
		ogr
	ROB:
		mapping.MIPython
		users.kris.Seismo.db.seismodb
"""


## Import standard python modules
import csv
import os
import sys
import platform
import datetime
import cPickle
import json
from collections import OrderedDict


## Import third-party modules
## Kludge because matplotlib is broken on seissrv3.
import numpy as np
import matplotlib
if platform.uname()[1] == "seissrv3":
	matplotlib.use('AGG')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, MaxNLocator
import mx.DateTime as mxDateTime


## Directories with MapInfo tables for named catalogs
if platform.uname()[0] == "Windows":
	GIS_root = r"D:\GIS-data"
else:
	GIS_root = os.path.join(os.environ.get("HOME", ""), "gis-data")



## Import ROB modules
from eqrecord import LocalEarthquake
from source_models import read_source_model
from completeness import *
from time_functions import timespan
import mapping.geotools.geodetic as geodetic



class EQCatalog:
	"""
	Class defining a collection of local earthquakes.

	:param eq_list:
		List containing instances of :class:`LocalEarthquake`
	:param start_date:
		datetime, start of catalog (default: None = datetime of oldest
		earthquake in catalog)
	:param end_date:
		datetime, end of catalog (default: None = datetime of youngest
		earthquake in catalog)
	:param region:
		(lon0, lon1, lat0, lat1) tuple with geographic coordinates of
		bounding box (default: None)
	:param name:
		String, catalog name (default: "")
	"""
	def __init__(self, eq_list, start_date=None, end_date=None, region=None, name=""):
		self.eq_list = eq_list[:]
		Tmin, Tmax = self.Tminmax()
		self.start_date = start_date
		if not start_date:
			self.start_date = Tmin
		self.end_date = end_date
		if not end_date:
			self.end_date = Tmax
		if isinstance(self.start_date, (datetime.datetime, datetime.date, mxDateTime.DateTimeType)):
			year, month, day = self.start_date.timetuple()[:3]
			self.start_date = mxDateTime.Date(year, month, day)
		if isinstance(self.end_date, (datetime.datetime, datetime.date, mxDateTime.DateTimeType)):
			year, month, day = self.end_date.timetuple()[:3]
			self.end_date = mxDateTime.Date(year, month, day)
		self.region = region
		self.name = name

	def __len__(self):
		"""
		Return number of earthquakes in collection.
		"""
		return len(self.eq_list)

	def __iter__(self):
		return self.eq_list.__iter__()

	def __getitem__(self, item):
		"""
		Indexing --> instance of :class:`LocalEarthquake`
		Slicing --> instance of :class:`EQCatalog`
		"""
		if isinstance(item, (int, np.int32, np.int64)):
			return self.eq_list.__getitem__(item)
		elif isinstance(item, slice):
			return EQCatalog(self.eq_list.__getitem__(item), start_date=self.start_date, end_date=self.end_date, region=self.region, name=self.name + " %s" % item)
		elif isinstance(item, (list, np.ndarray)):
			eq_list = []
			for index in item:
				eq_list.append(self.eq_list[index])
			return EQCatalog(eq_list, start_date=self.start_date, end_date=self.end_date, region=self.region, name=self.name + " %s" % item)

	@property
	def lons(self):
		return self.get_longitudes()

	@property
	def lats(self):
		return self.get_latitudes()

	@property
	def mags(self):
		return self.get_magnitudes()

	def print_report(self):
		"""
		Print some useful information about the catalog.
		"""
		print("Name: %s" % self.name)
		print("Earthquake number: %d" % len(self))
		print("Start time: %s" % self.start_date)
		print("End time :  %s" % self.end_date)
		for Mtype, count in self.get_Mtype_counts().items():
			mags = self.get_magnitudes(Mtype=Mtype, Mrelation={})
			mags = mags[np.isfinite(mags)]
			if len(mags):
				if mags.min() == 0:
					mags = mags[mags > 0]
				print("%s: n=%d, min=%.1f, max=%.1f" % (Mtype, count, mags.min(), mags.max()))
		lons = self.get_longitudes()
		lons = lons[np.isfinite(lons)]
		print("Longitude bounds: %.4f - %.4f" % (lons.min(), lons.max()))
		lats = self.get_latitudes()
		lats = lats[np.isfinite(lats)]
		print("Latitude bounds: %.4f - %.4f" % (lats.min(), lats.max()))
		depths = self.get_depths()
		depths = depths[np.isfinite(depths)]
		print("Depth range %.1f - %.1f km" % (depths.min(), depths.max()))

	def get_record(self, ID):
		"""
		Fetch record with given ID

		:param ID:
			Int, ID of earthquake in ROB database

		:return:
			instance of :class:`LocalEarthquake`
		"""
		events = [rec for rec in self if rec.ID == ID]
		if len(events) == 1:
			return events[0]
		else:
			return None

	@classmethod
	def from_json(cls, s):
		"""
		Generate instance of :class:`EQCatalog` from a json string

		:param s:
			String, json format
		"""
		dct = json.loads(s)
		if len(dct) == 1:
			class_name = dct.keys()[0]
			if class_name == "__EQCatalog__":
				return cls.from_dict(dct[class_name])

	@classmethod
	def from_dict(cls, dct):
		"""
		Generate instance of :class:`EQCatalog` from a dictionary

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
		if 'eq_list' in dct:
			dct['eq_list'] = [LocalEarthquake.from_dict(d["__LocalEarthquake__"]) for d in dct['eq_list']]
		return EQCatalog(**dct)

	def dump_json(self):
		"""
		Generate json string
		"""
		def json_handler(obj):
			if isinstance(obj, LocalEarthquake):
				key = '__%s__' % obj.__class__.__name__
				dct = {key: obj.__dict__}
				return dct
			elif isinstance(obj, (datetime.date, datetime.datetime)):
				return repr(obj)
			else:
				return obj.__dict__

		key = '__%s__' % self.__class__.__name__
		dct = {key: self.__dict__}
		return json.dumps(dct, default=json_handler)

	@classmethod
	def from_HY4(cls, filespec, Mtype='ML'):
		"""
		Read from HY4 earthquake catalog format used by SeismicEruption

		:param filespec:
			Str, full path to input file
		:param Mtype:
			Str, magnitude type, either 'ML', 'MS' or 'MW' (default: 'ML')
		"""
		eq_list = []
		fd = open(filespec, "rb")
		ID = 1
		while 1:
			bytes = fd.read(24)
			if bytes:
				h = HYPDAT(*struct.unpack("2ih2B2hl4B", bytes))
				eq_list.append(LocalEarthquake.from_HY4(h, Mtype=Mtype, ID=ID))
				ID += 1
			else:
				break
		return EQCatalog(eq_list)

	def get_time_delta(self):
		"""
		Return duration of catalog as timedelta object

		:return:
			instance of datetime.timedelta or mxDateTime.DateTimeDelta
		"""
		Tmin, Tmax = self.Tminmax()
		return Tmax - Tmin

	def get_time_deltas(self, start_datetime=None):
		"""
		Return time difference between a start time and each event.

		:param start_datetime:
			datetime object, start time (default: None, will take the
			start time of the catalog)

		:return:
			list of timedelta objects
		"""
		if not start_datetime:
			year, month, day = self.start_date.year, self.start_date.month, self.start_date.day
			start_datetime = mxDateTime.Date(year, month, day)
		return [eq.datetime - start_datetime for eq in self]

	def get_inter_event_times(self):
		"""
		Return time interval in days between each subsequent event

		:return:
			float array, inter-event times
		"""
		from time_functions import time_delta_to_days

		sorted_catalog = self.sort()
		date_times = sorted_catalog.get_datetimes()
		time_deltas = np.diff(date_times)
		return np.array([time_delta_to_days(td) for td in time_deltas])

	def timespan(self):
		"""
		Return total time span of catalog as number of fractional years.
		"""
		start_date, end_date = self.start_date, self.end_date
		return timespan(start_date, end_date)

	def get_datetimes(self):
		"""
		Return list of datetimes for all earthquakes in catalog
		"""
		return [eq.datetime for eq in self]

	def get_years(self):
		"""
		Return array of integer years for all earthquakes in catalog
		"""
		return np.array([date.year for date in self.get_datetimes()])

	def get_fractional_years(self):
		"""
		Return array with fractional years for all earthquakes in catalog
		"""
		years = np.array([eq.get_fractional_year() for eq in self])
		#years = [eq.datetime.year + (eq.datetime.month - 1.0) /12 + ((eq.datetime.day - 1.0) / 31) / 12 for eq in self]
		return years

	def get_magnitudes(self, Mtype="MW", Mrelation="default"):
		"""
		Return array of magnitudes for all earthquakes in catalog

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: "default", will select the default relation for the
			given Mtype)

		:return:
			1-D numpy float array, earthquake magnitudes
		"""
		Mags = [eq.get_or_convert_mag(Mtype, Mrelation) for eq in self]
		"""
		if Mtype.upper() == "ML":
			Mags = [eq.get_ML(Mrelation=Mrelation) for eq in self]
		elif Mtype.upper() == "MS":
			Mags = [eq.get_MS(Mrelation=Mrelation) for eq in self]
		elif Mtype.upper() == "MW":
			Mags = [eq.get_MW(Mrelation=Mrelation) for eq in self]
		"""
		return np.array(Mags)

	def get_magnitude_uncertainties(self, min_uncertainty=0.3):
		"""
		Return array with magnitude uncertainties

		:param min_uncertainty:
			Float, minimum uncertainty that will be used to replace zero values

		:return:
			1-D numpy float array, magnitude uncertainties
		"""
		Mag_uncertainties = np.array([eq.errM for eq in self])
		Mag_uncertainties[np.where(Mag_uncertainties == 0)] = min_uncertainty
		return Mag_uncertainties

	def get_Mtypes(self):
		"""
		Obtain list of magnitude types in catalog.

		:return:
			list of strings
		"""
		Mtypes = set()
		for eq in self:
			Mtypes.update(eq.get_Mtypes())
		return list(Mtypes)

	def get_Mtype_counts(self):
		"""
		Obtain number of earthquakes for each magnitude type in catalog

		:return:
			dict, mapping magnitude types to integers
		"""
		Mtype_counts = {}
		for eq in self:
			eq_Mtypes = eq.get_Mtypes()
			for Mtype in eq_Mtypes:
				if Mtype_counts.has_key(Mtype):
					Mtype_counts[Mtype] += 1
				else:
					Mtype_counts[Mtype] = 1
			if len(eq_Mtypes) > 1:
				comb_Mtype = '+'.join(sorted(eq_Mtypes))
				if Mtype_counts.has_key(comb_Mtype):
					Mtype_counts[comb_Mtype] += 1
				else:
					Mtype_counts[comb_Mtype] = 1
		return Mtype_counts

	def get_max_intensities(self):
		"""
		Return array with maximum intensities

		:return:
			1-D numpy int array, maximum intensities
		"""
		return np.array([eq.intensity_max for eq in self])

	def get_depths(self):
		"""
		Return array of focal depths for all earthquakes in catalog

		:return:
			1-D numpy float array, earthquake focal depths
		"""
		return np.array([eq.depth for eq in self])

	def get_longitudes(self):
		"""
		Return array of longitudes for all earthquakes in catalog

		:return:
			1-D numpy float array, epicenter longitudes
		"""
		return np.array([eq.lon for eq in self])

	def get_latitudes(self):
		"""
		Return array of latitudes for all earthquakes in catalog

		:return:
			1-D numpy float array, epicenter latitudes
		"""
		return np.array([eq.lat for eq in self])

	def get_cartesian_coordinates(self, proj="lambert1972"):
		"""
		Return cartesian coordinates

		:param proj:
			String, projection name: either "lambert1972" or "utm31"

		:return:
			List with (easting, northing) tuples
		"""
		import mapping.geotools.coordtrans as coordtrans
		lons, lats = self.get_longitudes(), self.get_latitudes()
		coord_list = zip(lons, lats)
		if proj == "lambert1972":
			return coordtrans.lonlat_to_lambert1972(coord_list)
		elif proj == "utm31N":
			return coordtrans.utm_to_lonlat(coord_list, proj)

	def Tminmax(self, Mmax=None, Mtype="MW", Mrelation="default"):
		"""
		Return tuple with oldest date and youngest date in catalog.

		:param Mmax:
			Float, maximum magnitude. Useful to check completeness periods.
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		"""
		DateTimes = self.get_datetimes()
		if Mmax != None:
			filtered_DateTimes = []
			Mags = self.get_magnitudes(Mtype=Mtype, Mrelation=Mrelation)
			for M, dt in zip(Mags, DateTimes):
				if M < Mmax:
					filtered_DateTimes.append(dt)
			DateTimes = filtered_DateTimes
		if DateTimes:
			return (min(DateTimes), max(DateTimes))
		else:
			return (None, None)

	def Mminmax(self, Mtype="MW", Mrelation="default"):
		"""
		Return tuple with minimum and maximum magnitude in catalog.

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		"""
		Mags = self.get_magnitudes(Mtype=Mtype, Mrelation=Mrelation)
		return (np.nanmin(Mags), np.nanmax(Mags))

	def depth_minmax(self):
		"""
		Return tuple with minimum and maximum depth in catalog.
		"""
		depths = self.get_depths()
		return (depths.min(), depths.max())

	def lon_minmax(self):
		"""
		Return tuple with minimum and maximum longitude in catalog.
		"""
		longitudes = self.get_longitudes()
		return (longitudes.min(), longitudes.max())

	def lat_minmax(self):
		"""
		Return tuple with minimum and maximum latitude in catalog.
		"""
		latitudes = self.get_latitudes()
		return (latitudes.min(), latitudes.max())

	def get_region(self):
		"""
		Return (w, e, s, n) tuple with geographic extent of catalog
		"""
		try:
			return self.lon_minmax() + self.lat_minmax()
		except:
			return None

	def get_Mmin(self, Mtype="MW", Mrelation="default"):
		"""
		Compute minimum magnitude in catalog

		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MW")
		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			Float, maximum observed magnitude
		"""
		return np.nanmin(self.get_magnitudes(Mtype, Mrelation))

	def get_Mmax(self, Mtype="MW", Mrelation="default"):
		"""
		Compute maximum magnitude in catalog

		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MW")
		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			Float, maximum observed magnitude
		"""
		if len(self) > 0:
			Mmax = np.nanmax(self.get_magnitudes(Mtype, Mrelation))
		else:
			Mmax = np.nan
		return Mmax

	def get_M0(self, Mrelation="default"):
		"""
		Return array with seismic moments for all earthquakes in catalog.

		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			1-D numpy float array, earthquake moments
		"""
		return np.array([eq.get_M0(Mrelation=Mrelation) for eq in self])

	def get_M0_total(self, Mrelation="default"):
		"""
		Compute total seismic moment.

		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MS" or "ML")
			(default: None, will select the default relation for the given Mtype)

		:return:
			Float, total seismic moment in N.m
		"""
		return np.add.reduce(self.get_M0(Mrelation=Mrelation))

	def get_M0rate(self, completeness=None, Mrelation="default"):
		"""
		Compute seismic moment rate.

		:param completeness:
			instance of :class:`Completeness` (default: None)
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MS" or "ML")
			(default: None, will select the default relation for the given Mtype)

		:return:
			Float, seismic moment rate in N.m/yr
		"""
		if completeness is None:
			M0rate = self.get_M0_total(Mrelation=Mrelation) / self.timespan()
		else:
			if completeness.Mtype != "MW":
				raise Exception("Completeness magnitude must be moment magnitude!")
			M0rate = 0.
			for subcatalog in self.split_completeness(completeness=completeness, Mtype="MW", Mrelation=Mrelation):
				M0rate += subcatalog.get_M0_total(Mrelation=Mrelation) / subcatalog.timespan()
		return M0rate

	def get_sorted(self, eq_attr, reverse=False):
		"""
		Get copy of catalog sorted by earthquake attribute.

		:param eq_attr:
			str, attribute of :class:`LocalEarthquake`
		:param reverse:
			bool, whether to sort ascending (False) or descending (True)
			(default: False)

		:return:
			instance of :class:`EQCatalog`
		"""
		eq_list = sorted(self.eq_list, key=lambda eq:getattr(eq, eq_attr, None),
						reverse=reverse)
		return EQCatalog(eq_list, start_date=self.start_date, end_date=self.end_date,
						region=self.region, name=self.name)

	def subselect(self, region=None, start_date=None, end_date=None, Mmin=None, Mmax=None, min_depth=None, max_depth=None, attr_val=(), Mtype="MW", Mrelation="default", include_right_edges=True, catalog_name=""):
		"""
		Make a subselection from the catalog.

		:param region:
			(w, e, s, n) tuple specifying rectangular region of interest in
			geographic coordinates (default: None)
		:param start_date:
			Int or date or datetime object specifying start of time window of interest
			If integer, start_date is interpreted as start year
			(default: None)
		:param end_date:
			Int date or datetime object specifying end of time window of interest
			If integer, end_date is interpreted as end year
			(default: None)
		:param Mmin:
			Float, minimum magnitude to extract (default: None)
		:param Mmax:
			Float, maximum magnitude to extract (default: None)
		:param min_depth:
			Float, minimum depth in km to extract (default: None)
		:param max_depth:
			Float, maximum depth in km to extract (default: None)
		:param attr_val:
			(attribute, value) tuple (default: ())
		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param include_right_edges:
			bool, whether or not to include earthquakes that have properties
			equal to the right-edge value of the different constraints
			(default: True)
		:param catalog_name:
			str, name of resulting catalog
			(default: "")

		:return:
			instance of :class:`EQCatalog`
		"""
		## Convert dates
		if isinstance(start_date, int):
			start_date = mxDateTime.Date(start_date, 1, 1)
		elif isinstance(start_date, datetime.datetime):
			start_date = start_date.date()
		elif isinstance(start_date, mxDateTime.DateTimeType):
			start_date = mxDateTime.Date(*start_date.timetuple()[:3])
		if isinstance(end_date, int):
			end_date = mxDateTime.Date(end_date, 12, 31)
		elif isinstance(end_date, datetime.datetime):
			end_date = end_date.date()
		elif isinstance(end_date, mxDateTime.DateTimeType):
			end_date = mxDateTime.Date(*end_date.timetuple()[:3])

		## Check each constraint separately
		eq_list = self.eq_list

		if region != None:
			w, e, s, n = region
			if include_right_edges:
				eq_list = [eq for eq in eq_list if w <= eq.lon <= e and s <= eq.lat <= n]
			else:
				eq_list = [eq for eq in eq_list if w <= eq.lon < e and s <= eq.lat < n]
		if start_date != None:
			eq_list = [eq for eq in eq_list if start_date <= eq.date]
		if end_date != None:
			if include_right_edges:
				eq_list = [eq for eq in eq_list if eq.date <= end_date]
			else:
				eq_list = [eq for eq in eq_list if eq.date < end_date]
		if Mmin != None:
			cat2 = EQCatalog(eq_list)
			Mags = cat2.get_magnitudes(Mtype, Mrelation)
			is_selected = Mmin <= Mags
			eq_list = [eq_list[i] for i in range(len(eq_list)) if is_selected[i]]
		if Mmax != None:
			cat2 = EQCatalog(eq_list)
			Mags = cat2.get_magnitudes(Mtype, Mrelation)
			if include_right_edges:
				is_selected = Mmax >= Mags
			else:
				is_selected = Mmax > Mags
			eq_list = [eq_list[i] for i in range(len(eq_list)) if is_selected[i]]
		if min_depth != None:
			eq_list = [eq for eq in eq_list if min_depth <= eq.depth]
		if max_depth != None:
			if include_right_edges:
				eq_list = [eq for eq in eq_list if eq.depth <= max_depth]
			else:
				eq_list = [eq for eq in eq_list if eq.depth < max_depth]
		if len(attr_val) == 2:
			attr, val = attr_val
			eq_list = [eq for eq in eq_list if getattr(eq, attr) == val]

		## Update catalog information
		if region is None:
			if self.region:
				region = self.region
			else:
				region = self.get_region()
		if start_date is None:
			start_date = self.start_date
		if end_date is None:
			end_date = self.end_date
		if not include_right_edges:
			end_date -= mxDateTime.DateTimeDelta(1)

		if not catalog_name:
			catalog_name = self.name + " (subselect)"

		return EQCatalog(eq_list, start_date=start_date, end_date=end_date, region=region, name=catalog_name)

	def subselect_declustering(self, method="Cluster", window="GardnerKnopoff1974", fa_ratio=0.5, Mtype="MW", Mrelation="default", return_triggered_catalog=False, catalog_name=""):
		"""
		Subselect earthquakes in the catalog that conform with the specified
		declustering method and params.

		:param method:
			String, declustering method: "Window" or "Cluster" (default: Cluster).
			The window method uses only the mainshock to determine the size of a
			cluster. The cluster method uses all earthquakes in a cluster.
		:param window:
			String, declustering window: "GardnerKnopoff1974", "Gruenthal2009"
			or "Uhrhammer1986" (default: GardnerKnopoff1974).
		:param fa_ratio:
			Float, foreshock/aftershock time window ratio (default: 0.50)
		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MW"). Note: some
			methods use params that are specified for a certain magnitude scale.
		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param return_triggered_catalog:
			Boolean, return also triggered catalog (default: False)
		:param catalog_name:
			str, name of resulting catalog
			(default: "")

		:return:
			instance of :class:`EQCatalog`
			if return_triggered_catalog = False also an triggered catalog is
			returned as instance of :class:`EQCatalog`
		"""
		from declustering import (WindowMethod, ClusterMethod,
			GardnerKnopoff1974Window, Gruenthal2009Window, Uhrhammer1986Window)

		methods = {
			"Window": WindowMethod(),
			"Cluster": ClusterMethod(),
			}
		windows = {
			"GardnerKnopoff1974": GardnerKnopoff1974Window(),
			"Gruenthal2009": Gruenthal2009Window(),
			"Uhrhammer1986": Uhrhammer1986Window(),
			}

		magnitudes = self.get_magnitudes(Mtype=Mtype, Mrelation=Mrelation)
		datetimes = np.array(self.get_datetimes())
		lons = self.get_longitudes()
		lats = self.get_latitudes()

		## Remove NaN magnitudes
		idxs = -np.isnan(magnitudes)
		magnitudes = magnitudes[idxs]
		datetimes = datetimes[idxs]
		lons = lons[idxs]
		lats = lats[idxs]

		d_index = methods[method].decluster(magnitudes, datetimes, lons, lats,
			windows[window], fa_ratio)

		dc = self.__getitem__(np.where(d_index == 1)[0])
		tc = self.__getitem__(np.where(d_index == 0)[0])

		if not catalog_name:
			dc.name = self.name + " (Declustered)"
			tc.name = self.name + " (Triggered)"
		else:
			dc.name = catalog_name
			tc.name = catalog_name

		if return_triggered_catalog:
			return dc, tc
		else:
			return dc

	def subselect_completeness(self, completeness=default_completeness, Mtype="MW", Mrelation="default", catalog_name="", verbose=True):
		"""
		Subselect earthquakes in the catalog that conform with the specified
		completeness criterion.

		:param completeness:
			instance of :class:`Completeness` (default: Completeness_MW_201303a)
		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MW")
		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param catalog_name:
			str, name of resulting catalog
			(default: "")
		:param verbose:
			Bool, whether or not some info should be printed (default: True)

		:return:
			instance of :class:`EQCatalog`
		"""
		if completeness:
			start_date = min(completeness.min_dates)
			if completeness.Mtype != Mtype:
				raise Exception("Magnitude type of completeness not compatible with specified Mtype!")
		else:
			start_date = self.start_date
		end_date = self.end_date

		## Select magnitudes according to completeness criteria
		if completeness:
			eq_list = []
			for eq in self.eq_list:
				M = eq.get_or_convert_mag(Mtype, Mrelation)
				if M >= completeness.get_completeness_magnitude(eq.date):
					eq_list.append(eq)
		else:
			eq_list = self.eq_list

		if verbose:
			print "Number of events constrained by completeness criteria: %d out of %d" % (len(eq_list), len(self.eq_list))

		if not catalog_name:
			catalog_name = self.name + " (completeness-constrained)"
		return EQCatalog(eq_list, start_date=start_date, end_date=end_date, region=self.region, name=catalog_name)

	def split_completeness(self, completeness=default_completeness, Mtype="MW", Mrelation="default"):
		"""
		Split catlog in subcatalogs according to completeness periods and magnitudes

		:param completeness:
			instance of :class:`Completeness` (default: Completeness_MW_201303a)
		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MW")
		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			list of instances of :class:`EQCatalog`
		"""
		completeness_catalogs = []
		min_mags = completeness.min_mags[::-1]
		max_mags = list(min_mags[1:]) + [None]
		start_dates = completeness.min_dates[::-1]
		for Mmin, Mmax, start_date in zip(min_mags, max_mags, start_dates):
			catalog = self.subselect(Mmin=Mmin, Mmax=Mmax, start_date=start_date, end_date=self.end_date)
			completeness_catalogs.append(catalog)
		return completeness_catalogs

	def bin_year(self, start_year, end_year, dYear, Mmin=None, Mmax=None, Mtype="MW", Mrelation="default"):
		"""
		Bin earthquakes into year intervals

		:param start_year:
			Int, lower year to bin (left edge of first bin)
		:param end_year:
			Int, upper year to bin (right edge of last bin)
		:param dYear:
			Int, bin interval in years
		:param Mmin:
			Float, minimum magnitude (inclusive) (default: None)
		:param Mmax:
			Float, maximum magnitude (inclusive) (default: None)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			tuple (bins_N, bins_Years)
			bins_N: array containing number of earthquakes for each bin
			bins_Years: array containing lower year of each interval
		"""
		bins_Years = np.arange(start_year, end_year+dYear, dYear)
		## Select years according to magnitude criteria
		subcatalog = self.subselect(start_date=start_year, end_date=end_year, Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		Years = subcatalog.get_years()
		bins_N, bins_Years = np.histogram(Years, bins_Years)
		return (bins_N, bins_Years[:-1])

	def bin_year_by_M0(self, start_year, end_year, dYear, Mmin=None, Mmax=None, Mrelation="default"):
		"""
		Bin earthquakes moments into year intervals

		:param start_year:
			Int, lower year to bin (left edge of first bin)
		:param end_year:
			Int, upper year to bin (right edge of last bin)
		:param dYear:
			Int, bin interval in years
		:param Mmin:
			Float, minimum magnitude (inclusive) (default: None)
		:param Mmax:
			Float, maximum magnitude (inclusive) (default: None)
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			tuple (bins_M0, bins_Years)
			bins_M0: array containing summed seismic moment in each bin
			bins_Years: array containing lower year of each interval
		"""
		bins_Years = np.arange(start_year, end_year+dYear, dYear)
		bins_M0 = np.zeros(len(bins_Years))
		## Select years according to magnitude criteria
		subcatalog = self.subselect(start_date=start_year, end_date=end_year, Mmin=Mmin, Mmax=Mmax, Mtype="MW", Mrelation=Mrelation)

		for eq in subcatalog:
			try:
				bin_id = np.where(bins_Years <= eq.datetime.year)[0][-1]
			except IndexError:
				## These are earthquakes that are younger
				pass
			else:
				bins_M0[bin_id] += eq.get_M0(Mrelation=Mrelation)
		return bins_M0, bins_Years


	def bin_day(self, start_date, end_date, dday, Mmin=None, Mmax=None, Mtype="MW", Mrelation="default"):
		"""
		Bin earthquakes into day intervals

		:param start_date:
			instance of datetime.date, start date
		:param end_date:
			instance of datetime.date, end date
		:param dday:
			Int, bin interval in days
		:param Mmin:
			Float, minimum magnitude (inclusive)
		:param Mmax:
			Float, maximum magnitude (inclusive)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			tuple (bins_N, bins_Days)
			bins_N: array containing number of earthquakes for each bin
			bins_Days: array containing lower day of each interval
				(relative to first day)
		"""
		bins_Days = np.arange(0, (end_date - start_date).days + dday, dday)
		## Select years according to magnitude criteria
		subcatalog = self.subselect(start_date=start_date, end_date=end_date, Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		Days = [(eq.date - start_date).days + (eq.date - start_date).seconds / 86400.0 for eq in subcatalog]
		bins_N, bins_Days = np.histogram(Days, bins_Days)
		return (bins_N, bins_Days[:-1])

	def bin_day_by_M0(self, start_date, end_date, dday, Mmin=None, Mmax=None, Mrelation="default"):
		"""
		Bin earthquake moments into day intervals.

		:param start_date:
			instance of datetime.date, start date
		:param end_date:
			instance of datetime.date, end date
		:param dday:
			Int, bin interval in days
		:param Mmin:
			Float, minimum magnitude (inclusive)
		:param Mmax:
			Float, maximum magnitude (inclusive)
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			tuple (bins_M0, bins_Days)
			bins_M0: array containing summed seismic moment in each bin
			bins_Days: array containing lower day of each interval
				(relative to first day)
		"""
		bins_Days = np.arange(0, (end_date - start_date).days + dday, dday)
		bins_M0 = np.zeros(len(bins_Days))
		## Select years according to magnitude criteria
		subcatalog = self.subselect(start_date=start_date, end_date=end_date, Mmin=Mmin, Mmax=Mmax, Mtype="MW", Mrelation=Mrelation)

		for eq in subcatalog:
			try:
				bin_id = np.where(bins_Days <= (eq.date - start_date).days)[0][-1]
			except IndexError:
				## These are earthquakes that are younger
				pass
			else:
				bins_M0[bin_id] += eq.get_M0(Mrelation=Mrelation)
		return bins_M0, bins_Days

	def bin_hour(self, Mmin=None, Mmax=None, Mtype="MW", Mrelation="default", start_year=None, end_year=None):
		"""
		Bin earthquakes into hour intervals

		:param Mmin:
			Float, minimum magnitude (inclusive) (default: None)
		:param Mmax:
			Float, maximum magnitude (inclusive) (default: None)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param start_year:
			Int, lower year to bin (default: None)
		:param end_year:
			Int, upper year to bin (default: None)

		:return:
			tuple (bins_N, bins_Hours)
			bins_N: array containing number of earthquakes for each bin
			bins_Hours: array containing lower limit of each hour interval
		"""
		subcatalog = self.subselect(start_date=start_year, end_date=end_year, Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		hours = np.array([eq.get_fractional_hour() for eq in subcatalog])
		bins_Hr = np.arange(25)
		bins_N, junk = np.histogram(hours, bins_Hr)
		return bins_N, bins_Hr[:-1]

	def bin_depth(self,
		min_depth=0,
		max_depth=30,
		bin_width=2,
		depth_error=None,
		Mmin=None,
		Mmax=None,
		Mtype="MW",
		Mrelation="default",
		start_date=None,
		end_date=None):
		"""
		Bin earthquakes into depth bins

		:param min_depth:
			Int, minimum depth in km (default: 0)
		:param max_depth:
			Int, maximum depth in km (default: 30)
		:param bin_width:
			Int, bin width in km (default: 2)
		:param depth_error:
			Float, maximum depth uncertainty (default: None)
		:param Mmin:
			Float, minimum magnitude (inclusive) (default: None)
		:param Mmax:
			Float, maximum magnitude (inclusive) (default: None)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param start_date:
			Int or datetime.date, lower year or date to bin (default: None)
		:param end_date:
			Int or datetime.date, upper year or date to bin (default: None)

		:return:
			tuple (bins_N, bins_depth)
			bins_N: array containing number of earthquakes for each bin
			bins_depth: array containing lower depth value of each interval
		"""
		subcatalog = self.subselect(start_date=start_date, end_date=end_date, Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		if depth_error:
			depths = [eq.depth for eq in subcatalog if not eq.depth in (None, 0) and 0 < eq.errz < depth_error]
		else:
			depths = [eq.depth for eq in subcatalog if not eq.depth in (None, 0)]
		bins_depth = np.arange(min_depth, max_depth + bin_width, bin_width)
		bins_N, _ = np.histogram(depths, bins_depth)
		return bins_N, bins_depth[:-1]

	def bin_depth_by_M0(self, min_depth=0, max_depth=30, bin_width=2, depth_error=None, Mmin=None, Mmax=None, Mrelation="default", start_year=None, end_year=None):
		"""
		Bin earthquake moments into depth bins

		:param min_depth:
			Int, minimum depth in km (default: 0)
		:param max_depth:
			Int, maximum depth in km (default: 30)
		:param bin_width:
			Int, bin width in km (default: 2)
		:param depth_error:
			Float, maximum depth uncertainty (default: None)
		:param Mmin:
			Float, minimum magnitude (inclusive) (default: None)
		:param Mmax:
			Float, maximum magnitude (inclusive) (default: None)
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param start_year:
			Int, lower year to bin (default: None)
		:param end_year:
			Int, upper year to bin (default: None)

		:return:
			tuple (bins_M0, bins_depth)
			bins_M0: array containing summed seismic moment in each bin
			bins_depth: array containing lower depth value of each interval
		"""
		bins_depth = np.arange(min_depth, max_depth + bin_width, bin_width)
		bins_M0 = np.zeros(len(bins_depth))
		subcatalog = self.subselect(start_date=start_year, end_date=end_year, Mmin=Mmin, Mmax=Mmax, Mtype="MW", Mrelation=Mrelation, min_depth=min_depth, max_depth=max_depth)
		if depth_error:
			min_depth_error = 0
		else:
			min_depth_error = -1
			depth_error = 100
		for eq in subcatalog:
			if eq.depth not in (None, 0) and min_depth_error < eq.errz < depth_error:
				try:
					#bin_id = np.where((bins_depth + bin_width) >= eq.depth)[0][0]
					bin_id = np.where(bins_depth <= eq.depth)[0][-1]
				except:
					## These are earthquakes that are deeper
					pass
				else:
					bins_M0[bin_id] += eq.get_M0(Mrelation=Mrelation)
		return bins_M0, bins_depth

	def bin_mag(self, Mmin, Mmax, dM=0.2, Mtype="MW", Mrelation="default", completeness=None, verbose=True):
		"""
		Bin all earthquake magnitudes in catalog according to specified magnitude interval.

		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude interval (default: 0.2)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes (default: None)
		:param verbose:
			Bool, whether or not to print additional information

		#:param large_eq_correction: (M, corr_factor) tuple, with M the lower magnitude for which to apply corr_factor
		#	This is to correct the frequency of large earthquakes having a return period which is longer
		#	than the catalog length

		:return:
			Tuple (bins_N, bins_Mag)
			bins_N: array containing number of earthquakes for each magnitude interval
			bins_Mag: array containing lower magnitude of each interval
		"""
		from hazard.rshalib.utils import seq

		## Set lower magnitude to lowermost threshold magnitude possible
		if completeness:
			Mmin = max(Mmin, completeness.get_completeness_magnitude(self.end_date))

		## Construct bins_Mag, including Mmax as right edge
		#Mmin = np.floor(Mmin / dM) * dM
		#Mmax = np.ceil(Mmax / dM) * dM
		#num_bins = int((Mmax - Mmin) / dM) + 1
		#bins_Mag = np.arange(num_bins + 1) * dM + Mmin
		bins_Mag = seq(Mmin, Mmax, dM)

		## Select magnitudes according to completeness criteria
		if completeness:
			cc_catalog = self.subselect_completeness(completeness, Mtype, Mrelation, verbose=verbose)
			Mags = cc_catalog.get_magnitudes(Mtype, Mrelation)
		else:
			Mags = self.get_magnitudes(Mtype, Mrelation)
		num_events = len(Mags)

		## Compute number of earthquakes per magnitude bin
		bins_N, bins_Mag = np.histogram(Mags, bins_Mag)
		bins_Mag = bins_Mag[:-1]

		return bins_N, bins_Mag

	def get_initial_completeness_dates(self, magnitudes, completeness=default_completeness):
		"""
		Compute initial date of completeness for list of magnitudes

		:param magnitudes:
			list or numpy array, magnitudes
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes. If None, use start year of
			catalog (default: Completeness_MW_201303a)

		:return:
			numpy array, completeness dates
		"""
		## Calculate year of completeness for each magnitude interval
		if completeness:
			completeness_dates = []
			for M in magnitudes:
				start_date = max(self.start_date, completeness.get_initial_completeness_date(M))
				#start_date = completeness.get_initial_completeness_date(M)
				completeness_dates.append(start_date)
		else:
			print("Warning: no completeness object provided. Using catalog length!")
			completeness_dates = [self.start_date] * len(magnitudes)
		completeness_dates = np.array(completeness_dates)
		return completeness_dates

	def get_uniform_completeness(self, Mmin, Mtype="MW"):
		"""
		Construct completeness object with uniform completeness
		since start of catalog.

		:param Mmin:
			float, minimum magnitude of completeness
		:param Mtype:
			string, magnitude type for :param:`Mmin`
			(default: "MW")

		:return:
			instance of :class:`Completeness`
		"""
		min_date = self.start_date.year
		return Completeness([min_date], [Mmin], Mtype=Mtype)

	def get_completeness_timespans(self, magnitudes, completeness=default_completeness):
		"""
		Compute completeness timespans for list of magnitudes

		:param magnitudes:
			list or numpy array, magnitudes
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes. If None, use start year of
			catalog (default: Completeness_MW_201303a)

		:return:
			numpy float array, completeness timespans (fractional years)
		"""
		return completeness.get_completeness_timespans(magnitudes, self.end_date)

	def get_incremental_MagFreq(self, Mmin, Mmax, dM=0.2, Mtype="MW", Mrelation="default", completeness=default_completeness, trim=False, verbose=True):
		"""
		Compute incremental magnitude-frequency distribution.

		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude interval
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes (default: Completeness_MW_201303a)
		:param trim:
			Bool, whether empty bins at start and end should be trimmed
			(default: False)
		:param verbose:
			Bool, whether or not to print additional information (default: True)

		:return:
			Tuple (bins_N_incremental, bins_Mag)
			bins_N_incremental: incremental annual occurrence rates
			bins_Mag: left edges of magnitude bins
		"""
		bins_N, bins_Mag = self.bin_mag(Mmin, Mmax, dM, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, verbose=verbose)
		bins_timespans = self.get_completeness_timespans(bins_Mag, completeness)

		bins_N_incremental = bins_N / bins_timespans

		## Optionally, trim empty trailing intervals
		if trim:
			last_non_zero_index = np.where(bins_N > 0)[0][-1]
			bins_N_incremental = bins_N_incremental[:last_non_zero_index+1]
			bins_Mag = bins_Mag[:last_non_zero_index+1]

		return bins_N_incremental, bins_Mag

	def get_incremental_MFD(self, Mmin, Mmax, dM=0.2, Mtype="MW", Mrelation="default", completeness=default_completeness, trim=False, verbose=True):
		"""
		Compute incremental magnitude-frequency distribution.

		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude interval
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes (default: Completeness_MW_201303a)
		:param trim:
			Bool, whether empty bins at start and end should be trimmed
			(default: False)
		:param verbose:
			Bool, whether or not to print additional information (default: True)

		:return:
			instance of nhlib :class:`EvenlyDiscretizedMFD`
		"""
		from hazard.rshalib.mfd import EvenlyDiscretizedMFD
		bins_N_incremental, bins_Mag = self.get_incremental_MagFreq(Mmin, Mmax, dM, Mtype, Mrelation, completeness, trim, verbose=verbose)
		## Mmin may have changed depending on completeness
		Mmin = bins_Mag[0]
		return EvenlyDiscretizedMFD(Mmin + dM/2, dM, list(bins_N_incremental), Mtype=Mtype)

	def get_cumulative_MagFreq(self, Mmin, Mmax, dM=0.1, Mtype="MW", Mrelation="default", completeness=default_completeness, trim=False, verbose=True):
		"""
		Compute cumulative magnitude-frequency distribution.

		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude bin width (default: 0.1)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes (default: Completeness_MW_201303a)
		:param trim:
			Bool, whether empty bins at start and end should be trimmed
			(default: False)
		:param verbose:
			Bool, whether or not to print additional information (default: True)

		:return:
			Tuple (bins_N_cumulative, bins_Mag)
			bins_N_cumulative: cumulative annual occurrence rates
			bins_Mag: left edges of magnitude bins
		"""
		bins_N_incremental, bins_Mag = self.get_incremental_MagFreq(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, trim=trim, verbose=verbose)
		## Reverse arrays for calculating cumulative number of events
		bins_N_incremental = bins_N_incremental[::-1]
		bins_N_cumulative = np.add.accumulate(bins_N_incremental)
		return bins_N_cumulative[::-1], bins_Mag

	def get_Bayesian_Mmax_pdf(self, prior_model="CEUS_COMP", Mmin_n=4.5, b_val=None, dM=0.1, truncation=(5.5, 8.25), Mtype='MW', Mrelation="default", completeness=default_completeness, verbose=True):
		"""
		Compute Mmax distribution following Bayesian approach.

		:param prior_model:
			String, indicating which prior model should be considered, one of:
			- "EPRI_extended": extended crust in EPRI (1994)
			- "EPRI_non_extended": non-extended crust in EPRI (1994)
			- "CEUS_COMP": composite prior in CEUS (2012)
			- "CEUS_MESE": Mesozoic and younger extension in CEUS (2012)
			- "CEUS_NMESE": Non-Mesozoic and younger extension in CEUS (2012)
			(default: "CEUS_COMP")
		:param Mmin_n:
			Float, lower magnitude, used to count n, the number of earthquakes
			between Mmin and Mmax_obs (corresponds to lower magnitude in PSHA).
			(default: 4.5)
		:param b_val:
			Float, b value of MFD (default: None, will compute b value
			using Weichert method)
		:param dM:
			Float, magnitude bin width.
			(default: 0.1)
		:param truncation:
			Int or tuple, representing truncation of prior distribution.
			If int, truncation is interpreted as the number of standard deviations.
			If tuple, elements are interpreted as minimum and maximum magnitude of
			the distribution
			(default: (5.5, 8.25), corresponding to the truncation applied in CEUS)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes (default: Completeness_MW_201303a)
		:param verbose:
			Bool, whether or not to print additional information (default: True)

		:return:
			(prior, likelihood, posterior, params) tuple
			- prior: instance of :class:`MmaxPMF`, prior distribution
			- likelihood: numpy array
			- posterior: instance of :class:`MmaxPMF`, posterior distribution
			- params: (observed Mmax, n, a, b) tuple
		"""
		## Mean Mmax of Global prior distributions
		if prior_model == "EPRI_extended":
			mean_Mmax = 6.4
		elif prior_model == "EPRI_non_extended":
			mean_Mmax = 6.3
		elif prior_model == "CEUS_COMP":
			mean_Mmax = 7.2
		elif prior_model == "CEUS_MESE":
			mean_Mmax = 7.35
		elif prior_model == "CEUS_NMESE":
			mean_Mmax = 6.7

		if len(self) > 0:
			Mmax_obs = self.get_Mmax()
			cc_catalog = self.subselect_completeness(completeness, verbose=verbose)
			n = len(cc_catalog.subselect(Mmin=Mmin_n))
			if not b_val:
				## Note: using lowest magnitude of completeness to compute Weichert
				## is more robust than using min_mag
				a_val, b_val, stda, stdb = self.calcGR_Weichert(Mmin=completeness.min_mag, Mmax=mean_Mmax, dM=dM, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, b_val=b_val, verbose=verbose)
			mfd = cc_catalog.get_incremental_MFD(Mmin=completeness.min_mag, Mmax=mean_Mmax, dM=dM, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, verbose=verbose)
		else:
			from hazard.rshalib.mfd import EvenlyDiscretizedMFD
			Mmax_obs = 0.
			n = 0.
			b_val = np.nan
			## Fake MFD
			mfd = EvenlyDiscretizedMFD(Mmin_n, dM, [1.0])

		prior, likelihood, posterior, params = mfd.get_Bayesian_Mmax_pdf(prior_model=prior_model, Mmax_obs=Mmax_obs, n=n, Mmin_n=Mmin_n, b_val=b_val, bin_width=dM, truncation=truncation, completeness=completeness, end_date=self.end_date, verbose=verbose)
		return (prior, likelihood, posterior, params)

	def plot_Bayesian_Mmax_pdf(self, prior_model="CEUS_COMP", Mmin_n=4.5, b_val=None, dM=0.1, truncation=(5.5, 8.25), Mtype='MW', Mrelation="default", completeness=default_completeness, num_discretizations=0, title=None, fig_filespec=None, verbose=True):
		"""
		Compute Mmax distribution following Bayesian approach.

		:param prior_model:
			String, indicating which prior model should be considered, one of:
			- "EPRI_extended": extended crust in EPRI (1994)
			- "EPRI_non_extended": non-extended crust in EPRI (1994)
			- "CEUS_COMP": composite prior in CEUS (2012)
			- "CEUS_MESE": Mesozoic and younger extension in CEUS (2012)
			- "CEUS_NMESE": Non-Mesozoic and younger extension in CEUS (2012)
			(default: "CEUS_COMP")
		:param Mmin_n:
			Float, lower magnitude, used to count n, the number of earthquakes
			between Mmin and Mmax_obs (corresponds to lower magnitude in PSHA).
			(default: 4.5)
		:param b_val:
			Float, b value of MFD (default: None, will compute b value
			using Weichert method)
		:param dM:
			Float, magnitude bin width.
			(default: 0.1)
		:param truncation:
			Int or tuple, representing truncation of prior distribution.
			If int, truncation is interpreted as the number of standard deviations.
			If tuple, elements are interpreted as minimum and maximum magnitude of
			the distribution
			(default: (5.5, 8.25), corresponding to the truncation applied in CEUS)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes (default: Completeness_MW_201303a)
		:param num_discretizations:
			int, number of portions to discretize the posterior in
			(default: 0)
		:param title:
			String, plot title (None = default title, "" = no title)
			(default: None)
		:param fig_filespec:
			String, full path of image to be saved.
			If None (default), histogram is displayed on screen.
		:param verbose:
			Bool, whether or not to print additional information (default: True)
		"""
		prior, likelihood, posterior, params = self.get_Bayesian_Mmax_pdf(prior_model, Mmin_n, b_val=b_val, dM=dM, truncation=truncation, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, verbose=verbose)
		mmax_obs, n, a_val, b_val = params
		mags = prior.values

		pylab.plot(mags, prior.weights, 'b', lw=2, label="Global prior")
		pylab.plot(mags, likelihood, 'g', lw=2, label="Regional likelihood")
		pylab.plot(mags, posterior.weights, 'r', lw=2, label="Posterior")
		if num_discretizations:
			binned_posterior = posterior.rebin_equal_weight(num_discretizations)
			weights = binned_posterior.weights.astype('d')
			## Apply some rescaling
			weights /=  posterior.max()
			# TODO: reconstruct left edges and widths of discretized portions, and plot as histogram
			pylab.plot(binned_posterior.values, weights, 'ro', label="Discretized posterior")
		font = FontProperties(size='large')
		pylab.legend(loc=0, prop=font)
		pylab.xlabel("Magnitude", fontsize="x-large")
		pylab.ylabel("Probability", fontsize="x-large")
		ax = pylab.gca()
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('large')
		if title != None:
			title += " (Mmax_obs = %.1f, n=%d, b=%.3f)" % (mmax_obs, n, b_val)
		pylab.title(title, fontsize="large")

		if fig_filespec:
			pylab.savefig(fig_filespec)
			pylab.clf()
		else:
			pylab.show()

	def plot_Mhistogram(self, Mmin, Mmax, dM=0.5, completeness=None, Mtype="MW",
		Mrelation="default", color="b", title=None, fig_filespec=None, verbose=False):
		"""
		Plot magnitude histogram of earthquakes in collection.
		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude interval
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes (default: None)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param color:
			matplotlib color specification (default: "b")
		:param title:
			String, plot title (None = default title, "" = no title)
			(default: None)
		:param fig_filespec:
			String, full path of image to be saved.
			If None (default), histogram is displayed on screen.
		"""
		bins_N, bins_Mag = self.bin_mag(Mmin, Mmax, dM, completeness=completeness, Mtype=Mtype, Mrelation=Mrelation, verbose=verbose)
		pylab.bar(bins_Mag, bins_N, width=dM, color=color)
		pylab.xlabel("Magnitude ($M_%s$)" % Mtype[1].upper(), fontsize="x-large")
		pylab.ylabel("Number of events", fontsize="x-large")
		ax = pylab.gca()
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('large')
		if title is None:
			num_events = pylab.add.reduce(bins_N)
			title = "%s (%d events)" % (self.name, num_events)
		pylab.title(title, fontsize="large")

		if fig_filespec:
			pylab.savefig(fig_filespec)
			pylab.clf()
		else:
			pylab.show()

	def plot_CumulativeYearHistogram(self, start_year, end_year, dYear, Mmin, Mmax, Mtype="MW", Mrelation="default", major_ticks=10, minor_ticks=1, completeness_year=None, regression_range=[], lang="en"):
		"""
		Plot cumulative number of earthquakes versus year

		:param start_year:
			Int, lower year to bin (left edge of first bin)
		:param end_year:
			Int, upper year to bin (right edge of last bin)
		:param dYear:
			Int, bin interval in years
		:param Mmin:
			Float, minimum magnitude (inclusive)
		:param Mmax:
			Float, maximum magnitude (inclusive)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param major_tick_interval:
			Int, interval in years for major ticks (default: 10)
		:param minor_tick_interval:
			Int, interval in years for minor ticks (default: 1)
		:param completeness_year:
			Int, year of completeness where arrow should be plotted
			(default: None)
		:param regression_range:
			List, range of years where regression should be computed adn plotted
			(default: [])
		:param lang:
			String, language of plot labels (default: "en")
		"""
		from matplotlib.patches import FancyArrowPatch
		catalog_start_year = self.start_date.year // dYear * dYear
		if start_year <= catalog_start_year:
			start_year = catalog_start_year
		bins_N, bins_Years = self.bin_year(catalog_start_year, end_year, dYear, Mmin, Mmax, Mtype=Mtype, Mrelation=Mrelation)
		bins_N_cumul = np.add.accumulate(bins_N)
		start_year_index = np.where(bins_Years == start_year)[0][0]
		bins_N = bins_N[start_year_index:]
		bins_Years = bins_Years[start_year_index:]
		bins_N_cumul = bins_N_cumul[start_year_index:]

		pylab.plot(bins_Years, bins_N_cumul, "b", label="_nolegend_")
		pylab.plot(bins_Years, bins_N_cumul, "bo", label="Cumulated number of events")

		## Optionally, plot regression for a particular range
		xmin, xmax, ymin, ymax = pylab.axis()
		if completeness_year != None:
			year_index = np.where(bins_Years == completeness_year)[0][0]
			if not regression_range:
				regression_range=[bins_Years[year_index], bins_Years[-1]]
			regression_xmin = np.where(bins_Years == regression_range[0])[0][0]
			regression_xmax = np.where(bins_Years == regression_range[1])[0][0]
			xfit = bins_Years[regression_xmin:regression_xmax+1]
			yfit = bins_N_cumul[regression_xmin:regression_xmax+1]
			m, b = pylab.polyfit(xfit, yfit, 1)
			#arr = pylab.Arrow(completeness_year, min(bins_N_cumul), 0, bins_N_cumul[year_index], facecolor='r', edgecolor='white', width=2)
			arr = FancyArrowPatch((completeness_year, min(bins_N_cumul)),(completeness_year, bins_N_cumul[year_index]), arrowstyle='-|>', mutation_scale=30, facecolor='r', edgecolor='r', lw=2)
			pylab.plot(xfit, m*xfit+b, 'r--', lw=2, label="Regression")

		pylab.xlabel({"en": "Time (years)", "nl": "Tijd (jaar)"}[lang], fontsize='x-large')
		pylab.ylabel({"en": "Cumulative number of events since", "nl": "Gecumuleerd aantal aardbevingen sinds"}[lang] + " %d" % self.start_date.year, fontsize='x-large')
		pylab.title("%s (M %.1f - %.1f)" % (self.name, Mmin, Mmax), fontsize='xx-large')
		majorLocator = MultipleLocator(major_ticks)
		minorLocator = MultipleLocator(minor_ticks)
		ax = pylab.gca()
		if completeness_year != None:
			ax.add_patch(arr)
		ax.xaxis.set_major_locator(majorLocator)
		ax.xaxis.set_minor_locator(minorLocator)
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('large')
		font = FontProperties(size='x-large')
		pylab.legend(loc=0, prop=font)
		#xmin, xmax, ymin, ymax = pylab.axis()
		pylab.axis((start_year, end_year, 0, ymax))
		pylab.show()

	def plot_CumulatedM0(self, start_date=None, end_date=None, bin_width=10, bin_width_spec="years", binned=False, histogram=True, Mrelation="default", M0max=None, fig_filespec=None):
		"""
		Plot cumulated seismic moment versus time.

		:param start_date:
			datetime.date object or integer, start date or start year
			(default: None)
		:param end_date:
			datetime.date object or integer, end date or end year
			(default: None)
		:param bin_width:
			Int, bin width of histogram and binned curve (default: 10)
		:param bin_width_spec:
			String, bin width specification, either "year" or "day" (default: "year")
		:param binned:
			Boolean, whether or not to plot binned curve (default: False)
		:param histogram:
			Boolean, whether or not to plot histogram (default: True)
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for MW)
		:param M0max:
			Float, maximum seismic moment for y axis (default: None)
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		"""
		if start_date == None:
			start_date = self.start_date
		if end_date == None:
			end_date = self.end_date

		if bin_width_spec.lower()[:4] == "year":
			bins_M0, bins_Dates = self.bin_year_by_M0(start_date.year, end_date.year, bin_width, Mrelation=Mrelation)
			#bins_Dates = np.arange(start_date.year, end_date.year+bin_width, bin_width)
			subcatalog = self.subselect(start_date=start_date.year, end_date=end_date.year)
			Dates = [eq.datetime.year for eq in subcatalog]
		elif bin_width_spec.lower()[:3] == "day":
			bins_M0, bins_Dates = self.bin_day_by_M0(start_date, end_date, bin_width, Mrelation=Mrelation)
			#bins_Dates = np.arange((end_date - start_date).days + 1)
			subcatalog = self.subselect(start_date=start_date, end_date=end_date)
			Dates = [(eq.datetime - start_dt).days + (eq.datetime - start_dt).seconds / 86400.0 for eq in subcatalog]
		bins_M0_cumul = np.add.accumulate(bins_M0)
		unbinned_M0 = subcatalog.get_M0(Mrelation=Mrelation)
		M0_cumul = np.add.accumulate(unbinned_M0)

		## Construct arrays with duplicate points in order to plot horizontal
		## lines between subsequent points
		bins_M0_cumul2 = np.array(zip(np.concatenate([np.zeros(1, 'd'), bins_M0_cumul[:-1]]), bins_M0_cumul)).flatten()
		M0_cumul2 = np.array(zip(np.concatenate([np.zeros(1, 'd'), M0_cumul[:-1]]), M0_cumul)).flatten()
		bins_Dates2 = np.array(zip(bins_Dates, bins_Dates)).flatten()
		Dates2 = np.array(zip(Dates, Dates)).flatten()

		## Plot
		if binned:
			pylab.plot(bins_Dates2, bins_M0_cumul2, 'r', lw=2, label="Cumulative (binned)")
			#pylab.plot(bins_Dates, bins_M0_cumul, 'ro', label="")
		else:
			pylab.plot(Dates2, M0_cumul2, 'g', lw=2, label="Cumulative")
			pylab.plot(Dates, M0_cumul, 'go', label="")
		if histogram:
			pylab.bar(bins_Dates, bins_M0, width=bin_width, label="Histogram")

		font = FontProperties(size='large')
		pylab.legend(loc=0, prop=font)
		pylab.xlabel("Time (%s)" % bin_width_spec, fontsize="x-large")
		pylab.ylabel("Seismic Moment (N.m)", fontsize="x-large")
		pylab.title(self.name)
		xmin, xmax, ymin, ymax = pylab.axis()
		if M0max:
			ymax = M0max
		pylab.axis((bins_Dates[0], bins_Dates[-1], ymin, ymax))

		ax = pylab.gca()
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('large')

		if fig_filespec:
			pylab.savefig(fig_filespec)
			pylab.clf()
		else:
			pylab.show()

	def plot_DateHistogram(self, start_date=None, end_date=None, ddate=1, ddate_spec="year", mag_limits=[2,3], Mtype="MW", Mrelation="default"):
		"""
		Plot histogram with number of earthqukes versus date
		for different magnitude ranges.

		:param start_date:
			Int or date or datetime object specifying start of time window to plot
			If integer, start_date is interpreted as start year
			(default: None)
		:param end_date:
			Int or date or datetime object specifying end of time window to plot
			If integer, start_date is interpreted as start year
			(default: None)
		:param ddate:
			Int, date interval (default: 1)
		:param ddate_spec:
			String, ddate specification, either "day" or "year"
		:param mag_limits:
			List, magnitude limits
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		"""
		subcatalog = self.subselect(start_date=start_date, end_date=end_date)
		start_date, end_date = subcatalog.start_date, subcatalog.end_date

		if ddate_spec.lower()[:4] == "year":
			bins_Dates = np.arange(start_date.year, end_date.year+ddate, ddate)
		elif ddate_spec.lower()[:3] == "day":
			bins_Dates = np.arange((end_date - start_date).days + 1)
		bins_Num = []
		mag_limits = np.array(mag_limits)
		Nmag = len(mag_limits) + 1
		for i in range(Nmag):
			bins_Num.append(np.zeros(len(bins_Dates), 'd'))

		for eq in subcatalog:
			M = eq.get_or_convert_mag(Mtype, Mrelation)
			try:
				im = np.where(M < mag_limits)[0][0]
			except IndexError:
				im = -1
			if ddate_spec.lower()[:4] == "year":
				id = np.where(eq.datetime.year == bins_Dates)[0][0]
			elif ddate_spec.lower()[:3] == "day":
				id = (eq.date - start_date).days
			bins_Num[im][id] += 1

		fig = pylab.figure()
		for i in range(Nmag):
			subplot_nr = Nmag * 100 + 10 + i + 1
			fig.add_subplot(subplot_nr)
			ax = pylab.bar(bins_Dates, bins_Num[Nmag-i-1], ddate)
			if Nmag > 1:
				if i == 0:
					label = "M > %.1f" % mag_limits[-1]
				elif i == Nmag - 1:
					label = "M < %.1f" % mag_limits[0]
				else:
					label = "%.1f <= M < %.1f" % (mag_limits[Nmag-i-2], mag_limits[Nmag-i-1])
				pylab.ylabel(label)
			xmin, xmax, ymin, ymax = pylab.axis()
			pylab.axis((bins_Dates[0], bins_Dates[-1], ymin, ymax))

		pylab.xlabel("Time (%s)" % ddate_spec)
		pylab.show()

	def plot_depth_magnitude(self, start_date=None, Mtype="MW", Mrelation="default", remove_undetermined=False, title=None, fig_filespec="", fig_width=0, dpi=300):
		"""
		Plot magnitude versus depth

		:param start_date:
			Int or date or datetime object specifying start of time window to plot
			If integer, start_date is interpreted as start year
			(default: None)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param remove_undetermined:
			Boolean, remove the earthquake for which depth equals zero if true (default: False)
		:param title:
			String, plot title (None = default title, "" = no title)
			(default: None)
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)
		"""
		subcatalog = self.subselect(start_date=start_date)
		magnitudes = subcatalog.get_magnitudes(Mtype, Mrelation)
		depths = subcatalog.get_depths()

		if remove_undetermined:
			i=depths.nonzero()
			depths=depths[i]
			magnitudes=magnitudes[i]

		pylab.plot(magnitudes, depths, '.')
		pylab.xlabel("Magnitude (%s)" % Mtype)
		pylab.ylabel("Depth (km)")
		ax = pylab.gca()
		ax.invert_yaxis()
		pylab.grid(True)

		if title is None:
			title='Depth-Magnitude function {0}-{1}, {2} events'.format(subcatalog.start_date.year, subcatalog.end_date.year, len(magnitudes))

		pylab.title(title)

		if fig_filespec:
			default_figsize = pylab.rcParams['figure.figsize']
			default_dpi = pylab.rcParams['figure.dpi']
			if fig_width:
				fig_width /= 2.54
				dpi = dpi * (fig_width / default_figsize[0])
			pylab.savefig(fig_filespec, dpi=dpi)
			pylab.clf()
		else:
			pylab.show()

	def plot_time_magnitude(self, Mtype="MW", Mrelation="default", lang="en"):
		"""
		Plot magnitude versus time

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param lang:
			String, language of plot labels (default: "en")
		"""
		magnitudes = self.get_magnitudes(Mtype, Mrelation)
		years = self.get_fractional_years()
		pylab.plot(magnitudes, years, '+')
		pylab.xlabel("Magnitude (%s)" % Mtype)
		pylab.ylabel({"en": "Time (years)", "nl": "Tijd (jaar)"}[lang])
		pylab.grid(True)
		pylab.show()

	def plot_magnitude_time(self, symbol='o', edge_color='k', fill_color=None, label=None, symbol_size=50, Mtype="MW", Mrelation="default", Mrange=(None, None), overlay_catalog=None, completeness=None, completeness_color="r", vlines=False, grid=True, plot_date=False, major_tick_interval=None, minor_tick_interval=1, title=None, lang="en", legend_location=0, fig_filespec=None, fig_width=0, dpi=300, ax=None):
		"""
		Plot time versus magnitude

		:param symbol:
			matplotlib marker specification, earthquake marker symbol
			(default: 'o')
		:param edge_color:
			matplotlib color specification, earthquake marker edge color
			(default: 'r')
		:param fill_color:
			matplotlib color specification, earthquake marker fill color
			(default: None)
		:param label:
			String, legend label for earthquake epicenters
			(default: None)
		:param symbol_size:
			Int or Float, symbol size in points (default: 50)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param Mrange:
			tuple of floats representing minimum and maximu magnitude in y axis
			(default: None, None)
		:param overlay_catalog:
			class:`EQCatalog` instance, second catalog to overlay on plot,
			e.g., a declustered catalog (default: None)
		:param completeness:
			class:`Completeness` instance, plot completeness (default: None)
		:param completeness_color:
			Str, color to plot completeness line (default: "r")
		:param vlines:
			List of years or datetime objects (or both), years/dates to plot as
			vertical lines (default: False)
		:param grid:
			Boolean, plot grid (default: False)
		:param plot_date:
			Boolean, whether or not to plot time axis as dates instead of
			fractional years (default: False)
		:param major_tick_interval:
			Int, interval in years for major ticks (default: None). If none, a
			maximum number of ticks at nice locations will be used.
		:param minor_tick_interval:
			Int, interval in years for minor ticks (default: 1)
		:param lang:
			String, language of plot labels (default: "en")
		:param title:
			String, plot title (default: None)
		:param legend_location:
			String or Int: location of legend (matplotlib location code):
				"best" 	0
				"upper right" 	1
				"upper left" 	2
				"lower left" 	3
				"lower right" 	4
				"right" 		5
				"center left" 	6
				"center right" 	7
				"lower center" 	8
				"upper center" 	9
				"center" 		10
			(default: 0)
		:param fig_filespec:
			String, full path of image to be saved.
			If None (default), map is displayed on screen.
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)
		:param ax:
			matplotlib Axes instance
			(default: None)
		"""
		# TODO: check plot_dates, style parameters and labels from decluster names
		catalogs = [self]
		edge_colors=[edge_color]
		if overlay_catalog:
			catalogs.append(overlay_catalog)
			if edge_color == 'k':
				edge_colors.append('r')
			else:
				edge_colors.append('k')
		plot_catalogs_magnitude_time(catalogs, symbols=[symbol], edge_colors=edge_colors, fill_colors=[fill_color], labels=[label], symbol_size=symbol_size, Mtype=Mtype, Mrelation=Mrelation, Mrange=Mrange, completeness=completeness, completeness_color=completeness_color, vlines=vlines, grid=grid, plot_date=plot_date, major_tick_interval=major_tick_interval, minor_tick_interval=minor_tick_interval, title=title, lang=lang, legend_location=legend_location, fig_filespec=fig_filespec, fig_width=fig_width, dpi=dpi, ax=ax)

	def DailyNightlyMean(self, Mmin=None, Mmax=None, Mtype="MW", Mrelation="default", start_year=None, end_year=None, day=(7, 19)):
		"""
		Report mean number of earthquakes for different times of day

		:param Mmin:
			Float, minimum magnitude (inclusive) (default: None)
		:param Mmax:
			Float, maximum magnitude (inclusive) (default: None)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param start_year:
			Int, lower year to bin (default: None)
		:param end_year:
			Int, upper year to bin (default: None)
		:param day:
			Tuple (min_hour, max_hour), default: (7, 19)

		:return:
			Tuple (mean, daily mean, nightly mean)
		"""
		bins_N, bins_Hr = self.bin_hour(Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation, start_year=start_year, end_year=end_year)
		mean = np.mean(bins_N)
		mean_day = np.mean(bins_N[day[0]:day[1]])
		mean_night = np.mean(np.concatenate([bins_N[:day[0]], bins_N[day[1]:]]))
		return (mean, mean_day, mean_night)

	def plot_HourHistogram(self, Mmin=None, Mmax=None, Mtype="MW", Mrelation="default", start_year=None, end_year=None):
		"""
		Plot histogram with number of earthquakes per hour of the day.

		:param Mmin:
			Float, minimum magnitude (inclusive) (default: None)
		:param Mmax:
			Float, maximum magnitude (inclusive) (default: None)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param start_year:
			Int, lower year to bin (default: None)
		:param end_year:
			Int, upper year to bin (default: None)
		"""
		bins_N, bins_Hr = self.bin_hour(Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation, start_year=start_year, end_year=end_year)
		pylab.bar(bins_Hr, bins_N)
		xmin, xmax, ymin, ymax = pylab.axis()
		pylab.axis((0, 24, ymin, ymax))
		pylab.xlabel("Hour of day", fontsize='x-large')
		pylab.ylabel("Number of events", fontsize='x-large')
		ax = pylab.gca()
		ax.invert_yaxis()
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('large')

		if not start_year:
			start_year = self.start_date.year
		if not end_year:
			end_year = self.end_date.year
		if Mmin is None:
			Mmin = self.get_Mmin()
		if Mmax is None:
			Mmax = self.get_Mmax()
		pylab.title("Hourly Histogram %d - %d, M %.1f - %.1f" % (start_year, end_year, Mmin, Mmax))
		pylab.show()

	def plot_depth_histogram(self,
		min_depth=0,
		max_depth=30,
		bin_width=2,
		depth_error=None,
		normalized=False,
		Mmin=None,
		Mmax=None,
		dM=None,
		Mtype="MW",
		Mrelation="default",
		start_date=None,
		end_date=None,
		color='b',
		title=None,
		fig_filespec="",
		fig_width=0,
		dpi=300,
		legend_location=0,
		ax=None):
		"""
		Plot histogram with number of earthquakes versus depth.

		:param min_depth:
			Float, minimum depth in km (default: 0)
		:param max_depth:
			Float, maximum depth in km (default: 30)
		:param bin_width:
			Float, bin width in km (default: 2)
		:param depth_error:
			Float, maximum depth uncertainty (default: None)
		:param normalized:
			Bool, whether or not bin numbers should be normalized (default: False)
		:param Mmin:
			Float, minimum magnitude (inclusive) (default: None)
		:param Mmax:
			Float, maximum magnitude (inclusive) (default: None)
		:param dM:
			Float, magnitude binning interval
			If set, a stacked histogram will be plotted
			(default: None)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param start_date:
			Int or datetime.date, lower year or date to bin (default: None)
		:param end_date:
			Int or datetime.date, upper year or date to bin (default: None)
		:param color:
			String, matplotlib color specification for histogram bars
			if :param:`dM` is set, this may also be a list of color specs
			(default: 'b')
		:param title:
			String, title (None = default title, empty string = no title)
			(default: None)
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)
		:param legend_location:
			Int, matplotlib legend location code
			(default: 0)
		:param ax:
			matplotlib Axes instance
			(default: None)
		"""
		if ax is None:
			ax = pylab.axes()
		else:
			fig_filespec = "hold"

		if dM:
			## Compute depth histogram for each magnitude bin
			assert not None in (Mmin, Mmax)
			_, bins_mag = self.bin_mag(Mmin, Mmax, dM=dM, Mtype=Mtype, Mrelation=Mrelation)
			bins_N = []
			for mmin in bins_mag[::-1]:
				mmax = mmin + dM
				bins_n, bins_depth = self.bin_depth(min_depth, max_depth, bin_width, depth_error, mmin, mmax, Mtype, Mrelation, start_date, end_date)
				bins_N.append(bins_n)
			if isinstance(color, (list, np.ndarray)):
				colors = color
			elif isinstance(color, matplotlib.colors.Colormap):
				values = np.linspace(0, 1, len(bins_mag))
				colors = color(values)
			else:
				colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

		else:
			bins_N, bins_depth = self.bin_depth(min_depth, max_depth, bin_width, depth_error, Mmin, Mmax, Mtype, Mrelation, start_date, end_date)
			bins_N = [bins_N]
			colors = [color]
			bins_mag = [Mmin]

		if normalized:
			total_num = np.sum(map(np.sum, bins_N)) * 1.0
			bins_N = [bins_n.astype('f') / total_num for bins_n in bins_N]

		left = 0
		for bins_n, mmin, color in zip(bins_N, bins_mag[::-1], colors):
			if dM:
				label = "M %.1f - %.1f" % (mmin, mmin + dM)
			else:
				label = "_nolegend_"
			ax.barh(bins_depth, bins_n, height=bin_width, left=left, color=color, label=label)
			left += bins_n

		xmin, xmax, ymin, ymax = ax.axis()
		ax.axis((xmin, xmax, min_depth, max_depth))
		ax.set_ylabel("Depth (km)", fontsize='x-large')
		xlabel = "Number of events"
		if normalized:
			xlabel += " (%)"
		ax.set_xlabel(xlabel, fontsize='x-large')
		ax.invert_yaxis()
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('large')

		if title is None:
			if Mmin is None:
				Mmin = self.get_Mmin(Mtype=Mtype, Mrelation=Mrelation)
			if Mmax is None:
				Mmax = self.get_Mmax(Mtype=Mtype, Mrelation=Mrelation)
			title = "Depth histogram: M %.1f - %.1f" % (Mmin, Mmax)
		ax.set_title(title)
		ax.legend(loc=legend_location)

		if fig_filespec == "hold":
			return
		elif fig_filespec:
			default_figsize = pylab.rcParams['figure.figsize']
			default_dpi = pylab.rcParams['figure.dpi']
			if fig_width:
				fig_width /= 2.54
				dpi = dpi * (fig_width / default_figsize[0])
			pylab.savefig(fig_filespec, dpi=dpi)
			pylab.clf()
		else:
			pylab.show()

	def plot_Depth_M0_Histogram(self, min_depth=0, max_depth=30, bin_width=2, depth_error=None, Mmin=None, Mmax=None, Mrelation="default", start_year=None, end_year=None, color='b', title=None, log=True, fig_filespec="", fig_width=0, dpi=300):
		"""
		Plot histogram with seismic moment versus depth.

		:param min_depth:
			Float, minimum depth in km (default: 0)
		:param max_depth:
			Float, maximum depth in km (default: 30)
		:param bin_width:
			Float, bin width in km (default: 2)
		:param depth_error:
			Float, maximum depth uncertainty (default: None)
		:param Mmin:
			Float, minimum magnitude (inclusive) (default: None)
		:param Mmax:
			Float, maximum magnitude (inclusive) (default: None)
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param start_year:
			Int, lower year to bin (default: None)
		:param end_year:
			Int, upper year to bin (default: None)
		:param color:
			String, matplotlib color specification for histogram bars
			(default: 'b')
		:param log:
			Boolean, whether or not log of seismic moment should be plotted
		:param title:
			String, title (None = default title, empty string = no title)
			(default: None)
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)
		"""
		bins_M0, bins_depth = self.bin_depth_by_M0(min_depth, max_depth, bin_width, depth_error, Mmin, Mmax, Mrelation, start_year, end_year)
		try:
			pylab.barh(bins_depth, bins_M0, height=bin_width, log=log, color=color)
		except:
			## This happens when all bins_M0 values are zero
			pass
		xmin, xmax, ymin, ymax = pylab.axis()
		pylab.axis((xmin, xmax, min_depth, max_depth))
		pylab.ylabel("Depth (km)", fontsize='x-large')
		pylab.xlabel("Summed seismic moment (N.m)", fontsize='x-large')
		ax = pylab.gca()
		ax.invert_yaxis()
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('large')
		if title is None:
			if not start_year:
				start_year = self.start_date.year
			if not end_year:
				end_year = self.end_date.year
			if Mmin is None:
				Mmin = self.get_Mmin()
			if Mmax is None:
				Mmax = self.get_Mmax()
			title = "Depth Histogram %d - %d, M %.1f - %.1f" % (start_year, end_year, Mmin, Mmax)
		pylab.title(title)

		if fig_filespec:
			default_figsize = pylab.rcParams['figure.figsize']
			default_dpi = pylab.rcParams['figure.dpi']
			if fig_width:
				fig_width /= 2.54
				dpi = dpi * (fig_width / default_figsize[0])
			pylab.savefig(fig_filespec, dpi=dpi)
			pylab.clf()
		else:
			pylab.show()

	def plot_map(self, symbol='o', edge_color='r', fill_color=None, edge_width=1, label="Epicenters", symbol_size=9, symbol_size_inc=4, Mtype="MW", Mrelation="default", region=None, projection="merc", resolution="i", dlon=1., dlat=1., source_model=None, sm_color='k', sm_line_style='-', sm_line_width=2, sm_label_colname="ShortName", title=None, legend_location=0, fig_filespec=None, fig_width=0, dpi=300):
		"""
		Plot map of catalog

		:param symbol:
			matplotlib marker specification, earthquake marker symbol
			(default: 'o')
		:param edge_color:
			matplotlib color specification, earthquake marker edge color
			(default: 'r')
		:param edge_width:
			earthquake marker edge width (default: 1)
		:param fill_color:
			matplotlib color specification, earthquake marker fill color
			(default: None)
		:param label:
			String, legend label for earthquake epicenters
			(default: "Epicenters")
		:param symbol_size:
			Int or Float, symbol size in points (default: 9)
		:param symbol_size_inc:
			Int or Float, symbol size increment per magnitude relative to M=3
			(default: 4)
		:param Mtype:
			String, magnitude type for magnitude scaling (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param region:
			(w, e, s, n) tuple specifying rectangular region to plot in
			geographic coordinates (default: None)
		:param projection:
			String, map projection. See Basemap documentation
			(default: "merc")
		:param resolution:
			String, map resolution (coastlines and country borders):
			'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high), 'f' (full)
			(default: 'i')
		:param dlon:
			Float, meridian interval in degrees (default: 1.)
		:param dlat:
			Float, parallel interval in degrees (default: 1.)
		:param source_model:
			String, name of source model to overlay on the plot
			or full path to GIS file containing source model
			(default: None)
		:param sm_color:
			matplotlib color specification to plot source model
			(default: 'k')
		:param sm_line_style:
			String, line style to plot source model (default: '-')
		:param sm_line_width:
			Int, line width to plot source model (default: 2)
		:param sm_label_colname:
			Str, column name of GIS table to use as label (default: "ShortName")
		:param title:
			String, plot title (default: None)
		:param legend_location:
			String or Int: location of legend (matplotlib location code):
				"best" 	0
				"upper right" 	1
				"upper left" 	2
				"lower left" 	3
				"lower right" 	4
				"right" 	5
				"center left" 	6
				"center right" 	7
				"lower center" 	8
				"upper center" 	9
				"center" 	10
			(default: 0)
		:param fig_filespec:
			String, full path of image to be saved.
			If None (default), map is displayed on screen.
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)
		"""
		if title is None:
			title = self.name

		plot_catalogs_map([self], symbols=[symbol], edge_colors=[edge_color], fill_colors=[fill_color], edge_widths=[edge_width], labels=[label], symbol_size=symbol_size, symbol_size_inc=symbol_size_inc, Mtype=Mtype, Mrelation=Mrelation, region=region, projection=projection, resolution=resolution, dlon=dlon, dlat=dlat, source_model=source_model, sm_color=sm_color, sm_line_style=sm_line_style, sm_line_width=sm_line_width, sm_label_colname=sm_label_colname, title=title, legend_location=legend_location, fig_filespec=fig_filespec, fig_width=fig_width, dpi=dpi)

	def calcGR_LSQ(self, Mmin, Mmax, dM=0.1, cumul=True, Mtype="MW", Mrelation="default", completeness=default_completeness, b_val=None, weighted=False, verbose=False):
		"""
		Calculate a and b values of Gutenberg-Richter relation using a linear regression (least-squares).

		:param Mmin:
			Float, minimum magnitude to use for binning
		:param Mmax:
			Float, maximum magnitude to use for binning
		:param dM:
			Float, magnitude interval to use for binning (default: 0.1)
		:param cumul:
			Bool, whether to use cumulative (True) or incremental (False)
			occurrence rates for linear regression (default: True)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` (default: Completeness_MW_201303a)
		:param b_val:
			Float, fixed b value to constrain MLE estimation (default: None)
			This parameter is currently ignored.
		:param weighted:
			bool, whether or not magnitude bins should be weighted by the
			number of events in them
			(default: False)
		:param verbose:
			Bool, whether some messages should be printed or not (default: False)

		Return value:
			Tuple (a, b, stda, stdb)
			- a: a value (intercept)
			- b: b value (slope, taken positive)
			- stda: standard deviation on a value
			- stdb: standard deviation on b value
		"""
		from calcGR import calcGR_LSQ
		if weighted:
			bins_N, bins_Mag = self.bin_mag(Mmin, Mmax, dM, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, verbose=False)
			weights = bins_N
		else:
			weights = None

		if cumul:
			rates, magnitudes = self.get_cumulative_MagFreq(Mmin, Mmax, dM, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, trim=False, verbose=verbose)
		else:
			rates, magnitudes = self.get_incremental_MagFreq(Mmin, Mmax, dM, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, trim=False, verbose=verbose)

		a, b, stda, stdb = calcGR_LSQ(magnitudes, rates, b_val=b_val, weights=weights, verbose=verbose)
		if not cumul:
			a += mfd.get_a_separation(b, dM)
		return a, b, stda, stdb

	def calcGR_Aki(self, Mmin=None, Mmax=None, dM=0.1, Mtype="MW", Mrelation="default", completeness=default_completeness, b_val=None, verbose=False):
		"""
		Calculate a and b values of Gutenberg-Richter relation using original
		maximum likelihood estimation by Aki (1965)

		:param Mmin:
			Float, minimum magnitude to use for binning (ignored)
		:param Mmax:
			Float, maximum magnitude to use for binning (ignored)
		:param dM:
			Float, magnitude interval to use for binning (default: 0.1)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` (default: Completeness_MW_201303a)
		:param b_val:
			Float, fixed b value to constrain MLE estimation (ignored)
		:param verbose:
			Bool, whether some messages should be printed or not (default: False)

		:return:
			Tuple (a, b, stdb)
			- a: a value
			- b: b value
			- stdb: standard deviation on b value
		"""
		return self.analyse_recurrence(dM=dM, method="MLE", aM=0., Mtype=Mtype, Mrelation=Mrelation, completeness=completeness)

	def calcGR_Weichert(self, Mmin, Mmax, dM=0.1, Mtype="MW", Mrelation="default", completeness=default_completeness, b_val=None, verbose=True):
		"""
		Calculate a and b values of Gutenberg-Richter relation using maximum likelihood estimation
		for variable observation periods for different magnitude increments.
		Adapted from calB.m and calBfixe.m Matlab modules written by Philippe Rosset (ROB, 2004),
		which is based on the method by Weichert, 1980 (BSSA, 70, Nr 4, 1337-1346).

		:param Mmin:
			Float, minimum magnitude to use for binning
		:param Mmax:
			Float, maximum magnitude to use for binning
		:param dM:
			Float, magnitude interval to use for binning (default: 0.1)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` (default: Completeness_MW_201303a)
		:param b_val:
			Float, fixed b value to constrain MLE estimation (default: None)
		:param verbose:
			Bool, whether some messages should be printed or not (default: False)

		:return:
			Tuple (a, b, stdb)
			- a: a value
			- b: b value
			- stda: standard deviation of a value
			- stdb: standard deviation of b value

		Note:
		This regression depends on the Mmax specified, as empty magnitude bins
		are taken into account. It is therefore important to specify Mmax as
		the evaluated Mmax for the specific region or source.
		"""
		from calcGR import calcGR_Weichert
		## Note: don't use get_incremental_MagFreq here, as completeness
		## is taken into account in the Weichert algorithm !
		bins_N, bins_Mag = self.bin_mag(Mmin, Mmax, dM, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, verbose=verbose)
		return calcGR_Weichert(bins_Mag, bins_N, completeness, self.end_date, b_val=b_val, verbose=verbose)

		"""
		bins_timespans = self.get_completeness_timespans(bins_Mag, completeness)
		bins_Mag += dM/2.0

		if not b_val:
			## Initial trial value
			BETA = 1.5
		else:
			## Fixed beta
			BETA = b_val * np.log(10)
		BETL = 0
		while(np.abs(BETA-BETL)) >= 0.0001:
			#print BETA

			SNM = 0.0
			NKOUNT = 0.0
			STMEX = 0.0
			SUMTEX = 0.0
			STM2X = 0.0
			SUMEXP = 0.0

			for k in range(len(bins_N)):
				SNM += bins_N[k] * bins_Mag[k]
				NKOUNT += bins_N[k]
				TJEXP = bins_timespans[k] * np.exp(-BETA * bins_Mag[k])
				TMEXP = TJEXP * bins_Mag[k]
				SUMEXP += np.exp(-BETA * bins_Mag[k])
				STMEX += TMEXP
				SUMTEX += TJEXP
				STM2X += bins_Mag[k] * TMEXP

			#print SNM, NKOUNT, STMEX, SUMTEX, STM2X, SUMEXP

			try:
				DLDB = STMEX / SUMTEX
			except:
				break
			else:
				D2LDB2 = NKOUNT * (DLDB*DLDB - STM2X/SUMTEX)
				DLDB = DLDB * NKOUNT - SNM
				BETL = BETA
				if not b_val:
					BETA -= DLDB/D2LDB2

		B = BETA / np.log(10)
		if not b_val:
			STDBETA = np.sqrt(-1.0/D2LDB2)
			STDB = STDBETA / np.log(10)
		else:
			STDB = 0
			STDBETA = 0
		FNGTMO = NKOUNT * SUMEXP / SUMTEX
		FN0 = FNGTMO * np.exp(BETA * (bins_Mag[0] - dM/2.0))
		FLGN0 = np.log10(FN0)
		A = FLGN0
		STDFN0 = FN0 / np.sqrt(NKOUNT)
		## Applying error propogation for base-10 logarithm
		STDA = STDFN0 / (2.303 * FN0)
		#print STDA
		## Note: the following formula in Philippe Rosset's program is equivalent
		#A = np.log10(FNGTMO) + B * (bins_Mag[0] - dM/2.0)
		## This is also equivalent to:
		#A = np.log10(FNGTMO * np.exp(-BETA * (0. - (bins_Mag[0] - (dM/2.0)))))

		if verbose:
			FN5 = FNGTMO * np.exp(-BETA * (5. - (bins_Mag[0] - dM/2.0)))
			STDFN5 = FN5 / np.sqrt(NKOUNT)
			print("Maximum-likelihood estimation (Weichert)")
			print("BETA=%.3f +/- %.3f; B=%.3f +/- %.3f" % (BETA, STDBETA, B, STDB))
			print("Total number of events: %d" % NKOUNT)
			print("LOG(annual rate above M0): %.3f" % FLGN0)
			print("Annual rate above M5: %.3f +/- %.3f" % (FN5, STDFN5))

		## Other parameters computed in Philippe Rosset's version
		#STDA = np.sqrt((bins_Mag[0]-dM/2.0)**2 * STDB**2 - (STDFNGTMO**2 / ((np.log(10)**2 * np.exp(2*(A+B*(bins_Mag[0]-dM/2.0))*np.log(10))))))
		#STDA = np.sqrt(abs(A)/NKOUNT)
		#ALPHA = FNGTMO * np.exp(-BETA * (bins_Mag[0] - dM/2.0))
		#STDALPHA = ALPHA / np.sqrt(NKOUNT)
		#if Mc !=None:
		#	LAMBDA_Mc = FNGTMO * np.exp(-BETA * (Mc - (bins_Mag[0] - dM/2.0)))
		#	STD_LAMBDA_Mc = np.sqrt(LAMBDA_Mc / NKOUNT)
		#if verbose:
		#	print "Maximum likelihood: a=%.3f ($\pm$ %.3f), b=%.3f ($\pm$ %.3f), beta=%.3f ($\pm$ %.3f)" % (A, STDA, B, STDB, BETA, STDBETA)
		#if Mc != None:
		#	return (A, B, BETA, LAMBDA_Mc, STDA, STDB, STDBETA, STD_LAMBDA_Mc)
		#else:
		#	return (A, B, BETA, STDA, STDB, STDBETA)

		return A, B, STDB
		"""

	#TODO: averaged Weichert method (Felzer, 2007)

	def get_estimated_MFD(self, Mmin, Mmax, dM=0.1, method="Weichert", Mtype="MW", Mrelation="default", completeness=default_completeness, b_val=None, verbose=True):
		"""
		Compute a and b values of Gutenberg Richter relation, and return
		as TruncatedGRMFD object.

		:param Mmin:
			Float, minimum magnitude to use for binning
		:param Mmax:
			Float, maximum magnitude to use for binning
		:param dM:
			Float, magnitude interval to use for binning (default: 0.1)
		:param method:
			String, computation method, either "Weichert", "Aki", "LSQc", "LSQc", "LSQi", "wLSQc" or "wLSQi"
			(default: "Weichert")
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` (default: Completeness_MW_201303a)
		:param b_val:
			Float, fixed b value to constrain MLE estimation
			Currently only supported by Weichert method (default: None)
		:param verbose:
			Bool, whether some messages should be printed or not (default: False)

		:return:
			instance of :class:`mfd.TruncatedGRMFD`
		"""
		from hazard.rshalib.mfd import TruncatedGRMFD

		kwargs = {}
		if "LSQc" in method:
			kwargs['cumul'] = True
		elif "LSQi" in method:
			kwargs['cumul'] = False
		if "LSQ" in method:
			if method[0] == 'w':
				kwargs['weighted'] = True
			else:
				kwargs['weighted'] = False
			method = "LSQ"
		calcGR_func = {"Weichert": self.calcGR_Weichert, "Aki": self.calcGR_Aki, "LSQ": self.calcGR_LSQ}[method]
		a, b, stda, stdb = calcGR_func(Mmin=Mmin, Mmax=Mmax, dM=dM, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, b_val=b_val, verbose=verbose, **kwargs)
		return TruncatedGRMFD(Mmin, Mmax, dM, a, b, stda, stdb, Mtype)

	def plot_MFD(self, Mmin, Mmax, dM=0.2, method="Weichert", Mtype="MW", Mrelation="default", completeness=default_completeness, b_val=None, num_sigma=0, color_observed="b", color_estimated="r", plot_completeness_limits=True, Mrange=(), Freq_range=(), title=None, lang="en", fig_filespec=None, fig_width=0, dpi=300, verbose=False):
		"""
		Compute GR MFD from observed MFD, and plot result

		:param Mmin:
			Float, minimum magnitude to use for binning
		:param Mmax:
			Float, maximum magnitude to use for binning
		:param dM:
			Float, magnitude interval to use for binning (default: 0.1)
		:param method:
			String, computation method, either "Weichert", "Aki", "LSQc", "LSQi", "wLSQc" or "wLSQi"
			(default: "Weichert"). If None, only observed MFD will be plotted.
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` (default: Completeness_MW_201303a)
		:param b_val:
			Float, fixed b value to constrain Weichert estimation (default: None)
		:param num_sigma:
			Int, number of standard deviations to consider for plotting uncertainty
			(default: 0)
		:param color_observed:
			matplotlib color specification for observed MFD
		:param color_estimated:
			matplotlib color specification for estimated MFD
		:param plot_completeness_limits:
			Bool, whether or not to plot completeness limits (default: True)
		:param Mrange:
			(Mmin, Mmax) tuple, minimum and maximum magnitude in X axis
			(default: ())
		:param Freq_range:
			(Freq_min, Freq_max) tuple, minimum and maximum values in frequency
			(Y) axis (default: ())
		:param title:
			String, plot title. If None, title will be automatically generated
			(default: None)
		:param lang:
			String, language of plot axis labels (default: "en")
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)
		:param verbose:
			Bool, whether some messages should be printed or not (default: False)
		"""
		from hazard.rshalib.mfd import plot_MFD

		mfd_list, labels, colors, styles = [], [], [], []
		cc_catalog = self.subselect_completeness(completeness, Mtype, Mrelation, verbose=verbose)
		observed_mfd = cc_catalog.get_incremental_MFD(Mmin, Mmax, dM=dM, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness)
		mfd_list.append(observed_mfd)
		label = {"en": "Observed", "nl": "Waargenomen"}[lang]
		labels.append(label)
		colors.append(color_observed)

		styles.append('o')
		if method:
			estimated_mfd = cc_catalog.get_estimated_MFD(Mmin, Mmax, dM=dM, method=method, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, b_val=b_val, verbose=verbose)
			mfd_list.append(estimated_mfd)
			a, b, stdb = estimated_mfd.a_val, estimated_mfd.b_val, estimated_mfd.b_sigma
			label = {"en": "Computed", "nl": "Berekend"}[lang]
			label += " (%s): " % method
			label += "a=%.3f, b=%.3f" % (a, b)
			if not b_val:
				label += " ($\pm$%.3f)" % stdb
			labels.append(label)
			colors.append(color_estimated)
			styles.append('-')
			if num_sigma:
				sigma_mfd1 = cc_catalog.get_estimated_MFD(Mmin, Mmax, dM=dM, method=method, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, b_val=b+num_sigma*stdb, verbose=verbose)
				mfd_list.append(sigma_mfd1)
				label = {"en": "Computed", "nl": "Berekend"}[lang]
				label += " $\pm$ %d sigma" % num_sigma
				labels.append(label)
				colors.append(color_estimated)
				styles.append('--')
				sigma_mfd2 = cc_catalog.get_estimated_MFD(Mmin, Mmax, dM=dM, method=method, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, b_val=b-num_sigma*stdb, verbose=verbose)
				mfd_list.append(sigma_mfd2)
				labels.append("_nolegend_")
				colors.append(color_estimated)
				styles.append('--')

		if title is None:
			num_events = len(cc_catalog)
			Mmax_obs = cc_catalog.get_Mmax(Mtype, Mrelation)
			title = "%s (%d events, Mmax=%.2f)" % (self.name, num_events, Mmax_obs)
		completeness_limits = {True: completeness, False: None}[plot_completeness_limits]
		end_year = self.end_date.year
		plot_MFD(mfd_list, colors=colors, styles=styles, labels=labels, completeness=completeness_limits, end_year=end_year, Mrange=Mrange, Freq_range=Freq_range, title=title, lang=lang, fig_filespec=fig_filespec, fig_width=fig_width, dpi=dpi)

	def export_ZMAP(self, filespec, Mtype="MW", Mrelation="default"):
		"""
		Export earthquake list to ZMAP format (ETH Zürich).

		:param filespec:
			String, full path specification of output file
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		"""
		f = open(filespec, "w")
		for eq in self.eq_list:
			M = eq.get_or_convert_mag(Mtype, Mrelation)
			f.write("%f  %f  %d  %d  %d  %.1f %.2f %d %d\n" % (eq.lon, eq.lat, eq.datetime.year, eq.datetime.month, eq.datetime.day, M, eq.depth, eq.datetime.hour, eq.datetime.minute))
		f.close()

	def export_csv(self, csv_filespec=None, Mtype=None, Mrelation="default"):
		"""
		Export earthquake list to a csv file.

		:param csv_filespec:
			String, full path specification of output csv file
			(default: None, will write to standard output)
		:param Mtype:
			Str, magnitude type, either 'ML', 'MS' or 'MW'.
			If None, magnitudes will not be converted (default: None)
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		"""
		if csv_filespec == None:
			f = sys.stdout
		else:
			f = open(csv_filespec, "w")

		if Mtype:
			f.write('ID,Date,Time,Name,Lon,Lat,Depth,%s,Intensity_max,Macro_radius\n' % Mtype)
		else:
			f.write('ID,Date,Time,Name,Lon,Lat,Depth,ML,MS,MW,Intensity_max,Macro_radius\n')
		for eq in self.eq_list:
			try:
				date = eq.date.isoformat()
			except AttributeError:
				date = eq.date.date
			time = eq.time.isoformat()
			if eq.name != None:
				eq_name = eq.name.encode('ascii', 'ignore')
			else:
				eq_name = ""
			if Mtype:
				f.write('%d,%s,%s,"%s",%.3f,%.3f,%.1f,%.2f,%s,%s\n' % (eq.ID, date, time, eq_name, eq.lon, eq.lat, eq.depth, eq.get_or_convert_mag(Mtype, Mrelation), eq.intensity_max, eq.macro_radius))
			else:
				f.write('%d,%s,%s,"%s",%.3f,%.3f,%.1f,%.2f,%.2f,%.2f,%s,%s\n' % (eq.ID, date, time, eq_name, eq.lon, eq.lat, eq.depth, eq.ML, eq.MS, eq.MW, eq.intensity_max, eq.macro_radius))
		if csv_filespec != None:
			f.close()

	def export_KML(self, kml_filespec=None, time_folders=True, instrumental_start_year=1910, color_by_depth=False):
		"""
		Export earthquake catalog to KML.

		:param kml_filespec:
			String, full path to output KML file. If None, kml is printed
			on screen (default: None)
		:param time_folders:
			Bool, whether or not to organize earthquakes in folders by time
			(default: True)
		:param instrumental_start_year:
			Int, start year of instrumental period (only applies when time_folders
			is True) (default: 1910)
		:param color_by_depth:
			Bool, whether or not to color earthquakes by depth (default: False)

		:return:
			str, KML code (if :param:`kml_filespec` is not set)
		"""
		import mapping.kml.mykml as mykml

		kmldoc = mykml.KML()
		#year, month, day = self.start_date.year, self.start_date.month, self.start_date.day
		#start_time = datetime.datetime(year, month, day)
		start_time = datetime.datetime.now()
		kmldoc.addTimeStamp(start_time)
		current_year = start_time.year
		eq_years = self.get_years()
		eq_centuries = sorted(set((eq_years // 100) * 100))

		event_types = set([eq.event_type for eq in self])
		nt_folder = None

		if time_folders:
			hist_folder = kmldoc.addFolder("Historical", visible=False, open=False)
			inst_folder = kmldoc.addFolder("Instrumental", visible=True, open=False)
			if len(event_types) > 1 and "ke" in event_types:
				nt_folder = kmldoc.addFolder("Non-tectonic", visible=True, open=False)

			folder_24h = kmldoc.createFolder("Past 24 hours", visible=True, open=False)
			inst_folder.appendChild(folder_24h)
			folder_2w = kmldoc.createFolder("Past 2 weeks", visible=True, open=False)
			inst_folder.appendChild(folder_2w)
			folder_lastyear = kmldoc.createFolder("Past year", visible=True, open=False)
			inst_folder.appendChild(folder_lastyear)

			decade_folders = {}
			for decade in range(max(instrumental_start_year, eq_years.min()), current_year, 10)[::-1]:
				folder_name = "%d - %d" % (decade, min(current_year - 1, decade + 9))
				decade_folder = kmldoc.createFolder(folder_name, visible=True, open=False)
				inst_folder.appendChild(decade_folder)
				decade_folders[decade] = decade_folder

			century_folders = {}
			last_century = ((instrumental_start_year - 1) // 100) * 100
			for century in eq_centuries:
				if century <= last_century:
					folder_name = "%d - %d" % (century, min(instrumental_start_year, century + 99))
					century_folder = kmldoc.createFolder(folder_name, visible=True, open=False)
					hist_folder.appendChild(century_folder)
					century_folders[century] = century_folder

		else:
			topfolder = kmldoc.addFolder("Earthquake catalog", visible=True, open=False)
			ke_folder = topfolder
			if len(event_types) > 1 and "ke" in event_types:
				ke_folder = kmldoc.addFolder("Tectonic", visible=True, open=False)
				topfolder.appendChild(ke_folder)
				nt_folder = kmldoc.addFolder("Non-tectonic", visible=True, open=False)
				topfolder.appendChild(nt_folder)

		for eq in self:
			if eq.event_type == "ke":
				if eq.datetime.year < instrumental_start_year:
					Mtype = "MS"
				else:
					Mtype = "ML"
			else:
				Mtype = "ML"

			if nt_folder and eq.event_type != "ke":
				folder = nt_folder
				visible = True
				color = (0, 0, 0)
				#Mtype = "ML"

			elif time_folders:
				if eq.datetime.year < instrumental_start_year:
					century = (eq.datetime.year // 100) * 100
					folder = century_folders[century]
					visible = True
					color = (0, 255, 0)
					#Mtype = "MS"
				else:
					visible = True
					#Mtype = "ML"
					if start_time - eq.datetime <= datetime.timedelta(1, 0, 0):
						folder = folder_24h
						color = (255, 0, 0)
					elif start_time - eq.datetime <= datetime.timedelta(14, 0, 0):
						folder = folder_2w
						color = (255, 128, 0)
					elif start_time - eq.datetime <= datetime.timedelta(365, 0, 0):
						folder = folder_lastyear
						color = (255, 255, 0)
					else:
						decade = (eq.datetime.year // 10) * 10
						folder = decade_folders[decade]
						if eq.datetime.year >= 2000:
							color = (192, 0, 192)
						else:
							color = (0, 0, 255)
			else:
				folder = ke_folder
				color = (255, 128, 0)
				visible = True

			if color_by_depth:
				if eq.depth == 0.:
					color = (255, 255, 255)
				elif 0 < eq.depth <= 2.:
					color = (205, 0 , 255)
				elif 2. < eq.depth <= 5.:
					color = (0, 0, 255)
				elif 5. < eq.depth <= 10.:
					color = (0, 255, 0)
				elif 10. < eq.depth <= 15.:
					color = (255, 255, 0)
				elif 15. < eq.depth <= 20.:
					color = (255, 156, 0)
				elif eq.depth > 20.:
					color = (255, 0, 0)

			t = eq.datetime.time()
			#url = '<a href="http://seismologie.oma.be/active.php?LANG=EN&CNT=BE&LEVEL=211&id=%d">ROB web page</a>' % eq.ID
			try:
				hash = eq.get_rob_hash()
			except:
				url = None
			else:
				url = '<a href="http://seismologie.oma.be/en/seismology/earthquakes-in-belgium/%s">ROB web page</a>' % hash

			values = OrderedDict()
			values['ID'] = eq.ID
			try:
				values['Date'] = eq.date.isoformat()
			except:
				values['Date'] = eq.date.date
			values['Time'] = "%02d:%02d:%02d" % (t.hour, t.minute, int(round(t.second + t.microsecond/1e+6)))
			values['Name'] = mykml.xmlstr(eq.name)
			values['ML'] = eq.ML
			values['MS'] = eq.MS
			values['MW'] = eq.MW
			values['Lon'] = eq.lon
			values['Lat'] = eq.lat
			values['Depth'] = eq.depth
			if len(event_types) > 1:
				values['Event type'] = eq.event_type
			values[None] = url
			name = "%s %.1f %s %s %s" % (Mtype, values[Mtype], values['Date'], values['Time'], values['Name'])
			labelstyle = kmldoc.createLabelStyle(scale=0)
			#iconstyle = kmldoc.createStandardIconStyle(palette="pal2", icon_nr=26, scale=0.75+(values[Mtype]-3.0)*0.15, rgb=color)
			icon_href = "http://kh.google.com:80/flatfile?lf-0-icons/shield3_nh.png"
			iconstyle = kmldoc.createIconStyle(href=icon_href, scale=0.75+(values[Mtype]-3.0)*0.15, rgb=color)
			style = kmldoc.createStyle(styles=[labelstyle, iconstyle])
			ts = kmldoc.createTimeSpan(begin=eq.datetime)
			kmldoc.addPointPlacemark(name, eq.lon, eq.lat, folder=folder, description=values, style=style, visible=visible, timestamp=ts)

		if kml_filespec:
			kmldoc.write(kml_filespec)
		else:
			return kmldoc.root.toxml()

	def export_VTK(self, vtk_filespec, proj="lambert1972", Mtype="MW", Mrelation="default"):
		"""
		Export earthquake catalog to VTK format for 3D viewing

		:param vtk_filespec:
			String, full path to output VTK file
		:param proj:
			String, projection name: either "lambert1972" or "utm31"
			(default: "lambert1972")
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		"""
		cartesian_coords = self.get_cartesian_coordinates(proj)
		num_rows = len(self)

		f = open(vtk_filespec, 'w')
		f.write("# vtk DataFile Version 2.0\n")
		f.write("%s\n" % self.name)
		f.write("ASCII\n")
		f.write("DATASET UNSTRUCTURED_GRID\n")
		f.write("POINTS %d float\n" % num_rows)
		for i, eq in enumerate(self):
			x, y = cartesian_coords[i]
			f.write("%.2f %.2f %.2f\n" % (x, y, eq.depth*-1000.0))
		f.write("CELLS %d %d\n" % (num_rows, num_rows * 2))
		for i in range(num_rows):
			f.write("1 %d\n" % i)
		f.write("CELL_TYPES %d\n" % num_rows)
		for i in range(num_rows):
			f.write("1\n")
		f.write("\n")
		f.write("POINT_DATA %d\n" % num_rows)
		f.write("SCALARS Depth float 1\n")
		f.write("LOOKUP_TABLE default\n")
		for eq in self:
			f.write("%.2f\n" % eq.depth)
		f.write("SCALARS Magnitude float 1\n")
		f.write("LOOKUP_TABLE default\n")
		for eq in self:
			f.write("%.2f\n" % eq.get_or_convert_mag(Mtype, Mrelation))
		f.close()

	def export_HY4(self, filespec):
		"""
		Export to SeismicEruption HY4 earthquake catalog format

		:param filespec:
			String, full path to output file
		"""
		from HY4 import HYPDAT

		ofd = open(filespec, "wb")
		for eq in self:
			hyp = eq.to_HY4()
			ofd.write("%s" % hyp.pack())
		ofd.close()

	def pickle(self, filespec):
		"""
		Dump earthquake collection to a pickled file.
		Required parameter:
			filespec: full path specification of output file
		"""
		f = open(filespec, "w")
		cPickle.dump(self, f)
		f.close()

	def export_sqlite(self, sqlite_filespec, table_name=None):
		"""
		Export catalog to sqlite (if possible: SpatiaLite) file.

		:param sqlite_filespec:
			string, full path to SQLite database
		:param table_name:
			string, name of database table containing catalog
			(default: None, assumes table name corresponds to
			basename of :param:`sqlite_filespec`)
		"""
		import db.simpledb as simpledb

		if not table_name:
			table_name = os.path.splitext(os.path.split(sqlite_filespec)[1])[0]

		if len(self):
			db = simpledb.SQLiteDB(sqlite_file)
			if table_name in db.list_tables():
				db.drop_table(table_name)
			eq = self.eq_list[-1]
			if isinstance(eq.ID, (int, long)):
				# Note: declare type as INT instead of INTEGER,
				# otherwise rowid will be replaced with ID!
				# See https://sqlite.org/lang_createtable.html#rowid
				col_info = [{'name': 'ID', 'type': 'INT', 'pk': 1}]
			else:
				col_info = [{'name': 'ID', 'type': 'TEXT', 'pk': 1}]
			col_info.extend([{'name': 'datetime', 'type': 'TIMESTAMP'},
							{'name': 'lon', 'type': 'NUMERIC'},
							{'name': 'lat', 'type': 'NUMERIC'},
							{'name': 'depth', 'type': 'NUMERIC'}])
			for Mtype in self.get_Mtypes():
				col_info.append({'name': Mtype, 'type': 'NUMERIC'})
			col_info.extend([{'name': 'name', 'type': 'TEXT'},
							{'name': 'intensity_max', 'type': 'NUMERIC'},
							{'name': 'macro_radius', 'type': 'NUMERIC'},
							{'name': 'errh', 'type': 'NUMERIC'},
							{'name': 'errz', 'type': 'NUMERIC'},
							{'name': 'errt', 'type': 'NUMERIC'},
							{'name': 'errM', 'type': 'NUMERIC'},
							{'name': 'zone', 'type': 'TEXT'},
							{'name': 'event_type', 'type': 'TEXT'}])
			db.create_table(table_name, col_info)

			recs = []
			for eq in self:
				dic = eq.to_dict()
				del dic['mag']
				for Mtype in eq.get_Mtypes():
					dic[Mtype] = eq.mag[Mtype]
				recs.append(dic)
			db.add_records(table_name, recs)

			if db.has_spatialite:
				db.init_spatialite("wgs84")
				db.add_geometry_column(table_name)
				db.create_points_from_columns(table_name, 'lon', 'lat')

			db.close()

	@classmethod
	def from_sqlite(cls, sqlite_filespec, table_name=None):
		"""
		Import catalog from SQLite database.

		:param sqlite_filespec:
			string, full path to SQLite database
		:param table_name:
			string, name of database table containing catalog
			(default: None, assumes table name corresponds to
			basename of :param:`sqlite_filespec`)

		:return:
			instance of :class:`EQCatalog`
		"""
		import db.simpledb as simpledb

		if not table_name:
			table_name = os.path.splitext(os.path.split(sqlite_filespec)[1])[0]

		eq_list = []
		db = simpledb.SQLiteDB(sqlite_filespec)
		for rec in db.query(table_name):
			dic = rec.to_dict()
			if dic.has_key('geom'):
				del dic['geom']
			eq = LocalEarthquake.from_dict(dic)
			eq_list.append(eq)

		return cls(eq_list, name=table_name)

	def get_bbox(self):
		"""
		Compute bounding box of earthquake catalog

		:return:
			(lonmin, lonmax, latmin, latmax) tuple
		"""
		lons = self.get_longitudes()
		lats = self.get_latitudes()
		return (lons.min(), lons.max(), lats.min(), lats.max())

	def subselect_distance(self, point, distance, catalog_name=""):
		"""
		Subselect earthquakes in a given radius around a given point

		:param point:
			(lon, lat) tuple
		:param distance:
			float, distance in km
		:param catalog_name:
			Str, name of resulting catalog
			(default: "")

		:return:
			instance of :class:`EQCatalog`
		"""
		from itertools import compress

		distances = self.get_epicentral_distances(*point)
		eq_list = list(compress(self.eq_list, distances <= distance))
		if not catalog_name:
			catalog_name = self.name + " (%s km radius from %s)" % (distance, point)
		region = self.get_bbox()
		return EQCatalog(eq_list, self.start_date, self.end_date, region, catalog_name)

	def subselect_polygon(self, poly_obj, catalog_name=""):
		"""
		Subselect earthquakes from catalog situated inside a polygon

		:param poly_obj:
			polygon or closed linestring object (ogr geometry object
			or oqhazlib.geo.polygon.Polygon object)
		:param catalog_name:
			Str, name of resulting catalog
			(default: "")

		:return:
			instance of :class:`EQCatalog`
		"""
		import osr, ogr

		if isinstance(poly_obj, ogr.Geometry):
			## Construct WGS84 projection system corresponding to earthquake coordinates
			wgs84 = osr.SpatialReference()
			wgs84.SetWellKnownGeogCS("WGS84")

			## Point object that will be used to test if earthquake is inside zone
			point = ogr.Geometry(ogr.wkbPoint)
			point.AssignSpatialReference(wgs84)

			if poly_obj.GetGeometryName() in ("POLYGON", "LINESTRING"):
				## Objects other than polygons or closed polylines will be skipped
				if poly_obj.GetGeometryName() == "LINESTRING":
					line_obj = poly_obj
					if line_obj.IsRing() and line_obj.GetPointCount() > 3:
						# Note: Could not find a way to convert linestrings to polygons
						# The following only works for linearrings (what is the difference??)
						#poly_obj = ogr.Geometry(ogr.wkbPolygon)
						#poly_obj.AddGeometry(line_obj)
						wkt = line_obj.ExportToWkt().replace("LINESTRING (", "POLYGON ((") + ")"
						poly_obj = ogr.CreateGeometryFromWkt(wkt)
					else:
						return None
				eq_list = []
				for i, eq in enumerate(self.eq_list):
					point.SetPoint(0, eq.lon, eq.lat)
					if point.Within(poly_obj):
						eq_list.append(eq)

				## Determine bounding box (region)
				linear_ring = poly_obj.GetGeometryRef(0)
				## In some versions of ogr, GetPoints method does not exist
				#points = linear_ring.GetPoints()
				points = [linear_ring.GetPoint(i) for i in range(linear_ring.GetPointCount())]
				lons, lats = zip(*points)[:2]

		else:
			import openquake.hazardlib as oqhazlib
			if isinstance(poly_obj, oqhazlib.geo.Polygon):
				mesh = oqhazlib.geo.Mesh(self.get_longitudes(), self.get_latitudes(), depths=None)
				intersects = poly_obj.intersects(mesh)
				idxs = np.argwhere(intersects == True)
				idxs = [idx[0] for idx in idxs]
				zone_catalog = self.__getitem__(idxs)
				lons = zone_catalog.get_longitudes()
				lats = zone_catalog.get_latitudes()
				eq_list = zone_catalog.eq_list

		if len(eq_list):
			region = (min(lons), max(lons), min(lats), max(lats))
		else:
			region = None
		if not catalog_name:
			catalog_name = self.name + " (inside polygon)"
		return EQCatalog(eq_list, self.start_date, self.end_date, region, catalog_name)

	def split_into_zones(self, source_model_name, ID_colname="", fix_mi_lambert=True, verbose=True):
		"""
		Split catalog into subcatalogs according to a
		source-zone model stored in a GIS (MapInfo) table.

		:param source_model_name:
			String, name of source-zone model containing area sources
			or else full path to GIS file containing area sources
			or rshalib SourceModel object
		:param ID_colname:
			String, name of GIS column containing record ID
			(default: "")
		:param fix_mi_lambert:
			bool, whether or not to apply spatial reference system fix for
			old MapInfo files in Lambert 1972 system
			(default: True)
		:param verbose:
			Boolean, whether or not to print information while reading
			GIS table (default: True)

		:return:
			ordered dict {String sourceID: EQCatalog}
		"""
		zone_catalogs = OrderedDict()

		if isinstance(source_model_name, (str, unicode)):
			## Read zone model from GIS file
			model_data = read_source_model(source_model_name, ID_colname=ID_colname, fix_mi_lambert=fix_mi_lambert, verbose=verbose)

			for zoneID, zone_data in model_data.items():
				zone_poly = zone_data['obj']
				if zone_poly.GetGeometryName() == "POLYGON" or zone_poly.IsRing():
					## Fault sources will be skipped
					zone_catalogs[zoneID] = self.subselect_polygon(zone_poly, catalog_name=zoneID)
		else:
			import hazard.rshalib as rshalib
			if isinstance(source_model_name, rshalib.source.SourceModel):
				source_model = source_model_name
				for src in source_model.sources:
					if isinstance(src, rshalib.source.AreaSource):
						zone_poly = src.polygon
						zoneID = src.source_id
						zone_catalogs[zoneID] = self.subselect_polygon(zone_poly, catalog_name=zoneID)

		return zone_catalogs

	def split_into_time_intervals(self, time_interval):
		"""
		:param time_interval:
			int (years) or timedelta object (precision of days)

		:return:
			list with instances of :class:`EQCatalog`
		"""
		def add_time_delta(date_time, time_delta):
			if isinstance(time_delta, int):
				time_tuple = list(date_time.timetuple())
				time_tuple[0] += time_delta
				out_date_time = mxDateTime.DateTimeFrom(*time_tuple[:6])
			else:
				out_date_time = mxDateTime.DateTimeFrom(date_time)
				out_date_time += time_delta
			return out_date_time

		subcatalogs = []
		start_date = self.start_date
		end_date = add_time_delta(self.start_date, time_interval)
		max_end_date = self.end_date + mxDateTime.DateTimeDelta(1)
		while start_date <= self.end_date:
			catalog = self.subselect(start_date=start_date, end_date=min(end_date, max_end_date), include_right_edges=False)
			subcatalogs.append(catalog)
			start_date = end_date
			end_date = add_time_delta(start_date, time_interval)

		return subcatalogs

	def generate_synthetic_catalogs(self, num_samples, num_sigma=2, random_seed=None):
		"""
		Generate synthetic catalogs by random sampling of the magnitude
		and epicenter of each earthquake.

		:param num_samples:
			Int, number of random synthetic catalogs to generate
		:param num_sigma:
			Float, number of standard deviations to consider
		:param random_seed:
			None or int, seed to initialize internal state of random number
			generator (default: None, will seed from current time)

		:return:
			list of instances of :class:`EQCatalog`
		"""
		import copy
		import scipy.stats

		np.random.seed(seed=random_seed)

		num_eq = len(self)
		ML = np.zeros((num_eq, num_samples))
		MS = np.zeros((num_eq, num_samples))
		MW = np.zeros((num_eq, num_samples))
		lons = np.zeros((num_eq, num_samples))
		lats = np.zeros((num_eq, num_samples))
		# depths relevant in declustering?
		depths = np.zeros((num_eq, num_samples))

		for i, eq in enumerate(self):
			# TODO: write function to generate errM, errh based on date (use completeness dates?)
			if not eq.errM:
				if eq.datetime.year < 1910:
					errM = 0.5
				elif 1910 <= eq.datetime.year < 1985:
					if eq.MS > 0:
						errM = 0.2
					else:
						errM = 0.3
				elif eq.datetime.year >= 1985:
					eq.errM = 0.2
			elif eq.errM >= 1.:
				# A lot of earthquakes have errM = 9.9 ???
				errM = 0.3
			else:
				errM = eq.errM

			if not eq.errh:
				if eq.datetime.year < 1650:
					errh = 25
				elif 1650 <= eq.datetime.year < 1910:
					errh = 15
				elif 1910 <= eq.datetime.year < 1930:
					errh = 10
				elif 1930 <= eq.datetime.year < 1960:
					errh = 5.
				elif 1960 <= eq.datetime.year < 1985:
					errh = 2.5
				elif eq.datetime.year >= 1985:
					errh = 1.5
			else:
				errh = eq.errh

			## Convert uncertainty in km to uncertainty in lon, lat
			errlon = errh / ((40075./360.) * np.cos(np.radians(eq.lat)))
			errlat = errh / (40075./360.)

			if not eq.errz:
				errz = 5.
			else:
				errz = eq.errz

			if eq.ML:
				ML[i,:] = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, eq.ML, errM, size=num_samples)
			if eq.MS:
				MS[i,:] = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, eq.MS, errM, size=num_samples)
			if eq.MW:
				MW[i,:] = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, eq.MW, errM, size=num_samples)
			lons[i,:] = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, eq.lon, errlon, size=num_samples)
			lats[i,:] = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, eq.lat, errlat, size=num_samples)
			depths[i,:] = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, eq.depth, errz, size=num_samples)
		depths.clip(min=0.)

		synthetic_catalogs = []
		for n in range(num_samples):
			eq_list = []
			for i, eq in enumerate(self):
				new_eq = copy.deepcopy(eq)
				new_eq.ML = ML[i,n]
				new_eq.MS = MS[i,n]
				new_eq.MW = MW[i,n]
				new_eq.lon = lons[i,n]
				new_eq.lat = lats[i,n]
				new_eq.depth = depths[i,n]
				eq_list.append(new_eq)
			synthetic_catalogs.append(EQCatalog(eq_list, self.start_date, self.end_date, region=self.region))

		return synthetic_catalogs

	def analyse_completeness_Stepp(self, dM=0.1, Mtype="MW", Mrelation="default", dt=5.0, increment_lock=True):
		"""
		Analyze catalog completeness with the Stepp method (1971). The GEM algorithm
		from the OQ hazard modeller's toolkit is used.

		:param dM:
			Float, magnitude bin width (default: 0.1)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param dt:
			Float, time interval (in years) (default: 5)
		:param increment_lock:
			Boolean, ensure completeness magnitudes always decrease with more
			recent bins (default: True)

		:return:
			instance of :class:`Completeness`
		"""
		from hmtk.seismicity.completeness.comp_stepp_1971 import Stepp1971
		ec = self.get_hmtk_catalogue(Mtype=Mtype, Mrelation=Mrelation)
		stepp_1971_algorithm = Stepp1971()
		result = stepp_1971_algorithm.completeness(ec, {'magnitude_bin': dM, 'time_bin': dt, 'increment_lock': increment_lock})
		Min_Years, Min_Mags = result[:, 0].astype('i'), result[:,1]
		return Completeness(Min_Years, Min_Mags, Mtype=Mtype)

	def analyse_completeness_CUVI(self, magnitudes, start_year, dYear, year1=None, year2=None, reg_line=None, Mtype="MW", Mrelation="default", title=None, fig_filespec="", fig_width=0, dpi=300):
		"""
		Analyze catalog completeness with the CUVI method (Mulargia, 1987).

		:param magnitudes:
			List of floats, magnitudes to analyze completeness for.
		:param start_year:
			Int, start year of analysis.
		:param dYear:
			Int, bin interval in years
		:param year1:
			Int, year to plot as completeness year (default=None)
		:param year2:
			Int, year to plot as next completeness year (default=None)
		:param reg_line:
			Float, magnitude to plot regression line for (default=None)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param title:
			str, title of plot (default: None, automatic title is used)
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)
		"""
		colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']
		max_mag = self.get_Mmax(Mtype=Mtype, Mrelation=Mrelation)
		start_year_index = None
		for i, magnitude in enumerate(magnitudes):
			bins_N, bins_Years = self.bin_year(self.start_date.year,
				self.end_date.year+1, dYear, magnitude, max_mag, Mtype=Mtype, Mrelation=Mrelation)
			bins_N_cumul = np.add.accumulate(bins_N)
			if not start_year_index:
				start_year_index = np.abs(bins_Years - start_year).argmin()
			bins_Years = bins_Years[start_year_index:]
			bins_N_cumul = bins_N_cumul[start_year_index:]
			plt.plot(bins_Years, bins_N_cumul, colors[i%len(colors)], label= '%.1f' % magnitude)
			plt.plot(bins_Years, bins_N_cumul, '%so' % colors[i%len(colors)], label='_nolegend_')
			if reg_line and np.allclose(magnitude, reg_line) and year1 != None:
				index = np.abs(bins_Years - year1).argmin()
				bins_Years = bins_Years[index:]
				bins_N_cumul = bins_N_cumul[index:]
				x = np.array([bins_Years, np.ones_like(bins_Years)])
				m, c = np.linalg.lstsq(x.T, bins_N_cumul)[0]
				plt.plot(bins_Years, m*bins_Years+c, color='k', linestyle='--', linewidth=5)
		minorLocator = MultipleLocator(dYear)
		plt.gca().xaxis.set_minor_locator(minorLocator)
		xmin, xmax, ymin, ymax = plt.axis()
		if year1:
			plt.vlines(year1, ymin, ymax, colors='r', linestyles='-', linewidth=5)
		if year2:
			plt.vlines(year2, ymin, ymax, colors='r', linestyles='--', linewidth=5)
		plt.axis((start_year, self.end_date.year, ymin, ymax))
		plt.xlabel('Time (years)', fontsize='large')
		plt.ylabel('Cumulative number of events since' + ' %d' % self.start_date.year,
			fontsize='large')
		title = title or 'CUVI completeness analysis for magnitudes %.1f - %.1f' % (magnitudes[0], magnitudes[-1])
		plt.title(title, fontsize='x-large')
		plt.legend(loc=0)
		plt.grid()
		if fig_filespec:
			default_figsize = pylab.rcParams['figure.figsize']
			default_dpi = pylab.rcParams['figure.dpi']
			if fig_width:
				fig_width /= 2.54
				dpi = dpi * (fig_width / default_figsize[0])
			pylab.savefig(fig_filespec, dpi=dpi)
			pylab.clf()
		else:
			pylab.show()

	def decluster_new(self, method="gardner-knopoff", window_opt="GardnerKnopoff", fs_time_prop=0., time_window=60., Mtype="MW", Mrelation="default"):
		"""
		Decluster catalog.
		This method is a wrapper for the declustering methods in the OQ
		hazard modeller's toolkit (new implementation).

		:param method:
			String, method name: "afteran" or "gardner-knopoff" (default: "afteran")
		:param window_opt:
			String, declustering window type: "GardnerKnopoff", "Gruenthal" or "Uhrhammer"
			(default: "GardnerKnopoff")
		:param fs_time_prop:
			Positive float, foreshock time window as a proportion of
			aftershock time window. Only applies to gardner_knopoff method
			(default: 0.)
		:param time_window:
			Positive float, length (in days) of moving time window.
			Only applies to afteran method (default: 60.)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:return:
			Tuple mainshock_catalog, foreshock_catalog, aftershock_catalog, cluster_catalogs
			mainshock_catalog: instance of class:`EQCatalog` containing main shocks
			foreshock_catalog: instance of class:`EQCatalog` containing foreshocks
			aftershock_catalog: instance of class:`EQCatalog` containing aftershocks
			cluster_catalog: list with instances of class:`EQCatalog` containing
				earthquakes belonging to the different clusters. The first element
				in this list represents earthquakes that do not belong to any cluster
		"""
		## TODO: revisit
		from hmtk.seismicity.declusterer.dec_gardner_knopoff import GardnerKnopoffType1
		from hmtk.seismicity.declusterer.dec_afteran import Afteran
		from hmtk.seismicity.declusterer.distance_time_windows import GardnerKnopoffWindow, GruenthalWindow, UhrhammerWindow
		from hmtk.seismicity.catalogue import Catalogue

		windows = {"GardnerKnopoff": GardnerKnopoffWindow, "Gruenthal": GruenthalWindow, "Uhrhammer": UhrhammerWindow}

		if method == "gardner-knopoff":
			decluster_func = GardnerKnopoffType1()
			decluster_param_name = "fs_time_prop"
			decluster_param = fs_time_prop
		elif method == "afteran":
			decluster_func = Afteran()
			decluster_param_name = "time_window"
			decluster_param = time_window

		ec = Catalogue()
		keys_int = ["year", "month", "day", "hour", "minute"]
		keys_flt = ["second", "magnitude", "longitude", "latitude"]
		data_int, data_flt = [], []
		for eq in self:
			mag = eq.get_or_convert_mag(Mtype, Mrelation)
			if not np.isnan(mag):
				data_int.append([
					int(eq.datetime.year),
					int(eq.datetime.month),
					int(eq.datetime.day),
					int(eq.datetime.hour),
					int(eq.datetime.minute),
				])
				data_flt.append([
					float(eq.datetime.second),
					float(mag),
					float(eq.lon),
					float(eq.lat),
				])
		ec.load_from_array(keys_int, np.array(data_int, dtype=np.int16))
		ec.load_from_array(keys_flt, np.array(data_flt, dtype=np.float64))

		vcl, flag_vector = decluster_func.decluster(ec, {"time_distance_window": windows[window_opt](), decluster_param_name: decluster_param})

		mainshock_catalog = self.__getitem__(np.where(flag_vector == 0)[0])
		foreshock_catalog = self.__getitem__(np.where(flag_vector == -1)[0])
		aftershock_catalog = self.__getitem__(np.where(flag_vector == 1)[0])

		return mainshock_catalog, foreshock_catalog, aftershock_catalog

	def analyse_Mmax(self, method='Cumulative_Moment', num_bootstraps=100, iteration_tolerance=None, maximum_iterations=100, num_samples=20, Mtype="MW", Mrelation="default"):
		"""
		Statistical analysis of maximum magnitude.
		This method is a wrapper for meth:`maximum_magnitude_analysis`
		in the OQhazard modeller's toolkit.

		:param method:
			String, either 'Kijko_Npg' or 'Cumulative_Moment'
			(default: 'Cumulative_Moment')
		:param num_bootstraps:
			Int, number of samples for bootstrapping (only applies to
			'Cumulative_Moment' method) (default: 100)
		:param iteration_tolerance:
			Float, integral tolerance (only applies to 'Kijko_Npg' method)
			(default: None)
		:param maximum_iterations:
			Int, maximum number of iterations (only applies to 'Kijko_Npg' method)
			(default: 100)
		:param num_samples:
			Int, number of sampling points of integral function (only applies to
			'Kijko_Npg' method) (default: 20)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			Tuple (Mmax, Mmax_sigma)
		"""
		# TODO: determine sensible default values
		from mtoolkit.scientific.maximum_magnitude import maximum_magnitude_analysis

		years = self.get_years()
		Mags = self.get_magnitudes(Mtype, Mrelation)
		Mag_uncertainties = self.get_magnitude_uncertainties(min_uncertainty=0.3)
		Mmax, Mmax_sigma = maximum_magnitude_analysis(years, Mags, Mag_uncertainties, method, iteration_tolerance, maximum_iterations, len(self), num_samples, num_bootstraps)
		return Mmax, Mmax_sigma

	def analyse_recurrence(self, dM=0.1, method="MLE", aM=0., dt=1., Mtype="MW", Mrelation="default", completeness=default_completeness):
		"""
		Analyse magnitude-frequency.
		This method is a wrapper for meth:`recurrence_analysis` in the
		OQhazard modeller's toolkit.

		:param dM:
			Float, magnitude bin width (default: 0.1)
		:param method:
			String, either "MLE" or "Weichert" (default: "MLE")
		:param aM:
			Float, reference magnitude for which a value should be computed
			(default: 0.)
		:param dt:
			Float, time bin width in number of years. Only applies to "Weichert"
			method (default: 1.)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` (default: Completeness_MW_201303a)

		:return:
			Tuple (a, b, stdb)
			a: a value
			b: b value
			stdb: standard deviation on b value

		Note:
		There seem to be problems with the "Weichert" method:
		- result does not depend on maximum magnitude (i.e., empty magnitude bins
			do not influence the result)
		- the a value computed for aM=0 appears to be wrong (partly fixed by
			replacing with aM = 0.5)
		"""
		from mtoolkit.scientific.recurrence import recurrence_analysis

		subcatalog = self.subselect_completeness(completeness, Mtype=Mtype, Mrelation=Mrelation)
		#years = subcatalog.get_years()
		years = subcatalog.get_fractional_years()
		Mags = subcatalog.get_magnitudes(Mtype, Mrelation)
		completeness_table = completeness.to_hmtk_table(Mmax=None)
		if method == "Weichert" and aM == 0.:
			aM = dM / 2.
		b, stdb, a, stda = recurrence_analysis(years, Mags, completeness_table, dM, method, aM, dt)
		return np.log10(a), b, stda, stdb

	def plot_3d(self, limits=None, Mtype=None, Mrelation="default"):
		"""
		Plot catalog in 3D. Points are colored by magnitude.

		:param limits:
			Tuple of six floats, defining respectively minumum and maximum for
			longitude scale, minumum and maximum for latitude scale and minumum,
			and minimum and maximum for depth scale (default: None). This param
			should be used to create plots with identical scales.
		:param Mtype:
			See :method: get_magnitudes.
		:param Mrelation:
			See :method: get_magnitudes.
		"""
		from mpl_toolkits.mplot3d.axes3d import Axes3D
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		kwargs = {}
		if Mtype:
			kwargs['Mtype'] = Mtype
		if Mrelation:
			kwargs['Mrelation'] = Mrelation
		p = ax.scatter(self.get_longitudes(), self.get_latitudes(), self.get_depths()*-1, c=self.get_magnitudes(**kwargs), cmap=plt.cm.jet)
		## set labels
		ax.set_xlabel('longitude')
		ax.set_ylabel('latitude')
		ax.set_zlabel('depth')
		## set limits
		if limits:
			ax.set_xlim(*limits[0:2])
			ax.set_ylim(*limits[2:4])
			ax.set_zlim(limits[5]*-1, limits[4])
		## create colorbar
		fig.colorbar(p)
		## plot
		plt.show()

	def get_hmtk_catalogue(self, Mtype='MW', Mrelation="default"):
		"""
		Convert ROB catalog to hmtk catalogue

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			instance of :class:`hmtk.seismicity.catalogue.Catalogue`
		"""
		from hmtk.seismicity.catalogue import Catalogue
		catalogue = Catalogue()
		keys_flt = ['second', 'longitude', 'latitude', 'depth', 'magnitude']
		keys_int = ['year', 'month', 'day', 'hour', 'minute']
		data_int, data_flt = [], []
		for eq in self:
			data_flt.append([
				float(eq.datetime.second),
				float(eq.lon),
				float(eq.lat),
				float(eq.depth),
				float(eq.get_or_convert_mag(Mtype=Mtype, Mrelation=Mrelation)),
			])
			data_int.append([
				int(eq.datetime.year),
				int(eq.datetime.month),
				int(eq.datetime.day),
				int(eq.datetime.hour),
				int(eq.datetime.minute),
			])
		catalogue.load_from_array(keys_flt, np.array(data_flt, dtype=np.float))
		catalogue.load_from_array(keys_int, np.array(data_int, dtype=np.int))
		return catalogue


	def get_hmtk_smoothed_source_model(self, spcx=0.1, spcy=0.1, Mtype='MW', Mrelation="default", completeness=default_completeness):
		"""
		"""
		from hmtk.seismicity.smoothing.smoothed_seismicity import SmoothedSeismicity
		xmin, xmax, ymin, ymax = self.get_region()
		zmin, zmax, spcz = 0., 0., 0.
		smoothed_seismicity = SmoothedSeismicity(grid_limits=[xmin, xmax, spcx, ymin, ymax, spcy, zmin, zmax, spcz])
		catalogue = self.get_hmtk_catalogue(Mtype=Mtype, Mrelation=Mrelation)
		config = {'Length_limit': 50., 'BandWidth': 25., 'increment': True}
		completeness_table = completeness.to_hmtk_table()
		data = smoothed_seismicity.run_analysis(catalogue=catalogue, config=config, completeness_table=completeness_table, smoothing_kernel=None, end_year=None)

	def plot_Poisson_test(self, Mmin, interval=100, nmax=0, Mtype='MW', Mrelation="default", completeness=default_completeness, title=None, fig_filespec=None, verbose=True):
		"""
		Plot catalog distribution versus Poisson distribution
		p(n, t, tau) = (t / tau)**n * exp(-t/tau) / n!

		First, the specified completeness constraint is applied to the catalog.
		The completeness-constrained catalog is then truncated to the
		specified minimum magnitude and corresponding year of completeness.
		The resulting catalog is divided into intervals of the specified
		length, the number of events in each interval is counted, and a
		histogram is computed of the number of intervals having the same
		number of events up to nmax.
		This histogram is compared to the theoretical Poisson distribution.
		It seems to work best if :param:`interval` is larger (2 to 4 times)
		than tau, the average return period.

		:param Mmin:
			Float, minimum magnitude to consider in analysis (ideally
			corresponding to one of the completeness magnitudes)
		:param interval:
			Int, length of interval (number of days) (default: 100)
		:param nmax:
			Int, maximum number of earthquakes in an interval to test
			(default: 0, will determine automatically)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes. If None, use start year of
			catalog (default: Completeness_MW_201303a)
		:param title:
			String, plot title. (None = default title, "" = no title)
			(default: None)
		:param fig_filespec:
			String, full path of image to be saved.
			If None (default), histogram is displayed on screen.
		:param verbose:
			Bool, whether or not to print additional information
		"""
		from scipy.misc import factorial
		from time_functions import time_delta_to_days

		def poisson(n, t, tau):
			## Probability of n events in period t
			## given average recurrence interval tau
			return (t / tau)**n * np.exp(-t/tau) / factorial(n)

		## Apply completeness constraint, and truncate result to completeness
		## year for specified minimum magnitude
		min_date = completeness.get_initial_completeness_date(Mmin)
		cc_catalog = self.subselect_completeness(Mtype=Mtype, Mrelation=Mrelation, completeness=completeness)
		catalog = cc_catalog.subselect(start_date=min_date, Mmin=Mmin)

		num_events = len(catalog)
		td = catalog.get_time_delta()
		catalog_num_days = time_delta_to_days(td)
		num_intervals = np.ceil(catalog_num_days / interval)

		## Real catalog distribution
		## Compute interval index for each event
		time_deltas = catalog.get_time_deltas()
		time_delta_days = np.array([time_delta_to_days(td) for td in time_deltas])
		interval_indexes = np.floor(time_delta_days / interval)
		## Compute number of events in each interval
		num_events_per_interval, _ = np.histogram(interval_indexes, np.arange(num_intervals))
		if not nmax:
			nmax = num_events_per_interval.max()
		## Compute number of intervals having n events
		bins_num_events, _ = np.histogram(num_events_per_interval, bins=np.arange(nmax+1))

		## Theoretical Poisson distribution
		n = np.arange(nmax)
		tau = catalog_num_days / num_events
		if verbose:
			print("Number of events in catalog: %d" % num_events)
			print("Number of days in catalog: %s" % catalog_num_days)
			print("Number of %d-day intervals: %d" % (interval, num_intervals))
			print("Average return period for M>=%s: %d days" % (Mmin, tau))
		poisson_probs = poisson(n, interval, tau)
		poisson_n = poisson_probs * num_intervals

		## Plot
		pylab.bar(n-0.5, bins_num_events, label="Catalog distribution")
		pylab.plot(n, poisson_n, 'r', lw=2, label="Poisson distribution")
		pylab.xlabel("Number of events per interval", fontsize="x-large")
		pylab.ylabel("Number of intervals", fontsize="x-large")
		pylab.legend()
		xmin, xmax, ymin, ymax = pylab.axis()
		pylab.axis((-0.5, nmax, ymin, ymax))

		ax = pylab.gca()
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('large')
		if title is None:
			title = r"Poisson test for $M\geq%.1f$ (t=%d, $\tau$=%.1f days, nt=%d)" % (Mmin, interval, tau, num_intervals)
		pylab.title(title, fontsize="x-large")

		if fig_filespec:
			pylab.savefig(fig_filespec)
			pylab.clf()
		else:
			pylab.show()

	def get_epicentral_distances(self, lon, lat):
		"""
		Compute epicentral distances between catalog earthquakes and a
		given point.

		:param lon:
			float, longitude of reference point
		:param lat:
			float, latitude of reference point

		:return:
			float, epicentral distance in km
		"""
		distances = geodetic.spherical_distance(lon, lat, self.get_longitudes(), self.get_latitudes())
		return distances / 1000.

	def get_hypocentral_distances(self, lon, lat, z=0):
		"""
		Compute hypocentral distances between catalog earthquakes and a
		given point.

		:param lon:
			float, longitude of reference point
		:param lat:
			float, latitude of reference point
		:param z:
			float, depth of reference point in km
			(default: 0)

		:return:
			float, hypocentral distance in km
		"""
		d_epi = self.get_epicentral_distances(lon, lat)
		d_hypo = np.sqrt(d_epi**2 + (self.get_depths() - z)**2)
		return d_hypo

	def sort(self, key="datetime", order="asc"):
		"""
		Sort catalog

		:param key:
			str, property of :class:`EQRecord` to use as sort key
		:param order:
			str, sorting order: "asc" or "desc"
			(default: "asc")

		:return:
			instance of :class:`EQCatalog`
		"""
		reverse = {"asc": False, "desc": True}[order]
		eq_list = sorted(self.eq_list, key=lambda eq: getattr(eq, key), reverse=reverse)
		return EQCatalog(eq_list, start_date=self.start_date, end_date=self.end_date, region=self.region, name=self.name)


EQCollection = EQCatalog


def concatenate_catalogs(catalog_list, name=""):
	"""
	Concatenate different catalogs into one new catalog

	:param catalog_list:
		list containing instances of :class:`EQCatalog`
	:param name:
		str, name of concatenated catalog

	:return:
		instance of :class:`EQCatalog`
	"""
	if not name:
		name = "Concatenated catalog"

	## Remove empty catalogs
	catalog_list = [catalog for catalog in catalog_list if catalog and len(catalog)]

	catalog0 = catalog_list[0]
	eq_list = catalog0.eq_list[:]
	start_date = catalog0.start_date
	end_date = catalog0.end_date
	try:
		region = list(catalog0.region)
	except TypeError:
		region = list(catalog0.get_region())
	for catalog in catalog_list[1:]:
		eq_list.extend(catalog.eq_list)
		if catalog.start_date < start_date:
			start_date = catalog.start_date
		if catalog.end_date > end_date:
			end_date = catalog.end_date
		try:
			w, e, s, n = catalog.region
		except:
			w, e, s, n = catalog.get_region()
		if w < region[0]:
			region[0] = w
		if e > region[1]:
			region[1] = e
		if s < region[2]:
			region[2] = s
		if n > region[3]:
			region[3] = n
	return EQCatalog(eq_list, start_date=start_date, end_date=end_date, region=region, name=name)


def read_catalogSQL(region=None, start_date=None, end_date=None, Mmin=None, Mmax=None, min_depth=None, max_depth=None, id_earth=None, sort_key="date", sort_order="asc", event_type="ke", convert_NULL=True, verbose=False, errf=None):
	"""
	Query ROB local earthquake catalog through the online database.

	Notes:
	Magnitude used for selection is based on MW first, then MS, then ML.
	NULL values in the database are converted to 0.0 (this may change in the future)
	Only real earthquakes are extracted (type = "ke" and is_true = 1).

	:param region:
		(w, e, s, n) tuple specifying rectangular region of interest in
		geographic coordinates (default: None)
	:param start_date:
		Int or date or datetime object specifying start of time window of interest
		If integer, start_date is interpreted as start year
		(default: None)
	:param end_date:
		Int or date or datetime object specifying end of time window of interest
		If integer, end_date is interpreted as end year
		(default: None)
	:param Mmin:
		Float, minimum magnitude to extract (default: None)
	:param Mmax:
		Float, maximum magnitude to extract (default: None)
	:param min_depth:
		Float, minimum depth in km to extract (default: None)
	:param max_depth:
		Float, maximum depth in km to extract (default: None)
	:param id_earth:
		Int or List, ID(s) of event to extract (default: None)
	:param sort_key":
		String, property name to sort results with: "date" (= "time")
		or "mag" (= "size") (default: "date")
	:param sort_order:
		String, sort order, either "asc" or "desc" (default: "asc")
	:param event_type:
		str, event type
		(default: "ke" = known earthquakes)
	:param convert_NULL:
		Bool, whether or not to convert NULL values to zero values
		(default: True)
	:param verbose:
		Bool, if True the query string will be echoed to standard output
	:param errf:
		File object, where to print errors

	:return:
		instance of :class:`EQCatalog`
	"""
	import seismodb
	return seismodb.query_ROB_LocalEQCatalog(region=region, start_date=start_date, end_date=end_date, Mmin=Mmin, Mmax=Mmax, min_depth=min_depth, max_depth=max_depth, id_earth=id_earth, sort_key=sort_key, sort_order=sort_order, event_type=event_type, convert_NULL=convert_NULL, verbose=verbose, errf=errf)


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
	from mapping.geotools.readGIS import read_GIS_file

	data = read_GIS_file(gis_filespec, verbose=verbose)
	eq_list = []
	skipped = 0
	for i, rec in enumerate(data):
		if column_map.has_key('ID'):
			ID = rec[column_map['ID']]
		else:
			ID = i
		if ID_prefix:
			ID = ID_prefix + str(ID)

		if column_map.has_key('date'):
			date = rec[column_map['date']]
			if date:
				year, month, day = [int(s) for s in date.split('/')]
			else:
				year, month, day = 0, 0, 0
		else:
			if column_map.has_key('year'):
				year = rec[column_map['year']]
			if column_map.has_key('month'):
				month = rec[column_map['month']]
				if month == 0 and fix_zero_days_and_months:
					month = 1
			else:
				month = 1
			if column_map.has_key('day'):
				day = rec[column_map['day']]
				if day == 0 and fix_zero_days_and_months:
					day = 1
			else:
				day = 1
		try:
			date = datetime.date(year, month, day)
		except:
			print year, month, day
			date = None

		if column_map.has_key('time'):
			time = rec[column_map['time']]
			hour, minute, second = [int(s) for s in time.split(':')]
		else:
			if column_map.has_key('hour'):
				hour = rec[column_map['hour']]
			else:
				hour = 0
			if column_map.has_key('minute'):
				minute = rec[column_map['minute']]
			else:
				minute = 0
			if column_map.has_key('second'):
				second = rec[column_map['second']]
			else:
				second = 0
			second = int(round(second))
			second = min(second, 59)
		try:
			time = datetime.time(hour, minute, second)
		except:
			print hour, minute, second
			time = None

		if column_map.has_key('lon'):
			lon = rec[column_map['lon']]
		else:
			lon = rec["obj"].GetX()

		if column_map.has_key('lat'):
			lat = rec[column_map['lat']]
		else:
			lat = rec["obj"].GetY()

		if column_map.has_key('depth'):
			depth = rec[column_map['depth']]
		else:
			depth = 0

		mag = {}
		if column_map.has_key('ML'):
			ML = rec[column_map['ML']]
			if convert_zero_magnitudes:
				ML = ML or np.nan
			mag['ML'] = ML

		if column_map.has_key('MS'):
			MS = rec[column_map['MS']]
			if convert_zero_magnitudes:
				MS = MS or np.nan
			mag['MS'] = MS

		if column_map.has_key('MW'):
			MW = rec[column_map['MW']]
			if convert_zero_magnitudes:
				MW = MW or np.nan
			mag['MW'] = MW

		if column_map.has_key('name'):
			name = rec[column_map['name']]
		else:
			name = ""

		if column_map.has_key('intensity_max'):
			intensity_max = rec[column_map['intensity_max']]
		else:
			intensity_max = None

		if column_map.has_key('macro_radius'):
			macro_radius = rec[column_map['macro_radius']]
		else:
			macro_radius = None

		if column_map.has_key('errh'):
			errh = rec[column_map['errh']]
		else:
			errh = 0.

		if column_map.has_key('errz'):
			errz = rec[column_map['errz']]
		else:
			errz = 0.

		if column_map.has_key('errt'):
			errt = rec[column_map['errt']]
		else:
			errt = 0.

		if column_map.has_key('errM'):
			errM = rec[column_map['errM']]
		else:
			errM = 0.

		if column_map.has_key('zone'):
			zone = rec[column_map['zone']]
		else:
			zone = ""

		#print ID, date, time, lon, lat, depth, ML, MS, MW
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


def read_named_catalog(catalog_name, fix_zero_days_and_months=False, verbose=True):
	"""
	Read a known catalog (corresponding files should be in standard location)

	:param catalog_name:
		Str, name of catalog ("SHEEC", "CENEC", "ISC-GEM", "CEUS-SCR", "BGS"):
	:param fix_zero_days_and_months:
		bool, if True, zero days and months are replaced with ones
		(default: False)
	:param verbose:
		Boolean, whether or not to print information while reading
		GIS table (default: True)

	:return:
		instance of :class:`EQCatalog`
	"""
	if catalog_name.upper() == "SHEEC":
		gis_filespec = os.path.join(GIS_root, "SHARE", "SHEEC", "Ver3.3", "SHAREver3.3.shp")
		column_map = {'lon': 'Lon', 'lat': 'Lat', 'year': 'Year', 'month': 'Mo', 'day': 'Da', 'hour': 'Ho', 'minute': 'Mi', 'second': 'Se', 'MW': 'Mw', 'depth': 'H', 'ID': 'event_id'}
		convert_zero_magnitudes = True
	elif catalog_name.upper() == "CENEC":
		gis_filespec = os.path.join(GIS_root, "Seismology", "Earthquake Catalogs", "CENEC", "CENEC 2008.TAB")
		column_map = {'lon': 'lon', 'lat': 'lat', 'date': 'Date', 'hour': 'hour', 'minute': 'minute', 'MW': 'Mw', 'depth': 'depth'}
		convert_zero_magnitudes = True
	elif catalog_name.upper() == "ISC-GEM":
		gis_filespec = os.path.join(GIS_root, "Seismology", "Earthquake Catalogs", "ISC-GEM", "isc-gem-cat.TAB")
		column_map = {'lon': 'lon', 'lat': 'lat', 'date': 'date', 'time': 'time', 'MW': 'mw', 'depth': 'depth', 'ID': 'eventid', 'errz': 'unc', 'errM': 'unc_2'}
		convert_zero_magnitudes = True
	elif catalog_name.upper() == "CEUS-SCR":
		gis_filespec = os.path.join(GIS_root, "Seismology", "Earthquake Catalogs", "CEUS-SCR", "CEUS_SCR_Catalog_2012.TAB")
		column_map = {'lon': 'Longitude', 'lat': 'Latitude', 'year': 'Year', 'month': 'Month', 'day': 'Day', 'hour': 'Hour', 'minute': 'Minute', 'second': 'Second', 'MW': 'E_M_', 'errM': 'sigma_M', 'zone': 'DN'}
		convert_zero_magnitudes = True
	elif catalog_name.upper() == "BGS":
		gis_filespec = os.path.join(GIS_root, "Seismology", "Earthquake Catalogs", "BGS", "Selection of SE-UK-BGS-earthquakes.TAB")
		column_map = {'lon': 'LON', 'lat': 'LAT', 'date': 'DY_MO_YEAR', 'hour': 'HR', 'minute': 'MN', 'second': 'SECS', 'depth': 'DEP', 'ML': 'ML', 'MS': 'MGMC', 'ID': 'ID', 'name': 'LOCALITY', 'intensity_max': 'INT'}
		convert_zero_magnitudes = True
	else:
		raise Exception("Catalog not recognized: %s" % catalog_name)

	if not os.path.exists(gis_filespec):
		raise Exception("Catalog file not found: %s" % gis_filespec)
	ID_prefix = catalog_name + "-"
	eqc = read_catalogGIS(gis_filespec, column_map, fix_zero_days_and_months=fix_zero_days_and_months,
						convert_zero_magnitudes=convert_zero_magnitudes, ID_prefix=ID_prefix, verbose=verbose)
	eqc.name = catalog_name
	return eqc


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
	from time_functions import parse_isoformat_datetime

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
				dt = parse_isoformat_datetime(line[column_map["datetime"]])
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
					date = mxDateTime.Date(year, month, day)
				except:
					print line
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
				time = mxDateTime.Time(hour, minute, second)

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


def get_catalogs_map(catalogs, catalog_styles=[], symbols=[], edge_colors=[], fill_colors=[],
					labels=[], mag_size_inc=4,  Mtype="MW", Mrelation="default",
					coastline_style={}, country_style={}, river_style=None, continent_style=None,
					source_model=None, sm_style={"line_color": 'k', "line_pattern": '-', "line_width": 2},
					sm_label_colname="ShortName",
					sites=[], site_style={"shape": 's', "fill_color": 'b', "size": 10}, site_legend="",
					circles=[], circle_styles=[],
					projection="merc", region=None, origin=(None, None), graticule_interval=(1., 1.), resolution="i",
					graticule_style={"annot_axes": "SE"}, title=None, legend_style={}, border_style={}):
	"""
	Construct map of multiple catalogs.

	:param catalogs:
		List containing instances of :class:`EQCatalog`
	:param catalog_styles:
		List with styles (instances of :class:`PointStyle` or dictionaries
		with subset of PointStyle attributes as keys) for each catalog.
		If list contains only 1 element, the same style will be used for
		all catalogs. If list is empty, a default style will be used.
		Point size refers to a magnitude-3 earthquake if :param:`mag_size_inc`
		is set (default: [])
	:param symbols:
		List containing point symbols (matplotlib marker specifications)
		for each catalog, overriding style given by :param:`catalog_styles`
		(default: [])
	:param edge_colors:
		List containing symbol edge colors (matplotlib color specifications)
		for each catalog, overriding style given by :param:`catalog_styles`
		(default: [])
	:param fill_colors:
		List containing symbol fill colors (matplotlib color specifications)
		for each catalog, overriding style given by :param:`catalog_styles`
		(default: [])
	:param labels:
		List containing plot labels, one for each catalog (default: [])
	:param mag_size_inc:
		Int or Float, symbol size increment per magnitude relative to M=3
		(default: 4)
	:param Mtype:
		String, magnitude type for magnitude scaling (default: "MW")
	:param Mrelation:
		{str: str} dict, mapping name of magnitude conversion relation
		to magnitude type ("MW", "MS" or "ML") (default: None, will
		select the default relation for the given Mtype)
	:param coastline_style:
		instance of :class:`LineStyle` or dictionary with subset of
		LineStyle attributes as keys, used to plot coastlines. If None,
		coastlines will not be drawn
		(default: {}, equivalent to default line style)
	:param country_style:
		instance of :class:`LineStyle` or dictionary with subset of
		LineStyle attributes as keys, used to plot country borders. If None,
		country borders will not be drawn
		(default: {}, equivalent to default line style)
	:param river_style:
		instance of :class:`LineStyle` or dictionary with subset of
		LineStyle attributes as keys, used to plot rivers. If None, rivers
		will not be drawn
		(default: None)
	:param continent_style:
		instance of :class:`PolygonStyle` or dictionary with subset of
		PolygonStyle attributes as keys, used to plot continents/oceans.
		If None, continents/oceans will not be drawn (default: None)
	:param source_model:
		String, name of source model to overlay on the plot
		or full path to GIS file containing source model
		(default: None)
	:param sm_style:
		instance of :class:`LineStyle`, :class:`PolygonStyle`,
		:class:`CompositeStyle` or dictionary with subset of attributes
		of LineStyle or PolygonStyle as keys, used to plot source model.
		(default: {"line_color": 'k', "line_style": '-', "line_width": 2, "fill_color": "None"}
	:param sm_label_colname:
		Str, column name of GIS table to use as label (default: "ShortName")
	:param sites:
		List of (lon, lat) tuples or instance of :class:`PSHASite`
	:param site_style:
		instance of :class:`PointStyle` or dictionary containing subset of
		PointStyle attributes as keys, used to plot sites
		(default: {"shape": 's', "fill_color": 'b', "size": 10})
	:param site_legend:
		String, common text referring to all sites to be placed in legend
		(default: "")
	:param circles:
		list with (lon, lat, radius) tuples defining center and radius
		(in km) of circles to plot (default: [])
	:param circle_styles:
		List with styles (instances of :class:`LineStyle` or dictionaries
		with subset of LineStyle attributes as keys) for each circle.
		If list contains only 1 element, the same style will be used for
		all circles. If list is empty, a default style will be used
		(default: [])
	:param projection:
		String, map projection. See Basemap documentation
		(default: "merc")
	:param region:
		(w, e, s, n) tuple specifying rectangular region to plot in
		geographic coordinates (default: None)
	:param origin:
		(lon, lat) tuple defining map origin. Needed for some
		projections (default: None)
	:param graticule_interval:
		(dlon, dlat) tuple defining meridian and parallel interval in
		degrees (default: (1., 1.)
	:param resolution:
		String, map resolution (coastlines and country borders):
		'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high), 'f' (full)
		(default: 'i')
	:param graticule_style:
		instance of :class:`GraticuleStyle` or dictionary containing
		GraticuleStyle attributes as keys, defining graticule style of map
		(default: {"annot_axes": "SE"}
		annot_axes: string, containing up to 4 characters ('W', 'E', 'S' and/or 'N'),
		defining which axes should be annotated
	:param title:
		String, plot title (default: None)
	:param legend_style:
		instance of :class:`LegendStyle` or dictionary containing
		LegendStyle attributes as keys, defining style of map legend
		(default: {})
	:param border_style:
		instance ov :class:`MapBorderStyle` or dictionary containing
		MapBorderStyle attributes as keys, defining style of map border
		(default: {})

	:return:
		instance of :class:`LayeredBasemap`
	"""
	from source_models import rob_source_models_dict
	import mapping.layeredbasemap as lbm

	layers = []

	## Continents/oceans
	if continent_style != None:
		data = lbm.BuiltinData("continents")
		if isinstance(continent_style, dict):
			style = lbm.PolygonStyle.from_dict(continent_style)
		else:
			style = continent_style
		layers.append(lbm.MapLayer(data, style, name="continents"))

	## Coastlines
	if coastline_style != None:
		data = lbm.BuiltinData("coastlines")
		if isinstance(coastline_style, dict):
			style = lbm.LineStyle.from_dict(coastline_style)
		else:
			style = coastline_style
		layers.append(lbm.MapLayer(data, style, name="coastlines"))

	## Country borders
	if country_style != None:
		data = lbm.BuiltinData("countries")
		if isinstance(country_style, dict):
			style = lbm.LineStyle.from_dict(country_style)
		else:
			style = country_style
		layers.append(lbm.MapLayer(data, style, name="countries"))

	## Rivers
	if river_style != None:
		data = lbm.BuiltinData("rivers")
		if isinstance(river_style, dict):
			style = lbm.LineStyle.from_dict(river_style)
		else:
			style = country_style
		layers.append(lbm.MapLayer(data, style, name="rivers"))

	## Source model
	if source_model:
		try:
			gis_filespec = rob_source_models_dict[source_model]["gis_filespec"]
		except:
			if isinstance(source_model, (str, unicode)):
				gis_filespec = source_model
				source_model_name = os.path.splitext(os.path.split(source_model)[1])[0]
			else:
				import hazard.rshalib as rshalib
				if isinstance(source_model, rshalib.source.SourceModel):
					gis_filespec = None
					source_model_name = source_model.name
		else:
			source_model_name = source_model

		if gis_filespec:
			data = lbm.GisData(gis_filespec, label_colname=sm_label_colname)
		else:
			# TODO: implement line and point sources too
			point_lons, point_lats, point_labels = [], [], []
			line_lons, line_lats, line_labels = [], [], []
			polygon_lons, polygon_lats, polygon_labels = [], [], []
			for src in source_model:
				if isinstance(src, rshalib.source.AreaSource):
					polygon_lons.append(src.polygon.lons)
					polygon_lats.append(src.polygon.lats)
					polygon_labels.append(getattr(src, sm_label_colname, ""))
				elif isinstance(src, rshalib.source.PointSource):
					point_lons.append(src.location.longitude)
					point_lats.append(src.location.latitude)
					point_labels.append(getattr(src, sm_label_colname, ""))
				elif isinstance(src, rshalib.source.SimpleFaultSource):
					line_lons.append([pt.lon for pt in src.fault_trace.points])
					line_lats.append([pt.lat for pt in src.fault_trace.points])
					line_labels.append(getattr(src, sm_label_colname, ""))
			point_data = lbm.MultiPointData(point_lons, point_lats, labels=point_labels)
			line_data = lbm.MultiLineData(line_lons, line_lats, labels=line_labels)
			polygon_data = lbm.MultiPolygonData(polygon_lons, polygon_lats, labels=polygon_labels)
			data = lbm.CompositeData(points=point_data, lines=line_data, polygons=polygon_data)
		if isinstance(sm_style, dict):
			if sm_style.has_key("fill_color") and not sm_style["fill_color"] in ("None", None):
				polygon_style = lbm.PolygonStyle.from_dict(sm_style)
				line_style = None
			else:
				line_style = lbm.LineStyle.from_dict(sm_style)
				polygon_style = None
		elif isinstance(sm_style, lbm.CompositeStyle):
			line_style = sm_style.line_style
			polygon_style = sm_style.polygon_style
		elif isinstance(sm_style, lbm.LineStyle):
			line_style = sm_style
			polygon_style = None
		elif isinstance(sm_style, lbm.PolygonStyle):
			polygon_style = sm_style
			line_style = None
		if line_style and not line_style.label_style:
			line_style.label_style = lbm.TextStyle(color=line_style.line_color, font_size=8)
		elif polygon_style and not polygon_style.label_style:
			polygon_style.label_style = lbm.TextStyle(color=polygon_style.line_color, font_size=8)
		style = lbm.CompositeStyle(line_style=line_style, polygon_style=polygon_style)
		legend_label = {'lines': source_model_name + " faults", 'polygons': source_model_name + " zones"}
		layer = lbm.MapLayer(data, style, legend_label=legend_label, name="source model")
		layers.append(layer)

	## Earthquakes
	if not labels:
		labels = [None] * len(catalogs)
	if catalog_styles in ([], None):
		catalog_styles = lbm.PointStyle(shape='o', size=9)
	if isinstance(catalog_styles, (lbm.PointStyle, dict)):
		catalog_styles = [catalog_styles]
	if len(catalog_styles) == 1:
		base_style = catalog_styles[0]
		if isinstance(base_style, dict):
			base_style = lbm.PointStyle.from_dict(base_style)
		if not symbols:
			symbols = ["o"]
		if not edge_colors:
			edge_colors = ("r", "g", "b", "c", "m", "k")
		if not fill_colors:
			fill_colors = ["None"]
		catalog_styles = []
		for i in range(len(catalogs)):
			style = lbm.PointStyle.from_dict(base_style.__dict__)
			style.shape = symbols[i%len(symbols)]
			style.line_color = edge_colors[i%len(edge_colors)]
			style.fill_color = fill_colors[i%len(fill_colors)]
			catalog_styles.append(style)

	for i in range(len(catalogs)):
		catalog = catalogs[i]
		style = catalog_styles[i]
		if isinstance(style, dict):
			style = lbm.PointStyle.from_dict(style)
		values = {}
		if mag_size_inc:
			## Magnitude-dependent size
			if i == 0:
				min_mag = np.floor(catalog.get_Mmin(Mtype, Mrelation))
				max_mag = np.ceil(catalog.get_Mmax(Mtype, Mrelation))
				mags = np.linspace(min_mag, max_mag, min(5, max_mag-min_mag+1))
				sizes = style.size + (mags - 3) * mag_size_inc
				sizes = sizes.clip(min=1)
				style.thematic_legend_style = lbm.LegendStyle(title="Magnitude", location=3, shadow=True, fancy_box=True, label_spacing=0.7)
			values['magnitude'] = catalog.get_magnitudes(Mtype, Mrelation)
			style.size = lbm.ThematicStyleGradient(mags, sizes, value_key="magnitude")

		# TODO: color by depth
		#values['depth'] = catalog.get_depths()
		#colorbar_style = ColorbarStyle(title="Depth (km)", location="bottom", format="%d")
		#style.fill_color = ThematicStyleRanges([0,1,10,25,50], ['red', 'orange', 'yellow', 'green'], value_key="depth", colorbar_style=colorbar_style)

		# TODO: color by age
		#values['year'] = [eq.datetime.year for eq in catalog]
		#style.fill_color = ThematicStyleRanges([1350,1910,2050], ['green', (1,1,1,0)], value_key="year")

		point_data = lbm.MultiPointData(catalog.get_longitudes(), catalog.get_latitudes(), values=values)

		layer = lbm.MapLayer(point_data, style, legend_label=labels[i], name="earthquakes")
		layers.append(layer)

	## Sites
	if sites:
		if isinstance(site_style, dict):
			site_style = lbm.PointStyle.from_dict(site_style)
			site_style.label_style = lbm.TextStyle()
		site_lons, site_lats, site_labels = [], [], []
		for i, site in enumerate(sites):
			try:
				lon, lat = site.longitude, site.latitude
				name = site.name
			except:
				lon, lat, name = site[:3]
			site_lons.append(lon)
			site_lats.append(lat)
			site_labels.append(name)

		point_data = lbm.MultiPointData(site_lons, site_lats, labels=site_labels)
		layer = lbm.MapLayer(point_data, site_style, site_legend, name="sites")
		layers.append(layer)

	## Circles
	if circles:
		if circle_styles == []:
			circle_styles = {}
		if isinstance(circle_styles, dict):
			circle_styles = lbm.LineStyle.from_dict(circle_styles)
		if isinstance(circle_styles, lbm.LineStyle):
			circle_styles = [circle_styles]
		for circle in circles:
			lon, lat, radius = circle
			circle_data = lbm.CircleData([lon], [lat], [radius])
			layer = lbm.MapLayer(circle_data, circle_styles[i%len(circle_styles)], name="circles")
			layers.append(layer)

	if isinstance(legend_style, dict):
		legend_style = lbm.LegendStyle.from_dict(legend_style)
	if isinstance(border_style, dict):
		border_style = lbm.MapBorderStyle.from_dict(border_style)
	if isinstance(graticule_style, dict):
		graticule_style = lbm.GraticuleStyle.from_dict(graticule_style)

	## Determine map extent if necessary
	if not region:
		west, east, south, north = 180, -180, 90, -90
		for catalog in catalogs:
			if catalog.region:
				w, e, s, n = list(catalog.region)
			else:
				w, e, s, n = list(catalog.get_region())
			if w < west:
				west = w
			if e > east:
				east = e
			if s < south:
				south = s
			if n > north:
				north = n
		region = (west, east, south, north)

	map = lbm.LayeredBasemap(layers, title, projection, region=region, origin=origin, graticule_interval=graticule_interval, resolution=resolution, graticule_style=graticule_style, legend_style=legend_style)
	return map


def plot_catalogs_map(catalogs, symbols=[], edge_colors=[], fill_colors=[], edge_widths=[], labels=[], symbol_size=9, symbol_size_inc=4, Mtype="MW", Mrelation="default", circle=None, region=None, projection="merc", resolution="i", dlon=1., dlat=1., source_model=None, sm_color='k', sm_line_style='-', sm_line_width=2, sm_label_size=11, sm_label_colname="ShortName", sites=[], site_symbol='o', site_color='b', site_size=10, site_legend="", title=None, legend_location=0, fig_filespec=None, fig_width=0, dpi=300):
	"""
	Plot multiple catalogs on a map

	:param catalogs:
		List containing instances of :class:`EQCatalog`
	:param symbols:
		List containing earthquake symbols for each catalog
		(matplotlib marker specifications)
	:param edge_colors:
		List containing symbol edge colors for each catalog
		(matplotlib color specifications)
		(default: [])
	:param fill_colors:
		List containing symbol fill colors for each catalog
		(matplotlib color specifications)
		(default: [])
	:param edge_widths:
		List containing symbol edge width for each catalog
		(default: [])
	:param labels:
		List containing plot labels, one for each catalog (default: [])
	:param symbol_size:
		Int or Float, symbol size in points (default: 9)
	:param symbol_size_inc:
		Int or Float, symbol size increment per magnitude relative to M=3
		(default: 4)
	:param Mtype:
		String, magnitude type for magnitude scaling (default: "MW")
	:param Mrelation:
		{str: str} dict, mapping name of magnitude conversion relation
		to magnitude type ("MW", "MS" or "ML") (default: None, will
		select the default relation for the given Mtype)
	:param circle:
		((lon, lat), float, string), respectively defining center, radius (in
		km) and color of circle to plot
	:param region:
		(w, e, s, n) tuple specifying rectangular region to plot in
		geographic coordinates (default: None)
	:param projection:
		String, map projection. See Basemap documentation
		(default: "merc")
	:param resolution:
		String, map resolution (coastlines and country borders):
		'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high), 'f' (full)
		(default: 'i')
	:param dlon:
		Float, meridian interval in degrees (default: 1.)
	:param dlat:
		Float, parallel interval in degrees (default: 1.)
	:param source_model:
		String, name of source model to overlay on the plot
		(default: None)
	:param sm_color:
		matplotlib color specification to plot source model
		(default: 'k')
	:param sm_line_style:
		String, line style to plot source model (default: '-')
	:param sm_line_width:
		Int, line width to plot source model (default: 2)
	:param sm_label_size:
		Int, font size of source labels. If 0 or None, no labels will
		be plotted (default: 11)
	:param sm_label_colname:
		Str, column name of GIS table to use as label (default: "ShortName")
	:param sites:
		List of (lon, lat) tuples or instance of :class:`PSHASite`
	:param site_symbol:
		matplotlib marker specifications for site symbols (default: 'o')
	:param site_color:
		matplotlib color specification for site symbols (default: 'b')
	:param site_size:
		Int, size to be used for site symbols (default: 10)
	:param site_legend:
		String, common text referring to all sites to be placed in legend
		(default: "")
	:param title:
		String, plot title (default: None)
	:param legend_location:
		String or Int: location of legend (matplotlib location code):
			"best" 	0
			"upper right" 	1
			"upper left" 	2
			"lower left" 	3
			"lower right" 	4
			"right" 	5
			"center left" 	6
			"center right" 	7
			"lower center" 	8
			"upper center" 	9
			"center" 	10
		(default: 0)
	:param fig_filespec:
		String, full path of image to be saved.
		If None (default), map is displayed on screen.
	:param fig_width:
		Float, figure width in cm, used to recompute :param:`dpi` with
		respect to default figure width (default: 0)
	:param dpi:
		Int, image resolution in dots per inch (default: 300)
	"""
	from mpl_toolkits.basemap import Basemap

	## Symbols, colors, and labels
	if not symbols:
		symbols = ["o"]
	if not edge_colors:
		edge_colors = ("r", "g", "b", "c", "m", "k")
	if not fill_colors:
		fill_colors = ["None"]
	if not edge_widths:
		edge_widths = [1]
	if not labels:
		labels = [None]

	## Determine map extent and center
	if not region:
		if catalogs[0].region:
			region = list(catalogs[0].region)
		else:
			region = list(catalogs[0].get_region())
			lon_range = region[1] - region[0]
			lat_range = region[3] - region[2]
			region[0] -= lon_range / 5.
			region[1] += lon_range / 5.
			region[2] -= lat_range / 5.
			region[3] += lat_range / 5.
	else:
		region = list(region)
	lon_0 = (region[0] + region[1]) / 2.
	lat_0 = (region[2] + region[3]) / 2.

	## Base map
	map = Basemap(projection=projection, resolution=resolution, llcrnrlon=region[0], llcrnrlat=region[2], urcrnrlon=region[1], urcrnrlat=region[3], lon_0=lon_0, lat_0=lat_0)
	map.drawcoastlines()
	map.drawcountries()

	## Meridians and parallels
	if dlon:
		first_meridian = np.ceil(region[0] / dlon) * dlon
		last_meridian = np.floor(region[1] / dlon) * dlon + dlon
		meridians = np.arange(first_meridian, last_meridian, dlon)
		map.drawmeridians(meridians, labels=[0,1,0,1])
	if dlat:
		first_parallel = np.ceil(region[2] / dlat) * dlat
		last_parallel = np.floor(region[3] / dlat) * dlat + dlat
		parallels = np.arange(first_parallel, last_parallel, dlat)
		map.drawparallels(parallels, labels=[0,1,0,1])

	## Source model
	if source_model:
		try:
			rob_source_models_dict[source_model_name]["gis_filespec"]
		except:
			source_model_name = os.path.splitext(os.path.split(source_model)[1])[0]
		else:
			source_model_name = source_model
		model_data = read_source_model(source_model)
		for i, zone_data in enumerate(model_data.values()):
			geom = zone_data['obj']
			lines = []
			if geom.GetGeometryName() == "LINESTRING":
				## In some versions of ogr, GetPoints method does not exist
				#points = linear_ring.GetPoints()
				points = [geom.GetPoint(i) for i in range(geom.GetPointCount())]
				lines.append(points)
				centroid = None
			elif geom.GetGeometryName() == "POLYGON":
				centroid = geom.Centroid()
				for linear_ring in geom:
					points = [linear_ring.GetPoint(i) for i in range(linear_ring.GetPointCount())]
					lines.append(points)
			for j, line in enumerate(lines):
				lons, lats, _ = zip(*line)
				x, y = map(lons, lats)
				if i == 0 and j == 0:
					label = source_model_name
				else:
					label = "_nolegend_"
				map.plot(x, y, ls=sm_line_style, lw=sm_line_width, color=sm_color, label=label)

				if centroid and sm_label_size:
					x, y = map(centroid.GetX(), centroid.GetY())
					if isinstance(sm_label_colname, (str, unicode)):
						zone_label = zone_data.get("sm_label_colname", "")
					else:
						zone_label = " / ".join([str(zone_data[colname]) for colname in sm_label_colname])
					pylab.text(x, y, zone_label, color=sm_color, fontsize=sm_label_size, fontweight='bold', ha='center', va='center')

	## Catalogs
	for i, catalog in enumerate(catalogs):
		if len(catalog):
			symbol = symbols[i%len(symbols)]
			edge_color = edge_colors[i%len(edge_colors)]
			if edge_color is None:
				edge_color = "None"
			fill_color = fill_colors[i%len(fill_colors)]
			if fill_color is None:
				fill_color = "None"
			edge_width = edge_widths[i%len(edge_widths)]
			label = labels[i%len(labels)]
			if label is None:
				label = catalog.name

			## Earthquake symbol size varying with magnitude
			if not symbol_size_inc:
				symbol_sizes = symbol_size ** 2
			else:
				magnitudes = catalog.get_magnitudes(Mtype, Mrelation)
				symbol_sizes = symbol_size + (magnitudes - 3.0) * symbol_size_inc
				symbol_sizes = symbol_sizes ** 2
				if symbol_sizes.min() <= 0:
					print "Warning: negative or zero symbol size encountered"
				#print symbol_sizes.min(), symbol_sizes.max()

			## Earthquake epicenters
			if len(catalog.eq_list) > 0:
				lons, lats = catalog.get_longitudes(), catalog.get_latitudes()
				x, y = map(lons, lats)
				map.scatter(x, y, s=symbol_sizes, marker=symbol, edgecolors=edge_color, facecolors=fill_color, linewidth=edge_width, label=label)

	## Sites
	for i, site in enumerate(sites):
		try:
			lon, lat = site.longitude, site.latitude
			name = site.name
		except:
			lon, lat, name = site[:3]
		x, y = map(lon, lat)
		if i == 0:
			label = site_legend
		else:
			label = None
		map.plot(x, y, site_symbol, markerfacecolor=site_color, markeredgecolor='k', markersize=site_size, label=label)

	## Circle
	if circle:
		from openquake.hazardlib.geo.geodetic import point_at
		center, radius, color = circle
		x, y = [], []
		for azimuth in range(0, 360):
			lon, lat = point_at(center[0], center[1], azimuth, radius)
			x.append(lon)
			y.append(lat)
		x.append(x[0])
		y.append(y[0])
		x, y = map(x,y)
		map.plot(x, y, c=color)

	## Map border and title
	map.drawmapboundary()
	if title:
		pylab.title(title)
	plt.legend(loc=legend_location)

	#plt.tight_layout()
	if fig_filespec:
		default_figsize = pylab.rcParams['figure.figsize']
		default_dpi = pylab.rcParams['figure.dpi']
		if fig_width:
			fig_width /= 2.54
			dpi = dpi * (fig_width / default_figsize[0])
		pylab.savefig(fig_filespec, dpi=dpi)
		pylab.clf()
	else:
		pylab.show()


#def plot_catalogs_time_magnitude(catalogs, symbols=[], edge_colors=[], fill_colors=[], labels=[], symbol_size=50, Mtype="MW", Mrelation=None, completeness=None, completeness_color="r", vlines=False, grid=True, plot_date=False, major_tick_interval=None, minor_tick_interval=1, title=None, lang="en", legend_location=0, fig_filespec=None, fig_width=0, dpi=300):
#	"""
#	:param catalogs:
#		List containing instances of :class:`EQCatalog`
#	:param symbols:
#		List containing earthquake symbols for each catalog
#		(matplotlib marker specifications)
#	:param edge_colors:
#		List containing symbol edge colors for each catalog
#		(matplotlib color specifications)
#		(default: [])
#	:param fill_colors:
#		List containing symbol fill colors for each catalog
#		(matplotlib color specifications)
#		(default: [])
#	:param labels:
#		List containing plot labels, one for each catalog (default: [])
#	:param symbol_size:
#		Int or Float, symbol size in points (default: 50)
#	:param Mtype:
#		String, magnitude type for magnitude scaling (default: "MW")
#	:param Mrelation:
#		{str: str} dict, mapping name of magnitude conversion relation
#		to magnitude type ("MW", "MS" or "ML") (default: None, will
#		select the default relation for the given Mtype)
#	:param completeness:
#		class:`Completeness` instance, plot completeness (default: None)
#	:param completeness_color:
#		Str, color to plot completeness line (default: "r")
#	:param vlines:
#		Boolean, plot vertical lines from data point to x-axis (default: False)
#	:param grid:
#		Boolean, plot grid (default: False)
#	:param plot_date:
#		Boolean, whether or not to plot time axis as dates instead of
#		fractional years (default: False)
#	:param major_tick_interval:
#		Int, interval in years for major ticks (default: None). If none, a
#		maximum number of ticks at nice locations will be used.
#	:param minor_tick_interval:
#		Int, interval in years for minor ticks (default: 1)
#	:param lang:
#		String, language of plot labels (default: "en")
#	:param title:
#		String, plot title (default: None)
#	:param legend_location:
#		String or Int: location of legend (matplotlib location code):
#			"best" 	0
#			"upper right" 	1
#			"upper left" 	2
#			"lower left" 	3
#			"lower right" 	4
#			"right" 		5
#			"center left" 	6
#			"center right" 	7
#			"lower center" 	8
#			"upper center" 	9
#			"center" 		10
#		(default: 0)
#	:param fig_filespec:
#		String, full path of image to be saved.
#		If None (default), map is displayed on screen.
#	:param fig_width:
#		Float, figure width in cm, used to recompute :param:`dpi` with
#		respect to default figure width (default: 0)
#	:param dpi:
#		Int, image resolution in dots per inch (default: 300)
#	"""
#	## Symbols, colors, and labels
#	if not symbols:
#		symbols = ["o"]
#	if not edge_colors:
#		edge_colors = ("r", "g", "b", "c", "m", "k")
#	if not fill_colors:
#		fill_colors = ["None"]
#	if not labels:
#		labels = [None]

#	for i, catalog in enumerate(catalogs):

#		symbol = symbols[i%len(symbols)]
#		edge_color = edge_colors[i%len(edge_colors)]
#		if edge_color is None:
#			edge_color = "None"
#		fill_color = fill_colors[i%len(fill_colors)]
#		if fill_color is None:
#			fill_color = "None"
#		label = labels[i%len(labels)]
#		if label is None:
#			label = catalog.name

#		y = catalog.get_magnitudes(Mtype, Mrelation)
#		if not plot_date:
#			x = catalog.get_fractional_years()
#			plt.scatter(x, y, s=symbol_size, edgecolors=edge_color, label=label, marker=symbol, facecolors=fill_color)
#			xmin, xmax, ymin, ymax = plt.axis()
#			xmin, xmax = catalog.start_date.year, catalog.end_date.year+1
#			plt.axis((xmin, xmax, 0, max(y)*1.1))
#		else:
#			x = pylab.date2num(catalog.get_datetimes())
#			plt.plot_date(x, y, ms=np.sqrt(symbol_size), mec=edge_color, color=fill_color, label=label, marker=symbol, mfc=fill_color)

#		## plot vlines
#		if vlines and not plot_date:
#			plt.vlines(x, ymin=ymin, ymax=y)

#	## plot ticks
#	ax = plt.gca()
#	if not plot_date:
#		if major_tick_interval:
#			majorLocator = MultipleLocator(major_tick_interval)
#		else:
#			majorLocator = MaxNLocator()
#		minorLocator = MultipleLocator(minor_tick_interval)
#		ax.xaxis.set_major_locator(majorLocator)
#		ax.xaxis.set_minor_locator(minorLocator)
#		ax.yaxis.set_minor_locator(MultipleLocator(0.1))
#	for label in ax.get_xticklabels() + ax.get_yticklabels():
#		label.set_size('large')
#	if plot_date:
#		for label in ax.get_xticklabels():
#			label.set_horizontalalignment('right')
#			label.set_rotation(30)

#	## plot x and y labels
#	if not plot_date:
#		xlabel = {"en": "Time (years)", "nl": "Tijd (jaar)"}[lang]
#	else:
#		xlabel = {"en": "Date", "nl": "Datum"}[lang]
#	plt.xlabel(xlabel, fontsize="x-large")
#	plt.ylabel("Magnitude ($M_%s$)" % Mtype[1].upper(), fontsize="x-large")

#	## plot completeness
#	if completeness and not plot_date:
#		x, y = completeness.min_years, completeness.min_mags
#		x = np.append(x, max([catalog.end_date for catalog in catalogs]).year+1)
#		plt.hlines(y, xmin=x[:-1], xmax=x[1:], colors=completeness_color)
#		plt.vlines(x[1:-1], ymin=y[1:], ymax=y[:-1], colors=completeness_color, lw=2)

#	if grid:
#		plt.grid()

#	if title:
#		plt.title(title)

#	plt.legend(loc=legend_location)

#	if fig_filespec:
#		default_figsize = plt.rcParams['figure.figsize']
#		default_dpi = plt.rcParams['figure.dpi']
#		if fig_width:
#			fig_width /= 2.54
#			dpi = dpi * (fig_width / default_figsize[0])
#		pylab.savefig(fig_filespec, dpi=dpi)
#		pylab.clf()
#	else:
#		pylab.show()

def plot_catalogs_magnitude_time(catalogs, symbols=[], edge_colors=[], fill_colors=[], edge_widths=[], labels=[], symbol_size=50, Mtype="MW", Mrelation="default", start_year=None, Mrange=(None, None), completeness=None, completeness_color="r", vlines=False, grid=True, plot_date=False, major_tick_interval=None, minor_tick_interval=1, tick_unit=None, tick_freq=None, tick_by=None, tick_form=None, title=None, lang="en", legend_location=0, fig_filespec=None, fig_width=0, dpi=300, ax=None):
	"""
	:param catalogs:
		List containing instances of :class:`EQCatalog`
	:param symbols:
		List containing earthquake symbols for each catalog
		(matplotlib marker specifications)
	:param edge_colors:
		List containing symbol edge colors for each catalog
		(matplotlib color specifications)
		(default: [])
	:param fill_colors:
		List containing symbol fill colors for each catalog
		(matplotlib color specifications)
		(default: [])
	:param edge_widths:
		List containing symbol edge widths for each catalog
		(default: [])
	:param labels:
		List containing plot labels, one for each catalog (default: [])
	:param symbol_size:
		Int or Float, symbol size in points (default: 50)
	:param Mtype:
		String, magnitude type for magnitude scaling (default: "MW")
	:param Mrelation:
		{str: str} dict, mapping name of magnitude conversion relation
		to magnitude type ("MW", "MS" or "ML") (default: None, will
		select the default relation for the given Mtype)
	:param start_year:
		float or int, year to start x axis (does not work when plot_date is True)
		(default: None)
	:param Mrange:
		tuple of floats representing minimum and maximu magnitude in y axis
		(default: None, None)
	:param completeness:
		class:`Completeness` instance, plot completeness (default: None)
	:param completeness_color:
		Str, color to plot completeness line (default: "r")
	:param vlines:
		List of years or datetime objects (or both), years/dates to plot as
		vertical lines (default: False)
	:param grid:
		Boolean, plot grid (default: False)
	:param plot_date:
		Boolean, whether or not to plot time axis as dates instead of
		fractional years (default: False)
	:param major_tick_interval:
		Int, interval in years for major ticks (default: None). If none, a
		maximum number of ticks at nice locations will be used.
	:param minor_tick_interval:
		Int, interval in years for minor ticks (default: 1)
	:param tick_unit:
		Str, unit of tick ('year', 'month', 'weekday' or 'monthday')
		(default: None)
	:param tick_freq:
		Int, interval of ticks (default: None)
	:param tick_by:
		List, fixed locations of ticks by month, weekday or monthday
		(default: None)
	:param tick_form:
		Str, tick format by strftime() (default: None)
	:param lang:
		String, language of plot labels (default: "en")
	:param title:
		String, plot title (default: None)
	:param legend_location:
		String or Int: location of legend (matplotlib location code):
			"best" 	0
			"upper right" 	1
			"upper left" 	2
			"lower left" 	3
			"lower right" 	4
			"right" 		5
			"center left" 	6
			"center right" 	7
			"lower center" 	8
			"upper center" 	9
			"center" 		10
		(default: 0)
	:param fig_filespec:
		String, full path of image to be saved.
		If None (default), map is displayed on screen.
	:param fig_width:
		Float, figure width in cm, used to recompute :param:`dpi` with
		respect to default figure width (default: 0)
	:param dpi:
		Int, image resolution in dots per inch (default: 300)
	:param ax:
		matplotlib Axes instance
		(default: None)
	"""
	if ax is None:
		ax = pylab.axes()

	## symbols, colors, and labels
	if not symbols:
		symbols = ["o"]
	if not edge_colors:
		edge_colors = ("r", "g", "b", "c", "m", "k")
	if not fill_colors:
		fill_colors = ["None"]
	if not edge_widths:
		edge_widths = [1]
	if not labels:
		labels = [None]

	## plot catalogs
	for i, catalog in enumerate(catalogs):
		symbol = symbols[i%len(symbols)]
		edge_color = edge_colors[i%len(edge_colors)]
		if edge_color is None:
			edge_color = "None"
		fill_color = fill_colors[i%len(fill_colors)]
		if fill_color is None:
			fill_color = "None"
		edge_width = edge_widths[i%len(edge_widths)]
		label = labels[i%len(labels)]
		if label is None:
			label = catalog.name
		y = catalog.get_magnitudes(Mtype, Mrelation)
		if plot_date:
			x = catalog.get_datetimes()
		else:
			x = catalog.get_fractional_years()
		ax.scatter(x, y, s=symbol_size, edgecolors=edge_color, label=label, marker=symbol, facecolors=fill_color, linewidth=edge_width)

	## crop X axis to data when using fractional years
	xmin, xmax, ymin, ymax = ax.axis()
	if not plot_date:
		if start_year:
			xmin = start_year
		else:
			xmin = min(catalog.start_date.year for catalog in catalogs)
		xmax = max(catalog.end_date.year for catalog in catalogs)+1

	## Set range of Y axis
	try:
		Mmin, Mmax = Mrange
	except:
		pass
	else:
		if Mmin != None:
			ymin = Mmin
		if Mmax != None:
			ymax = Mmax
	ax.axis((xmin, xmax, ymin, ymax))

	## plot ticks
	if plot_date:
		if tick_unit:
			tick_unit_loc_map = {'year': mdates.YearLocator, 'month': mdates.MonthLocator, 'weekday': mdates.WeekdayLocator, 'monthday': mdates.DayLocator}
			maj_loc = tick_unit_loc_map[tick_unit]
			maj_loc_kwargs = {}
			if tick_freq:
				maj_loc_kwargs[{'year': 'base'}.get(tick_unit, 'interval')] = tick_freq
			if tick_by:
				maj_loc_kwargs['by' + tick_unit] = tick_by
			maj_loc = maj_loc(**maj_loc_kwargs)
			ax.xaxis.set_major_locator(maj_loc)
			if tick_form:
				maj_fmt = DateFormatter(tick_form)
			else:
				maj_fmt = mdates.AutoDateFormatter(maj_loc)
			ax.xaxis.set_major_formatter(maj_fmt)
		for label in ax.get_xticklabels():
			label.set_horizontalalignment('right')
			label.set_rotation(30)
	else:
		if major_tick_interval:
			major_loc = MultipleLocator(major_tick_interval)
		else:
			major_loc = MaxNLocator()
		minor_loc = MultipleLocator(minor_tick_interval)
		ax.xaxis.set_major_locator(major_loc)
		ax.xaxis.set_minor_locator(minor_loc)
	ax.yaxis.set_minor_locator(MultipleLocator(0.1))
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size('large')

	## plot x and y labels
	if not plot_date:
		xlabel = {"en": "Time (years)", "nl": "Tijd (jaar)"}[lang]
	else:
		xlabel = {"en": "Date", "nl": "Datum"}[lang]
	ax.set_xlabel(xlabel, fontsize="x-large")
	ax.set_ylabel("Magnitude ($M_%s$)" % Mtype[1].upper(), fontsize="x-large")

	## plot vertical lines
	if vlines:
		for i, vline in enumerate(vlines):
			if isinstance(vline, int):
				if plot_date:
					vlines[i] = datetime.datetime(vline, 1, 1)
			else:
				if not plot_date:
					vlines[i] = vline.year
		ymin, ymax = ax.get_ylim()
		ax.vlines(vlines, ymin=ymin, ymax=ymax, colors='b')
		ax.set_ylim(ymin, ymax)

	## plot completeness
	# TODO: implement completeness dates rather than years
	if completeness:
		x, y = completeness.min_years, completeness.min_mags
		x = np.append(x, max([catalog.end_date for catalog in catalogs]).year+1)
		if plot_date:
			x = [datetime.datetime(year, 1, 1) for year in x]
		xmin, xmax, ymin, ymax = ax.axis()
		ax.hlines(y, xmin=x[:-1], xmax=x[1:], colors=completeness_color)
		ax.vlines(x[1:-1], ymin=y[1:], ymax=y[:-1], colors=completeness_color, lw=2)
		ax.axis((xmin, xmax, ymin, ymax))

	if grid:
		ax.grid()

	if title:
		ax.set_title(title)

	ax.legend(loc=legend_location)

	if fig_filespec == "hold":
		pass
	elif fig_filespec:
		plt.tight_layout()
		default_figsize = plt.rcParams['figure.figsize']
		default_dpi = plt.rcParams['figure.dpi']
		if fig_width:
			fig_width /= 2.54
			dpi = dpi * (fig_width / default_figsize[0])
		pylab.savefig(fig_filespec, dpi=dpi)
		pylab.clf()
	else:
		pylab.show()


def plot_depth_statistics(
	catalogs,
	labels=[],
	dmin=0,
	dmax=30,
	Mmin=None,
	Mtype="MW",
	Mrelation="default",
	title="",
	fig_filespec="",
	fig_width=0,
	dpi=300,
	ax=None):
	"""
	Plot depth statistics for different catalogs.

	:param catalogs:
		List containing instances of :class:`EQCatalog`
	:param labels:
		List containing catalog labels
		(default: [])
	:param dmin:
		float, minimum depth
		(default: 0)
	:param dmax:
		float, maximum depth
		(default: 30)
	:param Mtype:
		String, magnitude type for magnitude scaling (default: "MW")
	:param Mrelation:
		{str: str} dict, mapping name of magnitude conversion relation
		to magnitude type ("MW", "MS" or "ML")
		(default: "default", will select the default relation for the
		given Mtype)
	:param title:
		String, plot title (default: None)
	:param fig_filespec:
		String, full path of image to be saved.
		If None (default), map is displayed on screen.
	:param fig_width:
		Float, figure width in cm, used to recompute :param:`dpi` with
		respect to default figure width
		(default: 0)
	:param dpi:
		Int, image resolution in dots per inch
		(default: 300)
	:param ax:
		matplotlib Axes instance
		(default: None)
	"""
	if ax is None:
		ax = pylab.axes()
	else:
		fig_filespec = "hold"

	if not labels:
		labels = [catalog.name for catalog in catalogs]
	for i, catalog in enumerate(catalogs):
		if Mmin != None:
			catalog = catalog.subselect(Mmin=Mmin, Mtype=Mtype, Mrelation=Mrelation)
		depths = catalog.get_depths()

		ax.boxplot(depths, positions=[i], widths=0.15)
		ax.plot([i], [np.nanmean(depths)], marker='d', mfc='g', ms=8)
		ax.plot([i], [np.percentile(depths, 2.5)], marker='v', mfc='g', ms=8)
		ax.plot([i], [np.percentile(depths, 97.5)], marker='^', mfc='g', ms=8)

	ax.set_ylim(dmin, dmax)
	ax.invert_yaxis()
	ax.set_xlim(-0.5, len(catalogs) - 0.5)
	ax.set_xticks(np.arange(len(catalogs)))
	ax.set_xticklabels(labels)
	ax.set_xlabel("Catalog", fontsize="x-large")
	ax.set_ylabel("Depth (km)", fontsize="x-large")
	if title is None:
		title = 'Depth statistics  (M>=%.1f)' % Mmin
	ax.set_title(title)

	if fig_filespec == "hold":
		return
	elif fig_filespec:
		default_figsize = pylab.rcParams['figure.figsize']
		default_dpi = pylab.rcParams['figure.dpi']
		if fig_width:
			fig_width /= 2.54
			dpi = dpi * (fig_width / default_figsize[0])
		pylab.savefig(fig_filespec, dpi=dpi)
		pylab.clf()
	else:
		pylab.show()


# TODO: revise the following two functions


ZoneModelTables =	{"leynaud": "ROB Seismic Source Model (Leynaud, 2000)",
						"leynaud_updated": "Leynaud updated",
						"slz+rvg": "SLZ+RVG",
						"slz+rvg_split": "SLZ+RVG_split",
						"seismotectonic": "seismotectonic zones 1.2",
						"rvrs": "RVRS"}

def read_catalogMI(tabname="KSB-ORB_catalog", region=None, start_date=None, end_date=None, Mmin=None, Mmax=None, zone_model=None, zone_names=[], verbose=False):
	"""
	Read ROB local earthquake catalog through a running instance of MapInfo.
	Optional parameters:
		tabname: name of MapInfo table containing catalog, defaults to "KSB_ORB_Catalog".
			This table must be open in MapInfo.
		region: (w, e, s, n) tuple specifying rectangular region of interest
		start_date: date (date object or year) specifying start of time window of interest
		end_date: date (date object or year) specifying end of time window of interest
		Mmin: minimum magnitude to extract
		Mmax: minimum magnitude to extract
		zone_model: name of zone model to use for splitting catalog into subcollections, defaults to None
			The table corresponding to the zone model must also be open in MapInfo.
			Available zone models: "Leynaud", "SLZ+RVG", and "Seismotectonic"
		zone_names: list with names of zones to extract if a zone model is specified, defaults to [],
			which will extract all zones.
		verbose: bool, if True the query string will be echoed to standard output
	Magnitude for selection is based on ML only!
	Return value:
		EQCatalog object, or dictionary of EQCatalog objects if a zone model is specified.
	"""
	import mapping.MIPython as MI
	app = MI.Application(maximize=False)
	if type(start_date) == type(0):
		start_date = datetime.date(start_date, 1, 1)
	elif isinstance(start_date, datetime.datetime):
		start_date = start_date.date()
	if type(end_date) == type(0):
		end_date = datetime.date(end_date, 12, 31)
	elif isinstance(end_date, datetime.datetime):
		end_date = end_date.date()
	if not end_date:
		end_date = datetime.datetime.now().date()
	if not start_date:
		start_date = datetime.date(100, 1, 1)

	if zone_model:
		zonetab_name = ZoneModelTables[zone_model.lower()]
		zones = read_zonesMI(zonetab_name)
		if not zone_names:
			zone_names = zones.keys()
	else:
		zone_names = []

	tab_filespec = os.path.join(GIS_root, "KSB-ORB", os.path.splitext(tabname)[0] + ".TAB")
	tab = app.OpenTable(tab_filespec)
	tabname = tab.GetName()

	queries = []
	query = 'Select id_earth, date, time, longitude, latitude, depth, ML, MS, MW, name from %s' % tabname
	query += ' Where type = "ke" and is_true = 1'
	if start_date:
		query += ' and date >= "%02d/%02d/%04d"' % (start_date.day, start_date.month, start_date.year)
	if end_date:
		query += ' and date <= "%02d/%02d/%04d"' % (end_date.day, end_date.month, end_date.year)
	if region:
		query += ' and longitude >= %f and longitude <= %f and latitude >= %f and latitude <= %f' % region
	if Mmin:
		query += ' and ML >= %f' % Mmin
	if Mmax:
		query += ' and ML <= %f' % Mmax
	if zone_names:
		for zone_name in zone_names:
			queries.append(query + ' and obj Within Any(Select obj from %s where Name = "%s") into CatalogQuery Order By date,time asc noselect' % (app.GetTableName(zonetab_name), zone_name))
	else:
		queries.append(query + ' into CatalogQuery Order By date,time asc noselect')
		zone_names = ["ROB Catalog"]

	zone_catalogs = OrderedDict()
	for zone_name, query in zip(zone_names, queries):
		## Important: we loop over zone_names instead of zones, because the former is a function
		## argument, and can be specified by the caller
		name = "%s %s - %s" % (zone_name, start_date.isoformat(), end_date.isoformat())
		if verbose:
			print query
		app.Do(query)
		catalog_table = app.GetTable("CatalogQuery")
		catalog = []
		col_info = catalog_table.GetColumnInfo()
		for rec in catalog_table:
			values = rec.GetValues(col_info)
			h, m, s = [int(s) for s in values["time"].split(':')]
			time = datetime.time(h, m, s)
			eq = LocalEarthquake(values["id_earth"], values["date"], time, values["longitude"], values["latitude"], values["depth"], {}, values["ML"], values["MS"], values["MW"], values["name"])
			catalog.append(eq)
		catalog = EQCatalog(catalog, start_date, end_date, name=name)
		try:
			zone = zones[zone_name]
		except:
			pass
		else:
			catalog.Mmax_evaluated = float(zone.MS_max_evaluated)
		if zone_model:
			catalog.ID = int(zone.ID)
		zone_catalogs[zone_name] = catalog

	tab.Close()

	if not zone_model:
		return zone_catalogs.values()[0]
	else:
		return zone_catalogs


def read_zonesMI(tabname):
	"""
	Read zone model through MapInfo.
	Required parameter:
		tabname: name of MapInfo table containing zone model, must be open
	Return value:
		dictionary of shapely polygons
	"""
	import mapping.MIPython as MI
	app = MI.Application(maximize=False)
	coordsys = app.GetCurrentCoordsys()
	app.SetCoordsys(MI.Coordsys(1,0))
	zonetab = app.GetTable(tabname)
	zones = OrderedDict()
	for rec in zonetab:
		zone = DummyClass()
		geo = rec.GetGeography()
		for key, val in rec.GetValues().items():
			exec('zone.%s = "%s"' % (key, val))
		zone.geo = geo
		zones[rec.Name] = zone
	app.SetCoordsys(coordsys)
	return zones


def read_zonesTXT(filespec, fixed_depth=None):
	zones = OrderedDict()
	f = open(filespec)
	start_zone = True
	for line in f:
		if start_zone:
			name = line.strip()
			coords = []
			start_zone = 0
		elif line.strip() == "0":
			zones[name] = coords
			start_zone = 1
		else:
			try:
				x, y, z = line.split(',')
			except:
				pass
			else:
				x, y, z = float(x), float(y), float(z)
				if fixed_depth != None:
					z = fixed_depth
				coords.append((x, y, z))
	return zones


### The following functions are obsolete or have moved to other modules

def format_zones_CRISIS(zone_model, Mc=3.5, smooth=False, fixed_depth=None):
	import mapping.MIPython as MI
	app = MI.Application(maximize=False)
	coordsys = app.GetCurrentCoordsys()
	app.SetCoordsys(MI.Coordsys(1,0))
	zonetab = app.GetTable(ZoneModelTables[zone_model.lower()])

	if smooth:
		cmd = 'Objects Snap From %s Thin Bend 1.5 Distance 5 Units "km"' % zonetab.name
		app.Do(cmd)

	for i, rec in enumerate(zonetab):
		if rec.MS_max_evaluated >= Mc:
			## Sometimes smoothing results in more than one segment, so we take the longest one
			segments = rec.GetGeography()
			num_pts = 0
			for j in range(len(segments)):
				segment = segments[j]
				if len(segment) > num_pts:
					num_pts = len(segment)
					j_longest = j
			coords = rec.GetGeography()[j_longest]
			if coords[0] == coords[-1]:
				coords = coords[:-1]
			if rec.IsClockwise():
				coords.reverse()

			if fixed_depth == None:
				depth = rec.Source_Depth
			else:
				depth = fixed_depth

			if zone_model.lower() == "leynaud":
				alpha, beta, lambda0 = alphabetalambda(rec.a, rec.b, Mc)
			else:
				alpha, beta, lambda0 = alphabetalambda(rec.aMLEfixadj, rec.bMLEfix, Mc)

			print "%s" % rec.Name
			print "1,0,1,1"
			print " %d" % len(coords)
			for pt in coords:
				print "%f,%f,%.1f" % (pt.x, pt.y, depth)
			print "%.3f,%.3f,0,%.1f,0,%.1f,%.1f" % (lambda0, beta, rec.MS_max_evaluated, rec.MS_max_observed, Mc)
			print " 0"

	if smooth:
		zonetab.Revert()
	app.SetCoordsys(coordsys)


def distribute_avalues(zones, catalog, M=0):
	"""
	Distribute a values for a number of zones such that it is consistent with the a value
	for the complete catalog.
	"""
	N_tot = 0
	for zone in zones:
		N_tot += 10 ** (zone.a - zone.b * M)
	N_diff = 10 ** (catalog.a - catalog.b * M) - N_tot

	weights = []
	sum_weights = 0
	for zone in zones:
		weight = 1.0 - (zone.num_events * 1.0 / catalog.num_events)
		sum_weights += weight
		weights.append(weight)
	weights = [weight / sum_weights for weight in weights]

	for zone, weight in zip(zones, weights):
		zone.a_new = np.log10(10**zone.a + weight*N_diff)
		print "%s - a: %.3f  ->  %.3f" % (zone.name, zone.a, zone.a_new)

	N_zones = 0
	for zone in zones:
		N_zones += 10 ** zone.a_new
	N_diff = 10 ** catalog.a - N_zones


def Poisson(life_time=None, return_period=None, prob=None):
	"""
	Compute return period, life time or probability from any combination of two
		of the other parameters for a Poisson distribution
	Parameters:
		life_time: life time (default: None)
		return_period: return period (default: None)
		prob: probability (default: None)
	Two parameters need to be specified, the value will be computed for the remaining
		parameters
	"""
	if prob and prob > 1:
		prob /= 100.0
	if life_time:
		life_time = float(life_time)
	if return_period:
		return_period = float(return_period)
	if life_time and prob:
		return -life_time / np.log(1.0 - prob)
	elif return_period and prob:
		return -return_period * np.log(1.0 - prob)
	elif life_time and return_period:
		return 1.0 - np.exp(-life_time / return_period)
	else:
		raise TypeError("Need to specify 2 parameters")


def tr2tl(tr, prob):
	"""
	Convert return period to life time for a given probability
	Parameters:
		tr: return period
		prob: probability
	"""
	return -tr * np.log(1.0 - prob)


def tl2tr(tl, prob):
	"""
	Convert life time to return period for a given probability
	Parameters:
		tl: life time
		prob: probability
	"""
	return -tl / np.log(1.0 - prob)



if __name__ == "__main__":
	## Selection criteria
	start_date = datetime.date(1350, 1, 1)
	#end_date = datetime.date(2007, 12, 31)
	end_date = datetime.datetime.now().date()
	#region = (0, 8, 49, 52)
	region = (-1.25, 8.75, 49.15, 53.30)
	Mmin = 0.0
	Mmax = 7.0
	dM = 0.2
	completeness = Completeness_Rosset
	Mtype = "MS"
	Mrange = (1.5, 7.0)
	Freq_range = (1E-4, 10**1.25)
	Mc = 3.5

	## Read catalog from intranet database
	#region = (4.5, 4.65, 50.60, 50.70)
	#start_date = datetime.date(2008,7,12)
	catalog = read_catalogSQL(region=region, start_date=start_date, end_date=end_date)
	#catalog.plot_MagnitudeDate()

	## Bin magnitudes with verbose output
	#bins_N, bins_Mag, bins_Years, num_events, Mmax = catalog.bin_mag(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)

	## Plot magnitude histogram
	#catalog.plot_Mhistogram(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness)

	## Plot Magnitude/Frequency diagram
	#catalog.MagFreq(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)
	#fig_filespec = os.path.join(r"C:\PSHA\MagFreq", "MagFreq " + catalog.name + ".PNG")
	fig_filespec = None
	#catalog.plot_MagFreq(Mmin, Mmax, dM, discrete=True, Mtype=Mtype, completeness=completeness, verbose=True, Mrange=Mrange, Freq_range=Freq_range, want_exponential=True, fig_filespec=fig_filespec)

	## Plot histogram with number of events per year for a given magnitude range and year interval
	#catalog.plot_YearHistogram(1960, end_date.year, 1, 1.8, 3.0, Mtype=Mtype)

	## Plot histogram with number of events per hour of day for given magnitude range and year interval
	#catalog.plot_HourHistogram(Mmin=0.0, Mmax=2.5, Mtype=Mtype, start_year=1985, end_year=2007)

	## Export catalog to ZMAP format
	#catalog.export_ZMAP(r"C:\PSHA\MagFreq\ROB_Catalogue.dat")

	## Dump catalog to pickled file
	#catalog.pickle(r"Test\ROBcatalog.p")

	## Decluster
	catalog.decluster()

	## Analyse completeness with Stepp method
	#completeness = catalog.analyse_completeness_Stepp(ttol=0.1)
	#print completeness

	## Here we analyse hourly means for a section of the catalog
	"""
	start_year = 1985
	Mmin, Mmax, dM = 0.2, 3.6, 0.2
	means = []
	means_night = []
	means_day = []
	Mrange = my_arange(Mmin, Mmax, dM)[::-1]
	for M in Mrange:
		#catalog.plot_HourHistogram(Mmin=M, Mmax=7.0, Mtype=Mtype)
		mean, mean_day, mean_night = catalog.HourlyMean(Mmin=M, Mmax=7.0, Mtype=Mtype, start_year=start_year)
		means_night.append(mean_night)
		means_day.append(mean_day)
		means.append(mean)
		print Mmin, "%.1f" % mean_night, "%.1f" % (mean_night*24,)

	a, b = 2.731, 0.840
	num_years = end_date.year - start_year + 1
	expected = [10**(a - b*M)*num_years/24.0 for M in Mrange]
	#expected = np.array([17.6,26.0,38.3,56.6,83.6,123.4,182.2,269.0,397.1,586.3,865.6,1278.0,1886.8])
	pylab.plot(Mrange, means, 'r', label="Hourly mean (24h)")
	pylab.plot(Mrange, means_night, 'b', label="Hourly mean (night)")
	pylab.plot(Mrange, means_day, 'g', label="Hourly mean (day)")
	pylab.plot(Mrange, np.array(means_night)-np.array(means_day), 'k', label="Difference night-day")
	pylab.plot(Mrange, expected, 'm', label="Expected from G-R law")
	pylab.legend()
	pylab.xlabel("Magnitude (%s)" % Mtype)
	pylab.ylabel("Number of events")
	xmin, xmax, ymin, ymax = pylab.axis()
	pylab.axis((xmin, xmax, ymin, 80))
	pylab.title("Cumulative number of events per hour of day for period %d - %d" % (start_date.year, end_date.year))
	pylab.show()
	"""


	## Read catalog from MapInfo, and plot Magnitude/Frequency for entire catalog
	"""
	catalog = read_catalogMI(region=region, start_date=start_date, end_date=end_date, verbose=True)
	catalog.plot_MagFreq(Mmin, Mmax, dM, discrete=False, Mtype=Mtype, completeness=completeness, verbose=True)
	"""

	## Read catalog from pickled file and plot Magnitude/Frequency diagram
	#filespec = r"Test\ROBcatalog.p"
	#f = open(filespec, "r")
	#catalog = cPickle.load(f)
	#f.close()
	#catalog.plot_MagFreq(Mmin, Mmax, dM, discrete=True, Mtype=Mtype, completeness=completeness, verbose=True, Mrange=Mrange, Freq_range=Freq_range)

	## Report total seismic moment
	#historical_catalog = catalog.subselect(end_date=datetime.date(1909, 12, 31))
	#instrumental_catalog = catalog.subselect(start_date = datetime.date(1910, 1, 1))
	#M0_historical = historical_catalog.get_M0rate(Mrelation="geller")
	#M0_instrumental = instrumental_catalog.get_M0rate(Mrelation="hinzen")
	#M0_total = M0_historical + M0_instrumental
	#print "Total seismic moment: %.2E (historical) + %.2E (instrumental) = %.2E N.m" % (M0_historical, M0_instrumental, M0_total)
	#catalog.plot_CumulatedM0(ddate=10, ddate_spec="years", Mrelation=None)

	## Read full catalog from database, calculate a and b values, and store these for later use
	"""
	catalog = read_catalogSQL(region=region, start_date=start_date, end_date=end_date)
	cat = DummyClass()
	cat.a, cat.b, cat.beta, cat.stda, cat.stdb, cat.stdbeta = catalog.calcGR_mle(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)
	bins_N, bins_Mag, bins_Years, num_events, Mmax = catalog.bin_mag(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=False)
	cat.num_events = num_events
	"""

	## Read catalog from MapInfo, split according to source zone model, and generate magnitude/frequency plots
	"""
	zone_model = "Leynaud"
	zone_names = []
	catalogs = read_catalogMI(region=region, start_date=start_date, end_date=end_date, zone_model=zone_model, zone_names=zone_names, verbose=True)
	for catalog in catalogs.values():
		print catalog.name
		print "  Mmax: %.1f" % catalog.Mminmax()[1]
		if len(catalog) > 0:
			## Plot Mag/Freq for each zone and save figure to file
			dirname = os.path.join(r"C:\PSHA\MagFreq", zone_model)
			fig_filespec = os.path.join(dirname, "MagFreq " + catalog.name + ".PNG")
			try:
				catalog.plot_MagFreq(Mmin, Mmax, dM, discrete=False, Mtype=Mtype, completeness=completeness, verbose=True, fig_filespec=fig_filespec, Mrange=Mrange, Freq_range=Freq_range, fixed_beta=cat.beta)
			except:
				pass

			## Just print a and b values for each zone
			try:
				a, b, r = catalog.calcGR_lsq(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)
				a, b, beta, stda, stdb, stdbeta = catalog.calcGR_mle(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)
				BETA = cat.beta
				a, b, beta, stda, stdb, stdbeta = catalog.calcGR_mle(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True, beta=BETA)
			except:
				pass
			print
	"""

	## Read catalog from MapInfo, split according to source zone model, and calculate a and b values
	"""
	zone_model = "Seismotectonic"
	zone_names = []
	zones = []
	catalogs = read_catalogMI(region=region, start_date=start_date, end_date=end_date, zone_model=zone_model, zone_names=zone_names, verbose=True)
	for catalog in catalogs.values():
		print catalog.name
		print "  Mmax: %.1f" % catalog.Mminmax()[1]
		if len(catalog) > 0:

			## Just print a and b values for each zone
			BETA = cat.beta
			zone = DummyClass()
			zone.name = catalog.name
			try:
				a, b, r = catalog.calcGR_lsq(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)
				a, b, beta, stda, stdb, stdbeta = catalog.calcGR_mle(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)
				zone.a, zone.b, zone.beta, zone.stda, zone.stdb, zone.stdbeta = catalog.calcGR_mle(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True, beta=BETA)
				zone.num_events = catalog.bin_mag(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, trim=True, verbose=False)[3]
			except:
				pass
			else:
				zones.append(zone)
			print

	#M=completeness.get_completeness_magnitude(datetime.date.today())
	distribute_avalues(zones, cat, Mc)

	for zone in zones:
		print zone.name
		if "Single Large Zone" in zone.name:
			for a in split_avalues(zone.a_new, [0.793, 0.207]):
				alpha, beta, lambda0 = alphabetalambda(a, zone.b, Mc)
				print "%.3f  %.3f" % (lambda0, beta)
		else:
			alpha, beta, lambda0 = alphabetalambda(zone.a_new, zone.b, Mc)
			print "%.3f  %.3f" % (lambda0, beta)
	"""


	## Format zone model info for use in CRISIS
	"""
	zone_model = "Leynaud_updated"
	format_zones_CRISIS(zone_model, fixed_depth=3.5, smooth=False)
	"""


	## Calculate alpha, beta, lambda for different Mc
	"""
	for a in [2.4, 1.5, 1.7, 2.0, 2.3, 1.7, 1.4, 1.4, 1.5]:
		print "%.3f  %.3f" % tuple(alphabetalambda(a, 0.87, 2.0)[1:])

	print alphabetalambda(1.4, 0.87, 3.0)
	"""

