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
import os
import platform
import datetime
import cPickle
from collections import OrderedDict


## Import third-party modules
## Kludge because matplotlib is broken on seissrv3. Sigh...
import numpy as np
if platform.uname()[1] == "seissrv3":
	import matplotlib
	matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, MaxNLocator
from scipy import stats
import ogr
import osr


## Import ROB modules
import seismodb
import mfd
from thirdparty.recipes.my_arange import *



GIS_root = r"D:\GIS-data"

ZoneModelTables =	{"leynaud": "ROB Seismic Source Model (Leynaud, 2000)",
						"leynaud_updated": "Leynaud updated",
						"slz+rvg": "SLZ+RVG",
						"slz+rvg_split": "SLZ+RVG_split",
						"seismotectonic": "seismotectonic zones 1.2",
						"rvrs": "RVRS"}


class Completeness:
	"""
	Class defining completeness of earthquake catalog.

	:param min_years:
		list or array containing initial years of completeness, in
		chronological order (= from small to large)
	:param min_mags:
		list or array with corresponding lower magnitude for which
		catalog is assumed to be complete
	"""
	def __init__(self, min_years, min_mags):
		self.min_years = np.array(min_years)
		self.min_mags = np.array(min_mags)
		## Make sure ordering is chronologcal
		if self.min_years[0] > self.min_years[1]:
			self.min_years = self.min_years[::-1]
			self.min_mags = self.min_mags[::-1]

	def __len__(self):
		return len(self.min_years)

	def __str__(self):
		s = "\n".join(["%d, %.2f" % (year, mag) for (year, mag) in zip(self.min_years, self.min_mags)])
		return s

	@property
	def start_year(self):
		return self.min_years.min()

	def get_completeness_magnitude(self, year):
		"""
		Return completeness magnitude for given year, this is the lowest
		magnitude for which the catalog is complete.

		:param year:
			Int, year

		:return:
			Float, completeness magnitude
		"""
		try:
			index = np.where(year >= self.min_years)[0][-1]
		except IndexError:
			return None
		else:
			return self.min_mags[index]

	def get_completeness_year(self, M):
		"""
		Return initial year of completeness for given magnitude

		:param M:
			Float, magnitude

		:return:
			Int, initial year of completeness for given magnitude
		"""
		try:
			index = np.where(M >= self.min_mags)[0][0]
		except:
			return None
		else:
			return self.min_years[index]

	def to_table(self, Mmax=None):
		"""
		Convert to a 2-D completeness table
		"""
		n = len(self)
		if Mmax and Mmax > self.min_mags.max():
			n += 1
		table = np.zeros((n, 2), 'f')
		table[:len(self),0] = self.min_years[::-1]
		table[:len(self),1] = self.min_mags[::-1]
		if Mmax and Mmax > self.min_mags.max():
			table[-1,0] = self.min_years.min()
			table[-1,1] = Mmax
		return table


## NOTE: I think threshold magnitudes should be a multiple of dM (or dM/2)!
Completeness_Leynaud = Completeness([1350, 1911, 1985], [4.7, 3.3, 1.8])
#Completeness_Leynaud = Completeness([1350, 1911, 1985], [4.75, 3.25, 1.75])
Completeness_Rosset = Completeness([1350, 1926, 1960, 1985], [5.0, 4.0, 3.0, 1.8])
Completeness_Rosset_MW = Completeness([1350, 1926, 1960, 1985], [5.2, 4.5, 3.9, 3.1])


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
	:param name:
		String, catalog name (default: "")
	"""
	def __init__(self, eq_list, start_date=None, end_date=None, name=""):
		self.eq_list = eq_list[:]
		Tmin, Tmax = self.Tminmax()
		self.start_date = start_date
		if not start_date:
			self.start_date = Tmin
		self.end_date = end_date
		if not end_date:
			self.end_date = Tmax
		if isinstance(self.start_date, datetime.datetime):
			self.start_date = self.start_date.date()
		if isinstance(self.end_date, datetime.datetime):
			self.end_date = self.end_date.date()
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
		if isinstance(item, int):
			return self.eq_list.__getitem__(item)
		elif isinstance(item, slice):
			return EQCatalog(self.eq_list.__getitem__(item), name=self.name + " %s" % item)
		elif isinstance(item, (list, np.ndarray)):
			eq_list = []
			for index in item:
				eq_list.append(self.eq_list[index])
			return EQCatalog(eq_list, name=self.name + " %s" % item)

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

	def timespan(self):
		"""
		Return total time span of catalog as number of years (not fractional).
		"""
		start_date, end_date = self.start_date, self.end_date
		catalog_length = (end_date.year - start_date.year) * 1.0 + 1
		return catalog_length

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

	def get_magnitudes(self, Mtype="MS", Mrelation=None):
		"""
		Return array of magnitudes for all earthquakes in catalog

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			1-D numpy float array, earthquake magnitudes
		"""
		if Mtype.upper() == "ML":
			Mags = [eq.get_ML(Mrelation=Mrelation) for eq in self]
		elif Mtype.upper() == "MS":
			Mags = [eq.get_MS(Mrelation=Mrelation) for eq in self]
		elif Mtype.upper() == "MW":
			Mags = [eq.get_MW(Mrelation=Mrelation) for eq in self]
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
		import mapping.geo.coordtrans as coordtrans
		lons, lats = self.get_longitudes(), self.get_latitudes()
		coord_list = zip(lons, lats)
		if proj == "lambert1972":
			return coordtrans.lonlat_to_lambert1972(coord_list)
		elif proj == "utm31N":
			return coordtrans.utm_to_lonlat(coord_list, proj)

	def Tminmax(self, Mmax=None, Mtype="MS", Mrelation=None):
		"""
		Return tuple with oldest date and youngest date in catalog.

		:param Mmax:
			Float, maximum magnitude. Useful to check completeness periods.
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
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

	def Mminmax(self, Mtype="MS", Mrelation=None):
		"""
		Return tuple with minimum and maximum magnitude in catalog.

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		"""
		Mags = self.get_magnitudes(Mtype=Mtype, Mrelation=Mrelation)
		return (Mags.min(), Mags.max())

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
		return self.lon_minmax() + self.lat_minmax()

	def get_Mmax(self, Mtype="MS", Mrelation=None):
		"""
		Compute maximum magnitude in catalog

		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MS")
		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			Float, maximum observed magnitude
		"""
		return self.get_magnitudes(Mtype, Mrelation).max()

	def get_M0(self, Mrelation={"MS": "bungum", "ML": "hinzen"}):
		"""
		Compute total seismic moment.

		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MS" or "ML")
			(default: {"MS": "bungum", "ML": "hinzen"})

		:return:
			Float, total seismic moment in N.m
		"""
		return np.add.reduce(np.array([eq.get_M0(Mrelation=Mrelation) for eq in self]))

	def get_M0rate(self, Mrelation="Hinzen"):
		"""
		Compute seismic moment rate.

		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MS" or "ML")
			(default: {"MS": "geller", "ML": "hinzen"})

		:return:
			Float, seismic moment rate in N.m/yr
		"""
		return self.get_M0(Mrelation=Mrelation) / self.timespan()

	def subselect(self, region=None, start_date=None, end_date=None, Mmin=None, Mmax=None, min_depth=None, max_depth=None, Mtype="MS", Mrelation=None):
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
		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)

		:return:
			instance of :class:`EQCatalog`
		"""
		## Set default parameters
		if region is None:
			region = self.get_region()
		if isinstance(start_date, int):
			start_date = datetime.date(start_date, 1, 1)
		elif isinstance(start_date, datetime.datetime):
			start_date = start_date.date()
		if isinstance(end_date, int):
			end_date = datetime.date(end_date, 12, 31)
		elif isinstance(end_date, datetime.datetime):
			end_date = end_date.date()
		if start_date is None:
			start_date = self.start_date
		if end_date is None:
			end_date = self.end_date
		if Mmin is None:
			Mmin = 0.
		if Mmax is None:
			Mmax = 10.
		if min_depth is None:
			min_depth = 0.
		if max_depth is None:
			max_depth = 700.

		eq_list = []
		for eq in self:
			if start_date <= eq.datetime.date() <= end_date:
				w, e, s, n = region
				if w <= eq.lon <= e:
					if s <= eq.lat <= n:
						M = eq.get_M(Mtype, Mrelation)
						if Mmin <= M <= Mmax:
							if min_depth <= eq.depth <= max_depth:
								eq_list.append(eq)

		return EQCatalog(eq_list, start_date=start_date, end_date=end_date, name=self.name + " (subselect)")

	def subselect_declustering(self, method, params=None, Mtype="MS", Mrelation=None, return_triggered_catalog=False):
		"""
		Subselect earthquakes in the catalog that conform with the specified
		declustering method and params.

		:param method:
			instance of :class:`DeclusteringMethod`
		:param params:
			dict, mapping name of params needed for method to their value
		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MS"). Note: some
			methods use params that are specified for a certain magnitude scale.
		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param return_triggered_catalog:
			Boolean, return also triggered catalog (default: False)

		:return:
			instance of :class:`EQCatalog`
			if return_triggered_catalog = False also an triggered catalog is
			returned as instance of :class:`EQCatalog`
		"""
		magnitudes = self.get_magnitudes(Mtype=Mtype, Mrelation=Mrelation)
		datetimes = np.array(self.get_datetimes())
		lons = self.get_longitudes()
		lats = self.get_latitudes()

		d_index = method.decluster(magnitudes, datetimes, lons, lats, **params)

		dc = self.__getitem__(np.where(d_index == 1)[0])
		tc = self.__getitem__(np.where(d_index == 0)[0])

		if return_triggered_catalog:
			return dc, tc
		else:
			return dc

	def subselect_completeness(self, completeness=Completeness_Rosset, Mtype="MS", Mrelation=None, verbose=True):
		"""
		Subselect earthquakes in the catalog that conform with the specified
		completeness criterion.

		:param completeness:
			instance of :class:`Completeness` (default: Completeness_Rosset)
		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MS")
		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param verbose:
			Bool, whether or not some info should be printed (default: True)

		:return:
			instance of :class:`EQCatalog`
		"""
		if completeness:
			start_date = datetime.date(min(completeness.min_years), 1, 1)
		else:
			start_date = self.start_date
		end_date = self.end_date

		## Select magnitudes according to completeness criteria
		if completeness:
			eq_list = []
			for eq in self.eq_list:
				M = eq.get_M(Mtype, Mrelation)
				if M >= completeness.get_completeness_magnitude(eq.datetime.year):
					eq_list.append(eq)
		else:
			eq_list = self.eq_list

		if verbose:
			print "Number of events constrained by completeness criteria: %d out of %d" % (len(eq_list), len(self.eq_list))

		return EQCatalog(eq_list, start_date=start_date, end_date=end_date, name=self.name + " (completeness-constrained)")

	def bin_mag(self, Mmin, Mmax, dM=0.2, Mtype="MS", Mrelation=None, completeness=None):
		"""
		Bin all earthquake magnitudes in catalog according to specified magnitude interval.

		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude interval
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes (default: None)

		#:param large_eq_correction: (M, corr_factor) tuple, with M the lower magnitude for which to apply corr_factor
		#	This is to correct the frequency of large earthquakes having a return period which is longer
		#	than the catalog length

		:return:
			Tuple (bins_N, bins_Mag)
			bins_N: array containing number of earthquakes for each magnitude interval
			bins_Mag: array containing lower magnitude of each interval
		"""
		## Set lower magnitude to lowermost threshold magnitude possible
		if completeness:
			Mmin = max(Mmin, completeness.get_completeness_magnitude(self.end_date.year))

		## Construct bins_Mag, including Mmax as right edge
		Mmin = np.floor(Mmin / dM) * dM
		Mmax = np.ceil(Mmax / dM) * dM
		num_bins = int((Mmax - Mmin) / dM) + 1
		bins_Mag = np.linspace(Mmin, Mmax, num_bins)

		## Select magnitudes according to completeness criteria
		if completeness:
			cc_catalog = self.subselect_completeness(completeness, Mtype, Mrelation)
			Mags = cc_catalog.get_magnitudes(Mtype, Mrelation)
		else:
			Mags = self.get_magnitudes(Mtype, Mrelation)
		num_events = len(Mags)

		## Compute number of earthquakes per magnitude bin
		bins_N, bins_Mag = np.histogram(Mags, bins_Mag)
		bins_Mag = bins_Mag[:-1]

		return bins_N, bins_Mag

	def get_completeness_years(self, magnitudes, completeness=Completeness_Rosset):
		"""
		Compute year of completeness for list of magnitudes

		:param magnitudes:
			list or numpy array, magnitudes
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes. If None, use start year of
			catalog (default: completeness_Rosset)

		:return:
			numpy float array, completeness years
		"""
		## Calculate year of completeness for each magnitude interval
		if completeness:
			completeness_years = []
			for M in magnitudes:
				start_year = max(self.start_date.year, completeness.get_completeness_year(M))
				completeness_years.append(start_year)
		else:
			print("Warning: no completeness object provided. Using catalog length!")
			completeness_years = [self.start_date.year] * len(magnitudes)
		completeness_years = np.array(completeness_years, 'f')
		return completeness_years

	def get_completeness_timespans(self, magnitudes, completeness=Completeness_Rosset):
		"""
		Compute completeness timespans for list of magnitudes

		:param magnitudes:
			list or numpy array, magnitudes
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes. If None, use start year of
			catalog (default: completeness_Rosset)

		:return:
			numpy float array, completeness timespans (fractional years)
		"""
		completeness_years = self.get_completeness_years(magnitudes, completeness)
		completeness_timespans = [((self.end_date - datetime.date(int(start_year),1,1)).days + 1) / 365.25 for start_year in completeness_years]
		return np.array(completeness_timespans)

	def get_incremental_MagFreq(self, Mmin, Mmax, dM=0.2, Mtype="MS", Mrelation=None, completeness=Completeness_Rosset, trim=False):
		"""
		Compute incremental magnitude-frequency distribution.

		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude interval
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes (default: Completeness_Rosset)
		:param trim:
			Bool, whether empty bins at start and end should be trimmed
			(default: False)

		:return:
			Tuple (bins_N_incremental, bins_Mag)
			bins_N_incremental: incremental annual occurrence rates
			bins_Mag: left edges of magnitude bins
		"""
		bins_N, bins_Mag = self.bin_mag(Mmin, Mmax, dM, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness)
		bins_timespans = self.get_completeness_timespans(bins_Mag, completeness)

		bins_N_incremental = bins_N / bins_timespans

		## Optionally, trim empty trailing intervals
		if trim:
			last_non_zero_index = np.where(bins_N > 0)[0][-1]
			bins_N_incremental = bins_N_incremental[:last_non_zero_index+1]
			bins_Mag = bins_Mag[:last_non_zero_index+1]

		return bins_N_incremental, bins_Mag

	def get_incremental_MFD(self, Mmin, Mmax, dM=0.2, Mtype="MS", Mrelation=None, completeness=Completeness_Rosset, trim=False):
		"""
		Compute incremental magnitude-frequency distribution.

		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude interval
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes (default: Completeness_Rosset)
		:param trim:
			Bool, whether empty bins at start and end should be trimmed
			(default: False)

		:return:
			instance of nhlib :class:`EvenlyDiscretizedMFD`
		"""
		bins_N_incremental, bins_Mag = self.get_incremental_MagFreq(Mmin, Mmax, dM, Mtype, Mrelation, completeness, trim)
		return mfd.EvenlyDiscretizedMFD(Mmin + dM/2, dM, list(bins_N_incremental), Mtype=Mtype)

	def get_cumulative_MagFreq(self, Mmin, Mmax, dM=0.2, Mtype="MS", Mrelation=None, completeness=Completeness_Rosset, trim=False):
		"""
		Compute cumulative magnitude-frequency distribution.

		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude interval
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes (default: Completeness_Rosset)
		:param trim:
			Bool, whether empty bins at start and end should be trimmed
			(default: False)

		:return:
			Tuple (bins_N_cumulative, bins_Mag)
			bins_N_incremental: cumulative annual occurrence rates
			bins_Mag: left edges of magnitude bins
		"""
		bins_N_incremental, bins_Mag = self.get_incremental_MagFreq(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, trim=trim)
		## Reverse arrays for calculating cumulative number of events
		bins_N_incremental = bins_N_incremental[::-1]
		bins_N_cumulative = np.add.accumulate(bins_N_incremental)
		return bins_N_cumulative[::-1], bins_Mag

	def bin_year(self, start_year, end_year, dYear, Mmin, Mmax, Mtype="MS", Mrelation=None):
		"""
		Bin earthquakes into year intervals

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
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
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

	def bin_hour(self, Mmin, Mmax, Mtype="MS", Mrelation=None, start_year=None, end_year=None):
		"""
		Bin earthquakes into hour intervals

		:param Mmin:
			Float, minimum magnitude (inclusive)
		:param Mmax:
			Float, maximum magnitude (inclusive)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
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

	def bin_depth(self, Mmin, Mmax, Mtype="MS", Mrelation=None, start_year=None, end_year=None, min_depth=0, max_depth=30, bin_width=2):
		"""
		Bin earthquakes into depth bins

		:param Mmin:
			Float, minimum magnitude (inclusive)
		:param Mmax:
			Float, maximum magnitude (inclusive)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param start_year:
			Int, lower year to bin (default: None)
		:param end_year:
			Int, upper year to bin (default: None)
		:param min_depth:
			Int, minimum depth in km (default: 0)
		:param max_depth:
			Int, maximum depth in km (default: 30)
		:param bin_width:
			Int, bin width in km (default: 2)

		:return:
			tuple (bins_N, bins_depth)
			bins_N: array containing number of earthquakes for each bin
			bins_depth: array containing lower depth value of each interval
		"""
		subcatalog = self.subselect(start_date=start_year, end_date=end_year, Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		depths = [eq.depth for eq in subcatalog if not eq.depth in (None, 0)]
		bins_depth = np.arange(min_depth, max_depth + bin_width, bin_width)
		bins_N, junk = np.histogram(depths, bins_depth)
		return bins_N, bins_depth[:-1]

	def bin_depth_by_M0(self, Mmin, Mmax, Mtype="MW", Mrelation=None, start_year=None, end_year=None, min_depth=0, max_depth=30, bin_width=2):
		"""
		Bin earthquake moments into depth bins

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
		:param start_year:
			Int, lower year to bin (default: None)
		:param end_year:
			Int, upper year to bin (default: None)
		:param min_depth:
			Int, minimum depth in km (default: 0)
		:param max_depth:
			Int, maximum depth in km (default: 30)
		:param bin_width:
			Int, bin width in km (default: 2)

		:return:
			tuple (bins_M0, bins_depth)
			bins_M0: array containing summed seismic moment in each bin
			bins_depth: array containing lower depth value of each interval
		"""
		bins_depth = np.arange(min_depth, max_depth + bin_width, bin_width)
		bins_M0 = np.zeros(len(bins_depth))
		subcatalog = self.subselect(start_date=start_year, end_date=end_year, Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		for eq in subcatalog:
			if eq.depth not in (None, 0):
				try:
					bin_id = np.where((bins_depth + bin_width) >= eq.depth)[0][0]
				except:
					## These are earthquakes that are deeper
					pass
				else:
					bins_M0[bin_id] += eq.get_M0(Mrelation=Mrelation)
		return bins_M0, bins_depth

	def plot_Mhistogram(self, Mmin, Mmax, dM=0.5, completeness=Completeness_Rosset, Mtype="MS", Mrelation=None, fig_filespec=None, verbose=False):
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
			and corresponding minimum magnitudes (default: Completeness_Rosset)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param fig_filespec:
			String, full path of image to be saved.
			If None (default), histogram is displayed on screen.
		:param verbose:
			Bool, whether or not to print binning information (default: False)
		"""
		bins_N, bins_Mag = self.bin_mag(Mmin, Mmax, dM, completeness=completeness, Mtype=Mtype, Mrelation=Mrelation, verbose=verbose)
		pylab.bar(bins_Mag, bins_N, width=dM)
		pylab.xlabel("Magnitude (%s)" % Mtype)
		pylab.ylabel("Number of events")
		pylab.title("%s (%d events)" % (self.name, num_events))

		if fig_filespec:
			pylab.savefig(fig_filespec)
			pylab.clf()
		else:
			pylab.show()

	def plot_CumulativeYearHistogram(self, start_year, end_year, dYear, Mmin, Mmax, Mtype="MS", Mrelation=None, major_ticks=10, minor_ticks=1, completeness_year=None, regression_range=[], lang="en"):
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
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
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

	def plot_CumulatedM0(self, start_date=None, end_date=None, ddate=10, ddate_spec="years", Mrelation=None, M0max=None, fig_filespec=None):
		if start_date == None:
			start_date = self.start_date
		if end_date == None:
			end_date = self.end_date

		if ddate_spec.lower()[:4] == "year":
			bins_Dates = my_arange(start_date.year, end_date.year+ddate, ddate, decimals=1)
			Dates = [eq.datetime.year for eq in self]
		elif ddate_spec.lower()[:3] == "day":
			bins_Dates = my_arange((end_date - start_date).days + 1)
			Dates = [(eq.datetime - start_dt).days + (eq.datetime - start_dt).seconds / 86400.0 for eq in self]
		bins_M0 = np.zeros(len(bins_Dates), 'd')
		start_dt = datetime.datetime.combine(start_date, datetime.time(0))
		M0 = np.zeros(len(self), 'd')
		if ddate_spec.lower()[:4] == "year":
			for j, eq in enumerate(self):
				i = list(eq.datetime.year < bins_Dates).index(True) - 1
				bins_M0[i] += eq.get_M0(Mrelation=Mrelation)
				M0[j] = eq.get_M0(Mrelation=Mrelation)
		elif ddate_spec.lower()[:3] == "day":
			for j, eq in enumerate(self):
				i = (eq.datetime.date() - start_date).days
				bins_M0[i] += eq.get_M0(Mrelation=Mrelation)
				M0[j] = eq.get_M0(Mrelation=Mrelation)
		bins_M0_cumul = np.add.accumulate(bins_M0)
		M0_cumul = np.add.accumulate(M0)

		#bins_M02 = np.array(zip(np.concatenate([np.zeros(1, 'd'), bins_M0[:-1]]), bins_M0)).flatten()
		bins_M0_cumul2 = np.array(zip(np.concatenate([np.zeros(1, 'd'), bins_M0_cumul[:-1]]), bins_M0_cumul)).flatten()
		M0_cumul2 = np.array(zip(np.concatenate([np.zeros(1, 'd'), M0_cumul[:-1]]), M0_cumul)).flatten()
		bins_Dates2 = np.array(zip(bins_Dates, bins_Dates)).flatten()
		Dates2 = np.array(zip(Dates, Dates)).flatten()

		pylab.plot(bins_Dates2, bins_M0_cumul2, 'r', label="Cumulative")
		pylab.plot(bins_Dates, bins_M0_cumul, 'ro', label=" ")
		pylab.plot(Dates2, M0_cumul2, 'g', label="Cumulative (unbinned)")
		#pylab.plot(Dates, M0_cumul, 'go', label="")
		font = FontProperties(size='small')
		pylab.legend(loc=0, prop=font)
		#pylab.plot(bins_Dates2, bins_M02, 'b', label=" ")
		#pylab.plot(bins_Dates, bins_M0, 'bo', label="By date bin")
		pylab.bar(bins_Dates, bins_M0, width=ddate)
		pylab.xlabel("Time (%s)" % ddate_spec)
		pylab.ylabel("Seismic Moment (N.m)")
		pylab.title(self.name)
		xmin, xmax, ymin, ymax = pylab.axis()
		if M0max:
			ymax = M0max
		pylab.axis((bins_Dates[0], bins_Dates[-1], ymin, ymax))

		if fig_filespec:
			pylab.savefig(fig_filespec)
			pylab.clf()
		else:
			pylab.show()

	def plot_DateHistogram(self, start_date=None, end_date=None, ddate=1, ddate_spec="year", mag_limits=[2,3], Mtype="ML", Mrelation=None):
		"""
		Plot histogram with number of earthqukes versus date.

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
			String, magnitude type: "ML", "MS" or "MW" (default: "ML")
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
			M = eq.get_M(Mtype, Mrelation)
			try:
				im = np.where(M < mag_limits)[0][0]
			except IndexError:
				im = -1
			if ddate_spec.lower()[:4] == "year":
				id = np.where(eq.datetime.year == bins_Dates)[0][0]
			elif ddate_spec.lower()[:3] == "day":
				id = (eq.datetime.date() - start_date).days
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

	def plot_Magnitude_Date(self, Mtype="MS", Mrelation=None):
		"""
		Plot magnitude versus date
		"""
		dates = self.get_datetimes()
		magnitudes = self.get_magnitudes(Mtype, Mrelation)

		days = pylab.DayLocator()
		weeks = pylab.WeekdayLocator()
		months = pylab.MonthLocator()

		fig = pylab.figure()
		ax = fig.add_subplot(111)
		pylab.plot_date(pylab.date2num(dates), magnitudes)
		pylab.xlabel("Date")
		pylab.ylabel("Magnitude")
		ax.xaxis.set_major_locator(weeks)
		ax.xaxis.set_minor_locator(days)
		for label in ax.get_xticklabels():
			label.set_horizontalalignment('right')
			label.set_rotation(30)
		#ax.set(labels, rotation=30, fontsize=10)
		pylab.show()

	def plot_Magnitude_Time(self, Mtype="MS", Mrelation=None, lang="en"):
		"""
		Plot magnitude versus time

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
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

	def plot_Time_Magnitude(self, Mtype="MS", Mrelation=None, triggered_catalog=None, completeness=None, color="k", label=None, vlines=False, grid=False, major_ticks=None, minor_ticks=1, title="", lang="en"):
		"""
		Plot time versus magnitude

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param triggered_catalog:
			class:`EQCatalog` instance, plot triggered earthquakes (default: None)
		:param completeness:
			class:`Completeness` instance, plot completeness (default: None)
		:param color:
			Str, color to plot data points with (default: "k")
		:param label:
			Str, label of data points (default: None)
		:param vlines:
			Boolean, plot vertical lines from data point to x-axis (default: False)
		:param grid:
			Boolean, plot grid (default: False)
		:param major_ticks:
			Int, interval in years for major ticks (default: None). If none, a
			maximum number of ticks at nice locations will be used.
		:param minor_ticks:
			Int, interval in years for minor ticks (default: 1)
		:param title:
			Str, title of plot (default: "")
		:param lang:
			String, language of plot labels (default: "en")
		"""
		x = self.get_fractional_years()
		y = self.get_magnitudes(Mtype, Mrelation)
		plt.scatter(x, y, s=50, color=color, label=label, marker="s", facecolors='none')
		xmin, xmax, ymin, ymax = plt.axis()
		xmin, xmax = self.start_date.year, self.end_date.year+1
		plt.axis((xmin, xmax, 0, max(y)*1.1))

		## plot vlines
		if vlines:
			plt.vlines(x, ymin=ymin, ymax=y)

		## plot ticks
		if major_ticks:
			majorLocator = MultipleLocator(major_ticks)
		else:
			majorLocator = MaxNLocator()
		minorLocator = MultipleLocator(minor_ticks)
		ax = plt.gca()
		ax.xaxis.set_major_locator(majorLocator)
		ax.xaxis.set_minor_locator(minorLocator)
		ax.yaxis.set_minor_locator(MultipleLocator(0.1))

		## plot labels
		plt.xlabel({"en": "Time (years)", "nl": "Tijd (jaar)"}[lang])
		plt.ylabel("Magnitude (%s)" % Mtype)

		## plot declustering
		if triggered_catalog:
			x = triggered_catalog.get_fractional_years()
			y = triggered_catalog.get_magnitudes(Mtype, Mrelation)
			plt.scatter(x, y, s=50, color="r", marker="s", facecolors='none')

		## plot completeness
		if completeness:
			x, y = completeness.min_years, completeness.min_mags
			x = np.append(x, xmax)
			plt.hlines(y, xmin=x[:-1], xmax=x[1:], colors='r')
			plt.vlines(x[1:-1], ymin=y[1:], ymax=y[:-1], colors='r')

		if grid:
			plt.grid()

		plt.title(title)
		if label:
			plt.legend()
		plt.show()

	def HourlyMean(self, Mmin, Mmax, Mtype="MS", Mrelation=None, start_year=None, end_year=None, day=(10, 17), night=(19, 7)):
		bins_N, bins_Hr = self.bin_hour(Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation, start_year=start_year, end_year=end_year)
		mean = np.mean(bins_N)
		mean_day = np.mean(bins_N[day[0]:day[1]])
		mean_night = np.mean(np.concatenate((bins_N[:night[1]], bins_N[night[0]-24:])))
		return (mean, mean_day, mean_night)

	def plot_HourHistogram(self, Mmin, Mmax, Mtype="MS", Mrelation=None, start_year=None, end_year=None):
		"""
		Plot histogram with number of earthquakes per hour of the day.

		:param Mmin:
			Float, minimum magnitude (inclusive)
		:param Mmax:
			Float, maximum magnitude (inclusive)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
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
		pylab.xlabel("Hour of day")
		pylab.ylabel("Number of events")
		if not start_year:
			start_year = self.start_date.year
		if not end_year:
			end_year = self.end_date.year
		pylab.title("Hourly Histogram %d - %d, M %.1f - %.1f" % (start_year, end_year, Mmin, Mmax))
		pylab.show()

	def plot_DepthHistogram(self, Mmin, Mmax, Mtype="MS", start_year=None, end_year=None, min_depth=0, max_depth=30, bin_width=2, color='b', want_title=True, fig_filespec="", fig_width=0, dpi=300):
		bins_N, bins_depth = self.bin_depth(Mmin, Mmax, Mtype, start_year, end_year, min_depth, max_depth, bin_width)
		pylab.bar(bins_depth, bins_N, orientation="vertical", width=bin_width, color=color)
		xmin, xmax, ymin, ymax = pylab.axis()
		pylab.axis((min_depth, max_depth, ymin, ymax))
		pylab.xlabel("Depth (km)", fontsize='x-large')
		pylab.ylabel("Number of events", fontsize='x-large')
		ax = pylab.gca()
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('x-large')
		if not start_year:
			start_year = self.start_date.year
		if not end_year:
			end_year = self.end_date.year
		if want_title:
			pylab.title("Depth Histogram %d - %d, M %.1f - %.1f" % (start_year, end_year, Mmin, Mmax))

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

	def plot_Depth_M0_Histogram(self, Mmin, Mmax, Mtype="MS", start_year=None, end_year=None, min_depth=0, max_depth=30, bin_width=2, color='b', want_title=True, log=True, fig_filespec="", fig_width=0, dpi=300):
		bins_M0, bins_depth = self.bin_depth_by_M0(Mmin, Mmax, Mtype, start_year, end_year, min_depth, max_depth, bin_width)
		pylab.bar(bins_depth, bins_M0, orientation="vertical", width=bin_width, log=log, color=color)
		xmin, xmax, ymin, ymax = pylab.axis()
		pylab.axis((min_depth, max_depth, ymin, ymax))
		pylab.xlabel("Depth (km)", fontsize='x-large')
		pylab.ylabel("Cumulated seismic moment (N.m)", fontsize='x-large')
		ax = pylab.gca()
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('x-large')
		if not start_year:
			start_year = self.start_date.year
		if not end_year:
			end_year = self.end_date.year
		if want_title:
			pylab.title("Depth Histogram %d - %d, M %.1f - %.1f" % (start_year, end_year, Mmin, Mmax))

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

	def plot_map(self, region=None, symbol='o', edge_color='r', fill_color=None, symbol_size=10, symbol_size_inc=3, Mtype="MW", Mrelation=None, dlon=1., dlat=1., fig_filespec=None):
		from mpl_toolkits.basemap import Basemap
		if edge_color is None:
			edge_color = "None"
		if fill_color is None:
			fill_color = "None"
		if not region:
			region = list(self.get_region())
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

		map = Basemap(projection="cyl", resolution="i", llcrnrlon=region[0], llcrnrlat=region[2], urcrnrlon=region[1], urcrnrlat=region[3], lon_0=lon_0, lat_0=lat_0)
		map.drawcoastlines()
		map.drawcountries()
		first_meridian = numpy.ceil(region[0] / dlon) * dlon
		last_meridian = numpy.floor(region[1] / dlon) * dlon + dlon
		meridians = numpy.arange(first_meridian, last_meridian, dlon)
		map.drawmeridians(meridians, labels=[0,1,0,1])
		first_parallel = numpy.ceil(region[2] / dlat) * dlat
		last_parallel = numpy.floor(region[3] / dlat) * dlat + dlat
		parallels = numpy.arange(first_parallel, last_parallel, dlat)
		map.drawparallels(parallels, labels=[0,1,0,1])
		if not symbol_size_inc:
			symbol_sizes = symbol_size ** 2
		else:
			magnitudes = self.get_magnitudes(Mtype, Mrelation)
			symbol_sizes = symbol_size + (magnitudes - 3.0) * symbol_size_inc
			symbol_sizes = symbol_sizes ** 2
			print symbol_sizes.min(), symbol_sizes.max()
		map.scatter(self.get_longitudes(), self.get_latitudes(), s=symbol_sizes, marker=symbol, edgecolors=edge_color, facecolors=fill_color)
		map.drawmapboundary()

	def calcGR_LSQ(self, Mmin, Mmax, dM=0.2, Mtype="MS", completeness=Completeness_Rosset, verbose=False):
		"""
		Calculate a and b values of Gutenberg-Richter relation using a linear regression (least-squares).
		Parameters:
			Required:
				Mmin: minimum magnitude to use for binning
				Mmax: maximum magnitude to use for binning
				dM: magnitude interval to use for binning
			Optional:
				Mtype: magnitude type ("ML", "MS" or "MW"), defaults to "MS"
				completeness: Completeness object with initial years of completeness and corresponding
					minimum magnitudes, defaults to Completeness_Leynaud
				verbose: boolean indicating whether some messages should be printed or not, defaults to False
		Return value:
			(a, b, r) tuple
			a: a value (intercept)
			b: b value (slope, taken positive)
			r: correlation coefficient
		"""
		# TODO: constrained regression with fixed b
		# TODO: see also numpy.linalg.lstsq
		bins_N_cumul_log, bins_N_disc_log, bins_Mag, bins_Years, num_events, Mmax_obs = self.LogMagFreq(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=False)
		b, a, r, ttprob, stderr = stats.linregress(bins_Mag, bins_N_cumul_log)
		## stderr = standard error on b?
		if verbose:
			print "Linear regression: a=%.3f, b=%.3f (r=%.2f)" % (a, -b, r)
		return (a, -b, r)

	def calcGR_Aki(self, Mmin=None, Mmax=None, dM=0.1, Mtype="MS", Mrelation=None, completeness=Completeness_Rosset, b_val=None, verbose=False):
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
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` (default: Completeness_Rosset)
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

	def calcGR_Weichert(self, Mmin, Mmax, dM=0.1, Mtype="MS", Mrelation=None, completeness=Completeness_Rosset, b_val=None, verbose=False):
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
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` (default: Completeness_Rosset)
		:param b_val:
			Float, fixed b value to constrain MLE estimation (default: None)
		:param verbose:
			Bool, whether some messages should be printed or not (default: False)

		:return:
			Tuple (a, b, stdb)
			- a: a value
			- b: b value
			- stdb: standard deviation on b value

		Note:
		This regression depends on the Mmax specified, as empty magnitude bins
		are taken into account. It is therefore important to specify Mmax as
		the evaluated Mmax for the specific region or source.
		"""
		bins_N, bins_Mag = self.bin_mag(Mmin, Mmax, dM, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness)
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

	#TODO: averaged Weichert method

	def get_estimated_MFD(self, Mmin, Mmax, dM=0.1, method="Weichert", Mtype="MS", Mrelation=None, completeness=Completeness_Rosset, b_val=None, verbose=False):
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
			String, computation method, either "Weichert", "Aki" or "LSQ"
			(default: "Weichert")
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` (default: Completeness_Rosset)
		:param b_val:
			Float, fixed b value to constrain MLE estimation
			Currently only supported by Weichert method (default: None)
		:param verbose:
			Bool, whether some messages should be printed or not (default: False)

		:return:
			instance of :class:`mfd.TruncatedGRMFD`
		"""
		calcGR_func = {"Weichert": self.calcGR_Weichert, "Aki": self.calcGR_Aki, "LSQ": self.calcGR_LSQ}[method]
		a, b, stdb = calcGR_func(Mmin=Mmin, Mmax=Mmax, dM=dM, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, b_val=b_val, verbose=verbose)
		return mfd.TruncatedGRMFD(Mmin, Mmax, dM, a, b, stdb, Mtype)

	def plot_MFD(self, Mmin, Mmax, dM=0.2, method="Weichert", Mtype="MS", Mrelation=None, completeness=Completeness_Rosset, b_val=None, num_sigma=0, color_observed="b", color_estimated="r", plot_completeness_limits=True, Mrange=(), Freq_range=(), title="", lang="en", fig_filespec=None, fig_width=0, dpi=300, verbose=False):
		"""
		Compute GR MFD from observed MFD, and plot result

		:param Mmin:
			Float, minimum magnitude to use for binning
		:param Mmax:
			Float, maximum magnitude to use for binning
		:param dM:
			Float, magnitude interval to use for binning (default: 0.1)
		:param method:
			String, computation method, either "Weichert", "MLE" or "LSQ"
			(default: "Weichert")
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` (default: Completeness_Rosset)
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
			String, plot title (default: "")
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
			if num_sigma and method == "Weichert":
				sigma_mfd1 = cc_catalog.get_estimated_MFD(Mmin, Mmax, dM=dM, method=method, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, b_val=b+num_sigma*stdb, verbose=verbose)
				mfd_list.append(sigma_mfd1)
				label = {"en": "Computed", "nl": "Berekend"}[lang]
				label += " $\pm$ %d sigma" % num_sigma
				labels.append(label)
				colors.append(color_estimated)
				styles.append(':')
				sigma_mfd2 = cc_catalog.get_estimated_MFD(Mmin, Mmax, dM=dM, method=method, Mtype=Mtype, Mrelation=Mrelation, completeness=completeness, b_val=b-num_sigma*stdb, verbose=verbose)
				mfd_list.append(sigma_mfd2)
				labels.append("_nolegend_")
				colors.append(color_estimated)
				styles.append('--')

		if not title:
			num_events = len(cc_catalog)
			Mmax_obs = cc_catalog.get_Mmax(Mtype, Mrelation)
			title = "%s (%d events, Mmax=%.2f)" % (self.name, num_events, Mmax_obs)
		completeness_limits = {True: completeness, False: None}[plot_completeness_limits]
		end_year = self.end_date.year
		mfd.plot_MFD(mfd_list, colors=colors, styles=styles, labels=labels, completeness=completeness_limits, end_year=end_year, Mrange=Mrange, Freq_range=Freq_range, title=title, lang=lang, fig_filespec=fig_filespec, fig_width=fig_width, dpi=dpi)

	def export_ZMAP(self, filespec, Mtype="MS", Mrelation=None):
		"""
		Export earthquake list to ZMAP format (ETH Zürich).
		Parameters:
			Required:
				filespec: full path specification of output file
			Optional:
				Mtype: magnitude type ("ML", "MS" or "MW"), defaults to "MS"
		"""
		f = open(filespec, "w")
		for eq in self.eq_list:
			M = self.get_M(Mtype, Mrelation)
			f.write("%f  %f  %d  %d  %d  %.1f %.2f %d %d\n" % (eq.lon, eq.lat, eq.datetime.year, eq.datetime.month, eq.datetime.day, M, eq.depth, eq.datetime.hour, eq.datetime.minute))
		f.close()

	def export_csv(self, csv_filespec):
		"""
		Export earthquake list to a csv file.

		:param csv_filespec:
			String, full path specification of output csv file.
		"""
		f = open(csv_filespec, "w")
		f.write('ID,Date,Time,Name,Longitude,Latitude,Depth,ML,MS,MW,MS (converted),Intensity_max,Macro_radius\n')
		for eq in self.eq_list:
			date = eq.datetime.date().isoformat()
			time = eq.datetime.time().isoformat()
			f.write('%d,"%s","%s","%s",%.3f,%.3f,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%d\n' % (eq.ID, date, time, eq.name, eq.lon, eq.lat, eq.depth, eq.ML, eq.MS, eq.MW, eq.get_MS(), eq.intensity_max, eq.macro_radius))
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
		"""
		import mapping.kml.mykml as mykml

		kmldoc = mykml.KML()
		#year, month, day = self.start_date.year, self.start_date.month, self.start_date.day
		#start_time = datetime.datetime(year, month, day)
		start_time = datetime.datetime.now()
		kmldoc.addTimeStamp(start_time)

		if time_folders:
			hist_folder = kmldoc.addFolder("Historical", visible=False, open=False)
			inst_folder = kmldoc.addFolder("Instrumental", visible=True, open=False)

			folder_24h = kmldoc.createFolder("Past 24 hours", visible=True, open=False)
			inst_folder.appendChild(folder_24h)
			folder_2w = kmldoc.createFolder("Past 2 weeks", visible=True, open=False)
			inst_folder.appendChild(folder_2w)
			folder_lastyear = kmldoc.createFolder("Past year", visible=True, open=False)
			inst_folder.appendChild(folder_lastyear)
			folder_2000 = kmldoc.createFolder("2000 -", visible=True, open=False)
			inst_folder.appendChild(folder_2000)
			folder_1990 = kmldoc.createFolder("1990 - 2000", visible=True, open=False)
			inst_folder.appendChild(folder_1990)
			folder_1980 = kmldoc.createFolder("1980 - 1990", visible=True, open=False)
			inst_folder.appendChild(folder_1980)
			folder_1970 = kmldoc.createFolder("1970 - 1980", visible=True, open=False)
			inst_folder.appendChild(folder_1970)
			folder_1960 = kmldoc.createFolder("1960 - 1970", visible=True, open=False)
			inst_folder.appendChild(folder_1960)
			folder_1950 = kmldoc.createFolder("1950 - 1960", visible=True, open=False)
			inst_folder.appendChild(folder_1950)
			folder_1940 = kmldoc.createFolder("1940 - 1950", visible=True, open=False)
			inst_folder.appendChild(folder_1940)
			folder_1930 = kmldoc.createFolder("1930 - 1940", visible=True, open=False)
			inst_folder.appendChild(folder_1930)
			folder_1920 = kmldoc.createFolder("1920 - 1930", visible=True, open=False)
			inst_folder.appendChild(folder_1920)
			folder_1910 = kmldoc.createFolder("1910 - 1920", visible=True, open=False)
			inst_folder.appendChild(folder_1910)
			folder_1900 = kmldoc.createFolder("1900 - 1910", visible=True, open=False)
			inst_folder.appendChild(folder_1900)
		else:
			topfolder = kmldoc.addFolder("Earthquake catalog", visible=True, open=False)

		for eq in self:
			if eq.datetime.year < instrumental_start_year:
				Mtype = "MS"
			else:
				Mtype = "ML"
			if time_folders:
				if eq.datetime.year < instrumental_start_year:
					folder = hist_folder
					visible = False
					color = (0, 255, 0)
					Mtype = "MS"
				else:
					visible = True
					Mtype = "ML"
					if start_time - eq.datetime <= datetime.timedelta(1, 0, 0):
						folder = folder_24h
						color = (255, 0, 0)
					elif start_time - eq.datetime <= datetime.timedelta(14, 0, 0):
						folder = folder_2w
						color = (255, 128, 0)
					elif start_time - eq.datetime <= datetime.timedelta(365, 0, 0):
						folder = folder_lastyear
						color = (255, 255, 0)
					elif eq.datetime.year >= 2000:
						folder = folder_2000
						color = (192, 0, 192)
					elif 1990 <= eq.datetime.year < 2000:
						folder = folder_1990
						color = (0, 0, 255)
					elif 1980 <= eq.datetime.year < 1990:
						folder = folder_1980
						color = (0, 0, 255)
					elif 1970 <= eq.datetime.year < 1980:
						folder = folder_1970
						color = (0, 0, 255)
					elif 1960 <= eq.datetime.year < 1970:
						folder = folder_1960
						color = (0, 0, 255)
					elif 1950 <= eq.datetime.year < 1960:
						folder = folder_1950
						color = (0, 0, 255)
					elif 1940 <= eq.datetime.year < 1950:
						folder = folder_1940
						color = (0, 0, 255)
					elif 1930 <= eq.datetime.year < 1940:
						folder = folder_1930
						color = (0, 0, 255)
					elif 1920 <= eq.datetime.year < 1930:
						folder = folder_1920
						color = (0, 0, 255)
					elif 1910 <= eq.datetime.year < 1920:
						folder = folder_1910
						color = (0, 0, 255)
			else:
				folder = topfolder
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
			url = '<a href="http://seismologie.oma.be/active.php?LANG=EN&CNT=BE&LEVEL=211&id=%d">ROB web page</a>' % eq.ID
			values = OrderedDict()
			values['ID'] = eq.ID
			values['Date'] = eq.datetime.date().isoformat()
			values['Time'] = "%02d:%02d:%02d" % (t.hour, t.minute, int(round(t.second + t.microsecond/1e+6)))
			values['Name'] = mykml.xmlstr(eq.name)
			values['ML'] = eq.ML
			values['MS'] = eq.MS
			values['MW'] = eq.MW
			values['Lon'] = eq.lon
			values['Lat'] = eq.lat
			values['Depth'] = eq.depth
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

	def export_VTK(self, vtk_filespec, proj="lambert1972", Mtype="MW", Mrelation=None):
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
			f.write("%.2f\n" % eq.get_M(Mtype, Mrelation))
		f.close()

	def pickle(self, filespec):
		"""
		Dump earthquake collection to a pickled file.
		Required parameter:
			filespec: full path specification of output file
		"""
		f = open(filespec, "w")
		cPickle.dump(self, f)
		f.close()

	def split_into_zones(self, source_model_name, verbose=True):
		"""
		Split catalog into subcatalogs according to a
		source-zone model stored in a GIS (MapInfo) table.

		:param source_model_name:
			String, name of source-zone model containing area sources
		:param verbose:
			Boolean, whether or not to print information while reading
			GIS table (default: True)

		:return:
			ordered dict {String sourceID: EQCatalog}
		"""
		## Construct WGS84 projection system corresponding to earthquake coordinates
		wgs84 = osr.SpatialReference()
		wgs84.SetWellKnownGeogCS("WGS84")

		## Read zone model from MapInfo file
		#source_model_table = ZoneModelTables[source_model_name.lower()]
		#tab_filespec = os.path.join(GIS_root, "KSB-ORB", "Source Zone Models", source_model_table + ".TAB")
		from hazard.psha.openquake.rob_sourceModels import rob_source_models_dict
		tab_filespec = rob_source_models_dict[source_model_name]["tab_filespec"]
		mi = ogr.GetDriverByName("MapInfo File")
		ds = mi.Open(tab_filespec)
		if verbose:
			print("Number of layers: %d" % ds.GetLayerCount())
		layer = ds.GetLayer(0)
		## Set up transformation between table coordsys and wgs84
		tab_sr = layer.GetSpatialRef()
		coordTrans = osr.CoordinateTransformation(tab_sr, wgs84)
		## Loop over features in layer 1
		if verbose:
			print("Number of features in layer 1: %d" % layer.GetFeatureCount())
		zone_polygons = OrderedDict()
		for i in range(layer.GetFeatureCount()):
			feature = layer.GetNextFeature()
			zoneID = feature.GetField("ShortName")
			if verbose:
				print feature.GetField("Name")
			## Note: we need to clone the geometry returned by GetGeometryRef(),
			## otherwise python will crash
			## See http://trac.osgeo.org/gdal/wiki/PythonGotchas
			poly = feature.GetGeometryRef().Clone()
			poly.AssignSpatialReference(tab_sr)
			poly.Transform(coordTrans)
			poly.CloseRings()
			zone_polygons[zoneID] = poly

		## Point object that will be used to test if earthquake is inside zone
		point = ogr.Geometry(ogr.wkbPoint)
		point.AssignSpatialReference(wgs84)

		zone_catalogs = OrderedDict()
		for zoneID, zone_poly in zone_polygons.items():
			zone_eq_list = []
			for i, eq in enumerate(self.eq_list):
				point.SetPoint(0, eq.lon, eq.lat)
				if point.Within(zone_poly):
					zone_eq_list.append(eq)
			zone_catalogs[zoneID] = EQCatalog(zone_eq_list, self.start_date, self.end_date, zoneID)

		return zone_catalogs

	def generate_synthetic_catalogs(self, N, sigma=0.2):
		"""
		Generate synthetic catalogs by random sampling of the magnitude of each earthquake.
		Parameters:
			N: number of random synthetic catalogs to generate
			sigma: magnitude uncertainty (considered uniform for the entire catalog).
		Return value:
			list of EQCatalog objects
		"""
		import copy
		M_list = []
		for eq in self:
			M_list.append(np.random.normal(eq.get_MS(), sigma, N))

		synthetic_catalogs = []
		for i in range(N):
			eq_list = []
			for eq, Mrange in zip(self, M_list):
				new_eq = copy.deepcopy(eq)
				new_eq.MS = Mrange[i]
				eq_list.append(new_eq)
			synthetic_catalogs.append(EQCatalog(eq_list, self.start_date, self.end_date))

		return synthetic_catalogs

	def analyse_completeness_Stepp(self, Mmin=1.8, dM=1.0, Mtype="MS", Mrelation=None, dt=5, ttol=0.2):
		"""
		Analyze catalog completeness with the Stepp method algorithm from GEM (old
		implementation). This method is a wrapper for :meth:`stepp_analysis` in
		the OQ hazard modeller's toolkit.

		:param Mmin:
			Float, minimum magnitude (default: 1.8)
		:param dM:
			Float, magnitude bin width (default: 0.1)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param dt:
			Int, time interval (in years) (default: 5)
		:param ttol:
			Positive float, tolerance threshold (default: 0.2)

		:return:
			instance of :class:`Completeness`
		"""
		# TODO: determine sensible default values for dt and ttol

		from mtoolkit.scientific.completeness import stepp_analysis
		subcatalog = self.subselect(Mmin=Mmin, Mtype=Mtype, Mrelation=Mrelation)
		years = self.get_years()
		Mags = self.get_magnitudes(Mtype, Mrelation)
		result = stepp_analysis(years, Mags, dM, dt, ttol, iloc=True)
		Min_Years, Min_Mags = result[:,0].astype('i'), result[:,1]
		return Completeness(Min_Years[::-1], Min_Mags[::-1])

	def analyse_completeness_Stepp_new(self, dM=0.1, Mtype="MS", Mrelation=None, dt=5.0, increment_lock=True):
		"""
		Analyze catalog completeness with the Stepp method algorithm from GEM (new
		implementation). This method is a wrapper for :meth:`Step1971.completeness`
		in the OQ hazard modeller's toolkit.

		:param dM:
			Float, magnitude bin width (default: 0.1)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param dt:
			Float, time interval (in years) (default: 5)
		:param increment_lock:
			Boolean, ensure completeness magnitudes always decrease with more
			recent bins (default: True).
		:return:
			instance of :class:`Completeness`
		"""
		from hmtk.seismicity.catalogue import Catalogue
		from hmtk.seismicity.completeness.comp_stepp_1971 import Stepp1971

		ec = Catalogue()
		keys_int = ['year', 'month', 'day', 'hour', 'minute']
		keys_flt = ['second', 'magnitude']
		data_int, data_flt = [], []
		for eq in self:
			data_int.append([
				int(eq.datetime.year),
				int(eq.datetime.month),
				int(eq.datetime.day),
				int(eq.datetime.hour),
				int(eq.datetime.minute),
			])
			data_flt.append([
				float(eq.datetime.second),
				float(eq.get_M(Mtype, Mrelation)),
			])
		ec.load_from_array(keys_int, np.array(data_int, dtype=np.int16))
		ec.load_from_array(keys_flt, np.array(data_flt, dtype=np.float64))
		stepp = Stepp1971()
		result = stepp.completeness(ec, {'magnitude_bin': dM, 'time_bin': dt, 'increment_lock': increment_lock})
		Min_Years, Min_Mags = result[:, 0].astype('i'), result[:,1]
		return Completeness(Min_Years, Min_Mags)

	def completeness_Stepp(self, start_year=None, mags=[2.], dt=5, Mtype="MS", Mrelation=None):
		"""
		"""
		for mag in mags:
			eqc_m = self.subselect(start_date=start_year, Mmin=mag, Mtype=Mtype, Mrelation=Mrelation)
			bin_year = self.end_date.year - dt
			while bin_year > eqc_m.start_date.year:
				eqc_t = eqc_m.subselect(start_date=bin_year)
				N = len(eqc_t)/(eqc_t.timespan())
				print '%s-%s: %s' % (bin_year, self.end_date.year, N)
				bin_year -= dt

	def decluster(self, method="gardner-knopoff", window_opt="GardnerKnopoff", fs_time_prop=0., time_window=60., Mtype="MS", Mrelation=None):
		"""
		Decluster catalog.
		This method is a wrapper for the declustering methods in the OQ
		hazard modeller's toolkit (old implementation).

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
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
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
		from mtoolkit.scientific.declustering import afteran_decluster, gardner_knopoff_decluster

		if method == "afteran":
			decluster_func = afteran_decluster
			decluster_param = time_window
		elif method == "gardner-knopoff":
			decluster_func = gardner_knopoff_decluster
			decluster_param = fs_time_prop

		eq_matrix = np.zeros((len(self), 6) ,'f')
		for i, eq in enumerate(self):
			M = eq.get_M(Mtype, Mrelation)
			eq_matrix[i,:] = (eq.datetime.year, eq.datetime.month, eq.datetime.day, eq.lon, eq.lat, M)
		cluster_vector, mainshock_matrix, flag_vector = decluster_func(eq_matrix, window_opt, decluster_param)

		## cluster_vector: cluster number of each earthquake
		## main_shock_matrix: 2-D matrix containing only mainshocks
		## flag_vector: -1 = foreshock, 0 = mainshock, 1 = aftershock

		mainshock_catalog = self.__getitem__(np.where(flag_vector == 0)[0])
		foreshock_catalog = self.__getitem__(np.where(flag_vector == -1)[0])
		aftershock_catalog = self.__getitem__(np.where(flag_vector == 1)[0])
		cluster_catalogs = []
		cluster_IDs = set(cluster_vector)
		for cluster_ID in cluster_IDs:
			cluster_catalog = self.__getitem__(np.where(cluster_vector == cluster_ID)[0])
			cluster_catalogs.append(cluster_catalog)

		return mainshock_catalog, foreshock_catalog, aftershock_catalog, cluster_catalogs

	def decluster_new(self, method="gardner-knopoff", window_opt="GardnerKnopoff", fs_time_prop=0., time_window=60., Mtype="MS", Mrelation=None):
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
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
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
			data_int.append([
				int(eq.datetime.year),
				int(eq.datetime.month),
				int(eq.datetime.day),
				int(eq.datetime.hour),
				int(eq.datetime.minute),
			])
			data_flt.append([
				float(eq.datetime.second),
				float(eq.get_M(Mtype, Mrelation)),
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

	def analyse_Mmax(self, method='Cumulative_Moment', num_bootstraps=100, iteration_tolerance=None, maximum_iterations=100, num_samples=20, Mtype="MW", Mrelation=None):
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
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
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

	def analyse_recurrence(self, dM=0.1, method="MLE", aM=0., dt=1., Mtype="MS", Mrelation=None, completeness=Completeness_Rosset):
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
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML") (default: None, will
			select the default relation for the given Mtype)
		:param completeness:
			instance of :class:`Completeness` (default: Completeness_Rosset)

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
		years = subcatalog.get_years()
		Mags = subcatalog.get_magnitudes(Mtype, Mrelation)
		completeness_table = completeness.to_table(Mmax=None)
		if method == "Weichert" and aM == 0.:
			aM = dM / 2.
		b, stdb, a, stda = recurrence_analysis(years, Mags, completeness_table, dM, method, aM, dt)
		return np.log10(a), b, stdb

	def plot_2d(self):
		"""
		"""
		x = self.get_longitudes()
		y = self.get_latitudes()
		plt.scatter(x, y)
		for eq in self.eq_list:
			plt.text(eq.lon, eq.lat, '%s - %s' % (eq.datetime.month, eq.datetime.day), fontsize=10)
		plt.show()

	def plot_3d(self, limits=None, Mtype=None, Mrelation=None):
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


EQCollection = EQCatalog


class CompositeEQCatalog:
	"""
	Class representing a catalog that has been split into a number
	of non-overlapping subcatalogs (e.g., split according to different
	source zones).
	"""
	def __init__(self, master_catalog, zone_catalogs):
		self.master_catalog = master_catalog
		self.zone_catalogs = zone_catalogs

	def balance_MFD_by_moment_rate(self, Nsamples):
		"""
		For each zone catalog: MC sampling of b value, compute corresponding
		a value. Sum total moment rate of all zonee catalogs, and check that
		it falls within +/- 2 sigma of total moment rate in master catalog.
		"""
		pass

	def balance_MFD_by_num_eq(self, Nsamples):
		"""
		For each zone catalog: MC sampling of b value, compute corresponding
		a value. Then, for each magnitude bin, compute total number of
		earthquakes (or annual rate) up to Mmax - 1 or 2 bins of each zone,
		and check that sum of all zone catalogs falls within +/- 2 sigma of
		annual rate of master catalog.
		"""
		pass

	def balance_MFD_by_fixed_b_value(self):
		pass



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
			eq = seismodb.LocalEarthquake(values["id_earth"], values["date"], time, values["longitude"], values["latitude"], values["depth"], values["ML"], values["MS"], values["MW"], values["name"])
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


def read_catalogSQL(region=None, start_date=None, end_date=None, Mmin=None, Mmax=None, min_depth=None, max_depth=None, id_earth=None, sort_key="date", sort_order="asc", convert_NULL=True, verbose=False, errf=None):
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
	return seismodb.query_ROB_LocalEQCatalog(region=region, start_date=start_date, end_date=end_date, Mmin=Mmin, Mmax=Mmax, min_depth=min_depth, max_depth=max_depth, id_earth=id_earth, sort_key=sort_key, sort_order=sort_order, convert_NULL=convert_NULL, verbose=verbose, errf=errf)


def read_catalogTXT(filespec, column_map, skiprows=0, region=None, start_date=None, end_date=None, Mmin=None, Mmax=None, min_depth=None, max_depth=None, Mtype="MS", Mrelation=None):
	"""
	Read ROB local earthquake catalog from txt file.

	:param filespec:
		String, defining filespec of a txt file with columns defining at least
		the attributes id, year, month, day, hours, minutes, seconds, longitude,
		latitude and depth. ML, MS and MW are optional.
	:param column_map:
		Dictionary, mapping attributes to number of column (starting from 0).
		ML, MS and MW must be set to None if not given.
	:param skiprows:
		Integer, defining number of lines to skip at top of file (default: 0).
		To be used when header is present.

	See method EQCatalog.read_catalogSQL for other params.
	"""
	eq_list_txt = np.loadtxt(filespec, skiprows=skiprows)
	eq_list = []
	for eq_txt in eq_list_txt:
		id = eq_txt[column_map['id']]
		year = int(eq_txt[column_map['year']])
		month = int(eq_txt[column_map['month']])
		day = int(eq_txt[column_map['day']])
		date = datetime.date(year, month, day)
		hour = int(eq_txt[column_map['hour']])
		minute = int(eq_txt[column_map['minute']])
		second = int(eq_txt[column_map['second']])
		time = datetime.time(hour, minute, second)
		lon = eq_txt[column_map['lon']]
		lat = eq_txt[column_map['lat']]
		depth = eq_txt[column_map['depth']]
		ML = eq_txt[column_map['ML']]
		if column_map['ML']:
			ML = eq_txt[column_map['ML']]
		else:
			ML = 0
		if column_map['MS']:
			MS = eq_txt[column_map['MS']]
		else:
			MS = 0
		if column_map['MW']:
			MW = eq_txt[column_map['MW']]
		else:
			MW = 0
		eq_list.append(seismodb.LocalEarthquake(id, date, time, lon, lat, depth,
			ML, MS, MW))
	eqc = EQCatalog(eq_list)
	eqc = eqc.subselect(region, start_date, end_date, Mmin, Mmax, min_depth,
		max_depth, Mtype, Mrelation)
	return eqc


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

	#M=completeness.get_completeness_magnitude(datetime.datetime.now().year)
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

