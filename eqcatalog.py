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
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
from scipy import stats
import ogr
import osr


## Import ROB modules
import users.kris.Seismo.db.seismodb as seismodb
from thirdparty.recipes.dummyclass import *
from thirdparty.recipes.my_arange import *



__all__ = ["ZoneModelTables", "Completeness", "Completeness_Leynaud", "Completeness_Rosset", "EQCollection", "read_catalogMI", "read_catalogSQL", "read_zonesMI", "read_zonesTXT", "format_zones_CRISIS", "alphabetalambda", "distribute_avalues", "split_avalues"]

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
		list or array containing initial years of completeness
	:param min_mags:
		list or array with corresponding lower magnitude for which
		catalog is assumed to be complete
	"""
	def __init__(self, min_years, min_mags):
		self.min_years = np.array(min_years)
		self.min_mags = np.array(min_mags)

	def __str__(self):
		s = "\n".join(["%d, %.2f" % (year, mag) for (year, mag) in zip(self.min_years, self.min_mags)])
		return s

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


## NOTE: I think threshold magnitudes should be a multiple of dM (or dM/2)!
Completeness_Leynaud = Completeness([1350, 1911, 1985], [4.7, 3.3, 1.8])
#Completeness_Leynaud = Completeness([1350, 1911, 1985], [4.75, 3.25, 1.75])
Completeness_Rosset = Completeness([1350, 1926, 1960, 1985], [5.0, 4.0, 3.0, 1.8])


class EQCatalog:
	"""
	Class defining a collection of local earthquakes.
	Initialization parameters:
		Required:
			eq_list: list of LocalEarthquake objects
		Optional:
			start_date: datetime object of start of catalog, defaults to datetime of first earthquake in list
			end_date: datetime object of end of catalog, defaults to datetime of last earthquake in list
			name: catalog name, defaults to empty string
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
		return self.eq_list.__getitem__(item)

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

	def get_magnitudes(self, Mtype="MS", relation=None):
		"""
		Return array of magnitudes for all earthquakes in catalog

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param relation:
			String, magnitude conversion relation (default: None)

		:return:
			1-D numpy float array, earthquake magnitudes
		"""
		if Mtype.upper() == "ML":
			Mags = [eq.get_ML(relation=relation) for eq in self]
		elif Mtype.upper() == "MS":
			Mags = [eq.get_MS(relation=relation) for eq in self]
		elif Mtype.upper() == "MW":
			Mags = [eq.get_MW(relation=relation) for eq in self]
		return np.array(Mags)

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

	def Tminmax(self, Mmax=None, Mtype="MS", Mrelation=None):
		"""
		Return tuple with oldest date and youngest date in catalog.

		:param Mmax:
			Float, maximum magnitude. Useful to check completeness periods.
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param Mrelation:
			String, magnitude conversion relation (default: None)
		"""
		DateTimes = self.get_datetimes()
		if Mmax != None:
			filtered_DateTimes = []
			Mags = self.get_magnitudes(Mtype=Mtype, relation=Mrelation)
			for M, dt in zip(Mags, DateTimes):
				if M < Mmax:
					filtered_DateTimes.append(dt)
			DateTimes = filtered_DateTimes
		if DateTimes:
			return (min(DateTimes), max(DateTimes))
		else:
			return (None, None)

	def Mminmax(self, Mtype="MS", relation=None):
		"""
		Return tuple with minimum and maximum magnitude in catalog.

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MS")
		:param relation:
			String, magnitude conversion relation (default: None)
		"""
		Mags = self.get_magnitudes(Mtype=Mtype, relation=relation)
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

	def get_M0(self, relation="Hinzen"):
		"""
		Compute total seismic moment.

		:param relation:
			String, magnitude conversion relation (default: "Hinzen")

		:return:
			Float, total seismic moment in N.m
		"""
		return np.add.reduce(np.array([eq.get_M0(relation=relation) for eq in self]))

	def get_M0rate(self, relation="Hinzen"):
		"""
		Compute seismic moment rate.

		:param relation:
			String, magnitude conversion relation (default: "Hinzen")

		:return:
			Float, seismic moment rate in N.m/yr
		"""
		# TODO: check if this should not be N.m/sec
		return self.get_M0(relation=relation) / self.timespan()

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
			String, magnitude conversion relation (default: None)

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
			start_date = self.start_date.date()
		if end_date is None:
			end_date = self.end_date.date()
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

		return EQCollection(eq_list, start_date=start_date, end_date=end_date, name=self.name + " (subselect)")

	def subselect_completeness(self, completeness=Completeness_Rosset, Mtype="MS", Mrelation=None):
		"""
		Subselect earthquakes in the catalog that conform with the specified
		completeness criterion.

		:param completeness:
			instance of :class:`Completeness` (default: Completeness_Rosset)
		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MS")
		:param Mrelation":
			String, magnitude conversion relation (default: None)

		:return:
			instance of :class:`EQCatalog`
		"""
		start_date = datetime.date(min(completeness.min_years), 1, 1)
		end_date = self.end_date

		## Select magnitudes according to completeness criteria
		eq_list = []
		for eq in self.eq_list:
			M = eq.get_M(Mtype, Mrelation)
			if M >= completeness.get_completeness_magnitude(eq.datetime.year):
				eq_list.append(eq)

		return EQCollection(eq_list, start_date=start_date, end_date=end_date, name=self.name + " (completeness-constrained)")

	def Mbin(self, Mmin, Mmax, dM=0.2, completeness=Completeness_Rosset, Mtype="MS", Mrelation=None, trim=False, verbose=False):
		"""
		Bin all earthquake magnitudes in catalog according to specified magnitude interval.

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
			String, magnitude scaling relation (default: None)
		:param trim:
			Bool, whether empty bins at start and end should be trimmed
			(default: False)
		#:param large_eq_correction: (M, corr_factor) tuple, with M the lower magnitude for which to apply corr_factor
		#	This is to correct the frequency of large earthquakes having a return period which is longer
		#	than the catalog length
		:param verbose:
			Bool, whether some messages should be printed or not (default: False)

		:return:
			Tuple (bins_N, bins_Mag, bins_Years, num_events, Mmax_obs)
			bins_N: array containing number of earthquakes for each magnitude interval
			bins_Mag: array containing lower magnitude of each interval
			bins_timespans: array containing time span (in years) for each magnitude interval
			num_events: total number of events selected
			Mmax_obs: maximum observed magnitude in analyzed collection
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
		if num_events > 0:
			Mmax_obs = max(Mags)
		else:
			Mmax_obs = 0
		if verbose:
			print "Number of events constrained by completeness criteria: %d out of %d" % (num_events, len(self.eq_list))
			print "Mmax observed: %.1f" % Mmax_obs

		## Compute number of earthquakes per magnitude bin
		bins_N, bins_Mag = np.histogram(Mags, bins_Mag)
		bins_Mag = bins_Mag[:-1]

		## Calculate year of completeness for each magnitude interval
		if completeness:
			bins_Years = []
			for M in bins_Mag:
				start_year = max(self.start_date.year, completeness.get_completeness_year(M))
				bins_Years.append(start_year)
		else:
			bins_Years = [self.start_date.year] * len(bins_Mag)
		bins_Years = np.array(bins_Years, 'd')
		#for M, year in zip(bins_Mag, bins_Years):
		#	print M, year

		## Compute time spans for each magnitude bin
		bins_timespans = np.array([(self.end_date - datetime.date(int(start_year),1,1)).days / 365.25 + 1 for start_year in bins_Years])
		"""
		## Correction factor for large magnitudes with return period > catalog duration
		if large_eq_correction:
			Mcorr, corr_factor = large_eq_correction
			try:
				corr_index = list(bins_Mag).index(float(Mcorr))
			except:
				print "Magnitude for frequency correction must be multiple of binning interval (%.1f)" % dM
			else:
				for i in range (corr_index, len(bins_Years)):
					bins_timespans[i] *= corr_factor
		"""

		## Optionally, trim empty leading and trailing intervals
		if trim:
			non_zero_indexes = np.where(bins_N > 0)[0]
			start = non_zero_indexes[0]
			end = non_zero_indexes[-1] + 1
			bins_N = bins_N[start:end]
			bins_Mag = bins_Mag[start:end]
			bins_Years = bins_Years[start:end]
			bins_timespans = bins_timespans[start:end]

		if verbose:
			print "Mmin   N   Ncumul  Years"
			for M, N, Ncumul, Years in zip (bins_Mag, bins_N, np.add.accumulate(bins_N[::-1])[::-1], bins_Years):
				print "%.2f  %3d   %3d   %4d" % (M, N, Ncumul, Years)
			print

		return (bins_N, bins_Mag, bins_timespans, num_events, Mmax_obs)

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
			String, magnitude scaling relation (default: None)
		:param fig_filespec:
			String, full path of image to be saved.
			If None (default), histogram is displayed on screen.
		:param verbose:
			Bool, whether or not to print binning information (default: False)
		"""
		bins_N, bins_Mag, bins_timespans, num_events, Mmax_obs = self.Mbin(Mmin, Mmax, dM, completeness=completeness, Mtype=Mtype, Mrelation=Mrelation, verbose=verbose)
		pylab.bar(bins_Mag, bins_N, width=dM)
		pylab.xlabel("Magnitude (%s)" % Mtype)
		pylab.ylabel("Number of events")
		pylab.title("%s (%d events)" % (self.name, num_events))

		if fig_filespec:
			pylab.savefig(fig_filespec)
			pylab.clf()
		else:
			pylab.show()

	def YearBin(self, start_year, end_year, dYear, Mmin, Mmax, Mtype="MS", Mrelation=None):
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
			String, magnitude scaling relation (default: None)

		:return:
			tuple (bins_N, bins_Years)
			bins_N: array containing number of earthquakes for each bin
			bins_Years: array containing lower year of each interval
		"""
		bins_Years = np.arange(start_year, end_year+dYear, dYear)
		## Select years according to magnitude criteria
		Years = [eq.datetime.year for eq in self.subselect(start_date=start_year, end_date=end_year, Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)]
		bins_N, bins_Years = np.histogram(Years, bins_Years)
		return (bins_N, bins_Years[:-1])

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
			String, magnitude scaling relation (default: None)
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
		bins_N, bins_Years = self.YearBin(catalog_start_year, end_year, dYear, Mmin, Mmax, Mtype=Mtype, Mrelation=Mrelation)
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

	def plot_CumulatedM0(self, start_date=None, end_date=None, ddate=10, ddate_spec="years", relation=None, M0max=None, fig_filespec=None):
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
				bins_M0[i] += eq.get_M0(relation=relation)
				M0[j] = eq.get_M0(relation=relation)
		elif ddate_spec.lower()[:3] == "day":
			for j, eq in enumerate(self):
				i = (eq.datetime.date() - start_date).days
				bins_M0[i] += eq.get_M0(relation=relation)
				M0[j] = eq.get_M0(relation=relation)
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

	def plot_NumDate(self, start_date=None, end_date=None, ddate=1, ddate_spec="days"):
		if start_date == None:
			start_date = self.start_date
		if end_date == None:
			end_date = self.end_date

		if ddate_spec.lower()[:4] == "year":
			bins_Dates = my_arange(start_date.year, end_date.year+ddate, ddate)
		elif ddate_spec.lower()[:3] == "day":
			bins_Dates = my_arange((end_date - start_date).days + 1)
		bins_Num = []
		for i in range(3):
			bins_Num.append(np.zeros(len(bins_Dates), 'd'))
		if ddate_spec.lower()[:4] == "year":
			for eq in self:
				i = list(eq.datetime.year < bins_Dates).index(True) - 1
				if eq.ML < 2.0:
					bins_Num[0][i] += 1
				elif 2.0 <= eq.ML < 3.0:
					bins_Num[1][i] += 1
				elif eq.ML >= 3.0:
					bins_Num[2][i] += 1
		elif ddate_spec.lower()[:3] == "day":
			for eq in self:
				i = (eq.datetime.date() - start_date).days
				if eq.ML < 2.0:
					bins_Num[0][i] += 1
				elif 2.0 <= eq.ML < 3.0:
					bins_Num[1][i] += 1
				elif eq.ML >= 3.0:
					bins_Num[2][i] += 1

		fig = pylab.figure()
		fig.add_subplot(311)
		ax = pylab.bar(bins_Dates, bins_Num[2], ddate)
		pylab.ylabel("M > 3.0")
		fig.add_subplot(312)
		ax = pylab.bar(bins_Dates, bins_Num[1], ddate)
		pylab.ylabel("Number of earthquakes\n2.0 <= M < 3.0")
		fig.add_subplot(313)
		ax = pylab.bar(bins_Dates, bins_Num[0], ddate)
		print dir(ax)
		pylab.ylabel("M < 2.0")
		pylab.xlabel("Time (%s)" % ddate_spec)
		pylab.show()

	def plot_MagnitudeDate(self):
		dates = [eq.datetime for eq in self]
		values = [eq.ML for eq in self]

		days = pylab.DayLocator()
		weeks = pylab.WeekdayLocator()
		months = pylab.MonthLocator()

		fig = pylab.figure()
		ax = fig.add_subplot(111)
		pylab.plot_date(pylab.date2num(dates), values)
		pylab.xlabel("Date")
		pylab.ylabel("Magnitude")
		ax.xaxis.set_major_locator(weeks)
		ax.xaxis.set_minor_locator(days)
		for label in ax.get_xticklabels():
			label.set_horizontalalignment('right')
			label.set_rotation(30)
		#ax.set(labels, rotation=30, fontsize=10)
		pylab.show()

	def plot_MagTime(self, Mtype="MS", lang="en"):
		if Mtype.upper() == "ML":
			Mags = [eq.ML for eq in self]
		elif Mtype.upper() == "MS":
			Mags = [eq.get_MS() for eq in self]
		elif Mtype.upper() == "MW":
			Mags = [eq.get_MW() for eq in self]
		Years = [eq.datetime.year + (eq.datetime.month - 1.0) /12 + ((eq.datetime.day - 1.0) / 31) / 12 for eq in self]
		pylab.plot(Mags, Years, '+')
		pylab.xlabel("Magnitude (%s)" % Mtype)
		pylab.ylabel({"en": "Time (years)", "nl": "Tijd (jaar)"}[lang])
		pylab.grid(True)
		pylab.show()

	def HourBin(self, Mmin, Mmax, Mtype="MS", start_year=None, end_year=None):
		hours = [(eq.datetime.hour + eq.datetime.minute/60.0 + eq.datetime.second/3600.0) for eq in self.subselect(start_date=start_year, end_date=end_year, Mmin=Mmin, Mmax=Mmax, Mtype=Mtype)]
		bins_Hr = range(1, 25)
		bins_N, junk = np.histogram(hours, bins_Hr, new=False)
		#print bins_N
		#print bins_Hr
		return bins_N, bins_Hr[:-1]

	def HourlyMean(self, Mmin, Mmax, Mtype="MS", start_year=None, end_year=None, day=(10, 17), night=(19, 7)):
		bins_N, bins_Hr = self.HourBin(Mmin, Mmax, Mtype, start_year, end_year)
		mean = np.mean(bins_N)
		mean_day = np.mean(bins_N[day[0]:day[1]])
		mean_night = np.mean(np.concatenate((bins_N[:night[1]], bins_N[night[0]-24:])))
		return (mean, mean_day, mean_night)

	def plot_HourHistogram(self, Mmin, Mmax, Mtype="MS", start_year=None, end_year=None):
		bins_N, bins_Hr = self.HourBin(Mmin, Mmax, Mtype, start_year, end_year)
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

	def DepthBin(self, Mmin, Mmax, Mtype="MS", start_year=None, end_year=None, min_depth=0, max_depth=30, bin_width=2):
		depths = [eq.depth for eq in self.subselect(start_date=start_year, end_date=end_year, Mmin=Mmin, Mmax=Mmax, Mtype=Mtype) if not eq.depth in (None, 0)]
		bins_depth = np.arange(min_depth, max_depth + bin_width, bin_width)
		bins_N, junk = np.histogram(depths, bins_depth)
		print bins_depth
		print bins_N
		return bins_N, bins_depth[:-1]

	def DepthBinByM0(self, Mmin, Mmax, Mtype="MS", start_year=None, end_year=None, min_depth=0, max_depth=30, bin_width=2):
		bins_depth = np.arange(min_depth, max_depth + bin_width, bin_width)
		bins_M0 = np.zeros(len(bins_depth))
		print bins_depth
		for eq in self.subselect(start_date=start_year, end_date=end_year, Mmin=Mmin, Mmax=Mmax, Mtype=Mtype):
			if eq.depth not in (None, 0):
				try:
					bin_id = np.where((bins_depth + bin_width) >= eq.depth)[0][0]
				except:
					## These are earthquakes that are deeper
					pass
				else:
					bins_M0[bin_id] += eq.get_M0()
		return bins_M0, bins_depth

	def plot_DepthHistogram(self, Mmin, Mmax, Mtype="MS", start_year=None, end_year=None, min_depth=0, max_depth=30, bin_width=2, color='b', want_title=True, fig_filespec="", fig_width=0, dpi=300):
		bins_N, bins_depth = self.DepthBin(Mmin, Mmax, Mtype, start_year, end_year, min_depth, max_depth, bin_width)
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
		bins_M0, bins_depth = self.DepthBinByM0(Mmin, Mmax, Mtype, start_year, end_year, min_depth, max_depth, bin_width)
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

	def MagFreq(self, Mmin, Mmax, dM=0.2, Mtype="MS", completeness=Completeness_Rosset, trim=False, verbose=False):
		"""
		Calculate cumulative and discrete magnitude / frequency for binned magnitude intervals.
		Parameters:
			Required:
				Mmin: minimum magnitude to bin
				Mmax: maximum magnitude to bin
				dM: magnitude interval
			Optional:
				Mtype: magnitude type ("ML", "MS" or "MW"), defaults to "MS"
				completeness: Completeness object with initial years of completeness and corresponding
					minimum magnitudes, defaults to Completeness_Leynaud
				trim: boolean indicating whether empty bins at start and end should be trimmed, defaults to False
				verbose: boolean indicating whether some messages should be printed or not, defaults to False
		Return value:
			(bins_N_cumulative, bins_N_discrete, bins_Mag, bins_Years, num_events, Mmax_obs) tuple
			bins_N_cumulative: array with cumulative frequency of M higher than or equal to minimum magnitude in each magnitude interval
			bins_N_discrete: array with discrete frequency for each magnitude interval
			bins_Mag: array containing lower magnitude of each interval
			bins_Years: array containing time span (in years) for each magnitude interval
			num_events: total number of events selected
			Mmax_obs: maximum observed magnitude in analyzed collection
			Note that order of arrays is reversed with respect to Mbin !
		"""
		bins_N, bins_Mag, bins_Years, num_events, Mmax_obs = self.Mbin(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, trim=trim, verbose=verbose)
		## Chop off last element of bins_Mag, as it is not a lower bin value
		bins_Mag = bins_Mag[:-1]
		## We need to normalize n values to maximum time span !
		max_span = bins_Years[-1] * 1.0
		bins_N_discrete = np.array([n * max_span / span for (n, span) in zip(bins_N, bins_Years)])
		## Reverse arrays for calculating cumulative number of events
		bins_N_discrete = bins_N_discrete[::-1]
		bins_Mag = bins_Mag[::-1]
		bins_Years = bins_Years[::-1]
		bins_N_cumulative = np.add.accumulate(bins_N_discrete)

		if verbose:
			print "Mmin   N   Ncumul"
			for Mag, N_disc, N_cumul in zip (bins_Mag, bins_N_discrete, bins_N_cumulative):
				print "%.2f  %3d   %3d" % (Mag, N_disc, N_cumul)
			print

		return (bins_N_cumulative/max_span, bins_N_discrete/max_span, bins_Mag, bins_Years, num_events, Mmax_obs)

	def LogMagFreq(self, Mmin, Mmax, dM=0.2, Mtype="MS", completeness=Completeness_Rosset, verbose=False):
		"""
		Calculate log10 of cumulative and discrete magnitude / frequency for binned magnitude intervals.
		Parameters:
			Required:
				Mmin: minimum magnitude to bin
				Mmax: maximum magnitude to bin
				dM: magnitude interval
			Optional:
				Mtype: magnitude type ("ML", "MS" or "MW"), defaults to "MS"
				completeness: Completeness object with initial years of completeness and corresponding
					minimum magnitudes, defaults to Completeness_Leynaud
				verbose: boolean indicating whether some messages should be printed or not, defaults to False
		Return value:
			(bins_N_cumul_log, bins_N_disc_log, bins_Mag, bins_Years, num_events, Mmax_obs) tuple
			bins_N_cumulative: array with log10 of cumulative frequency of M higher than or equal to minimum magnitude in each magnitude interval
			bins_N_discrete: array with log10 of discrete frequency for each magnitude interval
			bins_Mag: array containing lower magnitude of each interval
			bins_Years: array containing time span (in years) for each magnitude interval
			num_events: total number of events selected
			Mmax_obs: maximum observed magnitude in analyzed collection
			Note that order of arrays is reversed with respect to Mbin !
		"""
		bins_N_cumulative, bins_N_discrete, bins_Mag, bins_Years, num_events, Mmax_obs = self.MagFreq(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, trim=True, verbose=verbose)
		bins_N_disc_log = np.log10(bins_N_discrete)
		bins_N_cumul_log = np.log10(bins_N_cumulative)

		if verbose:
			print "Mmin   Ncumul/yr   log(Ncumul/yr)"
			for Mag, Ncumul, logNcumul in zip(bins_Mag, bins_N_cumulative, bins_N_cumul_log):
				print "%.2f   %.6f   %.3f" % (Mag, Ncumul, logNcumul)
			print

		return (bins_N_cumul_log, bins_N_disc_log, bins_Mag, bins_Years, num_events, Mmax_obs)

	def LogMagFreqExp(self, bins_Mag, a, b, Mmax):
		"""
		Compute exponential form of truncated GR
		"""
		#beta = b * math.log(10)
		#bins_N_cumul_log = a - b * bins_Mag
		#bins_N_cumul = 10 ** bins_N_cumul_log
		#bins_N_cumul_mle_ln = np.log(bins_N_cumul * np.exp(-beta * bins_Mag) * (1. - np.exp(-(Mmax - bins_Mag))))
		#return bins_N_cumul_mle_ln / math.log(10)
		Mmin = min(bins_Mag)
		alpha, beta, lamda = alphabetalambda(a, b, Mmin)
		bins_N_cumul_mle = lamda * (np.exp(-beta*bins_Mag) - np.exp(-beta*Mmax)) / (np.exp(-beta*Mmin) - np.exp(-beta*Mmax))
		return np.log10(bins_N_cumul_mle)

	def calcGR_lsq(self, Mmin, Mmax, dM=0.2, Mtype="MS", completeness=Completeness_Rosset, verbose=False):
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
		bins_N_cumul_log, bins_N_disc_log, bins_Mag, bins_Years, num_events, Mmax_obs = self.LogMagFreq(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=False)
		b, a, r, ttprob, stderr = stats.linregress(bins_Mag, bins_N_cumul_log)
		## stderr = standard error on b?
		if verbose:
			print "Linear regression: a=%.3f, b=%.3f (r=%.2f)" % (a, -b, r)
		return (a, -b, r)

	def calcGR_mle(self, Mmin, Mmax, dM=0.2, Mtype="MS", completeness=Completeness_Rosset, beta=None, Mc=None, verbose=False):
		"""
		Calculate a and b values of Gutenberg-Richter relation using maximum likelihood estimation (mle).
		Adapted from calB.m and calBfixe.m Matlab modules written by Philippe Rosset (ROB, 2004),
		which is based on the method by Weichert, 1980 (BSSA, 70, N°4, 1337-1346).
		Parameters:
			Required:
				Mmin: minimum magnitude to use for binning
				Mmax: maximum magnitude to use for binning
				dM: magnitude interval to use for binning
			Optional:
				Mtype: magnitude type ("ML", "MS" or "MW"), defaults to "MS"
				completeness: Completeness object with initial years of completeness and corresponding
					minimum magnitudes, defaults to Completeness_Leynaud
				beta: fixed beta (= b * ln(10)) value to use for calculating a, defaults to None
				Mc: cutoff magnitude = magnitude for which to calculate lambda (defaults to None)
				verbose: boolean indicating whether some messages should be printed or not, defaults to False
		Return value:
			if Mc == None:
				(A, B, BETA, STDA, STDB, STDBETA) tuple
			else:
				(A, B, BETA, LAMBDA_Mc, STDA, STDB, STDBETA, STD_LAMBDA_Mc) tuple
			A: a value (intercept)
			B: b value (slope, taken positive)
			BETA: beta value (= b * ln(10))
			LAMBDA_Mc : lambda for cutoff magnitude Mc
			STDA: standard error on a value
			STDB: standard error on b value
			STDBETA: standard error on beta value
			STD_LAMBDA_Mc: standard error on lambda_Mc
		IMPORTANT NOTE:
			This regression depends very strongly on the Mmax specified. Empty bins are taken into account.
			It is therefore important to specify Mmax not larger than the evaluated Mmax for the specific area.
		"""
		bins_N, bins_Mag, bins_Years, num_events, Mmax_obs = self.Mbin(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, trim=False, verbose=verbose)
		bins_Mag += dM/2.0

		if not beta:
			BETA = 1.5
		else:
			BETA = beta
		BETL = 0
		#i = 0
		while(abs(BETA-BETL)) >= 0.0001:
			SNM = 0.0
			NKNOUT = 0.0
			STMEX = 0.0
			SUMTEX = 0.0
			STM2X = 0.0
			SUMEXP = 0.0

			for k in range(len(bins_N)):
				SNM += bins_N[k] * bins_Mag[k]
				NKNOUT += bins_N[k]
				TJEXP = bins_Years[k] * math.exp(-BETA * bins_Mag[k])
				TMEXP = TJEXP * bins_Mag[k]
				SUMEXP += math.exp(-BETA * bins_Mag[k])
				STMEX += TMEXP
				SUMTEX += TJEXP
				STM2X += bins_Mag[k] * TMEXP

			try:
				DLDB = STMEX / SUMTEX
			#if np.isnan(DLDB):
			except:
				break
			else:
				D2LDB2 = NKNOUT * (DLDB**2 - STM2X/SUMTEX)
				DLDB = DLDB*NKNOUT - SNM
				BETL = BETA
				if not beta:
					BETA -= DLDB/D2LDB2
			#i += 1

		STDBETA = math.sqrt(-1.0/D2LDB2)
		B = BETA / math.log(10)
		STDB = STDBETA / math.log(10)
		FNGTMO = NKNOUT * SUMEXP / SUMTEX
		STDFNGTMO = math.sqrt(FNGTMO/NKNOUT)
		#A = math.log10(FNGTMO) + B*bins_Mag[0]
		A = math.log10(FNGTMO) + B*(bins_Mag[0] - dM/2.0)
		STDA = math.sqrt((bins_Mag[0]-dM/2.0)**2 * STDB**2 - (STDFNGTMO**2 / ((math.log(10)**2 * math.exp(2*(A+B*(bins_Mag[0]-dM/2.0))*math.log(10))))))
		#STDA = math.sqrt(abs(A)/NKNOUT)
		ALPHA = FNGTMO * math.exp(-BETA * (bins_Mag[0] - dM/2.0))
		STDALPHA = ALPHA / math.sqrt(NKNOUT)
		if Mc !=None:
			LAMBDA_Mc = FNGTMO * math.exp(-BETA * (Mc - (bins_Mag[0] - dM/2.0)))
			STD_LAMBDA_Mc = math.sqrt(LAMBDA_Mc / NKNOUT)
		if verbose:
			print "Maximum likelihood: a=%.3f ($\pm$ %.3f), b=%.3f ($\pm$ %.3f), beta=%.3f ($\pm$ %.3f)" % (A, STDA, B, STDB, BETA, STDBETA)
		if Mc != None:
			return (A, B, BETA, LAMBDA_Mc, STDA, STDB, STDBETA, STD_LAMBDA_Mc)
		else:
			return (A, B, BETA, STDA, STDB, STDBETA)

	def plot_MagFreq(self, Mmin, Mmax, dM=0.2, Mtype="MS", cumul=True, discrete=False, completeness=Completeness_Rosset, Mrange=(), Freq_range=(), fixed_beta=None, num_sigma=0, lang="en", color=True, want_lsq=True, want_completeness_limits=False, want_exponential=False, title=None, fig_filespec=None, fig_width=0, dpi=300, verbose=False):
		"""
		Plot magnitude / log10(frequency) diagram for collection.
		Parameters:
			Required:
				Mmin: minimum magnitude to use for binning
				Mmax: maximum magnitude to use for binning
				dM: magnitude interval to use for binning
			Optional:
				Mtype: magnitude type ("ML", "MS" or "MW"), defaults to "MS"
				completeness: Completeness object with initial years of completeness and corresponding
					minimum magnitudes, defaults to Completeness_Leynaud
				cumul: boolean indicating whether cumulative frequency should be plotted, defaults to True
				discrete: boolean indicating whether discrete frequency should be plotted, defaults to False
				verbose: boolean indicating whether some messages should be printed or not, defaults to False
				fig_filespec: filespec of image to be saved. If None (default), diagram is displayed on screen.
				Mrange: (Mmin, Mmax) tuple of magnitude range in plot, defaults to ()
				Freq_range: (Freq_min, Freq_max) tuple of frequency range in plot, defaults to ()
				fixed_beta: fixed beta value to use for maximum-likelihood estimation
				color: boolean indicating whether or not color should be used
				want_lsq: boolean indicating whether or not least-squares regression should be shown
				want_completeness_limits: boolean indicating whether or not completeness limits should be plotted.
				want_exponential: boolean indicating whether or not to plot the exponential form of the GR
				fig_filespec:
				fig_width:
				dpi:
		"""
		bins_N_cumul_log, bins_N_disc_log, bins_Mag, bins_Years, num_events, Mmax_obs = self.LogMagFreq(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=verbose)

		## Remove values that are infinite (log(0)), pylab can't cope with it
		for i in range(len(bins_N_disc_log)):
			if abs(bins_N_disc_log[i]) == np.infty:
				bins_N_disc_log[i] = -12.0

		## Plot
		fig = pylab.figure()

		if discrete:
			label = {"en": "Observed (incremental)", "nl": "Waargenomen (incrementeel)", "fr": "Observe (incremental)"}[lang.lower()]
			symbol = 's'
			symbol_color = 'r'
			if not color:
				symbol_color = 'w'
			ax = pylab.semilogy(bins_Mag, 10**bins_N_disc_log, symbol, label=label)
			pylab.setp(ax, markersize=10.0, markeredgewidth=1.0, markeredgecolor="k", markerfacecolor=symbol_color)
		if cumul:
			label = {"en": "Observed (cumulative)", "nl" :"Waargenomen (cumulatief)", "fr": "Observe (cumulatif)"}[lang.lower()]
			symbol = 'bo'
			if not color:
				symbol = 'ko'
			pylab.semilogy(bins_Mag, 10**bins_N_cumul_log, symbol, label=label)

		#xmin, xmax = bins_Mag[0], bins_Mag[-1]
		xmin, xmax = Mmin, Mmax
		## Linear Regression
		if want_lsq:
			line_style = 'g--'
			if not color:
				line_style = 'k:'
			a, b, r = self.calcGR_lsq(Mmin, Mmax, dM, Mtype, completeness, verbose=verbose)
			plot_label="LSQ: a=%.3f, b=%.3f (r=%.2f)" % (a, b, r)
			ymin = a - b * xmin
			ymax = a - b * xmax
			pylab.semilogy([xmin, xmax], [10**ymin, 10**ymax], line_style, lw=2, label=plot_label)

		## Maximum likelihood
		try:
			a, b, beta, stda, stdb, stdbeta = self.calcGR_mle(Mmin, Mmax, dM, Mtype, completeness, beta=None, verbose=verbose)
		except:
			print "Failed calculating MLE regression for %s" % self.name
		else:
			ymin, ymax = a - b * xmin, a - b * xmax
			if num_sigma:
				betamin, betamax = beta - num_sigma*stdbeta, beta + num_sigma*stdbeta
				amin, bmin = self.calcGR_mle(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, beta=betamin, verbose=verbose)[0:2]
				amax, bmax = self.calcGR_mle(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, beta=betamax, verbose=verbose)[0:2]
				ymin_min, ymax_min = amin - bmin * xmin, amin - bmin * xmax
				ymin_max, ymax_max = amax - bmax * xmin, amax - bmax * xmax

			line_color = 'b'
			if not color:
				line_color = 'k'
			plot_label="MLE: a=%.3f ($\pm$ %.3f), b=%.3f ($\pm$ %.3f)" % (a, stda, b, stdb)
			if want_exponential:
				bins_Mag_full = np.arange(Mmin, Mmax + dM, dM)
				bins_N_cumul_mle_log = self.LogMagFreqExp(bins_Mag_full, a, b, Mmax)
				pylab.semilogy(bins_Mag_full, 10**bins_N_cumul_mle_log, '%c--' % line_color, lw=3, label=plot_label)
			else:
				pylab.semilogy([xmin, xmax], [10**ymin, 10**ymax], '%c--' % line_color, lw=3, label=plot_label)
			if num_sigma:
				pylab.semilogy([xmin, xmax], [10**ymin_min, 10**ymax_min], '%c:' % line_color, lw=2, label="MLE: $\pm$ %d sigma" % num_sigma)
				pylab.semilogy([xmin, xmax], [10**ymin_max, 10**ymax_max], '%c:' % line_color, lw=2, label="_nolegend_")

		## Maximum likelihood with fixed beta
		if fixed_beta:
			try:
				a, b, beta, stda, stdb, stdbeta = self.calcGR_mle(Mmin, Mmax, dM, Mtype, completeness, beta=fixed_beta, verbose=verbose)
			except:
				print "Failed calculating MLE regression for %s" % self.name
			else:
				ymin = a - b * xmin
				ymax = a - b * xmax
				label = {"en": "MLE w. fixed beta: ", "nl": "MLE (b vast): "}[lang.lower()]
				plot_label=label + "a=%.3f ($\pm$ %.3f), b=%.3f" % (a, stda, b)
				if want_exponential:
					bins_Mag_full = np.arange(Mmin, Mmax + dM, dM)
					bins_N_cumul_mle_log = self.LogMagFreqExp(bins_Mag_full, a, b, Mmax)
					pylab.semilogy(bins_Mag_full, 10**bins_N_cumul_mle_log, 'm--', lw=3, label=plot_label)
				else:
					pylab.semilogy([xmin, xmax], [10**ymin, 10**ymax], 'm--', lw=3, label=plot_label)

		if not Mrange:
			Mrange = pylab.axis()[:2]
		if not Freq_range:
			Freq_range = pylab.axis()[2:]

		## Plot limits of completeness
		if want_completeness_limits:
			annoty = Freq_range[0] * 10**0.5
			bbox_props = dict(boxstyle="round,pad=0.4", fc="w", ec="k", lw=1)
			ax = pylab.gca()
			min_mags = completeness.Min_Mags[:]
			min_mags.sort()
			for i in range(1, len(min_mags)):
				pylab.plot([min_mags[i], min_mags[i]], Freq_range, 'k--', lw=1, label="_nolegend_")
				ax.annotate("", xy=(min_mags[i-1], annoty), xycoords='data', xytext=(min_mags[i], annoty), textcoords='data', arrowprops=dict(arrowstyle="<->"),)
				label = "%s - %s" % (completeness.get_completeness_year(min_mags[i-1]), self.end_date.year)
				ax.text(np.mean([min_mags[i-1], min_mags[i]]), annoty*10**-0.25, label, ha="center", va="center", size=12, bbox=bbox_props)
			ax.annotate("", xy=(min_mags[i], annoty), xycoords='data', xytext=(min(Mmax, Mrange[1]), annoty), textcoords='data', arrowprops=dict(arrowstyle="<->"),)
			label = "%s - %s" % (completeness.get_completeness_year(min_mags[i]), self.end_date.year)
			ax.text(np.mean([min_mags[i], Mmax]), annoty*10**-0.25, label, ha="center", va="center", size=12, bbox=bbox_props)

		## Apply plot limits
		pylab.axis((Mrange[0], Mrange[1], Freq_range[0], Freq_range[1]))

		pylab.xlabel("Magnitude ($M_%s$)" % Mtype[1].upper(), fontsize="x-large")
		label = {"en": "Number of earthquakes per year", "nl": "Aantal aardbevingen per jaar", "fr": "Nombre de seismes par annee"}[lang.lower()]
		pylab.ylabel(label, fontsize="x-large")
		if title is None:
			title = "%s (%d events, Mmax=%.2f)" % (self.name, num_events, Mmax_obs)
		#pylab.title("%s (%d events, Mmax=%.2f)" % (unicode(self.name, errors="replace"), num_events, Mmax_obs))
		pylab.title(title, fontsize='x-large')
		pylab.grid(True)
		font = FontProperties(size='medium')
		pylab.legend(loc=1, prop=font)
		ax = pylab.gca()
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('large')

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

	def export_ZMAP(self, filespec, Mtype="MS"):
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
			if Mtype.upper() == "ML":
				M = eq.ML
			elif Mtype.upper() == "MS":
				M = eq.get_MS()
			elif Mtype.upper() == "MW":
				M = eq.get_MW()
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
			ordered dict {String sourceID: EQCollection}
		"""
		## Construct WGS84 projection system corresponding to earthquake coordinates
		wgs84 = osr.SpatialReference()
		wgs84.SetWellKnownGeogCS("WGS84")

		## Read zone model from MapInfo file
		source_model_table = ZoneModelTables[source_model_name.lower()]
		tab_filespec = os.path.join(GIS_root, "KSB-ORB", "Source Zone Models", source_model_table + ".TAB")
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
			zone_catalogs[zoneID] = EQCollection(zone_eq_list, self.start_date, self.end_date, zoneID)

		return zone_catalogs

	def generate_synthetic_catalogs(self, N, sigma=0.2):
		"""
		Generate synthetic catalogs by random sampling of the magnitude of each earthquake.
		Parameters:
			N: number of random synthetic catalogs to generate
			sigma: magnitude uncertainty (considered uniform for the entire catalog).
		Return value:
			list of EQCollection objects
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
			synthetic_catalogs.append(EQCollection(eq_list, self.start_date, self.end_date))

		return synthetic_catalogs

	def analyse_completeness_Stepp(self, Mmin=1.8, dM=1.0, Mtype="MS", dt=5, ttol=0.2):
		from thirdparty.oq_hazard_modeller.mtoolkit.scientific.completeness import stepp_analysis
		subcatalog = self.subselect(Mmin=Mmin, Mtype=Mtype)
		years = np.array([eq.datetime.year for eq in subcatalog])
		if Mtype.upper() == "ML":
			Mags = [eq.ML for eq in subcatalog]
		elif Mtype.upper() == "MS":
			Mags = [eq.get_MS() for eq in subcatalog]
		elif Mtype.upper() == "MW":
			Mags = [eq.get_MW() for eq in subcatalog]
		Mags = np.array(Mags)
		result = stepp_analysis(years, Mags, dM, dt, ttol, iloc=True)
		Min_Years, Min_Mags = result[:,0].astype('i'), result[:,1]
		return Completeness(Min_Years, Min_Mags)

	def decluster(self, Mtype="MS"):
		from thirdparty.oq_hazard_modeller.mtoolkit.scientific.declustering import afteran_decluster
		eq_catalog = np.zeros((len(self), 6) ,'f')
		for i, eq in enumerate(self):
			if Mtype.upper() == "ML":
				M = eq.ML
			elif Mtype.upper() == "MS":
				M = eq.get_MS()
			elif Mtype.upper() == "MW":
				M = eq.get_MW()
			eq_catalog[i,:] = (eq.datetime.year, eq.datetime.month, eq.datetime.day, eq.lon, eq.lat, M)
		cluster_vector, main_shock_catalog, flag_vector = afteran_decluster(eq_catalog)
		print len(eq_catalog)
		print len(cluster_vector), len(main_shock_catalog), len(flag_vector)
		print len(np.where(cluster_vector != 0.)[0]), len(np.where(flag_vector != 0.)[0])
		for i in range(len(eq_catalog)):
			if flag_vector[i] != 0:
				print i, cluster_vector[i], flag_vector[i]


EQCollection = EQCatalog


#TODO: moment balancing with MFD objects

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
		EQCollection object, or dictionary of EQCollection objects if a zone model is specified.
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
		catalog = EQCollection(catalog, start_date, end_date, name=name)
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


def read_catalogSQL(region=None, start_date=None, end_date=None, Mmin=None, Mmax=None, min_depth=None, max_depth=None, verbose=False):
	"""
	Query ROB local earthquake catalog through the online database.
	Parameters:
		region: (w, e, s, n) tuple specifying rectangular region of interest
		start_date: date (date object or year) specifying start of time window of interest
		end_date: date (date object or year) specifying end of time window of interest
		Mmin: minimum magnitude to extract
		Mmax: minimum magnitude to extract
		verbose: bool, if True the query string will be echoed to standard output
	Magnitude used for selection is based on MW first, then MS, then ML.
	Note that NULL values in the database are converted to 0.0 (this may change in the future)
	Return value:
		EQCollection object
	"""
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
	catalog = seismodb.query_ROB_LocalEQCatalog(region=region, start_date=start_date, end_date=end_date, Mmin=Mmin, Mmax=Mmax, min_depth=min_depth, max_depth=max_depth, verbose=verbose)
	name = "ROB Catalog %s - %s" % (start_date.isoformat(), end_date.isoformat())
	return EQCollection(catalog, start_date, end_date, name=name)


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


def alphabetalambda(a, b, M0):
	"""
	Calculate alpha, beta, lambda from a, b, and M0.
	Parameters:
		a: a value of Gutenberg-Richter relation
		b: b value of Gutenberg-Richter relation
		M0: threshold magnitude
	Return value:
		(alpha, beta, lambda) tuple
	"""
	alpha = a * math.log(10)
	beta = b * math.log(10)
	lambda0 = math.exp(alpha - beta*M0)
	# This is identical
	# lambda0 = 10**(a - b*M0)
	return (alpha, beta, lambda0)


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
		zone.a_new = math.log10(10**zone.a + weight*N_diff)
		print "%s - a: %.3f  ->  %.3f" % (zone.name, zone.a, zone.a_new)

	N_zones = 0
	for zone in zones:
		N_zones += 10 ** zone.a_new
	N_diff = 10 ** catalog.a - N_zones


def split_avalues(a, weights):
	N = 10**a
	avalues = []
	for w in weights:
		aw = math.log10(w*N)
		avalues.append(aw)
	return avalues


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
	#bins_N, bins_Mag, bins_Years, num_events, Mmax = catalog.Mbin(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)

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
	#M0_historical = historical_catalog.get_M0rate(relation="geller")
	#M0_instrumental = instrumental_catalog.get_M0rate(relation="hinzen")
	#M0_total = M0_historical + M0_instrumental
	#print "Total seismic moment: %.2E (historical) + %.2E (instrumental) = %.2E N.m" % (M0_historical, M0_instrumental, M0_total)
	#catalog.plot_CumulatedM0(ddate=10, ddate_spec="years", relation=None)

	## Read full catalog from database, calculate a and b values, and store these for later use
	"""
	catalog = read_catalogSQL(region=region, start_date=start_date, end_date=end_date)
	cat = DummyClass()
	cat.a, cat.b, cat.beta, cat.stda, cat.stdb, cat.stdbeta = catalog.calcGR_mle(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)
	bins_N, bins_Mag, bins_Years, num_events, Mmax = catalog.Mbin(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=False)
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
				zone.num_events = catalog.Mbin(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, trim=True, verbose=False)[3]
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

