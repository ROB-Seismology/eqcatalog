"""
Module containing different declustering methods.

Original implementation by Bart Vleminckx
Completely rewritten by Kris Vanneste
Royal Observatory of Belgium
"""

# TODO: allow taking into account location errors in ClusterMethod and WindowMethod

# TODO: appropriate catalog names for returned catalogs (aftershocks, foreshocks, ...)

# TODO: better handling of MW, maybe add Mtype to DeclusteringWindow?

# TODO: make decluster methods accept dc_window as string


from __future__ import absolute_import, division, print_function, unicode_literals


import abc
import datetime
from collections import OrderedDict

import numpy as np

from .time_functions_np import SECS_PER_DAY
from .moment import (moment_to_mag, mag_to_moment)
from .eqrecord import LocalEarthquake
from .eqcatalog import EQCatalog


class DeclusteringWindow():
	"""
	Class implementing declustering window
	"""

	__metaclass__ = abc.ABCMeta

	name = ''

	@abc.abstractmethod
	def get_time_window(self, magnitude):
		pass

	@abc.abstractmethod
	def get_dist_window(self, magnitude):
		pass

	def get_windows(self, magnitude):
		"""
		:param magnitude:
			float, magnitude for wich to calculate window.

		:returns:
			(float, float), defining time window (in days) and distance window (in km)
		"""
		t_window = self.get_time_window(magnitude)
		s_window = self.get_dist_window(magnitude)
		return (t_window, s_window)


class GardnerKnopoff1974Window(DeclusteringWindow):
	"""
	Class implementing Gardner-Knopoff (1974) declustering window
	"""
	name = "GardnerKnopoff1974"

	def get_time_window(self, magnitude):
		if np.isscalar(magnitude):
			if magnitude >= 6.5:
				t_window = 10**(0.0320*magnitude + 2.7389)
			else:
				t_window = 10**(0.5409*magnitude - 0.5470)
			return np.timedelta64(int(round(t_window * SECS_PER_DAY)), 's')
		else:
			magnitude = np.asarray(magnitude)
			t_window = np.zeros(len(magnitude))
			idxs = np.where(magnitude >= 6.5)
			t_window[idxs] = 10**(0.0320*magnitude[idxs] + 2.7389)
			idxs = np.where(magnitude < 6.5)
			t_window[idxs] = 10**(0.5409*magnitude[idxs] - 0.5470)
			return np.round(t_window * SECS_PER_DAY).astype('m8[s]')

	def get_dist_window(self, magnitude):
		return 10**(0.1238*magnitude + 0.983)


class Uhrhammer1986Window(DeclusteringWindow):
	"""
	Class implementing Uhrhammer (1986) declustering window
	"""
	name = "Uhrhammer1986"

	def get_time_window(self, magnitude):
		t_window = np.exp(-2.870 + 1.235*magnitude)
		if np.isscalar(t_window):
			return np.timedelta64(int(round(t_window * SECS_PER_DAY)), 's')
		else:
			return np.round(t_window * SECS_PER_DAY).astype('m8[s]')

	def get_dist_window(self, magnitude):
		return np.exp(-1.024 + 0.804*magnitude)


class Gruenthal2009Window(DeclusteringWindow):
	"""
	Class implementing Gruenthal (2009) declustering window
	"""
	name = "Gruenthal2009"

	def get_time_window(self, magnitude):
		if np.isscalar(magnitude):
			if magnitude >= 6.5:
				t_window = 10**(2.8 + 0.024*magnitude)
			elif magnitude > 0:
				t_window = np.exp(-3.95 + np.sqrt(0.62 + 17.32*magnitude))
			else:
				t_window = 0
			return np.timedelta64(int(round(t_window * SECS_PER_DAY)), 's')
		else:
			magnitude = np.asarray(magnitude)
			t_window = np.zeros(len(magnitude))
			idxs = np.where(magnitude >= 6.5)
			t_window[idxs] = 10**(2.8 + 0.024*magnitude[idxs])
			idxs = np.where((magnitude > 0) & (magnitude < 6.5))
			t_window[idxs] = np.exp(-3.95 + np.sqrt(0.62 + 17.32*magnitude[idxs]))
			return np.round(t_window * SECS_PER_DAY).astype('m8[s]')

	def get_dist_window(self, magnitude):
		s_window = np.maximum(0, np.exp(1.77 + np.sqrt(0.037 + 1.02*magnitude)))
		return s_window


class Reasenberg1985Window(DeclusteringWindow):
	"""
	Implements time and distance windows used in declustering
	method of Reasenberg (1985). This should only be used
	in combination with :class:`ReasenbergMethod`
	"""
	name = "Reasenberg1985"

	# TODO: add XK, XMEFF, rfact, dsigma to __init__ as properties?
	def get_time_window(self, mag, time_delta=0,
						tau_min=2880., tau_max=14400.):
		"""
		Time window is a function of the magnitude of
		and the time since the largest event in the cluster

		:param mag:
			float or array, magnitude of largest event in cluster
		:param time_delta:
			numpy timedelta64 or float or array,
			time difference between event and largest event in cluster
			(in minutes if float)
			(default: 0 = case where no cluster has been defined yet)
		:param tau_min:
			float, minimum length of time window (in minutes)
			(default: 2880. = 2 days)
		:param tau_max:
			float, maximum length of time window (in minutes)
			(default: 144000. = 10 days)

		:return:
			numpy timedelta64
		"""
		from .time_functions_np import fractional_time_delta, is_np_timedelta

		is_scalar = np.isscalar(mag)
		if is_scalar:
			mag = np.array([mag])
		else:
			mag = np.asarray(mag)
		# TODO: xmeff should be function argument
		P1, XK, XMEFF = 0.99, 0.5, 1.5
		if is_np_timedelta(time_delta):
			td = fractional_time_delta(time_delta, 'm')
		else:
			td = float(time_delta)
		if (np.isscalar(td) or len(td) == 1):
			td = np.repeat(td, len(mag))
		tau = np.ones_like(mag) * tau_min
		pos_idxs = (td > 0)
		if pos_idxs.any():
			## deltam: delta in magnitude
			deltam = (1. - XK) * mag[pos_idxs] - XMEFF
			deltam = np.maximum(0, deltam)
			## denom: expected rate of aftershocks
			denom = 10**((deltam - 1.) * 2./3)
			top = -np.log(1. - P1) * td[pos_idxs]
			_tau = top / denom
			## Truncate tau to not exceed TAUMAX or drop below TAUMIN
			tau[pos_idxs] = np.maximum(tau_min, np.minimum(tau_max, _tau))
		tau_td = np.round(tau * 60).astype('m8[s]')
		if is_scalar:
			tau_td = tau_td[0]
		return tau_td

	def get_dist_window(self, mag, cluster_mag=0,
						rfact=10, dsigma=30, rmax=30):
		"""
		Calculate interaction radius for an event and for the largest
		event in the cluster associated with this event.

		The interaction distance about an event is defined as
		r = rfact * a(M, dsigma)
		where a(M, dsigma) is the radius of a circular crack in km
		(Kanamori & Anderson, 1975) corresponding to an earthquake of
		magnitude M and stress drop dsigma.

		:param mag:
			float, magnitude of event
		:param cluster_mag:
			float, magnitude of largest event in cluster associated
			with same event
			(default: 0 = case where no cluster has been defined yet)
		:param rfact:
			float, number of crack radii (see Kanamori and Anderson,
			1975) surrounding each earthquake within which to
			consider linking a new event into cluster
			(default: 10)
		:param dsigma:
			float, stress drop (in bars)
			(default: 30)
		:param rmax:
			float, maximum interaction distance (in km)
			(default: 30, i.e. one crustal thickness)

		:return:
			float, distance (in km)
		"""
		if dsigma == 30:
			dsigma_term = 0.011
		else:
			dsigma_term = 10**((-np.log10(dsigma))/3. - 1.45)
		r1 = rfact * dsigma_term * 10**(0.4 * mag)
		rmain = dsigma_term * 10**(0.4 * cluster_mag)
		## Limit interaction distance to one crustal thickness
		rtest = np.minimum(rmax, r1 + rmain)
		return rtest


class Cluster():
	"""
	Class representing an earthquake cluster.

	:param eq_list:
		list with instances of :class:`LocalEarthquake`,
		earthquakes composing the cluster
	:param ID:
		int or str, ID of cluster
	:param Mrelation:
		dict specifying how to convert cluster magnitudes to MW
	:param distance_metric:
		str, metric used to compute inter-event distances,
		either "hypocentral" or "epicentral"
		(default: "hypocentral")
	"""
	def __init__(self, eq_list, ID, Mrelation, distance_metric="hypocentral"):
		self.eq_list = eq_list
		self.ID = ID
		self.Mrelation = Mrelation
		self.distance_metric = distance_metric

	def __len__(self):
		return len(self.eq_list)

	def __iter__(self):
		return self.eq_list.__iter__()

	def __getitem__(self, item):
		subselection = np.array(self.eq_list).__getitem__(item)
		if isinstance(subselection, LocalEarthquake):
			return subselection
		else:
			eq_list = list(subselection)
			return self.__class__(eq_list, self.ID, self.Mrelation,
								self.distance_metric)

	def __contains__(self, eq):
		assert isinstance(eq, LocalEarthquake)
		return eq.ID in self.get_ids()

	def print_info(self):
		print("Cluster #%d" % self.ID)
		print("Number of events: %d" % len(self))
		print("Mainshock: %s (M=%.1f)" % (self.datetime1(), self.mag1()))
		print("First event: %s (M=%.1f)" % (self.datetime0(), self.mag0()))
		print("Duration: %d days" % self.duration().astype('<m8[D]').astype('int'))
		print("Equivalent magnitude: %.1f" % self.mag)

	def append(self, eq):
		"""
		Append earthquake to cluster

		:param eq:
			instance of :class:`LocalEarthquake`
		"""
		self.eq_list.append(eq)

	def extend(self, other_cluster):
		"""
		Extend cluster with another cluster

		:param other_cluster:
			instance of :class:`Cluster`
		"""
		self.eq_list.extend(other_cluster.eq_list)

	def get_ids(self):
		"""
		:return:
			list containing ID of all events in cluster
		"""
		return [eq.ID for eq in self.eq_list]

	def get_longitudes(self):
		"""
		:return:
			array containing longitude of all events in cluster
		"""
		lons = np.array([eq.lon for eq in self])
		return lons

	def get_latitudes(self):
		"""
		:return:
			array containing latitude of all events in cluster
		"""
		lats = np.array([eq.lat for eq in self])
		return lats

	def get_depths(self):
		"""
		:return:
			array containing depth of all events in cluster
		"""
		depths = np.array([eq.depth for eq in self])
		return depths

	def get_magnitudes(self):
		"""
		:return:
			array containing magnitude of all events in cluster
		"""
		mags = np.array([eq.get_MW(Mrelation=self.Mrelation) for eq in self])
		mags[np.isnan(mags)] = min(0, np.nanmin(mags))
		return mags

	def get_datetimes(self):
		"""
		:return:
			array containing datetime of all events in cluster
		"""
		datetimes = np.array([eq.datetime for eq in self])
		return datetimes

	def get_timedeltas(self):
		"""
		:return:
			array containing time delta with respect to first event
		"""
		return self.get_datetimes() - self.datetime0()

	def get_distances_to(self, reference="centroid"):
		"""
		:param reference:
			str, one of "centroid" or "mainshock"
			(default: "centroid")

		:return:
			array containing distance (in km) with respect to reference
		"""
		if reference == "centroid":
			eq = self.get_equivalent_event()
		elif reference == "mainshock":
			eq = self.get_mainshock()
		lons = self.get_longitudes()
		lats = self.get_latitudes()
		depths = self.get_depths()
		return eq.hypocentral_distance((lons, lats, depths))

	@property
	def lon(self):
		"""
		:return:
			float, centroid longitude of cluster (in degrees)
		"""
		return np.mean(self.get_longitudes())

	@property
	def lat(self):
		"""
		:return:
			float, centroid latitude of cluster (in degrees)
		"""
		return np.mean(self.get_latitudes())

	@property
	def depth(self):
		"""
		:return:
			float, centroid depth of cluster (in degrees)
		"""
		return np.mean(self.get_depths())

	@property
	def moment(self):
		"""
		:return:
			float, total seismic moment of cluster (in N.m)
		"""
		moments = mag_to_moment(self.get_magnitudes())
		return np.sum(moments)

	@property
	def mag(self):
		"""
		:return:
			float, equivalent magnitude of cluster
		"""
		return moment_to_mag(self.moment)

	def argsort_mag(self):
		"""
		Return indexes that would sort cluster by descending magnitude

		:return:
			(mags, order) tuple:
			- mags: array, unsorted magnitudes
			- order: array, ordered indexes
		"""
		mags = self.get_magnitudes()
		nan_idxs = np.isnan(mags)
		ordered_idxs = np.hstack([np.argsort(mags[~nan_idxs])[::-1], nan_idxs])
		return (mags, ordered_idxs)

	def sort_mag(self):
		"""
		Get copy of cluster sorted by descending magnitude

		:return:
			instance of :class:`Cluster`
		"""
		mags, order = self.argsort_mag()
		return Cluster(self.__getitem__(order), self.ID, self.Mrelation,
						self.distance_metric)

	def argsort_datetime(self):
		"""
		Return indexes that would sort cluster by (ascending) datetime

		:return:
			(mags, order) tuple:
			- mags: array, unsorted datetimes
			- order: array, ordered indexes
		"""
		datetimes = self.get_datetimes()
		return (datetimes, np.argsort(datetimes))

	def sort_datetime(self):
		"""
		Get copy of cluster sorted by ascending datetime

		:return:
			instance of :class:`Cluster`
		"""
		datetimes, order = self.argsort_datetime()
		return Cluster(self.__getitem__(order), self.ID, self.Mrelation,
						self.distance_metric)

	def get_mainshock_idx(self, rank=0):
		"""
		Determine index of mainshock or n'th largest event

		:param rank:
			int, 0 for largest, 1 for second largest, -1 for smallest, ...
			(default: 0)

		:return:
			int, index
		"""
		mags, order = self.argsort_mag()
		return order[rank]

	def get_mainshock(self, rank=0):
		"""
		Return mainshock of n'th largest event

		:param rank:
			int, 0 for largest, 1 for second largest, -1 for smallest, ...
			(default: 0)

		:return:
			instance of :class:`LocalEarthquake`
		"""
		return self.eq_list[self.get_mainshock_idx(rank=rank)]

	def mag1(self, rank=0):
		"""
		Determine n'th largest magnitude in cluster

		:param rank:
			int, 0 for largest, 1 for second largest, -1 for smallest, ...
			(default: 0)

		:return:
			float, magnitude
		"""
		mags, order = self.argsort_mag()
		return mags[order[rank]]

	def mag0(self):
		"""
		Return magnitude of first event in cluster

		:return:
			float
		"""
		datetimes, order = self.argsort_datetime()
		mags = self.get_magnitudes()
		return mags[order[0]]

	def datetime1(self, rank=0):
		"""
		Determine datetime of mainshock or n'th largest event

		:param rank:
			int, 0 for largest, 1 for second largest, -1 for smallest, ...
			(default: 0)

		:return:
			numpy datetime64
		"""
		return self.get_mainshock(rank=rank).datetime

	def datetime0(self):
		"""
		Return datetime of first event in cluster

		:return:
			numpy datetime64
		"""
		datetimes, order = self.argsort_datetime()
		return datetimes[order[0]]

	def duration(self):
		"""
		Determine total duration of cluster

		:return:
			numpy timedelta64
		"""
		datetimes, order = self.argsort_datetime()
		start_dt, end_dt = datetimes.min(), datetimes.max()
		return end_dt - start_dt

	def get_equivalent_event(self):
		"""
		Return "equivalent event" to cluster, composed as follows:
		- datetime is datetime of largest event in cluster
		- magnitude reflects summed moment of cluster
		- hypocenter is centroid of cluster (not weighted by magnitude)
		- error parameters are from largest event in cluster

		:return:
			instance of :class:`LocalEarthquake`
		"""
		from .eqrecord import LocalEarthquake
		from .time_functions_np import (as_np_date, to_py_time)

		## Use datetime and error parameters from largest event in cluster
		mainshock = self.get_mainshock()
		datetime = mainshock.datetime
		date, time = as_np_date(datetime), to_py_time(datetime)
		errh, errz, errt = mainshock.errh, mainshock.errz, mainshock.errt
		errM = mainshock.errM
		eq = LocalEarthquake(self.ID, date, time, self.lon, self.lat, self.depth,
							{'MW': self.mag}, errh=errh, errz=errz, errt=errt,
							errM=errM)
		return eq

	def get_foreshocks(self):
		"""
		Get foreshocks in cluster

		:return:
			instance of :class:`EQCatalog`
		"""
		mainshock_dt = self.datetime1()
		foreshocks = [eq for eq in self.sort_datetime()
						if eq.datetime < mainshock_dt]
		return EQCatalog(foreshocks, name=self.catalog.name + ' (foreshocks)',
						default_Mrelations={'MW': self.Mrelation})

	def get_aftershocks(self):
		"""
		Get aftershocks in cluster

		:return:
			instance of :class:`EQCatalog`
		"""
		mainshock_dt = self.datetime1()
		aftershocks = [eq for eq in self.sort_datetime()
						if eq.datetime > mainshock_dt]
		return EQCatalog(aftershocks, name=self.catalog.name + ' (aftershocks)',
						default_Mrelations={'MW': self.Mrelation})

	def to_catalog(self):
		"""
		Convert to EQCatalog

		:return:
			instance of :class:`EQCatalog`
		"""
		return EQCatalog(self.eq_list, name="Cluster #%s" % self.ID,
						default_Mrelations={'MW': self.Mrelation})

	def get_end_of_time_windows(self, dc_window):
		"""
		Return end of time window of all events in cluster

		:param dc_window:
			instance of :class:`DeclusteringWindow`

		:return:
			numpy datetime64 array
		"""
		mags = self.get_magnitudes()
		datetimes = self.get_datetimes()
		time_deltas = dc_window.get_time_window(mags)
		return datetimes + time_deltas

	def get_combined_time_window(self, dc_window):
		"""
		Return start and end of combined time window determined by
		all events in cluster

		:param dc_window:
			instance of :class:`DeclusteringWindow`

		:return:
			(start_datetime, end_datetime) tuple of numpy datetime64
		"""
		datetimes = self.get_datetimes()
		start_datetime = datetimes.min()
		end_datetimes = self.get_end_of_time_windows(dc_window)
		return (start_datetime, end_datetimes.max())

	def is_in_combined_time_window(self, catalog, dc_window):
		"""
		Determine which events in a catalog are inside the combined
		time window of the cluster. Note that this does not mean that
		these events indeed belong to the same cluster.
		Use :meth:`is_in_time_window` for that.
		This is useful, however, to limit the potential aftershocks
		in a catalog.

		:param catalog:
			instance of :class:`EQCatalog`
		:param dc_window:
			instance of :class:`DeclusteringWindow`

		:return:
			bool array
		"""
		start_dt, end_dt = self.get_combined_time_window(dc_window)
		datetimes = catalog.get_datetimes()
		return (datetimes >= start_dt) & (datetimes <= end_dt)

	def is_in_time_window(self, eq, dc_window):
		"""
		Determine whether a particular earthquake is in the time window
		of any of the events in the cluster.

		:param eq:
			instance of :class:`LocalEarthquake`
		:param dc_window:
			instance of :class:`DeclusteringWindow`

		:return:
			bool array
		"""
		datetimes = self.get_datetimes()
		end_datetimes = self.get_end_of_time_windows(dc_window)
		return ((datetimes <= eq.datetime) & (eq.datetime <= end_datetimes))

	def is_in_dist_window(self, eq, dc_window):
		"""
		Determine whether a particular earthquake is in the distance window
		of any of the events in the cluster.

		:param eq:
		:param dc_window:
			see :meth:`is_in_time_window`

		:return:
			bool array
		"""
		mags = self.get_magnitudes()
		distance_limits = dc_window.get_dist_window(mags)
		lons, lats = self.get_longitudes(), self.get_latitudes()
		if self.distance_metric == "hypocentral":
			depths = self.get_depths()
			distances = eq.hypocentral_distance((lons, lats, depths))
		else:
			distances = eq.epicentral_distance((lons, lats))
		return (distances <= distance_limits)

	def is_in_window(self, eq, dc_window):
		"""
		Determine whether a particular earthquake is in the time and
		distance window of any of the events in the cluster.

		:param eq:
		:param dc_window:
			see :meth:`is_in_time_window`

		:return:
			bool
		"""
		#return (self.is_in_time_window(eq, dc_window)
		#		& self.is_in_dist_window(eq, dc_window)).any()
		in_t_window = self.is_in_time_window(eq, dc_window)
		subcluster = self.__getitem__(in_t_window)
		in_d_window = subcluster.is_in_dist_window(eq, dc_window)
		in_window = in_t_window
		in_window[in_t_window] &= in_d_window
		return in_window.any()


class DeclusteringResult():
	"""
	Class representing full declustering result

	:param catalog:
		instance of :class:`EQCatalog`, original, undeclustered catalog
	:param dc_idxs:
		1-D int array, containing cluster index for each earthquake
		in :param:`catalog`,
		clusters are numbered starting from 0,
		-1 indicates earthquake does not belong to a cluster
	:param Mrelation:
		dict specifying how to convert catalog magnitudes to MW
	:param distance_metric:
		str, metric that was used to compute inter-event distances,
		either "hypocentral" or "epicentral"
	"""
	# TODO: set names of returned catalogs
	def __init__(self, catalog, dc_idxs, Mrelation, distance_metric):
		self.catalog = catalog
		self.dc_idxs = dc_idxs
		self.Mrelation = Mrelation
		self.distance_metric = distance_metric
		self._fix_indexes()

	def print_info(self):
		print("Number of clusters identified: %d" % self.get_num_clusters())
		print("Max. cluster length: %d" % self.get_max_cluster_length())
		print("Num. clustered/unclustered events: %d / %d"
			% (self.get_num_clustered_events(), self.get_num_unclustered_events()))
		print("Num. dependent/independent events: %d / %d"
			% (self.get_num_dependent_events(), self.get_num_independent_events()))

	def _fix_indexes(self):
		"""
		Cluster index array may contain gaps (when clusters were merged
		during the declustering process). This reorders cluster indexes
		such that they are contiguous.
		"""
		current_idxs = self.get_cluster_ids()
		ordered_idxs = np.arange(self.get_num_clusters())
		for current_idx, ordered_idx in zip(current_idxs, ordered_idxs):
			if current_idx != ordered_idx:
				self.dc_idxs[self.dc_idxs == current_idx] = ordered_idx

	def get_cluster_ids(self):
		"""
		:return:
			array containing IDs (= indexes) of identified clusters
		"""
		return np.sort(np.unique(self.dc_idxs[self.dc_idxs >= 0]))

	def argsort_clusters(self, sort_key='datetime0', order='asc'):
		"""
		Return sorted cluster indexes

		:param sort_key:
			str, sort key: 'datetime0', 'datetime1', 'size', 'mag' or 'duration'
			(default: 'datetime0')
		:param order:
			str, sort order: 'asc' or 'desc'
			(default: 'asc')

		:return:
			1-D int array
		"""
		cluster_idxs = self.get_cluster_ids()
		if sort_key == 'datetime0':
			cluster_props = [self.get_cluster(idx).datetime0() for idx in cluster_idxs]
		elif sort_key == 'datetime1':
			cluster_props = [self.get_cluster(idx).datetime1() for idx in cluster_idxs]
		elif sort_key == 'size':
			cluster_props = [len(self.get_cluster(idx)) for idx in cluster_idxs]
		elif sort_key == 'mag':
			cluster_props = [self.get_cluster(idx).mag1() for idx in cluster_idxs]
		elif sort_key == 'duration':
			cluster_props = [self.get_cluster(idx).duration() for idx in cluster_idxs]

		ordered_prop_idxs = np.argsort(cluster_props)
		if order == 'desc':
			ordered_prop_idxs = ordered_prop_idxs[::-1]

		return cluster_idxs[ordered_prop_idxs]

	def get_cluster_lengths(self):
		"""
		:return:
			int array, length of each identified cluster
		"""
		return np.array([len(clust) for clust in self.get_clusters()])

	def get_max_cluster_length(self):
		"""
		:return:
			int, length of largest cluster
		"""
		return self.get_cluster_lengths().max()

	def get_num_clusters(self):
		"""
		:return:
			int, total number of identified clusters
		"""
		return len(self.get_cluster_ids())

	def get_num_clustered_events(self):
		"""
		:return:
			int, total number of events in catalog that belong
			to a cluster
		"""
		return len(self.dc_idxs[self.dc_idxs >= 0])

	def get_num_unclustered_events(self):
		"""
		:return:
			int, total number of events in catalog that do not belong
			to a cluster
		"""
		return len(self.dc_idxs[self.dc_idxs < 0])

	def get_num_independent_events(self):
		"""
		:return:
			int, total number of independent events
			(= number of clusters + number of unclustered events)
		"""
		return self.get_num_clusters() + self.get_num_unclustered_events()

	def get_num_dependent_events(self):
		"""
		:return:
			int, total number of dependent events
			(= number of clustered events - number of clusters)
		"""
		return self.get_num_clustered_events() - self.get_num_clusters()

	def get_clusters(self):
		"""
		:return:
			list with instances of :class:`Cluster`
		"""
		clusters = []
		for cluster_id in self.get_cluster_ids():
			cluster = self.get_cluster(cluster_id)
			clusters.append(cluster)
		return clusters

	def get_cluster(self, cluster_idx):
		"""
		Fetch cluster with given index

		:param cluster_idx:
			int, cluster index

		:return:
			instance of :class:`Cluster`
		"""
		eq_list = (self.catalog[self.dc_idxs == cluster_idx]).eq_list
		Mrelation = self.Mrelation or self.catalog.default_Mrelations['MW']
		cluster = Cluster(eq_list, cluster_idx, Mrelation, self.distance_metric)
		return cluster

	def get_cluster_by_eq_idx(self, eq_idx):
		"""
		Fetch cluster corresponding to given index of earthquake in catalog

		:param eq_idx:
			int, index of earthquake in :prop:`catalog`

		:return:
			instance of :class:`Cluster`
		"""
		cluster_idx = self.dc_idxs[eq_idx]
		return self.get_cluster(cluster_idx)

	def get_cluster_by_eq(self, eq):
		"""
		Fetch the cluster that a particular earthquake belongs to

		:param eq:
			instance of :class:`LocalEarthquake`

		:return:
			instance of :class:`Cluster`
		"""
		eq_idx = self.catalog.index(eq)
		return self.get_cluster_by_eq_idx(eq_idx)

	def get_unclustered_events(self):
		"""
		:return:
			instance of :class:`EQCatalog` containing unclustered events
		"""
		return self.catalog[self.dc_idxs < 0]

	def get_clustered_events(self):
		"""
		:return:
			instance of :class:`EQCatalog` containing all clustered events
		"""
		return self.catalog[self.dc_idxs >= 0]

	def get_declustered_catalog(self, replace_clusters_with="mainshocks"):
		"""
		:param replace_clusters_with:
			str, one of "mainshocks" or "equivalent_events"
			(default: "mainshocks")

		:return:
			instance of :class:`EQCatalog` containing declustered catalog
		"""
		if replace_clusters_with == "mainshocks":
			mainshocks = self.get_mainshocks()
		elif replace_clusters_with == "equivalent_events":
			mainshocks = self.get_equivalent_events()
		unclustered_events = self.get_unclustered_events()
		declustered_catalog = mainshocks + unclustered_events
		declustered_catalog.sort()
		return declustered_catalog

	get_independent_events = get_declustered_catalog

	def get_dependent_events(self):
		"""
		:return:
			instance of :class:`EQCatalog` containing dependent events
			(= clustered events minus mainshocks)
		"""
		clustered_events = self.get_clustered_events()
		mainshocks = self.get_mainshocks()
		return clustered_events - mainshocks

	def get_mainshocks(self):
		"""
		:return:
			instance of :class:`EQCatalog` containing mainshocks of
			each cluster
		"""
		clusters = self.get_clusters()
		mainshocks = [cluster.get_mainshock() for cluster in clusters]
		return EQCatalog(mainshocks, name=self.catalog.name + ' (mainshocks)',
						default_Mrelations={'MW': self.Mrelation})

	def get_equivalent_events(self):
		"""
		:return:
			instance of :class:`EQCatalog` containing equivalent events
			of each cluster
		"""
		clusters = self.get_clusters()
		equivalent_events = [cluster.get_equivalent_event() for cluster in clusters]
		return EQCatalog(equivalent_events, name=self.catalog.name + ' (equiv. events)',
						default_Mrelations={'MW': self.Mrelation})


class DeclusteringMethod():
	"""
	Class implementing a declustering method, which splits earthquakes in
	dependent and independent ones.
	"""

	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def decluster_catalog(self):
		"""
		:returns:
			instance of :class:`EQCatalog`, declustered catalog
		"""
		pass

	@abc.abstractmethod
	def analyze_clusters(self):
		"""
		:returns:
			instance of :class:`DeclusteringResult`
		"""
		pass


class WindowMethod(DeclusteringMethod):
	"""
	Class implementing simple window declustering method.
	In this method, aftershocks are not considered to generate
	additional aftershocks.

	Note that the use of this method is not recommended, because
	it is conceptually wrong.
	For real use, the cluster (= linked-window) method is preferred.

	:param fa_ratio:
		float, ratio between foreshock and aftershock time windows
		(default: 0.17, cf. SHARE project, Danciu pers. comm.)
	:param distance_metric:
		str, metric used to compute inter-event distances,
		either "hypocentral" or "epicentral"
		(default: "hypocentral"
	"""
	def __init__(self, fa_ratio=0.17, distance_metric="hypocentral"):
		self.fa_ratio = fa_ratio
		self.distance_metric = distance_metric
		print("Warning: Using the window method for declustering "
			"is not recommended!")

	def is_in_time_window(self, main_event, main_mag, catalog, dc_window):
		"""
		Determine if catalog events are in time window of main event

		:param main_event:
			instance of :class:`LocalEarthquake`
		:param main_mag:
			float, magnitude of main event
		:param catalog:
			instance of :class:`EQCatalog` (or :class:`Cluster`)
		:param dc_window:
			instance of :class:`DeclusteringWindow`

		:return:
			bool array
		"""
		## Main event and catalog parameters
		main_datetime = main_event.datetime
		datetimes = catalog.get_datetimes()

		## Get time window
		t_window = dc_window.get_time_window(main_mag)

		## Create time window index
		in_t_window = ((datetimes >= (main_datetime - self.fa_ratio * t_window))
						& (datetimes <= (main_datetime + t_window)))
		return in_t_window

	def is_in_dist_window(self, main_event, main_mag, catalog, dc_window):
		"""
		Determine if catalog events are in distance window of main event

		:param main_event:
		:param main_mag:
		:param catalog:
		:param dc_window:
			see :meth:`is_in_time_window`

		:return:
			bool array
		"""
		## Main event and catalog parameters
		lons, lats = catalog.get_longitudes(), catalog.get_latitudes()

		## Get distance window
		d_window = dc_window.get_dist_window(main_mag)

		## Create distance window index
		if self.distance_metric == "hypocentral":
			depths = catalog.get_depths()
			in_d_window = (main_event.hypocentral_distance((lons, lats, depths))
							<= d_window)
		else:
			in_d_window = main_event.epicentral_distance((lons, lats)) <= d_window
		return in_d_window

	def is_in_window(self, main_event, main_mag, catalog, dc_window):
		"""
		Determine if catalog events are in time and distance window
		of main event

		:param main_event:
		:param main_mag:
		:param catalog:
		:param dc_window:
			see :meth:`is_in_time_window`

		:return:
			bool array
		"""
		in_t_window = self.is_in_time_window(main_event, main_mag, catalog, dc_window)
		#in_d_window = self.is_in_dist_window(main_event, main_mag, catalog, dc_window)
		#in_window = (in_t_window & in_d_window)
		in_d_window = self.is_in_dist_window(main_event, main_mag, catalog[in_t_window],
											dc_window)
		in_window = in_t_window
		in_window[in_t_window] &= in_d_window
		return in_window

	def get_dependent_events(self, main_event, catalog, dc_window, Mrelation):
		"""
		Extract all events from earthquake catalog that depend on given
		main event (excluding the main event itself).
		It is not necessary that the main event is in the catalog.

		:param main_event:
			instance of :class:`LocalEarthquake`
		:param catalog:
			instance of :class:`EQCatalog` (or :class:`Cluster`)
		:param dc_window:
			instance of :class:`DeclusteringWindow`
		:param Mrelation:
			dict specifying how to convert catalog magnitudes to MW

		:return:
			instance of :class:`EQCatalog` containing dependent events
		"""
		Mrelation = Mrelation or catalog.default_Mrelations['MW']
		main_mag = main_event.get_MW(Mrelation)
		in_window = self.is_in_window(main_event, main_mag, catalog, dc_window)
		return catalog[in_window & (catalog.get_ids() != main_event.ID)]

	def get_independent_events(self, main_event, catalog, dc_window, Mrelation):
		"""
		Extract all events from earthquake catalog that do not depend
		on given main event.
		It is not necessary that the main event is in the catalog.

		:param main_event:
		:param catalog:
		:param dc_window:
		:param Mrelation:
			see :meth:`get_dependent_events`

		:return:
			instance of :class:`EQCatalog` containing independent events
		"""
		Mrelation = Mrelation or catalog.default_Mrelations['MW']
		main_mag = main_event.get_MW(Mrelation)
		in_window = self.is_in_window(main_event, main_mag, catalog, dc_window)
		return catalog[~in_window]

	def get_foreshocks(self, main_event, catalog, dc_window, Mrelation):
		"""
		Extract foreshocks of given main event from earthquake catalog
		It is not necessary that the main event is in the catalog.

		:param main_event:
		:param catalog:
		:param dc_window:
		:param Mrelation:
			see :meth:`get_dependent_events`

		:return:
			instance of :class:`EQCatalog` containing foreshocks
		"""
		dependent_events = self.get_dependent_events(main_event, catalog, dc_window,
													Mrelation)
		foreshock_idxs = [i for i in range(len(dependent_events))
						if dependent_events[i].datetime < main_event.datetime]
		return dependent_events[foreshock_idxs]

	def get_aftershocks(self, main_event, catalog, dc_window, Mrelation):
		"""
		Extract aftershocks of given main event from earthquake catalog
		It is not necessary that the main event is in the catalog.

		:param main_event:
		:param catalog:
		:param dc_window:
		:param Mrelation:
			see :meth:`get_dependent_events`

		:return:
			instance of :class:`EQCatalog` containing aftershocks
		"""
		fa_ratio = self.fa_ratio
		self.fa_ratio = 0.
		return self.get_dependent_events(main_event, catalog, dc_window, Mrelation)
		self.fa_ratio = fa_ratio

	def analyze_clusters(self, catalog, dc_window, Mrelation,
						include_larger_aftershocks=1, verbose=False):
		"""
		Full analysis of individual clusters in earthquake catalog.
		Note that earthquakes that could be aftershocks of an earthquake
		that is itself an aftershock of a larger event, are not
		considered as a cluster! This illustrates that the simple window
		method is conceptually wrong.

		:param catalog:
			instance of :class:`EQCatalog`, undeclustered catalog
		:param dc_window:
			instance of :class:`DeclusteringWindow`
		:param Mrelation:
			dict specifying how to convert catalog magnitudes to MW
		:param include_larger_aftershocks:
			int, how to deal with larger earthquakes that are in the
			aftershock window of a smaller event
			0: do not include larger aftershocks in cluster, whatever
				their nature
			1: if there is a larger aftershock that is isolated (i.e.,
				not the mainshock of another cluster), assign it to
				the cluster
			2: if there is a larger aftershock that is the mainshock of
				another cluster, merge the two clusters.
				This possibly corresponds to:
				"reset the dimensions of the window to the largest shock
				in the series, whether the largest was the first or a
				subsequent shock" (Knopoff & Gardner, 1972)
			(default: 1)
		:param verbose:
			bool, whether or not to print progress information

		:return:
			instance of :class:`DeclusteringResult`
		"""
		## Array storing cluster index, -1 means not belonging to any cluster
		dc_idxs = -np.ones(len(catalog), dtype='int')
		ncl = 0

		Mrelation = Mrelation or catalog.default_Mrelations['MW']
		magnitudes = catalog.get_magnitudes(Mtype='MW', Mrelation=Mrelation)
		## Set NaN magnitudes to 0 or lowest magnitude in catalog
		## (necessary because magnitude comparison fails with NaN magnitudes)
		nan_idxs = np.isnan(magnitudes)
		if np.sum(nan_idxs):
			msg = "Warning: setting %d NaN magnitudes to lowest magnitude!"
			msg %= np.sum(nan_idxs)
			print(msg)
			magnitudes[nan_idxs] = min(0, np.nanmin(magnitudes))

		## Order by descending magnitude and ascending time
		mag_dates = np.rec.fromarrays([-magnitudes, catalog.get_datetimes()],
										names=str('magnitude,datetime'))
		order = np.argsort(mag_dates, order=['magnitude', 'datetime'])

		for i in order:
			main_mag = magnitudes[i]
			## if earthquake is not marked as triggered
			if dc_idxs[i] == -1:
				main_event = catalog[i]
				in_window = self.is_in_window(main_event, main_mag, catalog, dc_window)
				if np.sum(in_window) > 1:
					## > 1 because main_event is always in window

					## Do not include earthquakes that are already marked as
					## belonging to the cluster of a larger mainshock
					in_window_and_not_yet_clustered = (in_window & (dc_idxs == -1))

					if include_larger_aftershocks == 2:
						in_window_and_already_clustered = in_window & (dc_idxs >= 0)
						## If there is another cluster (belonging to a larger mainshock)
						## in the window of this one, append to that cluster
						## This possibly corresponds to:
						## "reset the dimensions of the window to the largest shock
						## in the series, whether the largest was the first or a
						## subsequent shock" (Knopoff & Gardner, 1972)
						if np.sum(in_window_and_already_clustered) > 0:
							_ncl = dc_idxs[in_window_and_already_clustered][0]
							if verbose:
								print("Appending %d events to cluster #%d"
								% (np.sum(in_window_and_not_yet_clustered), _ncl))
							dc_idxs[in_window_and_not_yet_clustered] = _ncl
							in_window_and_not_yet_clustered[:] = False

					if include_larger_aftershocks == 0:
						## Do not include larger aftershocks, whatever their nature
						in_window_and_not_yet_clustered &= (magnitudes <= main_mag)

					if np.sum(in_window_and_not_yet_clustered) > 1:
						dc_idxs[in_window_and_not_yet_clustered] = ncl
						if verbose:
							print("Found new cluster #%d (n=%d)"
									% (ncl, np.sum(in_window_and_not_yet_clustered)))
						ncl += 1

		return DeclusteringResult(catalog, dc_idxs, Mrelation, self.distance_metric)

	def decluster_catalog(self, catalog, dc_window, Mrelation, include_larger_aftershocks=True):
		"""
		Decluster catalog quickly, without identifying individual clusters.
		Based on original :meth:`decluster` by Bart Vleminckx.

		Note that this method does not capture larger aftershocks that are
		in the window of a smaller event, in contrast to :meth:`analyze_clusters`

		:param catalog:
			instance of :class:`EQCatalog`
		:param dc_window:
			instance of :class:`DeclusteringWindow`
		:param Mrelation:
			dict specifying how to convert catalog magnitudes to MW
		:param include_larger_aftershocks:
			bool, whether or not to mark earthquakes as dependent
			if there is a larger earthquake in their aftershock window
			(regardless if it is an isolated earthquake or the mainshock
			of another cluster)
			Note that this method corresponds to :meth:`analyze_clusters`
			only if this parameter is False (or zero) in both cases.
			If set to True, there is no correspondance, as there is no
			way to determine whether the larger aftershock is an isolated
			event or the mainshock of another cluster
			(default: True)

		:return:
			instance of :class:`EQCatalog`, declustered catalog
		"""
		## Create array storing declustering status
		## True means mainshock or unclustered, False means dependent
		dc_idxs = np.ones(len(catalog), dtype='bool')

		Mrelation = Mrelation or catalog.default_Mrelations['MW']
		magnitudes = catalog.get_magnitudes(Mtype='MW', Mrelation=Mrelation)
		## Set NaN magnitudes to 0 or lowest magnitude in catalog
		nan_idxs = np.isnan(magnitudes)
		if np.sum(nan_idxs):
			msg = "Warning: setting %d NaN magnitudes to lowest magnitude!"
			msg %= np.sum(nan_idxs)
			print(msg)
			magnitudes[nan_idxs] = min(0, np.nanmin(magnitudes))

		## Order by descending magnitude and ascending time
		mag_dates = np.rec.fromarrays([-magnitudes, catalog.get_datetimes()],
										names=str('magnitude,datetime'))
		order = np.argsort(mag_dates, order=['magnitude', 'datetime'])

		for i in order:
			main_mag = magnitudes[i]
			## If earthquake is not marked as triggered
			if dc_idxs[i] == True:
				## Create window index
				main_event = catalog[i]
				in_window = self.is_in_window(main_event, main_mag, catalog, dc_window)
				if np.sum(in_window) > 1:
					in_window_and_independent = (in_window & dc_idxs)
					in_window_and_not_yet_clustered = (in_window_and_independent
													& (magnitudes <= main_mag))
					if include_larger_aftershocks:
						if (np.sum(in_window_and_independent)
							== np.sum(in_window_and_not_yet_clustered)):
							## No larger events in window
							## All events in window belong to new cluster
							## Current earthquake is mainshock, so remove from window index
							in_window_and_not_yet_clustered[i] = False
						else:
							## There is a larger earthquake in the window
							## Impossible to tell if it is an isolated event
							## or the mainshock of another cluster
							## Mark current earthquake as clustered, leaving the
							## larger earthquake as it is
							in_window_and_not_yet_clustered[i] = True

					else:
						## Ignore larger aftershocks, if any
						in_window_and_not_yet_clustered[i] = False

					## Apply window to declustering index
					if np.sum(in_window_and_not_yet_clustered):
						dc_idxs[in_window_and_not_yet_clustered] = False

		return catalog[dc_idxs]

	def decluster_legacy(self, magnitudes, datetimes, lons, lats, window, fa_ratio):
		"""
		Legacy decluster method written by Bart Vleminckx.
		Left here for illustrative and repeatability purposes.
		Contains a bug, which could result in a mainshock being
		marked as an aftershock of a smaller earthquake,
		particularly when fa_ratio is small (see note).

		:param magnitudes:
			np array, List of magnitudes in chronological order
		:param datetimes:
			np array, List of datetimes in chronological order
		:param lons:
			np array, List of longitudes in chronological order
		:param lats:
			np array, List of latitudes in chronological order
		:param window:
			instance of :class:`DeclusteringWindow`
		:fa_ratio:
			float, ratio of foreshock window to aftershock window
		"""
		from openquake.hazardlib.geo.geodetic import geodetic_distance as distance

		## get order of descending magnitudes
		order = np.argsort(magnitudes)[::-1]
		## create declustering index
		d_index = np.ones_like(order)
		## loop over magnitudes in descending order
		for i in order:
			## if earthquake is not marked as triggered
			if d_index[i] == 1:
				## get time and distance window
				t_window, d_window = window.get_windows(magnitudes[i])
				## make datetime object from time window
				#t_window = datetime.timedelta(days=t_window)
				## create time window index
				in_t_window = np.logical_and(
					#datetimes >= (datetimes[i] - datetime.timedelta(
					#			seconds=fa_ratio*t_window.total_seconds())),
					datetimes >= (datetimes[i] - fa_ratio * t_window),
					datetimes <= (datetimes[i] + t_window),
				)
				## create distance window index
				in_d_window = distance(lons[i], lats[i], lons, lats) <= d_window
				## create window index
				in_window = np.logical_and(in_t_window, in_d_window)
				## remove earthquake from window index
				## Note (KVN) when fa_ratio is small, it is possible that this
				## mainshock becomes an aftershock of a smaller event,
				## and would then be marked as triggered...!
				in_window[i] = 0
				## apply window to declustering index
				d_index[in_window] = 0

		return d_index


class LinkedWindowMethod(DeclusteringMethod):
	"""
	Class implementing cluster (= linked-window) declustering method.
	In this method, aftershocks may generate additional aftershocks,
	possibly extending the duration and spatial extent of a cluster
	beyond that predicted for the mainshock only.

	Note that compared to the original implementation by Bart Vleminckx,
	and compared to the Window method, the fa_ratio parameter has been
	removed, because it is conceptually wrong to identify foreshocks
	in a space/time window based on the mainshock (breaks causality).
	Instead, foreshocks are identified as such if there is a larger
	event (mainshock) in their space/time window.

	:param distance_metric:
		str, metric used to compute inter-event distances,
		either "hypocentral" or "epicentral"
		(default: "hypocentral")
	"""
	def __init__(self, distance_metric="hypocentral"):
		self.distance_metric = distance_metric

	def is_in_time_window(self, main_event, main_mag, catalog, dc_window):
		"""
		Determine if catalog events are in time window of main event

		:param main_event:
			instance of :class:`LocalEarthquake`
		:param main_mag:
			float, magnitude of main event
		:param catalog:
			instance of :class:`EQCatalog` (or :class:`Cluster`)
		:param dc_window:
			instance of :class:`DeclusteringWindow`

		:return:
			bool array
		"""
		## Main event and catalog parameters
		main_datetime = main_event.datetime
		datetimes = catalog.get_datetimes()

		## Get time window
		t_window = dc_window.get_time_window(main_mag)

		## Create time window index
		in_t_window = ((datetimes >= main_datetime)
						& (datetimes <= (main_datetime + t_window)))
		return in_t_window

	def is_in_dist_window(self, main_event, main_mag, catalog, dc_window):
		"""
		Determine if catalog events are in distance window of main event

		:param main_event:
		:param main_mag:
		:param catalog:
		:param dc_window:
			see :meth:`is_in_time_window`

		:return:
			bool array
		"""
		## Main event and catalog parameters
		lons, lats = catalog.get_longitudes(), catalog.get_latitudes()

		## Get distance window
		d_window = dc_window.get_dist_window(main_mag)

		## Create distance window index
		if self.distance_metric == "hypocentral":
			depths = catalog.get_depths()
			in_d_window = (main_event.hypocentral_distance((lons, lats, depths))
							<= d_window)
		else:
			in_d_window = main_event.epicentral_distance((lons, lats)) <= d_window
		return in_d_window

	def is_in_window(self, main_event, main_mag, catalog, dc_window):
		"""
		Determine if catalog events are in time and distance window
		of main event

		:param main_event:
		:param main_mag:
		:param catalog:
		:param dc_window:
			see :meth:`is_in_time_window`

		:return:
			bool array
		"""
		in_t_window = self.is_in_time_window(main_event, main_mag, catalog, dc_window)
		#in_d_window = self.is_in_dist_window(main_event, main_mag, catalog, dc_window)
		#in_window = (in_t_window & in_d_window)
		in_d_window = self.is_in_dist_window(main_event, main_mag, catalog[in_t_window],
											dc_window)
		in_window = in_t_window
		in_window[in_t_window] &= in_d_window

		return in_window

	def _find_aftershocks(self, main_event, catalog, dc_window, Mrelation):
		"""
		Re-entrant function used in :meth:`get_aftershocks`
		"""
		Mrelation = Mrelation or catalog.default_Mrelations['MW']
		magnitudes = catalog.get_magnitudes(Mtype='MW', Mrelation=Mrelation)
		main_mag = main_event.get_MW(Mrelation)
		in_window = self.is_in_window(main_event, main_mag, catalog, dc_window)
		time_deltas = catalog.get_datetimes() - main_event.datetime
		order = np.argsort(time_deltas)
		is_later = time_deltas > np.timedelta64(0, 's')

		later_and_in_window = order[is_later & in_window]
		aftershock_list = catalog[later_and_in_window].eq_list
		after_aftershock_list = []
		for eq, idx in zip(aftershock_list, later_and_in_window):
			remaining_catalog = catalog[idx+1:]
			if len(remaining_catalog):
				after_aftershocks = self._find_aftershocks(eq, remaining_catalog,
														dc_window, Mrelation)
				for aas in after_aftershocks:
					if not (aas in aftershock_list or aas in after_aftershock_list):
						after_aftershock_list.append(aas)

		return aftershock_list + after_aftershock_list

	def get_aftershocks(self, main_event, catalog, dc_window, Mrelation):
		"""
		Extract aftershocks of given main event from earthquake catalog
		It is not necessary that the main event is in the catalog.

		:param main_event:
		:param catalog:
		:param dc_window:
		:param Mrelation:
			see :meth:`get_dependent_events`

		:return:
			instance of :class:`EQCatalog` containing aftershocks
		"""
		# TODO: catalog name, start and end date
		aftershocks = self._find_aftershocks(main_event, catalog, dc_window, Mrelation)
		if len(aftershocks):
			return EQCatalog(aftershocks, name=catalog.name + ' (aftershocks)',
						default_Mrelations={'MW': self.Mrelation})

	def get_foreshocks(self, main_event, catalog, dc_window, Mrelation):
		# TODO
		mainshock_idx = catalog.index(main_event)
		subcat = catalog[:mainshock_idx + 1]

	def analyze_clusters(self, catalog, dc_window, Mrelation, verbose=False):
		"""
		Full analysis of individual clusters in earthquake catalog.

		:param catalog:
			instance of :class:`EQCatalog`, undeclustered catalog
		:param dc_window:
			instance of :class:`DeclusteringWindow`
		:param Mrelation:
			dict specifying how to convert catalog magnitudes to MW
		:param verbose:
			bool, whether or not to print progress information

		:return:
			instance of :class:`DeclusteringResult`
		"""
		## Make sure catalog is ordered by date (ascending)
		catalog = catalog.get_sorted()
		Mrelation = Mrelation or catalog.default_Mrelations['MW']
		magnitudes = catalog.get_magnitudes(Mtype='MW', Mrelation=Mrelation)
		if np.sum(np.isnan(magnitudes)):
			print("Warning: Catalog contains NaN magnitudes!")
			print("Better remove these first.")

		## Array storing cluster index, -1 means not belonging to any cluster
		dc_idxs = -np.ones(len(catalog), dtype='int')
		clusters = []

		for i, eqi in enumerate(catalog):
			## Skip NaN magnitudes
			if np.isnan(eqi.get_MW(Mrelation)):
				continue

			## Create temporary cluster with 1 event
			clust = Cluster([eqi], -1, Mrelation, self.distance_metric)

			## Keep getting jth events until we are out of cluster's time window
			for j in range(i+1, len(catalog)):
				## Skip the jth event if it is already identified as being part
				## of the cluster associated with the ith event
				if (dc_idxs[i] == dc_idxs[j]) and (dc_idxs[i] > -1):
					continue

				eqj = catalog[j]
				end_dt = clust.get_combined_time_window(dc_window)[1]
				if eqj.datetime > end_dt:
					## Out of time window, proceed to next event i
					break
				else:
					if clust.is_in_window(eqj, dc_window):
						## Cluster declared
						if dc_idxs[i] == -1 and dc_idxs[j] == -1:
							## Initialize new cluster
							clust.append(eqj)
							ncl = len(clusters)
							clust.ID = ncl
							clusters.append(clust)
							dc_idxs[i] = dc_idxs[j] = k = ncl
							if verbose:
								print("Initializing new cluster %d" % (ncl))
						elif dc_idxs[j] == -1:
							## If event i is already associated with a cluster,
							## add event j to it
							k = dc_idxs[i]
							clusters[k].append(eqj)
							dc_idxs[j] = k
						elif dc_idxs[i] == -1:
							## If event j is already associated with a cluster,
							## add event i to it
							k = dc_idxs[j]
							clusters[k].append(eqi)
							dc_idxs[i] = k
						else:
							## Combine existing clusters by merging into earlier cluster
							k = min(dc_idxs[i], dc_idxs[j])
							l = max(dc_idxs[i], dc_idxs[j])
							clusters[k].extend(clusters[l])
							clusters[l].eq_list = []
							dc_idxs[dc_idxs == l] = k
							if verbose:
								print("Combining clusters %d and %d" % (k, l))

						## Set clust back to cluster eqi belongs to
						clust = clusters[k]

		return DeclusteringResult(catalog, dc_idxs, Mrelation, self.distance_metric)

	def decluster_catalog(self, catalog, dc_window, Mrelation):
		"""
		Decluster catalog quickly, without identifying individual clusters.
		This implementation catches aftershocks of aftershocks,
		but fails to catch foreshocks if the mainshock is not in the direct
		space/time window...

		Left here for illustrative purposes.

		:param catalog:
			instance of :class:`EQCatalog`
		:param dc_window:
			instance of :class:`DeclusteringWindow`
		:param Mrelation:
			dict specifying how to convert catalog magnitudes to MW

		:return:
			instance of :class:`EQCatalog`, declustered catalog
		"""
		print("Warning: this method doesn't work correctly.")
		print("Use analyze_clusters method instead!")
		## Create array storing declustering status,
		## True means mainshock or unclustered, False means dependent
		dc_idxs = np.ones(len(catalog), dtype='bool')

		Mrelation = Mrelation or catalog.default_Mrelations['MW']
		magnitudes = catalog.get_magnitudes(Mtype='MW', Mrelation=Mrelation)
		## Set NaN magnitudes to 0 or lowest magnitude in catalog
		nan_idxs = np.isnan(magnitudes)
		if np.sum(nan_idxs):
			msg = "Warning: setting %d NaN magnitudes to lowest magnitude!"
			msg %= np.sum(nan_idxs)
			print(msg)
			magnitudes[nan_idxs] = min(0, np.nanmin(magnitudes))
		## Order by descending magnitude and ascending time
		mag_dates = np.rec.fromarrays([-magnitudes, catalog.get_datetimes()],
										names=str('magnitude,datetime'))
		order = np.argsort(mag_dates, order=['magnitude', 'datetime'])

		for i in order:
			main_mag = magnitudes[i]
			if not np.isnan(main_mag):
				main_event = catalog[i]
				is_clustered = not dc_idxs[i]
				## Find aftershocks of event i in catalog
				in_window = self.is_in_window(main_event, main_mag, catalog, dc_window)
				## Mark smaller afterschocks as clustered in declustering index
				## Larger aftershocks are not yet marked, as one of them will
				## turn out to be the mainshock
				## Note that if there is more than 1 larger aftershock, they are
				## guaranteed to be in each other's window, given that they are
				## in the window of a smaller event
				dc_idxs[in_window & (magnitudes <= main_mag)] = False
				## Set event i as unclustered, except if it is a foreshock
				## If it turns out to be an aftershock of a higher magnitude,
				## this will be overwritten later
				#if (np.sum(in_window) == 1)
				if not(magnitudes[in_window] > main_mag).any():
					## No larger magnitude in window, hence no foreshock...
					## Note: this fails if the mainshock is not in the
					## window of this event, but in the window of an aftershock
					## of this event...
					## Set as independent, except if marked as clustered before
					if not is_clustered:
						dc_idxs[i] = True
				else:
					dc_idxs[i] = False

		"""
		## Note: looping over catalog in reverse time order and
		## marking all aftershocks in turn doesn't work, because
		## mainshocks are marked as clustered if they have a foreshock!
		order = catalog.argsort("datetime", "desc")
		for i in order:
			main_mag = magnitudes[i]
			if not np.isnan(main_mag):
				main_event = catalog[i]
				## Find aftershocks of event i in catalog
				in_window = self.is_in_window(main_event, main_mag, catalog, dc_window)
				## Mark afterschocks as clustered in declustering index
				dc_idxs[in_window] = False
				## Set event i as unclustered. This will be overwritten
				## if it is an aftershock of an earlier earthquake
				dc_idxs[i] = True
		"""

		return catalog[dc_idxs]

	def decluster_legacy(self, magnitudes, datetimes, lons, lats, window, fa_ratio):
		"""
		Legacy decluster method written by Bart Vleminckx.
		Left here for illustrative and repeatability purposes,
		but is not correct!

		:param magnitudes:
			np array, List of magnitudes in chronological order
		:param datetimes:
			np array, List of datetimes in chronological order
		:param lons:
			np array, List of longitudes in chronological order
		:param lats:
			np array, List of latitudes in chronological order
		:param window:
			instance of :class:`DeclusteringWindow`
		:fa_ratio:
			float, ratio of foreshock window to aftershock window
		"""
		from openquake.hazardlib.geo.geodetic import geodetic_distance as distance

		## create declustering index
		declustering = np.ones_like(magnitudes)
		## loop over magnitudes
		for i, magnitude in enumerate(magnitudes):
			## get time and distance window
			t_window, d_window = window.get_windows(magnitude)
			## make datetime object from time window
			#t_window = datetime.timedelta(days=t_window)
			## create window index for equal or smaller earthquakes
			smaller_magnitude = magnitudes <= magnitude
			## create time window index
			in_t_window = np.logical_and(
				#datetimes >= (datetimes[i] - datetime.timedelta(
				#				seconds=fa_ratio*t_window.total_seconds())),
				datetimes >= (datetimes[i] - fa_ratio * t_window),
				datetimes <= (datetimes[i] + t_window),
			)
			## create distance window index
			in_d_window = distance(lons[i], lats[i], lons, lats) <= d_window
			## create window index
			in_window = np.logical_and(in_t_window, in_d_window)
			## remove earthquake from window index if not marked as triggered
			if declustering[i] == 1:
				in_window[i] = 0
			## create window index for smaller earthquakes
			smaller_magnitude_window = np.logical_and(in_window, smaller_magnitude)
			## apply window to declustering index
			declustering[smaller_magnitude_window] = 0
		return declustering


ClusterMethod = LinkedWindowMethod


class ReasenbergMethod(DeclusteringMethod):
	"""
	Cluster2000 method by Reasenberg,
	adapted to python from FORTRAN program cluster2000x.f
	"""
	distance_metric = 'hypocentral'

	def is_in_dist_window(self, eq1, eq2, mag1, cmag1, dc_window,
					rfact=10, dsigma=30, rmax=30, ignore_location_errors=True):
		"""
		Determine whether event1 and event2 are clustered according
		to the radial distance criterion.
		Corresponds to subroutine ctest in cluster2000x.f.

		:param eq1:
			instance of :class:`LocalEarthquake`, first event of
			event pair to be compared
		:param eq2:
			instance of :class:`LocalEarthquake`, second event of
			event pair to be compared
		:param mag1:
			float, magnitude of first event
		:param cmag1:
			float, highest magnitude of cluster first event belongs to.
			Can be zero if no cluster has been defined yet.
		:param dc_window:
			instance of :class:`DeclusteringWindow`
		:param rfact:
		:param dsigma:
		:param rmax:
			see :meth:`Reasenberg1985Window.get_dist_window`
		:param ignore_location_errors:
			bool, whether location errors should be ignored or
			subtracted from the distance between event 1 and event 2

		:return:
			bool
		"""
		## Determine hypocentral distance between events
		d = eq1.epicentral_distance(eq2)
		z = np.abs(eq1.depth - eq2.depth)

		## Reduce hypocentral distance by location uncertainty of both events
		## Note that r can be negative when location uncertainties exceed
		## hypocentral distance
		if not ignore_location_errors:
			#if not np.nan in (eq1.errh, eq1.errz, eq2.errh, eq2.errh):
				## Note use of radial distance in arctan2 funciton in cluster2000x.f
				#alpha = np.arctan2(r, d)
				#ca, sa = np.cos(alpha), np.sin(alpha)
				#e1 = np.sqrt(eq1.errh*eq1.errh * ca*ca + eq1.errz*eq1.errz * sa*sa)
				#e2 = np.sqrt(eq2.errh*eq2.errh * ca*ca + eq2.errz*eq2.errz * sa*sa)
				#r = r - e1 - e2
			if not np.isnan(eq1.errh):
				d -= eq1.errh
			if not np.isnan(eq2.errh):
				d -= eq2.errh
			d = max(0, d)
			if not np.isnan(eq1.errz):
				z -= eq1.errz
			if not np.isnan(eq2.errz):
				z -= eq2.errz
			z = max(0, z)

		r = np.sqrt(d*d + z*z)

		## Calculate interaction radius for the first event of the pair and for
		## the largest event in the cluster associated with the first event
		rtest = dc_window.get_dist_window(mag1, cmag1, rfact=rfact,
										dsigma=dsigma, rmax=rmax)

		if r <= rtest:
			return True
		else:
			return False

	def analyze_clusters(self, catalog, Mrelation, dc_window=Reasenberg1985Window(),
				tau_min=2880., tau_max=14400., rfact=10, dsigma=30, rmax=30,
				ignore_location_errors=True, verbose=False):
		"""
		Full analysis of individual clusters in earthquake catalog.

		:param catalog:
			instance of :class:`EQCatalog`
		:param Mrelation:
			dict specifying how to convert catalog magnitudes to MW
		:param dc_window:
			instance of :class:`DeclusteringWindow`
			(default: instance of :class:`Reasenberg1985Window`)
		:param tau_min:
		:param tau_max:
			see :meth:`Reasenberg1985Window.get_time_window`
		:param rfact:
		:param dsigma:
		:param rmax:
			see :meth:`Reasenberg1985Window.get_dist_window`
		:param ignore_location_errors:
			see :meth:`is_in_dist_window`
		:param verbose:
			bool, whether or not to print information during declustering

		:return:
			instance of :class:`DeclusteringResult`
		"""
		## Make sure catalog is ordered by date (ascending)
		catalog = catalog.get_sorted()

		Mrelation = Mrelation or catalog.default_Mrelations['MW']

		dc_idxs = -np.ones(len(catalog), dtype='int')
		clusters = []

		for i, eqi in enumerate(catalog):
			itime = eqi.datetime
			magi = eqi.get_MW(Mrelation)

			## Calculate tau (minutes), the look-ahead time for event i
			if dc_idxs[i] == -1:
				## Event is not (yet) clustered
				#cmag1 = 0.
				cmag1 = magi
				time_delta = 0.
			else:
				## When event i belongs to a cluster, tau is a function of
				## the magnitude of and the time since the largest event
				## in the cluster
				ctim1 = clusters[dc_idxs[i]].datetime1()
				cmag1 = clusters[dc_idxs[i]].mag1()
				time_delta = itime - ctim1
			tau = dc_window.get_time_window(cmag1, time_delta, tau_min, tau_max)

			## Keep getting jth events until time_delta_ij > tau
			for j in range(i+1, len(catalog)):
				## Skip the jth event if it is already identified as being part
				## of the cluster associated with the ith event
				if (dc_idxs[i] == dc_idxs[j]) and (dc_idxs[i] > -1):
					continue

				eqj = catalog[j]
				## Test for temporal clustering
				jtime = eqj.datetime
				time_delta_ij = jtime - itime
				if time_delta_ij <= tau:
					## Test for spatial clustering
					## Note that cmag1 may change throughout the loop
					## as events j are added to the cluster
					if dc_idxs[i] != -1:
						cmag1 = clusters[dc_idxs[i]].mag1()
					else:
						#cmag1 = 0.
						cmag1 = magi
					is_clustered = self.is_in_dist_window(eqi, eqj, magi, cmag1,
								dc_window, rfact=rfact, dsigma=dsigma, rmax=rmax,
								ignore_location_errors=ignore_location_errors)
					if is_clustered:
						## Cluster declared
						if dc_idxs[i] == dc_idxs[j] == -1:
							## Initialize new cluster
							ncl = len(clusters)
							cluster = Cluster([eqi, eqj], ncl, Mrelation,
												self.distance_metric)
							clusters.append(cluster)
							dc_idxs[i] = dc_idxs[j] = ncl
							if verbose:
								print("Initializing new cluster %d" % (ncl))
						elif dc_idxs[j] == -1:
							## If event i is already associated with a cluster,
							## add event j to it
							k = dc_idxs[i]
							clusters[k].append(eqj)
							dc_idxs[j] = k
						elif dc_idxs[i] == -1:
							## If event j is already associated with a cluster,
							## add event i to it
							k = dc_idxs[j]
							clusters[k].append(eqi)
							dc_idxs[i] = k
						else:
							## Combine existing clusters by merging into earlier cluster
							k = min(dc_idxs[i], dc_idxs[j])
							l = max(dc_idxs[i], dc_idxs[j])
							clusters[k].extend(clusters[l])
							clusters[l].eq_list = []
							dc_idxs[dc_idxs == l] = k
							if verbose:
								print("Combining clusters %d and %d" % (k, l))
				else:
					## Assuming catalog is sorted by time
					break

		return DeclusteringResult(catalog, dc_idxs, Mrelation,
								distance_metric="hypocentral")

	def decluster_catalog(self, catalog, Mrelation, dc_window=Reasenberg1985Window(),
				tau_min=2880., tau_max=14400., rfact=10, dsigma=30, rmax=30,
				ignore_location_errors=True):
		"""
		Decluster catalog.
		Note that this implementation is currently based on
		:meth:`analyze_clusters`, and is thus not faster

		:param catalog:
		:param Mrelation:
		:param dc_window:
		:param tau_min:
		:param tau_max:
		:param rfact:
		:param dsigma:
		:param rmax:
		:param ignore_location_errors:
			see :meth:`analyze_clusters`

		:return:
			instance of :class:`DeclusteringResult`
		"""
		dc_result = self.analyze_clusters(catalog, Mrelation, dc_window,
						tau_min, tau_max, rfact, dsigma, rmax,
						ignore_location_errors)
		return dc_result.get_declustered_catalog()


def get_available_windows():
	"""
	Function to get all declustering windows
	"""
	import sys, inspect

	def is_window(member):
		r_val = False
		if inspect.isclass(member):
			if issubclass(member, DeclusteringWindow) and member != DeclusteringWindow:
				return True
		return r_val

	windows = inspect.getmembers(sys.modules[__name__], is_window)
	windows = [(name.replace('Window', ''), val) for (name, val) in windows]
	windows = OrderedDict(windows)
	return windows


def get_window_by_name(window_name):
	return get_available_windows()[window_name]()


def plot_declustering_windows(fig_filespec=None, dpi=300, **kwargs):
	"""
	Plot time and distance windows of available declustering windows
	"""
	import matplotlib.pyplot as plt
	from .time_functions_np import fractional_time_delta
	from .plot import plot_xy
	#from .moment import calc_rupture_radius

	dc_windows = get_available_windows()

	Mmin, Mmax, dM = 0., 9., 0.1
	mags = np.arange(Mmin, Mmax, dM)

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)

	tlines, dlines = [], []
	labels = []
	for dc_window_name, dc_window in dc_windows.items():
		if dc_window_name != 'Reasenberg1985':
			dc_window = dc_window()
			tw = fractional_time_delta(dc_window.get_time_window(mags), 'D')
			tlines.append((mags, tw))
			dw = dc_window.get_dist_window(mags)
			dlines.append((mags, dw))
			labels.append(dc_window_name)

	xlabel ='Magnitude ($M_W$)'
	xgrid = ygrid = 1
	xtick_interval = (1.0, 0.5)

	## Plot time windows
	ylabel = 'Time (days)'
	ytick_interval = (250, 50)
	ymax = 1500
	title = 'Time window'
	legend_location = 2
	plot_xy(tlines, labels=labels, linewidths=[2], xlabel=xlabel, ylabel=ylabel,
			ytick_interval=ytick_interval, ymax=ymax, xgrid=xgrid, ygrid=ygrid,
			title=title, ax=ax1, legend_location=legend_location, **kwargs)

	## Plot space windows
	ylabel = 'Distance (km)'
	ytick_interval = (25, 5)
	ymax = 150
	title = 'Distance window'
	plot_xy(tlines, linewidths=[2], xlabel=xlabel, ylabel=ylabel, ytick_interval=ytick_interval,
			ymax=ymax, xgrid=xgrid, ygrid=ygrid, title=title, ax=ax2, **kwargs)

	fig.subplots_adjust(wspace=0.5)
	#fig.legend(lines, labels, 9, bbox_to_anchor=(0.50,0.95), ncol=2, prop={'size': 12})
	#plt.tight_layout(pad=0.25, w_pad=1.0)

	if fig_filespec:
		plt.savefig(fig_filespec, dpi=dpi)
		plt.clf()
	else:
		plt.show()


def generate_linked_cluster(dc_window, Mrange=(1, 6), num_events=50):
	"""
	Generate a synthetic cluster where each earthquake (apart from the
	first one) is in the space/time window of the previous one.
	Note that this does not at all resemble a natural earthquake
	cluster! This is only useful to test the performance of the
	linked_window declustering method

	:param dc_window:
		instance of :class:`DeclusteringWindow`, the window definition
		used to generate the cluster
	:param Mrange:
		(min_mag, max_mag) tuple of floats, magnitude range of cluster
		(default: (1, 6))
	:param num_events:
		int, the number of earthquakes the cluster should consist of
		(default: 50)

	:return:
		instance of :class:`EQCatalog`
	"""
	from mapping.geotools.geodetic import spherical_point_at
	from .eqrecord import LocalEarthquake
	from .eqcatalog import EQCatalog
	from .time_functions_np import (to_py_date, to_py_time)

	mags = np.random.random(num_events)
	min_mag, max_mag = Mrange
	mags = min_mag + mags * (max_mag - min_mag)
	max_distances = dc_window.get_dist_window(mags[:-1])
	max_time_deltas = dc_window.get_time_window(mags[:-1])
	distances = np.random.random(num_events-1) * max_distances * 1000
	time_deltas = np.random.random(num_events-1) * max_time_deltas
	azimuths = np.random.random(num_events-1) * 360

	## First earthquake in sequence
	start_dt = np.datetime64('now')
	start_date, start_time = to_py_date(start_dt), to_py_time(start_dt)
	start_lon, start_lat = 0, 0
	depth = 0
	eq0 = LocalEarthquake(0, start_date, start_time, start_lon, start_lat,
						depth, {'MW': mags[0]}, intensity_max=np.nan)
	eq_list = [eq0]

	## Linked aftershocks
	for i in range(num_events-1):
		foreshock = eq_list[-1]
		mag = mags[i+1]
		time_delta = time_deltas[i]
		dist, azimuth = distances[i], azimuths[i]
		lon, lat = spherical_point_at(foreshock.lon, foreshock.lat, dist, azimuth)
		dt = foreshock.datetime + np.timedelta64(time_delta, 's')
		eq = LocalEarthquake(i+1, to_py_date(dt), to_py_time(dt), lon, lat, depth,
							{'MW': mag}, intensity_max=np.nan)
		eq_list.append(eq)

	return EQCatalog(eq_list, name='Synthetic cluster (%s window)' % dc_window.name)
