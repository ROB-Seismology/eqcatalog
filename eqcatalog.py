# -*- coding: iso-Latin-1 -*-

"""
Module for processing earthquake catalogs.
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

from __future__ import absolute_import, division, print_function, unicode_literals

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str


## Import standard python modules
import os
import sys
import platform
import datetime
import json
from collections import OrderedDict


## Import third-party modules
import numpy as np
import matplotlib
if platform.uname()[1] == "seissrv3":
	## Kludge because matplotlib is broken on seissrv3.
	matplotlib.use('AGG')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, MaxNLocator


## Import ROB modules
import mapping.geotools.geodetic as geodetic


## Import package submodules
from .eqrecord import LocalEarthquake
from .completeness import Completeness
from . import time as timelib


# TODO: re-implement interface with (Hazard) Modelers' Toolkit

class EQCatalog(object):
	"""
	Class defining a collection of local earthquakes.

	:param eq_list:
		List containing instances of :class:`LocalEarthquake`
	:param start_date:
		instance of :class:`np.datetime64` or :class:`datetime.datetime`,
		start datetime of catalog
		(default: None = datetime of oldest earthquake in catalog)
	:param end_date:
		instance of :class:`np.datetime64` or :class:`datetime.datetime`,
		end datetme of catalog
		(default: None = datetime of youngest earthquake in catalog)
	:param region:
		(lon0, lon1, lat0, lat1) tuple with geographic coordinates of
		bounding box
		(default: None)
	:param name:
		String, catalog name
		(default: "")
	:param default_Mrelations:
		dict, mapping Mtype (str) to Mrelations (ordered dicts,
		in turn mapping Mtype to name of magnitude conversion relations):
		default conversion relations for different magnitude types
		(default: {})
	:param default_completeness:
		instance of :class:`Completeness`, default catalog completeness
		(default: None)
	"""
	def __init__(self, eq_list, start_date=None, end_date=None, region=None,
				name="", default_Mrelations={}, default_completeness=None):
		self.eq_list = eq_list[:]
		Tmin, Tmax = self.Tminmax()
		if not start_date:
			self.start_date = Tmin
		else:
			self.start_date = timelib.as_np_datetime(start_date, unit='ms')
		if not end_date:
			self.end_date = Tmax
		else:
			self.end_date = timelib.as_np_datetime(end_date, unit='ms')
		self.region = region
		self.name = name
		self.default_Mrelations = default_Mrelations
		self.default_completeness = default_completeness

	def __repr__(self):
		txt = '<EQCatalog "%s" | %s - %s | n=%d>'
		txt %= (self.name, self.start_date, self.end_date, len(self))
		return txt

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
		Slicing / Index array --> instance of :class:`EQCatalog`
		"""
		if isinstance(item, (int, np.int32, np.int64)):
			return self.eq_list.__getitem__(item)
		elif isinstance(item, slice):
			return EQCatalog(self.eq_list.__getitem__(item), start_date=self.start_date,
							end_date=self.end_date, region=self.region,
							name=self.name + " %s" % item,
							default_Mrelations=self.default_Mrelations,
							default_completeness=self.default_completeness)
		elif isinstance(item, (list, np.ndarray)):
			## item can contain indexes or bool
			eq_list = []
			if len(item):
				idxs = np.arange(len(self))
				idxs = idxs[np.asarray(item)]
				for idx in idxs:
					eq_list.append(self.eq_list[idx])
			return EQCatalog(eq_list, start_date=self.start_date, end_date=self.end_date,
							region=self.region, name=self.name + " %s" % item,
							default_Mrelations=self.default_Mrelations,
							default_completeness=self.default_completeness)

	def __contains__(self, eq):
		"""
		Determine whether or not given earthquake is in catalog
		(based on its ID)

		:param eq:
			instance of :class:`LocalEarthquake`

		:return:
			bool
		"""
		assert isinstance(eq, LocalEarthquake)
		return eq.ID in self.get_ids()

	def __eq__(self, other_catalog):
		return sorted(self.get_ids()) == sorted(other_catalog.get_ids())

	def __add__(self, other_catalog):
		return self.get_union(other_catalog)

	def __sub__(self, other_catalog):
		return self.get_difference(other_catalog)

	def append(self, eq):
		"""
		Append earthquake to catalog

		:param eq:
			instance of :class:`LocalEarthquake`
		"""
		self.eq_list.append(eq)

	def copy(self):
		"""
		Copy catalog

		:return:
			instance of :class:`EQCatalog`
		"""
		eq_list = [eq.copy() for eq in self]
		start_date, end_date = self.start_date, self.end_date
		region = self.region
		name = self.name
		default_Mrelations = self.default_Mrelations
		default_completeness = self.default_completeness
		return self.__class__(eq_list, start_date=start_date, end_date=end_date,
							region=region, name=name,
							default_Mrelations=default_Mrelations,
							default_completeness=default_completeness)

	@property
	def lons(self):
		return self.get_longitudes()

	@property
	def lats(self):
		return self.get_latitudes()

	@property
	def mags(self):
		return self.get_magnitudes()

	## Methods related to earthquake IDs

	def get_ids(self):
		"""
		Return earthquake IDs

		:return:
			list of strings or integers
		"""
		return [eq.ID for eq in self]

	def get_unique_ids(self):
		"""
		Return unqiue earthquake IDs
		"""
		return np.unique(self.get_ids())

	def has_unique_ids(self):
		"""
		Determine whether or not earthquakes in catalog have unique IDs

		:return:
			bool
		"""
		num_unique_ids = len(self.get_unique_ids())
		if num_unique_ids == len(self):
			return True
		else:
			return False

	def index(self, id):
		"""
		Get index of event with given ID

		:param id:
			int or str, earthquake ID

		:return:
			int, index in catalog
		"""
		str_ids = list(map(str, self.get_ids()))
		try:
			idx = str_ids.index(str(id))
		except ValueError:
			return None
		else:
			return idx

	def get_event_by_id(self, id):
		"""
		Extract event with given ID

		:param id:
			int or str, earthquake ID

		:return:
			instance of :class:`LocalEarthquake`
		"""
		idx = self.index(id)
		if idx is not None:
			return self.__getitem__(idx)

	def get_duplicate_idxs(self):
		"""
		Determine indexes of duplicate earthquakes in catalog

		:return:
			int array, indexes of duplicates
		"""
		ids = self.get_ids()
		unique_ids, duplicate_idxs = [], []
		for i, id in enumerate(ids):
			if not id in unique_ids:
				unique_ids.append(id)
			else:
				duplicate_idxs.append(i)
		return duplicate_idxs

	def remove_duplicates(self):
		"""
		Return catalog with duplicates removed

		:return:
			instance of :class:`EQCatalog`
		"""
		duplicate_idxs = self.get_duplicate_idxs()
		return self.remove_events_by_index(duplicate_idxs)

	def remove_events_by_index(self, idxs):
		"""
		Return catalog with earthquakes corresponding to indexes removed

		:param idxs:
			int list or array, indexes of earthquakes to be removed

		:return:
			instance of :class:`EQCatalog`
		"""
		all_idxs = np.arange(len(self))
		remaining_idxs = set(all_idxs) - set(idxs)
		remaining_idxs = np.array(sorted(remaining_idxs))
		return self.__getitem__(remaining_idxs)

	def remove_events_by_id(self, IDs):
		"""
		Return catalog with earthquakes with given IDs removed

		:param IDs:
			list of ints or strings, IDs of earthquakes to be removed

		:return:
			instance of :class:`EQCatalog`
		"""
		idxs = []
		for ID in IDs:
			idxs.append(self.index(ID))
		return self.remove_events_by_index(idxs)

	def get_union(self, other_catalog, name=None):
		"""
		Return catalog of all events in catalog and in other catalog
		(without duplicates)

		:param other_catalog:
			instance of :class:`EQCatalog`
		:param name:
			str, name of resulting catalog
			(default: None, will generate name automatically)

		:return:
			instance of :class:`EQCatalog`
		"""
		assert isinstance(other_catalog, EQCatalog)
		ids = set(self.get_ids())
		other_ids = set(other_catalog.get_ids())
		union = list(ids.union(other_ids))
		cat1 = self.subselect(attr_val=('ID', union))
		cat2 = other_catalog.subselect(attr_val=('ID', union))
		if name is None:
			name = "Union(%s, %s)" % (self.name, other_catalog.name)
		return concatenate_catalogs([cat1, cat2], name=name)

	# TODO: set catalog start and end dates!
	def get_intersection(self, other_catalog, name=None):
		"""
		Return catalog of events that are both in catalog and in
		other catalog

		:param other_catalog:
			instance of :class:`EQCatalog`
		:param name:
			str, name of resulting catalog
			(default: None, will generate name automatically)

		:return:
			instance of :class:`EQCatalog`
		"""
		assert isinstance(other_catalog, EQCatalog)
		ids = set(self.get_ids())
		other_ids = set(other_catalog.get_ids())
		intersection = list(ids.intersection(other_ids))
		catalog = self.subselect(attr_val=('ID', intersection))
		if name is None:
			name = "Intersection(%s, %s)" % (self.name, other_catalog.name)
		catalog.name = name
		return catalog

	def get_difference(self, other_catalog, name=None):
		"""
		Return catalog of events that are in catalog but not in
		other catalog

		:param other_catalog:
			instance of :class:`EQCatalog`
		:param name:
			str, name of resulting catalog
			(default: None, will generate name automatically)

		:return:
			instance of :class:`EQCatalog`
		"""
		assert isinstance(other_catalog, EQCatalog)
		ids = set(self.get_ids())
		other_ids = set(other_catalog.get_ids())
		diff = list(ids - other_ids)
		catalog = self.subselect(attr_val=('ID', diff))
		if name is None:
			name = "Difference(%s, %s)" % (self.name, other_catalog.name)
		catalog.name = name
		return catalog

	def get_symmetric_difference(self, other_catalog, name=None):
		"""
		Return catalog of events that are only in catalog or in
		other catalog

		:param other_catalog:
			instance of :class:`EQCatalog`
		:param name:
			str, name of resulting catalog
			(default: None, will generate name automatically)

		:return:
			instance of :class:`EQCatalog`
		"""
		assert isinstance(other_catalog, EQCatalog)
		ids = set(self.get_ids())
		other_ids = set(other_catalog.get_ids())
		symdiff = list(ids.symmetric_difference(other_ids))
		cat1 = self.subselect(attr_val=('ID', symdiff))
		cat2 = other_catalog.subselect(attr_val=('ID', symdiff))
		if name is None:
			name = "Symmetric Difference(%s, %s)" % (self.name, other_catalog.name)
		return concatenate_catalogs([cat1, cat2], name=name)

	def print_info(self, as_html=False):
		"""
		Print some useful information about the catalog.

		:param as_html:
			bool, whether to return HTML or to print plain text
			(default: False)

		:return:
			str or instance of :class:`PrettyTable`
		"""
		try:
			from prettytable import PrettyTable
		except:
			has_prettytable = False
		else:
			has_prettytable = True
			tab = PrettyTable(["Parameter", "Value"])

		rows = []
		rows.append(["Catalog name", self.name])
		rows.append(["Earthquake number", "%d" % len(self)])
		rows.append(["Start time", "%s" % self.start_date])
		rows.append(["End time", "%s" % self.end_date])

		lonmin, lonmax = self.lon_minmax()
		rows.append(["Longitude bounds", "%.4f / %.4f" % (lonmin, lonmax)])
		latmin, latmax = self.lat_minmax()
		rows.append(["Latitude bounds", "%.4f / %.4f" % (latmin, latmax)])
		depth_min, depth_max = self.depth_minmax()
		rows.append(["Depth range", "%.1f / %.1f km" % (depth_min, depth_max)])

		for Mtype, count in self.get_Mtype_counts().items():
			mags = self.get_magnitudes(Mtype=Mtype, Mrelation={})
			mags = mags[np.isfinite(mags)]
			if len(mags):
				if mags.min() == 0:
					mags = mags[mags > 0]
				rows.append([Mtype,
					"n=%d, min=%.1f, max=%.1f" % (count, mags.min(), mags.max())])

		etype_num_dict = self.count_num_by_event_type()
		etype_str = ', '.join(["%s (n=%d)" % (etype, etype_num_dict[etype])
											for etype in etype_num_dict])
		rows.append(["Event types", etype_str])

		if has_prettytable:
			for row in rows:
				tab.add_row(row)
			if as_html:
				return tab.get_html_string()
			else:
				print(tab)
				return tab
		else:
			for row in tab:
				print(' :\t'.join(row))

	def get_formatted_table(self, max_name_width=30, lonlat_decimals=3,
							depth_decimals=1, mag_decimals=1,
							padding_width=1):
		"""
		Return formatted table of earthquakes in catalog.

		:param max_name_width:
			int, max. width for 'name' column
			(default: 30)
		:param lonlat_decimals:
			int, number of decimals for longitudes and latitudes
			(default: 3)
		:param depth_decimals:
			int, number of decimals for depths
			(default: 1)
		:param mag_decimals:
			int, number of decimals for magnitudes
			(default: 1)

		:return:
			instance of :class:`PrettyTable`
		"""
		import prettytable as pt

		def remove_nan_values(ar):
			return [val if not np.isnan(val) else '' for val in ar]

		tab = pt.PrettyTable()
		tab.add_column('ID', self.get_ids(), align='r', valign='m')
		tab.add_column('Date', [eq.date for eq in self], valign='m')
		tab.add_column('Time', [eq.time for eq in self], valign='m')
		names = [eq.name for eq in self]
		if sum([len(name) for name in names]):
			tab.add_column('Name', names, valign='m')
		lons = remove_nan_values(self.get_longitudes())
		tab.add_column('Lon', lons, align='r', valign='m')
		lats = remove_nan_values(self.get_latitudes())
		tab.add_column('Lat', lats, align='r', valign='m')
		depths = remove_nan_values(self.get_depths())
		tab.add_column('Z', depths, align='r', valign='m')
		Mtypes = self.get_Mtypes()
		for Mtype in Mtypes:
			#mags = self.get_magnitudes(Mtype, Mrelation={})
			mags = np.array([eq.mag.get(Mtype, np.nan) for eq in self])
			if not np.isnan(mags).all():
				mags = remove_nan_values(mags)
				tab.add_column(Mtype, mags, align='r', valign='m')
		intensities = self.get_max_intensities()
		if not ((intensities == 0).all() or np.isnan(intensities).all()):
			intensities = remove_nan_values(intensities)
			tab.add_column('Imax', intensities, align='r', valign='m')
		event_types = [eq.event_type for eq in self]
		if len(set(event_types)) > min(1, len(self)-1):
			tab.add_column('Type', event_types, valign='m')

		tab.padding_width = padding_width
		tab.max_width['Name'] = max_name_width
		tab.float_format['Lon'] = tab.float_format['Lat'] = '.%d' % lonlat_decimals
		tab.float_format['Z'] = '.%d' % depth_decimals
		for Mtype in Mtypes:
			if Mtype in tab.field_names:
				tab.float_format[Mtype] = '.%d' % mag_decimals

		return tab

	def get_formatted_list(self, as_html=False, max_name_width=30,
					lonlat_decimals=3, depth_decimals=1, mag_decimals=1,
					padding_width=1):
		"""
		Return string representing formatted table of earthquakes
		in catalog.

		:param as_html:
			bool, whether to return HTML or to print plain text
			(default: False)
		:param max_name_width:
		:param lonlat_decimals:
		:param depth_decimals:
		:param mag_decimals:
			see :meth:`get_formatted_table`

		:return:
			str
		"""
		tab = self.get_formatted_table(max_name_width=max_name_width,
					lonlat_decimals=lonlat_decimals, depth_decimals=depth_decimals,
					mag_decimals=mag_decimals, padding_width=padding_width)
		if as_html:
			return tab.get_html_string()
		else:
			return tab.get_string()

	def print_list(self, out_file=None, max_name_width=30, lonlat_decimals=3,
					depth_decimals=1, mag_decimals=1, padding_width=1):
		"""
		Print list of earthquakes in catalog

		:param out_file:
			str, full path to output file
			If extension starts with .htm, the list is exported in HTML
			format, else plain text
			(default: None, will print list to screen)
		:param max_name_width:
		:param lonlat_decimals:
		:param depth_decimals:
		:param mag_decimals:
			see :meth:`get_formatted_list`

		:return:
			None or str if :param:`as_html` is True
		"""
		try:
			from prettytable import PrettyTable
		except:
			from pprint import pprint
			has_prettytable = False
		else:
			has_prettytable = True

		if has_prettytable:
			tab = self.get_formatted_table(max_name_width=max_name_width,
						lonlat_decimals=lonlat_decimals, depth_decimals=depth_decimals,
						mag_decimals=mag_decimals, padding_width=padding_width)
			if out_file:
				of = open(out_file, 'w')
				if os.path.splitext(out_file)[-1].lower()[:4] == '.htm':
					of.write(tab.get_html_string())
				else:
					of.write(tab.get_string())
				of.close()
			else:
				print(tab)

		else:
			pass

		"""
		try:
			from prettytable import PrettyTable
		except:
			has_prettytable = False
		else:
			has_prettytable = True

		col_names = ["ID", "Date", "Time", "Name", "Lon", "Lat", "Depth",
					"ML", "MS", "MW"]
		if has_prettytable:
			tab = PrettyTable(col_names)
		else:
			tab = [col_names]

		for eq in self.eq_list:
			row = [str(eq.ID), str(eq.date), str(eq.time), eq.name,
					"%.4f" % eq.lon, "%.4f" % eq.lat, "%.1f" % eq.depth,
					"%.1f" % eq.ML, "%.1f" % eq.MS, "%.1f" % eq.MW]
			if has_prettytable:
				tab.add_row(row)
			else:
				tab.append(row)

		if has_prettytable:
			if as_html:
				return tab.get_html_string()
			else:
				print(tab)
				return tab
		else:
			for row in tab:
				print('\t'.join(row))
		"""

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
		if 'eq_list' in dct:
			dct['eq_list'] = [LocalEarthquake.from_dict(d["__LocalEarthquake__"])
							for d in dct['eq_list']]
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
			elif isinstance(obj, (datetime.time, datetime.date)):
				return repr(obj)
			elif isinstance(obj, np.datetime64):
				return str(obj)
			else:
				return obj.__dict__

		key = '__%s__' % self.__class__.__name__
		dct = {key: self.__dict__}
		return json.dumps(dct, default=json_handler)

	## Methods related to event_type

	def get_event_types(self):
		"""
		Return list of event types for all earthquakes in catalog
		"""
		return [eq.event_type for eq in self]

	def get_unique_event_types(self):
		"""
		Return list of unique event types in catalog
		"""
		return sorted(set(self.get_event_types()))

	def count_num_by_event_type(self):
		"""
		Count number of events for each event type

		:return:
			dict, mapping event type (str) to number of events (int)
		"""
		etype_num_dict = {}
		for eq in self:
			etype = eq.event_type
			if not etype in etype_num_dict:
				etype_num_dict[etype] = 1
			else:
				etype_num_dict[etype] += 1
		return etype_num_dict

	## Time methods
	@property
	def start_year(self):
		return int(timelib.to_year(self.start_date))

	@property
	def end_year(self):
		return int(timelib.to_year(self.end_date))

	def get_datetimes(self):
		"""
		Return list of datetimes for all earthquakes in catalog
		"""
		return np.array([eq.datetime for eq in self])

	def get_years(self):
		"""
		Return array of integer years for all earthquakes in catalog
		"""
		return timelib.to_year(self.get_datetimes())

	def get_fractional_years(self):
		"""
		Return array with fractional years for all earthquakes in catalog
		"""
		years = timelib.to_fractional_year(self.get_datetimes())
		return years

	def Tminmax(self, Mmax=None, Mtype="MW", Mrelation={}):
		"""
		Return tuple with oldest date and youngest date in catalog.

		:param Mmax:
			Float, maximum magnitude. Useful to check completeness periods.
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		"""
		datetimes = self.get_datetimes()
		if Mmax != None:
			Mags = self.get_magnitudes(Mtype=Mtype, Mrelation=Mrelation)
			datetimes = datetimes[np.where(Mags < Mmax)]
		try:
			return (datetimes.min(), datetimes.max())
		except ValueError:
			return (None, None)

	def get_time_delta(self, from_events=False):
		"""
		Return duration of catalog as timedelta object

		:param from_events:
			bool, if True, compute time between first and last event
			else use start and end date of catalog
			(default: False)

		:return:
			instance of :class:`np.timedelta64`
		"""
		if from_events:
			Tmin, Tmax = self.Tminmax()
		else:
			Tmin, Tmax = self.start_date, self.end_date
		return Tmax - Tmin

	def get_time_deltas(self, start_datetime=None):
		"""
		Return time difference between a start time and each event.

		:param start_datetime:
			instance of :class:`np.datetime64` or :class:`datetime.datetime`,
			start time
			(default: None, will take the start time of the catalog)

		:return:
			array with instances of :class:`np.timedelta64`
		"""
		if not start_datetime:
			start_datetime = self.start_date
		return self.get_datetimes() - start_datetime

	def get_inter_event_times(self, unit='D'):
		"""
		Return time interval in fractions of specified unit between each
		subsequent event

		:param unit:
			str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
			(year|week|day|hour|minute|second|millisecond|microsecond)
			(default: 'D')

		:return:
			float array, inter-event times
		"""
		sorted_catalog = self.get_sorted()
		date_times = sorted_catalog.get_datetimes()
		time_deltas = np.diff(date_times)
		return timelib.fractional_time_delta(time_deltas, unit=unit)

	def timespan(self, unit='Y'):
		"""
		Return total time span of catalog as fraction of specified unit

		:param unit:
			str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
			(year|week|day|hour|minute|second|millisecond|microsecond)
			(default: 'Y')
		"""
		start_date, end_date = self.start_date, self.end_date
		return timelib.timespan(start_date, end_date, unit=unit)

	## Magnitude / moment methods

	def get_magnitudes(self, Mtype="MW", Mrelation={}):
		"""
		Return array of magnitudes for all earthquakes in catalog

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {}, will select the default relation for the
			given Mtype in :prop:`default_Mrelations`)

		:return:
			1-D numpy float array, earthquake magnitudes
		"""
		# TODO: it should also be possible to get magnitudes without any conversion...
		if not Mrelation:
			Mrelation = self.default_Mrelations.get(Mtype, {})
		Mags = [eq.get_or_convert_mag(Mtype, Mrelation) for eq in self]
		return np.array(Mags)

	def Mminmax(self, Mtype="MW", Mrelation={}):
		"""
		Return tuple with minimum and maximum magnitude in catalog.

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		"""
		Mags = self.get_magnitudes(Mtype=Mtype, Mrelation=Mrelation)
		return (np.nanmin(Mags), np.nanmax(Mags))

	def get_Mmin(self, Mtype="MW", Mrelation={}):
		"""
		Compute minimum magnitude in catalog

		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MW")
		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})

		:return:
			Float, maximum observed magnitude
		"""
		return np.nanmin(self.get_magnitudes(Mtype, Mrelation))

	def get_Mmax(self, Mtype="MW", Mrelation={}):
		"""
		Compute maximum magnitude in catalog

		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MW")
		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})

		:return:
			Float, maximum observed magnitude
		"""
		if len(self) > 0:
			Mmax = np.nanmax(self.get_magnitudes(Mtype, Mrelation))
		else:
			Mmax = np.nan
		return Mmax

	def convert_magnitudes(self, Mtype="MW", Mrelation={}):
		"""
		Convert magnitude to given magnitude type and store in
		earthquake objects.

		:param Mtype:
		:param Mrelation:
			see :meth:`get_magnitudes`
		"""
		mags = self.get_magnitudes(Mtype, Mrelation)
		for eq, mag in zip(self.eq_list, mags):
			eq.set_mag(Mtype, mag)

	def get_magnitude_uncertainties(self, min_uncertainty=0.3):
		"""
		Return array with magnitude uncertainties

		:param min_uncertainty:
			Float, minimum uncertainty that will be used to replace zero
			and nan values. If None, no values will be replaced.
			(default: 0.3)

		:return:
			1-D numpy float array, magnitude uncertainties
		"""
		Mag_uncertainties = np.array([eq.errM for eq in self])
		if not min_uncertainty is None:
			Mag_uncertainties[Mag_uncertainties == 0] = min_uncertainty
			Mag_uncertainties[np.isnan(Mag_uncertainties)] = min_uncertainty
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
				if Mtype in Mtype_counts:
					Mtype_counts[Mtype] += 1
				else:
					Mtype_counts[Mtype] = 1
			if len(eq_Mtypes) > 1:
				comb_Mtype = '+'.join(sorted(eq_Mtypes))
				if comb_Mtype in Mtype_counts:
					Mtype_counts[comb_Mtype] += 1
				else:
					Mtype_counts[comb_Mtype] = 1
		return Mtype_counts

	def subselect_Mtype(self, Mtypes, catalog_name=""):
		"""
		Subselect earthquakes according to their magnitude type(s)

		:param Mtypes:
			str or list, one or more magnitude types
		:param catalog_name:
			str, name of output catalog
			(default: "")

		:return:
			instance of :class:`EQCatalog`
		"""
		if isinstance(Mtypes, basestring):
			Mtypes = [Mtypes]
		else:
			Mtypes = sorted(Mtypes)

		eq_list = []
		for eq in self:
			eq_Mtypes = set(eq.get_Mtypes())
			if sorted(eq_Mtypes.intersection(Mtypes)) == Mtypes:
				eq_list.append(eq)

		if not catalog_name:
			catalog_name = self.name + " (%s)" % ', '.join(Mtypes)

		subcat = EQCatalog(eq_list, self.start_date, self.end_date, self.region,
						catalog_name, default_Mrelations=self.default_Mrelations,
						default_completeness=self.default_completeness)
		return subcat

	def get_M0(self, Mrelation={}):
		"""
		Return array with seismic moments for all earthquakes in catalog.

		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})

		:return:
			1-D numpy float array, earthquake moments
		"""
		return np.array([eq.get_M0(Mrelation=Mrelation) for eq in self])

	def get_M0_total(self, Mrelation={}):
		"""
		Compute total seismic moment.

		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MS" or "ML")
			(default: {})

		:return:
			Float, total seismic moment in N.m
		"""
		return np.sum(self.get_M0(Mrelation=Mrelation))

	def get_M0_rate(self, completeness=None, Mrelation={}, time_unit='Y'):
		"""
		Compute seismic moment rate.

		:param completeness:
			instance of :class:`Completeness` (default: None)
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MS" or "ML")
			(default: {})
		:param time_unit:
			str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
			(year|week|day|hour|minute|second|millisecond|microsecond)
			timespan unit
			(default: 'Y')

		:return:
			Float, seismic moment rate in N.m per unit of :param:`time_unit`
		"""
		completeness = completeness or self.default_completeness
		if completeness is None:
			M0rate = self.get_M0_total(Mrelation=Mrelation) / self.timespan(time_unit)
		else:
			if completeness.Mtype != "MW":
				raise Exception("Completeness magnitude must be moment magnitude!")
			M0rate = 0.
			for subcatalog in self.split_completeness(completeness=completeness,
											Mtype="MW", Mrelation=Mrelation):
				M0rate += (subcatalog.get_M0_total(Mrelation=Mrelation)
						 	/ subcatalog.timespan(time_unit))
		return M0rate

	## Intensity methods

	def get_max_intensities(self):
		"""
		Return array with maximum intensities

		:return:
			1-D numpy array, maximum intensities
		"""
		return np.array([eq.intensity_max or np.nan for eq in self])

	def get_macro_radii(self):
		"""
		Return array with macroseismic radii

		:return:
			1-D numpy array, macroseismic radii
		"""
		return np.array([eq.macro_radius or np.nan for eq in self])

	## Coordinate-related methods

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
		coord_list = list(zip(lons, lats))
		if proj.lower() == "lambert1972":
			return coordtrans.lonlat_to_lambert1972(coord_list)
		elif proj.lower() == "utm31n":
			return coordtrans.utm_to_lonlat(coord_list, proj)

	def depth_minmax(self):
		"""
		Return tuple with minimum and maximum depth in catalog.
		"""
		depths = self.get_depths()
		depths = depths[np.isfinite(depths)]
		if len(depths):
			return (depths.min(), depths.max())
		else:
			return (np.nan, np.nan)

	def lon_minmax(self):
		"""
		Return tuple with minimum and maximum longitude in catalog.
		"""
		longitudes = self.get_longitudes()
		longitudes = longitudes[np.isfinite(longitudes)]
		if len(longitudes):
			return (longitudes.min(), longitudes.max())
		else:
			return (np.nan, np.nan)

	def lat_minmax(self):
		"""
		Return tuple with minimum and maximum latitude in catalog.
		"""
		latitudes = self.get_latitudes()
		latitudes = latitudes[np.isfinite(latitudes)]
		if len(latitudes):
			return (latitudes.min(), latitudes.max())
		else:
			return (np.nan, np.nan)

	def get_region(self):
		"""
		Return (w, e, s, n) tuple with geographic extent of catalog
		"""
		try:
			return self.lon_minmax() + self.lat_minmax()
		except:
			return None

	def get_bbox(self):
		"""
		Compute bounding box of earthquake catalog

		:return:
			(ll_lon, ll_lat, ur_lon, ur_lat) tuple
		"""
		ll_lon, ur_lon = self.lon_minmax()
		ll_lat, ur_lat = self.lat_minmax()
		return (ll_lon, ll_lat, ur_lon, ur_lat)

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
		depths = np.nan_to_num(self.get_depths())
		d_hypo = np.sqrt(d_epi**2 + (depths - z)**2)
		return d_hypo

	## Sorting

	def argsort(self, key="datetime", order="asc"):
		"""
		Return indexes that would sort catalog by given key

		:param key:
			str, property of :class:`LocalEarthquake` to use as sort key
			Multiple sorting criteria can be specified, separated by a comma
			(default: "datetime")
		:param order:
			str, sorting order: "asc" or "desc"
			If :param:`key` contains multiple sorting criteria, a different
			order can be specified for each, separated by a comma
			(default: "asc")

		:return:
			int array, indexes that sort catalog
		"""
		keys = key.split(',')
		orders = order.split(',')
		if len(orders) == 1 and len(keys) > 1:
			orders = orders * len(keys)
		arrays = [np.array([getattr(eq, key) for eq in self]) for key in keys]
		for i, order in enumerate(orders):
			if order == "desc":
				if isinstance(arrays[i][0], np.datetime64):
					arrays[i] = timelib.seconds_since_epoch(arrays[i])
				arrays[i] = -arrays[i]
		rec = np.rec.fromarrays(arrays, names=keys)
		idxs = np.argsort(rec, order=keys)
		return np.asarray(idxs)

	def get_sorted(self, key="datetime", order="asc"):
		"""
		Get copy of catalog sorted by earthquake attribute.

		:param key:
			str, property of :class:`LocalEarthquake` to use as sort key
			(default: "datetime")
		:param order:
			str, sorting order: "asc" or "desc"
			(default: "asc")

		:return:
			instance of :class:`EQCatalog`
		"""
		idxs = self.argsort(key=key, order=order)
		return self.__getitem__(idxs)

	def sort(self, key="datetime", order="asc"):
		"""
		Sort catalog in place

		:param key:
			str, property of :class:`LocalEarthquake` to use as sort key
			(default: "datetime")
		:param order:
			str, sorting order: "asc" or "desc"
			(default: "asc")

		:return:
			None, catalog is sorted in place
		"""
		idxs = self.argsort(key=key, order=order)
		self.eq_list = list(np.array(self.eq_list)[idxs])

	## Subselection and splitting methods

	def subselect(self,
		region=None,
		start_date=None, end_date=None,
		Mmin=None, Mmax=None,
		min_depth=None, max_depth=None,
		attr_val=(),
		Mtype="MW", Mrelation={},
		include_right_edges=True,
		catalog_name=""):
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
		:param attr_val:
			(attribute, value) tuple (default: ())
		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
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
			start_date = timelib.time_tuple_to_np_datetime(start_date, 1, 1)
		else:
			start_date = timelib.as_np_datetime(start_date)
		if isinstance(end_date, int):
			end_date = timelib.time_tuple_to_np_datetime(end_date, 12, 31, 23, 59, 59, 999999)
		else:
			end_date = timelib.as_np_datetime(end_date)
		## If start_date and end_date are the same,
		## set time of end date to end of the day
		if start_date and end_date and end_date - start_date == np.timedelta64(0):
			end_time = datetime.time(23, 59, 59, 999999)
			end_date = timelib.combine_np_date_and_py_time(end_date, end_time)

		## Check each constraint separately
		eq_list = self.eq_list

		if region != None:
			w, e, s, n = region
			if include_right_edges:
				eq_list = [eq for eq in eq_list if w <= eq.lon <= e and s <= eq.lat <= n]
			else:
				eq_list = [eq for eq in eq_list if w <= eq.lon < e and s <= eq.lat < n]
		if start_date != None:
			eq_list = [eq for eq in eq_list if start_date <= eq.datetime]
		if end_date != None:
			if include_right_edges:
				eq_list = [eq for eq in eq_list if eq.datetime <= end_date]
			else:
				eq_list = [eq for eq in eq_list if eq.datetime < end_date]

		Mrelation = Mrelation or self.default_Mrelations.get(Mtype, {})

		if Mmin != None:
			cat2 = EQCatalog(eq_list)
			Mags = cat2.get_magnitudes(Mtype, Mrelation)
			Mags[np.isnan(Mags)] = Mmin - 1
			is_selected = Mmin <= Mags
			eq_list = [eq_list[i] for i in range(len(eq_list)) if is_selected[i]]
		if Mmax != None:
			cat2 = EQCatalog(eq_list)
			Mags = cat2.get_magnitudes(Mtype, Mrelation)
			Mags[np.isnan(Mags)] = Mmax + 1
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
			if isinstance(val, (list, np.ndarray)):
				eq_list = [eq for eq in eq_list if getattr(eq, attr) in val]
			else:
				eq_list = [eq for eq in eq_list if getattr(eq, attr) == val]

		## Update catalog information
		if region is None:
			if self.region:
				region = self.region
			else:
				region = self.get_region()
		if start_date is None:
			start_date = self.start_date
		else:
			start_date = timelib.as_np_datetime(start_date)
		if end_date is None:
			end_date = self.end_date
		else:
			end_date = timelib.as_np_datetime(end_date)
		if not include_right_edges:
			unit = timelib.get_datetime_unit(end_date)
			end_date -= np.timedelta64(1, unit)

		if not catalog_name:
			catalog_name = self.name + " (subselect)"

		return EQCatalog(eq_list, start_date=start_date, end_date=end_date,
						region=region, name=catalog_name,
						default_Mrelations=dict(self.default_Mrelations, **{Mtype: Mrelation}),
						default_completeness=self.default_completeness)

	def subselect_distance(self, point, distance, catalog_name=""):
		"""
		Subselect earthquakes in a given radius around a given point
		If point contains Z coordinate, hypocentral distances are used,
		else epicentral distances.

		:param point:
			(lon, lat, [z]) tuple
		:param distance:
			float, distance in km
		:param catalog_name:
			Str, name of resulting catalog
			(default: "")

		:return:
			instance of :class:`EQCatalog`
		"""
		from itertools import compress

		if len(point) == 2:
			distances = self.get_epicentral_distances(*point)
		elif len(point) == 3:
			distances = self.get_hypocentral_distances(*point)
		eq_list = list(compress(self.eq_list, distances <= distance))
		if not catalog_name:
			catalog_name = self.name + " (%s km radius from %s)" % (distance, point)
		lons, lats = geodetic.spherical_point_at(*point, distance=distance*1000,
												azimuth=np.arange(0, 360, 90))
		region = (lons.min(), lons.max(), lats.min(), lats.max())
		subcat = EQCatalog(eq_list, self.start_date, self.end_date, region, catalog_name,
						default_Mrelations=self.default_Mrelations,
						default_completeness=self.default_completeness)
		return subcat

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
		from osgeo import ogr

		if isinstance(poly_obj, ogr.Geometry):
			## Construct WGS84 projection system corresponding to earthquake coordinates
			from mapping.geotools.coordtrans import WGS84

			## Point object that will be used to test if earthquake is inside zone
			point = ogr.Geometry(ogr.wkbPoint)
			point.AssignSpatialReference(WGS84)

			if poly_obj.GetGeometryName() in ("MULTIPOLYGON", "POLYGON", "LINESTRING"):
				## Objects other than polygons or closed polylines will be skipped
				if poly_obj.GetGeometryName() == "LINESTRING":
					line_obj = poly_obj
					if line_obj.IsRing() and line_obj.GetPointCount() > 3:
						## Note: Could not find a way to convert linestrings to polygons
						## The following only works for linearrings (what is the difference??)
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
				envelope = poly_obj.GetEnvelope()
				lons, lats = envelope[:2], envelope[2:]
				"""
				linear_ring = poly_obj.GetGeometryRef(0)
				## Note: in some versions of ogr, GetPoints method does not exist
				#points = linear_ring.GetPoints()
				points = [linear_ring.GetPoint(i) for i in range(linear_ring.GetPointCount())]
				lons, lats = zip(*points)[:2]
				"""
			else:
				msg = 'Warning: %s not a polygon geometry!'
				msg %= poly_obj.GetGeometryName()
				print(msg)

		else:
			import openquake.hazardlib as oqhazlib
			if isinstance(poly_obj, oqhazlib.geo.Polygon):
				mesh = oqhazlib.geo.Mesh(self.get_longitudes(), self.get_latitudes(),
										depths=None)
				intersects = poly_obj.intersects(mesh)
				#idxs = np.argwhere(intersects == True)
				#idxs = [idx[0] for idx in idxs]
				idxs = np.where(intersects == True)[0]
				zone_catalog = self.__getitem__(idxs)
				lons = zone_catalog.get_longitudes()
				lats = zone_catalog.get_latitudes()
				eq_list = zone_catalog.eq_list
			else:
				raise Exception("poly_obj not recognized!")

		if len(eq_list):
			region = (min(lons), max(lons), min(lats), max(lats))
		else:
			region = None
		if not catalog_name:
			catalog_name = self.name + " (inside polygon)"
		return EQCatalog(eq_list, self.start_date, self.end_date, region, catalog_name,
						default_Mrelations=self.default_Mrelations,
						default_completeness=self.default_completeness)

	def subselect_fault(self, oq_fault, max_dist, catalog_name=''):
		"""
		Subselect earthquakes located within given distance from a fault

		:param oq_fault:
			instance of :class:`rshalib.source.SimpleFaultSource`
		:param max_dist:
			float, maximum distance around fault (in km)
		:param catalog_name:
			str, name of resulting catalog
			(default: "")

		:return:
			instance of :class:`EQCatalog`
		"""
		import openquake.hazardlib as oqhazlib
		import hazard.rshalib as rshalib

		lons, lats = self.get_longitudes(), self.get_latitudes()
		depths = self.get_depths()
		eq_mesh = oqhazlib.geo.mesh.Mesh(lons, lats, depths)
		fault_mesh = oq_fault.get_mesh()
		distances = fault_mesh.get_min_distance(eq_mesh)
		idxs = np.where(distances <= max_dist)[0]
		fault_catalog = self.__getitem__(idxs)
		if not catalog_name:
			catalog_name = self.name
			if hasattr(oq_fault, 'name'):
				catalog_name += (' (%s)' % oq_fault.name)
			else:
				catalog_name += " (fault)"
		fault_catalog.name = catalog_name
		return fault_catalog

	def subselect_line(self, start_pt, end_pt, distance, catalog_name=''):
		"""
		Subselect earthquakes located within given distance from a
		straight line (or cross section)

		:param start_pt:
			(lon, lat) tuple
		:param end_pt:
			(lon, lat) tuple
		:param distance:
			float, distance to line in km
		:param catalog_name:
			Str, name of resulting catalog
			(default: "")

		:return:
			instance of :class:`EQCatalog`
		"""
		import mapping.geotools.geodetic as geodetic
		from mapping.layeredbasemap import PolygonData

		## Note: we don't use a buffer, as this would also include a region
		## left and right of the endpoints
		lon1, lat1 = start_pt[:2]
		lon2, lat2 = end_pt[:2]
		line_az = geodetic.spherical_azimuth(lon1, lat1, lon2, lat2)
		pg_lons, pg_lats = [], []
		distance *= 1000
		lons, lats = geodetic.spherical_point_at(np.array([lon1, lon2]),
								np.array([lat1, lat2]), distance, line_az - 90)
		pg_lons.extend(lons)
		pg_lats.extend(lats)
		pg_lons.append(lon2)
		pg_lats.append(lat2)
		lons, lats = geodetic.spherical_point_at(np.array([lon2, lon1]),
								np.array([lat2, lat1]), distance, line_az + 90)
		pg_lons.extend(lons)
		pg_lats.extend(lats)
		pg_lons.append(lon1)
		pg_lats.append(lat1)
		pg_lons.append(pg_lons[0])
		pg_lats.append(pg_lats[0])

		pg = PolygonData(pg_lons, pg_lats)

		if not catalog_name:
			catalog_name = self.name + " (along line)"

		return self.subselect_polygon(pg.to_ogr_geom(), catalog_name=catalog_name)

	def split_into_zones(self,
		source_model_name, ID_colname="",
		fix_mi_lambert=True,
		verbose=True):
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
		from .rob.source_models import read_source_model

		zone_catalogs = OrderedDict()

		if isinstance(source_model_name, basestring):
			## Read zone model from GIS file
			model_data = read_source_model(source_model_name, ID_colname=ID_colname,
								fix_mi_lambert=fix_mi_lambert, verbose=verbose)

			for zoneID, zone_data in model_data.items():
				zone_poly = zone_data['obj']
				if zone_poly.GetGeometryName() == "POLYGON" or zone_poly.IsRing():
					## Fault sources will be skipped
					zone_catalogs[zoneID] = self.subselect_polygon(zone_poly,
															catalog_name=zoneID)
		else:
			import hazard.rshalib as rshalib
			if isinstance(source_model_name, rshalib.source.SourceModel):
				source_model = source_model_name
				for src in source_model.sources:
					if isinstance(src, rshalib.source.AreaSource):
						zone_poly = src.polygon
						zoneID = src.source_id
						zone_catalogs[zoneID] = self.subselect_polygon(zone_poly,
															catalog_name=zoneID)

		return zone_catalogs

	def split_into_time_intervals(self, time_interval, start_date=None):
		"""
		:param time_interval:
			int (years) or instance of :class:`datetime.timedelta` or
			:class:`np.timedelta64` (precision of days or better)
		:param start_date:
			Int or date or datetime object specifying start of time window of interest
			If integer, start_date is interpreted as start year
			(default: None)

		:return:
			list with instances of :class:`EQCatalog`
		"""
		subcatalogs = []
		if start_date is None:
			start_date = self.start_date
		if isinstance(time_interval, int):
			time_interval = np.timedelta64(time_interval, 'Y')
		else:
			time_interval = timelib.as_np_timedelta(time_interval)
		end_date = self.start_date + time_interval
		unit = timelib.get_datetime_unit(end_date)
		max_end_date = self.end_date + np.timedelta64(1, unit)
		while start_date <= self.end_date:
			catalog = self.subselect(start_date=start_date, end_date=min(end_date, max_end_date),
									include_right_edges=False)
			subcatalogs.append(catalog)
			start_date = end_date
			end_date = start_date + time_interval

		return subcatalogs

	def subselect_completeness(self,
		completeness=None,
		Mtype="MW", Mrelation={},
		catalog_name="",
		verbose=True):
		"""
		Subselect earthquakes in the catalog that conform with the specified
		completeness criterion.

		:param completeness:
			instance of :class:`Completeness`
			(default: None)
		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML"
			(default: "MW")
		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param catalog_name:
			str, name of resulting catalog
			(default: "")
		:param verbose:
			Bool, whether or not some info should be printed (default: True)

		:return:
			instance of :class:`EQCatalog`
		"""
		completeness = completeness or self.default_completeness
		Mrelation = Mrelation or self.default_Mrelations.get(Mtype, {})

		if completeness:
			start_date = min(completeness.min_dates)
			if completeness.Mtype != Mtype:
				raise Exception("Magnitude type of completeness "
								"not compatible with specified Mtype!")
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
			print("Number of events constrained by completeness criteria: %d out of %d"
					% (len(eq_list), len(self.eq_list)))

		if not catalog_name:
			catalog_name = self.name + " (completeness-constrained)"
		return EQCatalog(eq_list, start_date=start_date, end_date=end_date,
						region=self.region, name=catalog_name,
						default_Mrelations=dict(self.default_Mrelations, **{Mtype: Mrelation}),
						default_completeness=completeness)

	def split_completeness(self,
		completeness=None,
		Mtype="MW", Mrelation={}):
		"""
		Split catlog in subcatalogs according to completeness periods and magnitudes

		:param completeness:
			instance of :class:`Completeness`
			(default: None)
		:param Mtype:
			String, magnitude type: "MW", "MS" or "ML" (default: "MW")
		:param Mrelation":
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})

		:return:
			list of instances of :class:`EQCatalog`
		"""
		completeness = completeness or self.default_completeness
		assert Mtype == completeness.Mtype
		completeness_catalogs = []
		min_mags = completeness.min_mags[::-1]
		max_mags = list(min_mags[1:]) + [None]
		start_dates = completeness.min_dates[::-1]
		for Mmin, Mmax, start_date in zip(min_mags, max_mags, start_dates):
			catalog = self.subselect(Mmin=Mmin, Mmax=Mmax, start_date=start_date,
								end_date=self.end_date, include_right_edges=False)
			completeness_catalogs.append(catalog)
		return completeness_catalogs

	def subselect_declustering(self,
		dc_method,
		dc_window,
		Mrelation={},
		catalog_name=None):
		"""
		Decluster catalog using the given method and window definition

		:param dc_method:
			str or instance of :class:`declustering.DeclusteringMethod`
		:param dc_window:
			str or instance of :class:`declustering.DeclusteringWindow`
		:param Mrelation:
			dict specifying how to convert catalog magnitudes to MW
			(default: {})
		:param catalog_name:
			str, name of resultig catalog
			(default: None)

		:return:
			instance of :class:`EQCatalog`, declustered catalog
		"""
		if isinstance(dc_method, basestring):
			from . import declustering
			if not 'Method' in dc_method:
				dc_method += 'Method'
			dc_method = getattr(declustering, dc_method)()

		if isinstance(dc_window, basestring):
			from . import declustering
			if not 'Window' in dc_window:
				dc_window += 'Window'
			dc_window = getattr(declustering, dc_window)()

		#dc_cat = dc_method.decluster_catalog(self, dc_window, Mrelation)
		dc_result = dc_method.analyze_clusters(self, dc_window, Mrelation)
		dc_cat = dc_result.get_declustered_catalog()
		if catalog_name != None:
			dc_cat.name = catalog_name
		return dc_cat

	def subselect_declustering_legacy(self,
		method="Cluster",
		window="GardnerKnopoff1974",
		fa_ratio=0.5,
		Mtype="MW", Mrelation={},
		return_triggered_catalog=False,
		catalog_name=""):
		"""
		Subselect earthquakes in the catalog that conform with the specified
		declustering method and params.

		This is the original implementation written by Bart Vleminckx,
		which calls his original methods in declustering.py, but
		contain bugs.

		Left here for repeatability purposes.

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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
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
		from .declustering import (WindowMethod, ClusterMethod,
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
		datetimes = self.get_datetimes()
		lons = self.get_longitudes()
		lats = self.get_latitudes()

		## Remove NaN magnitudes
		idxs = ~np.isnan(magnitudes)
		magnitudes = magnitudes[idxs]
		datetimes = datetimes[idxs]
		lons = lons[idxs]
		lats = lats[idxs]

		d_index = methods[method].decluster_legacy(magnitudes, datetimes, lons, lats,
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


	## Various binning methods

	def bin_by_time_interval(self,
		start_datetime, end_datetime, time_delta, include_incomplete=True,
		Mmin=None, Mmax=None,
		Mtype="MW", Mrelation={}):
		"""
		Bin earthquakes into time intervals.
		Note that this may include an incomplete interval at the end!

		:param start_datetime:
			str or instance of :class:`datetime.date` or :class:`datetime.datetime`
			or :class:`np.datetime64`, start date and time (left edge of first bin)
		:param end_datetime:
			str or instance of :class:`datetime.date` or :class:`datetime.datetime`
			or :class:`np.datetime64`, end date and time (right edge of last bin)
		:param time_delta:
			instance of :class:`datetime.timedelta` or :class:`np.timedelta64`,
			time interval
			Note that datetimes and time_delta must have compatible base time units!
		:param include_incomplete:
			bool, whether or not last interval should be included if it
			is empty
			(default: True)
		:param Mmin:
			Float, minimum magnitude (inclusive)
		:param Mmax:
			Float, maximum magnitude (inclusive)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})

		:return:
			tuple (bins_N, bins_times)
			bins_N: array containing number of earthquakes for each bin
			bins_times: datetime64 array containing start datetime of each interval
		"""
		start_datetime = timelib.as_np_datetime(start_datetime)
		end_datetime = timelib.as_np_datetime(end_datetime)
		time_delta = timelib.as_np_timedelta(time_delta)
		bins_times = np.arange(start_datetime, end_datetime, time_delta)
		## Select earthquakes according to magnitude criteria
		subcatalog = self.subselect(start_date=start_datetime, end_date=end_datetime,
						Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		bin_idxs = ((subcatalog.get_datetimes() - start_datetime) / time_delta)
		bin_idxs = bin_idxs.astype('int')
		bins_N = np.zeros(len(bins_times))
		for i in range(len(bins_N)):
			bins_N[i] = np.sum(bin_idxs == i)
		if not include_incomplete:
			if subcatalog.end_date - bins_times[-1] < time_delta:
				bins_N = bins_N[:-1]
				bins_times = bins_times[:-1]
		return (bins_N, bins_times)

	def bin_M0_by_time_interval(self,
		start_datetime, end_datetime, time_delta,
		Mmin=None, Mmax=None,
		Mtype="MW", Mrelation={}):
		"""
		Bin earthquake moments into time intervals.

		:param start_datetime:
		:param end_datetime:
		:param time_delta:
		:param Mmin:
		:param Mmax:
		:param Mtype:
		:param Mrelation:
			see :meth:`bin_by_time_interval`

		:return:
			tuple (bins_M0, bins_times)
			bins_M0: array containing summed seismic moment in each bin
			bins_times: datetime64 array containing start datetime of each interval
		"""
		start_datetime = timelib.as_np_datetime(start_datetime)
		end_datetime = timelib.as_np_datetime(end_datetime)
		time_delta = timelib.as_np_timedelta(time_delta)
		bins_times = np.arange(start_datetime, end_datetime, time_delta)
		subcatalog = self.subselect(start_date=start_datetime, end_date=end_datetime,
						Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		bin_idxs = ((subcatalog.get_datetimes() - start_datetime) / time_delta)
		bin_idxs = bin_idxs.astype('int')
		bins_M0 = np.zeros(len(bins_times))
		M0 = subcatalog.get_M0(Mrelation=Mrelation)
		for i in range(len(bins_M0)):
			bins_M0[i] = np.nansum(M0[bin_idxs == i])
		return (bins_M0, bins_times)

	def bin_by_year(self,
		start_year, end_year, dYear,
		Mmin=None, Mmax=None,
		Mtype="MW", Mrelation={}):
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})

		:return:
			tuple (bins_N, bins_Years)
			bins_N: array containing number of earthquakes for each bin
			bins_Years: array containing lower year of each interval
		"""
		bins_Years = np.arange(start_year, end_year+dYear, dYear)
		subcatalog = self.subselect(start_date=start_year, end_date=end_year,
						Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		Years = subcatalog.get_years()
		bins_N, _ = np.histogram(Years, bins_Years)
		return (bins_N, bins_Years[:-1])

	def bin_M0_by_year(self,
		start_year, end_year, dYear,
		Mmin=None, Mmax=None,
		Mrelation={}):
		"""
		Bin earthquake moments into year intervals

		:param start_year:
		:param end_year:
		:param dYear:
		:param Mmin:
		:param Mmax:
		:param Mrelation:
			see :meth:`bin_by_year`

		:return:
			tuple (bins_M0, bins_Years)
			bins_M0: array containing summed seismic moment in each bin
			bins_Years: array containing lower year of each interval
		"""
		bins_Years = np.arange(start_year, end_year, dYear)
		bins_M0 = np.zeros(len(bins_Years))
		## Select years according to magnitude criteria
		subcatalog = self.subselect(start_date=start_year, end_date=end_year,
						Mmin=Mmin, Mmax=Mmax, Mtype="MW", Mrelation=Mrelation)
		## Note: idx 0 returned by np.digitize corresponds to years < start_date
		bin_idxs = np.digitize(subcatalog.get_fractional_years(), bins_Years) - 1
		M0 = subcatalog.get_M0(Mrelation=Mrelation)
		for i in range(len(bins_Years)):
			bins_M0[i] = np.nansum(M0[bin_idxs == i])
		return bins_M0, bins_Years

	def bin_by_day(self,
		start_date, end_date, dday,
		Mmin=None, Mmax=None,
		Mtype="MW", Mrelation={}):
		"""
		Bin earthquakes into day intervals

		:param start_date:
			instance of :class:`datetime.date` or :class:`np.datetime64`, start date
		:param end_date:
			instance of :class:`datetime.date` or :class:`np.datetime64`, end date
			(right edge of last bin)
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})

		:return:
			tuple (bins_N, bins_Days)
			bins_N: array containing number of earthquakes for each bin
			bins_Days: datetime64 array containing start day of each interval
		"""
		time_delta = np.timedelta64(dday, 'D')
		return self.bin_by_time_interval(start_date, end_date, time_delta)
		#bins_Days = np.arange(0, int(timelib.timespan(start_date, end_date, 'D')) + dday, dday)
		## Select years according to magnitude criteria
		#subcatalog = self.subselect(start_date=start_date, end_date=end_date,
		#				Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		#Days = timelib.timespan(start_date, subcatalog.get_datetimes(), 'D')
		#bins_N, bins_Days = np.histogram(Days, bins_Days)
		#return (bins_N, bins_Days[:-1])

	def bin_M0_by_day(self,
		start_date, end_date, dday,
		Mmin=None, Mmax=None,
		Mrelation={}):
		"""
		Bin earthquake moments into day intervals.

		:param start_date:
		:param end_date:
		:param dday:
		:param Mmin:
		:param Mmax:
		:param Mrelation:
			see :meth:`bin_by_day`

		:return:
			tuple (bins_M0, bins_Days)
			bins_M0: array containing summed seismic moment in each bin
			bins_Days: datetime64 array containing start day of each interval
		"""
		time_delta = np.timedelta64(dday, 'D')
		return self.bin_M0_by_time_interval(start_date, end_date, time_delta)
		#bins_Days = np.arange(0, int(timelib.timespan(start_date, end_date, 'D')) + dday, dday)
		#bins_M0 = np.zeros(len(bins_Days))
		## Select years according to magnitude criteria
		#subcatalog = self.subselect(start_date=start_date, end_date=end_date,
		#				Mmin=Mmin, Mmax=Mmax, Mtype="MW", Mrelation=Mrelation)
		#eq_days = [timelib.timespan(start_date, eq.date, 'D') for eq in subcatalog]
		#bin_idxs = np.digitize(eq_days, bins_Days) - 1
		#M0 = subcatalog.get_M0(Mrelation=Mrelation)
		#for i in range(len(bins_Days)):
		#	bins_M0[i] = np.nansum(M0[bin_idxs == i])
		#return bins_M0, bins_Days

	def bin_by_hour(self,
		Mmin=None, Mmax=None,
		Mtype="MW", Mrelation={},
		start_year=None, end_year=None):
		"""
		Bin earthquakes into hour intervals [0 - 24]

		:param Mmin:
			Float, minimum magnitude (inclusive) (default: None)
		:param Mmax:
			Float, maximum magnitude (inclusive) (default: None)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param start_year:
			Int, lower year to bin (default: None)
		:param end_year:
			Int, upper year to bin (default: None)

		:return:
			tuple (bins_N, bins_Hours)
			bins_N: array containing number of earthquakes for each bin
			bins_Hours: array containing lower limit of each hour interval
		"""
		subcatalog = self.subselect(start_date=start_year, end_date=end_year,
						Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		hours = np.array([eq.get_fractional_hour() for eq in subcatalog])
		bins_Hr = np.arange(25)
		bins_N, _ = np.histogram(hours, bins_Hr)
		return bins_N, bins_Hr[:-1]

	def bin_M0_by_hour(self,
		Mmin=None, Mmax=None,
		Mtype="MW", Mrelation={},
		start_year=None, end_year=None):
		"""
		Bin earthquakes into hour intervals [0 - 24]

		:param Mmin:
		:param Mmax:
		:param MW:
		:param Mrelation:
		:param start_year:
		:param end_year:
			see :meth:`bin_by_hour`
		"""
		subcatalog = self.subselect(start_date=start_year, end_date=end_year,
						Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		hours = np.array([eq.get_fractional_hour() for eq in subcatalog])
		bins_Hr = np.arange(24)
		bin_idxs = np.digitize(hours, bins_Hr) - 1
		bins_M0 = np.zeros(len(bins_Hr))
		M0 = subcatalog.get_M0(Mrelation=Mrelation)
		for i in range(len(bins_Hr)):
			bins_M0[i] = np.nansum(M0[bin_idxs == i])
		return bins_M0, bins_Hr

	def get_daily_nightly_mean(self,
		Mmin=None, Mmax=None,
		Mtype="MW", Mrelation={},
		start_year=None, end_year=None,
		day=(7, 19)):
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param start_year:
			Int, lower year to bin (default: None)
		:param end_year:
			Int, upper year to bin (default: None)
		:param day:
			Tuple (min_hour, max_hour), default: (7, 19)

		:return:
			Tuple (mean, daily mean, nightly mean)
		"""
		bins_N, _ = self.bin_by_hour(Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation,
									start_year=start_year, end_year=end_year)
		mean = np.mean(bins_N)
		mean_day = np.mean(bins_N[day[0]:day[1]])
		mean_night = np.mean(np.concatenate([bins_N[:day[0]], bins_N[day[1]:]]))
		return (mean, mean_day, mean_night)

	def bin_by_depth(self,
		min_depth=0, max_depth=30, bin_width=2,
		max_depth_error=None,
		Mmin=None, Mmax=None,
		Mtype="MW", Mrelation={},
		start_date=None, end_date=None,
		include_right_edge=False):
		"""
		Bin earthquakes into depth bins

		:param min_depth:
			Int, minimum depth in km (left edge of first bin)
			(default: 0)
		:param max_depth:
			Int, maximum depth in km (right edge of last bin)
			(default: 30)
		:param bin_width:
			Int, bin width in km
			(default: 2)
		:param max_depth_error:
			Float, maximum depth uncertainty
			(default: None)
		:param Mmin:
			Float, minimum magnitude (inclusive)
			(default: None)
		:param Mmax:
			Float, maximum magnitude (inclusive)
			(default: None)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default:  çà)
		:param start_date:
			Int or instance of :class:`datetime.date` or :class:`np.datetime64`,
			lower year or date to bin
			(default: None)
		:param end_date:
			Int or instance of :class:`datetime.date` or :class:`np.datetime64`,
			upper year or date to bin
			(default: None)
		:param include_right_edge:
			bool, whether or not right edge should be included in depth bins
			(default: False)

		:return:
			tuple (bins_N, bins_depth)
			bins_N: array containing number of earthquakes for each bin
			bins_depth: array containing lower depth value of each interval
		"""
		subcatalog = self.subselect(start_date=start_date, end_date=end_date,
						Mmin=Mmin, Mmax=Mmax, Mtype=Mtype, Mrelation=Mrelation)
		if max_depth_error:
			## Note: NaN errz values are ignored
			depths = [eq.depth for eq in subcatalog if not eq.depth in (None, np.nan)
							and 0 <= eq.errz <= max_depth_error]
		else:
			## Ignore NaN depth values
			depths = subcatalog.get_depths()
			depths = depths[~np.isnan(depths)]
		bins_depth = np.arange(min_depth, max_depth + bin_width, bin_width)
		bins_N, _ = np.histogram(depths, bins_depth)
		if not include_right_edge:
			bins_depth = bins_depth[:-1]
		return bins_N, bins_depth

	def bin_M0_by_depth(self,
		min_depth=0, max_depth=30, bin_width=2,
		max_depth_error=None,
		Mmin=None, Mmax=None,
		Mrelation={},
		start_date=None, end_date=None):
		"""
		Bin earthquake moments into depth bins

		:param min_depth:
		:param max_depth:
		:param bin_width:
		:param max_depth_error:
		:param Mmin:
		:param Mmax:
		:param Mrelation:
		:param start_date:
		:param end_date:
			see :meth:`bin_by_depth`

		:return:
			tuple (bins_M0, bins_depth)
			bins_M0: array containing summed seismic moment in each bin
			bins_depth: array containing lower depth value of each interval
		"""
		bins_depth = np.arange(min_depth, max_depth, bin_width)
		bins_M0 = np.zeros(len(bins_depth))
		subcatalog = self.subselect(start_date=start_date, end_date=end_date,
						Mmin=Mmin, Mmax=Mmax, Mtype="MW", Mrelation=Mrelation,
						min_depth=min_depth, max_depth=max_depth)

		if not max_depth_error:
			max_depth_error = 100
		M0 = subcatalog.get_M0(Mrelation=Mrelation)
		for e, eq in enumerate(subcatalog):
			if (eq.depth not in (None, np.nan)
				and 0 <= eq.errz <= max_depth_error):
				[bin_idx] = np.digitize([eq.depth], bins_depth) - 1
				if 0 <= bin_idx < len(bins_M0):
					bins_M0[bin_idx] += np.nansum(M0[e])
		return bins_M0, bins_depth

	def bin_by_mag(self,
		Mmin, Mmax, dM=0.2,
		Mtype="MW", Mrelation={},
		completeness=None,
		include_right_edge=False,
		verbose=True):
		"""
		Bin all earthquake magnitudes in catalog according to specified
		magnitude interval.

		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude interval
			(default: 0.2)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness` containing initial years of
			completeness and corresponding minimum magnitudes
			(default: None, will not apply completeness even if
			catalog default_completeness is defined)
		:param include_right_edge:
			bool, whether or not right edge should be included in magnitude bins
			(default: False)
		:param verbose:
			Bool, whether or not to print additional information

		:return:
			Tuple (bins_N, bins_Mag)
			bins_N: array containing number of earthquakes for each magnitude interval
			bins_Mag: array containing lower magnitude of each interval
		"""
		from hazard.rshalib.utils import seq

		## Note: do not take into account catalog default_completeness
		#completeness = completeness or self.default_completeness

		## Set lower magnitude to lowermost threshold magnitude possible
		if completeness:
			#Mmin = max(Mmin, completeness.get_completeness_magnitude(self.end_date))
			Mmin = max(Mmin, completeness.get_lowest_completeness_magnitude(self.end_date))

		## Construct bins_Mag, including Mmax as right edge
		#num_bins = int(np.floor((Mmax - Mmin) / dM)) + 1
		#bins_Mag = Mmin + np.arange(num_bins) * dM
		bins_Mag = seq(Mmin, Mmax, dM)

		## Select magnitudes according to completeness criteria
		if completeness:
			cc_catalog = self.subselect_completeness(completeness, Mtype,
												Mrelation, verbose=verbose)
			Mags = cc_catalog.get_magnitudes(Mtype, Mrelation)
		else:
			Mags = self.get_magnitudes(Mtype, Mrelation)
		Mags[np.isnan(Mags)] = Mmin - 1

		## Compute number of earthquakes per magnitude bin
		bins_N, bins_Mag = np.histogram(Mags, bins_Mag)
		if not include_right_edge:
			bins_Mag = bins_Mag[:-1]

		return bins_N, bins_Mag

	## Completeness methods

	def get_initial_completeness_dates(self, magnitudes,
									completeness=None):
		"""
		Compute initial date of completeness for list of magnitudes

		:param magnitudes:
			list or numpy array, magnitudes
		:param completeness:
			instance of :class:`Completeness` containing initial years of
			completeness and corresponding minimum magnitudes.
			If None, use start year of catalog.
			(default: None)

		:return:
			numpy array, completeness dates
		"""
		## Calculate year of completeness for each magnitude interval
		completeness = completeness or self.default_completeness
		if completeness:
			completeness_dates = []
			for M in magnitudes:
				start_date = max(self.start_date,
								completeness.get_initial_completeness_date(M))
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
		min_date = self.start_year
		return Completeness([min_date], [Mmin], Mtype=Mtype)

	def get_completeness_timespans(self, magnitudes,
								completeness=None, unit='Y'):
		"""
		Compute completeness timespans for list of magnitudes

		:param magnitudes:
			list or numpy array, magnitudes
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes. If None, use start year of
			catalog
			(default: None)
		:param unit:
			str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
			(year|week|day|hour|minute|second|millisecond|microsecond)
			(default: 'Y')

		:return:
			numpy float array, completeness timespans in fractions of :param:`unit`
		"""
		completeness = completeness or self.default_completeness
		if not completeness:
			min_mag = np.min(magnitudes)
			completeness = self.get_uniform_completeness(min_mag, Mtype="MW")
		return completeness.get_completeness_timespans(magnitudes, self.end_date, unit=unit)

	def calc_return_period(self, Mmin, Mtype='MW', Mrelation={},
							completeness=None, time_unit='Y'):
		"""
		Compute return period for magnitudes above given lower value

		:param Mmin:
			float, lower magnitude
		:param Mtype:
		:param Mrelation:
		:param completeness:
			see ...
		:param time_unit:
			str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
			(year|week|day|hour|minute|second|millisecond|microsecond)
			(default: 'Y')

		:return:
			float, return period
		"""
		## Apply completeness constraint, and truncate result to
		## completeness year for specified minimum magnitude
		completeness = completeness or self.default_completeness
		if completeness:
			min_date = completeness.get_initial_completeness_date(Mmin)
			cc_catalog = self.subselect_completeness(Mtype=Mtype, Mrelation=Mrelation,
													completeness=completeness)
		else:
			min_date = self.start_date
			cc_catalog = self
		catalog = cc_catalog.subselect(start_date=min_date, Mmin=Mmin,
									Mtype=Mtype, Mrelation=Mrelation)

		num_events = len(catalog)
		td = catalog.get_time_delta(from_events=False)
		td_units = timelib.fractional_time_delta(td, time_unit)
		tau = td_units / float(num_events)

		return tau

	## MFD methods

	def get_incremental_mag_freqs(self,
		Mmin, Mmax, dM=0.2,
		Mtype="MW", Mrelation={},
		completeness=None,
		trim=False,
		verbose=True):
		"""
		Compute incremental magnitude-frequency distribution.

		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude interval
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes
			(default: None)
		:param trim:
			Bool, whether empty bins at start and end should be trimmed
			(default: False)
		:param verbose:
			Bool, whether or not to print additional information
			(default: True)

		:return:
			Tuple (bins_N_incremental, bins_Mag)
			bins_N_incremental: incremental annual occurrence rates
			bins_Mag: left edges of magnitude bins
		"""
		bins_N, bins_Mag = self.bin_by_mag(Mmin, Mmax, dM, Mtype=Mtype,
				Mrelation=Mrelation, completeness=completeness, verbose=verbose)
		bins_timespans = self.get_completeness_timespans(bins_Mag, completeness)

		bins_N_incremental = bins_N / bins_timespans

		## Optionally, trim empty trailing intervals
		if trim:
			last_non_zero_index = np.where(bins_N > 0)[0][-1]
			bins_N_incremental = bins_N_incremental[:last_non_zero_index+1]
			bins_Mag = bins_Mag[:last_non_zero_index+1]

		return bins_N_incremental, bins_Mag

	def get_incremental_mfd(self,
		Mmin, Mmax, dM=0.2,
		Mtype="MW", Mrelation={},
		completeness=None,
		trim=False,
		verbose=True):
		"""
		Compute incremental magnitude-frequency distribution.

		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude interval
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes
			(default: None)
		:param trim:
			Bool, whether empty bins at start and end should be trimmed
			(default: False)
		:param verbose:
			Bool, whether or not to print additional information
			(default: True)

		:return:
			instance of :class:`hazard.rshalib.EvenlyDiscretizedMFD`
		"""
		from hazard.rshalib.mfd import EvenlyDiscretizedMFD
		bins_N_incremental, bins_Mag = self.get_incremental_mag_freqs(Mmin, Mmax,
					dM, Mtype, Mrelation, completeness, trim, verbose=verbose)
		## Mmin may have changed depending on completeness
		Mmin = bins_Mag[0]
		return EvenlyDiscretizedMFD(Mmin + dM/2, dM, list(bins_N_incremental),
									Mtype=Mtype)

	def get_cumulative_mag_freqs(self,
		Mmin, Mmax, dM=0.1,
		Mtype="MW", Mrelation={},
		completeness=None,
		trim=False,
		verbose=True):
		"""
		Compute cumulative magnitude-frequency distribution.

		:param Mmin:
			Float, minimum magnitude to bin
		:param Mmax:
			Float, maximum magnitude to bin
		:param dM:
			Float, magnitude bin width
			(default: 0.1)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes
			(default: None)
		:param trim:
			Bool, whether empty bins at start and end should be trimmed
			(default: False)
		:param verbose:
			Bool, whether or not to print additional information
			(default: True)

		:return:
			Tuple (bins_N_cumulative, bins_Mag)
			bins_N_cumulative: cumulative annual occurrence rates
			bins_Mag: left edges of magnitude bins
		"""
		bins_N_incremental, bins_Mag = self.get_incremental_mag_freqs(Mmin, Mmax,
				dM, Mtype=Mtype, completeness=completeness, trim=trim, verbose=verbose)
		## Reverse arrays for calculating cumulative number of events
		bins_N_incremental = bins_N_incremental[::-1]
		bins_N_cumulative = np.cumsum(bins_N_incremental)
		return bins_N_cumulative[::-1], bins_Mag

	def calcGR_LSQ(self,
		Mmin, Mmax, dM=0.1,
		cumul=True,
		Mtype="MW", Mrelation={},
		completeness=None,
		b_val=None,
		weighted=False,
		verbose=False):
		"""
		Calculate a and b values of Gutenberg-Richter relation using
		linear regression (least-squares).

		:param Mmin:
			Float, minimum magnitude to use for binning
		:param Mmax:
			Float, maximum magnitude to use for binning
		:param dM:
			Float, magnitude interval to use for binning
			(default: 0.1)
		:param cumul:
			Bool, whether to use cumulative (True) or incremental (False)
			occurrence rates for linear regression
			(default: True)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness`
			(default: None)
		:param b_val:
			Float, fixed b value to constrain MLE estimation
			(default: None)
			This parameter is currently ignored.
		:param weighted:
			bool, whether or not magnitude bins should be weighted by the
			number of events in them
			(default: False)
		:param verbose:
			Bool, whether some messages should be printed or not
			(default: False)

		Return value:
			Tuple (a, b, stda, stdb)
			- a: a value (intercept)
			- b: b value (slope, taken positive)
			- stda: standard deviation on a value
			- stdb: standard deviation on b value
		"""
		from hazard.rshalib.mfd.truncated_gr import get_a_separation
		from .calcGR import calcGR_LSQ

		completeness = completeness or self.default_completeness

		if weighted:
			bins_N, _ = self.bin_by_mag(Mmin, Mmax, dM, Mtype=Mtype, Mrelation=Mrelation,
										completeness=completeness, verbose=False)
			weights = bins_N
		else:
			weights = None

		if cumul:
			rates, magnitudes = self.get_cumulative_mag_freqs(Mmin, Mmax, dM,
									Mtype=Mtype, Mrelation=Mrelation, completeness=completeness,
									trim=False, verbose=verbose)
		else:
			rates, magnitudes = self.get_incremental_mag_freqs(Mmin, Mmax, dM,
									Mtype=Mtype, Mrelation=Mrelation, completeness=completeness,
									trim=False, verbose=verbose)

		a, b, stda, stdb = calcGR_LSQ(magnitudes, rates, b_val=b_val, weights=weights,
									verbose=verbose)
		if not cumul:
			a += get_a_separation(b, dM)
		return a, b, stda, stdb

	def calcGR_Aki(self,
		Mmin=None, Mmax=None, dM=0.1,
		Mtype="MW", Mrelation={},
		completeness=None,
		b_val=None,
		verbose=False):
		"""
		Calculate a and b values of Gutenberg-Richter relation using original
		maximum likelihood estimation by Aki (1965)

		:param Mmin:
			Float, minimum magnitude to use for binning (ignored)
		:param Mmax:
			Float, maximum magnitude to use for binning (ignored)
		:param dM:
			Float, magnitude interval to use for binning
			(default: 0.1)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness`
			(default: None)
		:param b_val:
			Float, fixed b value to constrain MLE estimation (ignored)
		:param verbose:
			Bool, whether some messages should be printed or not
			(default: False)

		:return:
			Tuple (a, b, stdb)
			- a: a value
			- b: b value
			- stdb: standard deviation on b value
		"""
		completeness = completeness or self.default_completeness

		return self.analyse_recurrence(dM=dM, method="MLE", aM=0., Mtype=Mtype,
								Mrelation=Mrelation, completeness=completeness)

	def calcGR_Weichert(self,
		Mmin, Mmax, dM=0.1,
		Mtype="MW", Mrelation={},
		completeness=None,
		b_val=None,
		verbose=True):
		"""
		Calculate a and b values of Gutenberg-Richter relation using maximum
		likelihood estimation for variable observation periods for different
		magnitude increments.
		Adapted from calB.m and calBfixe.m Matlab modules written by Philippe
		Rosset (ROB, 2004), which is based on the method by Weichert, 1980
		(BSSA, 70, Nr 4, 1337-1346).

		:param Mmin:
			Float, minimum magnitude to use for binning
		:param Mmax:
			Float, maximum magnitude to use for binning
		:param dM:
			Float, magnitude interval to use for binning
			(default: 0.1)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness`
			(default: None)
		:param b_val:
			Float, fixed b value to constrain MLE estimation
			(default: None)
		:param verbose:
			Bool, whether some messages should be printed or not
			(default: False)

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
		from .calcGR import calcGR_Weichert
		## Note: don't use get_incremental_mag_freqs here, as completeness
		## is taken into account in the Weichert algorithm !
		completeness = completeness or self.default_completeness

		bins_N, bins_Mag = self.bin_by_mag(Mmin, Mmax, dM, Mtype=Mtype, Mrelation=Mrelation,
										completeness=completeness, verbose=verbose)
		return calcGR_Weichert(bins_Mag, bins_N, completeness, self.end_date,
								b_val=b_val, verbose=verbose)

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
			#print(BETA)

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

			#print(SNM, NKOUNT, STMEX, SUMTEX, STM2X, SUMEXP)

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
		#print(STDA)
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
		#	print("Maximum likelihood: a=%.3f ($\pm$ %.3f), b=%.3f ($\pm$ %.3f), beta=%.3f ($\pm$ %.3f)"
		# 		% (A, STDA, B, STDB, BETA, STDBETA))
		#if Mc != None:
		#	return (A, B, BETA, LAMBDA_Mc, STDA, STDB, STDBETA, STD_LAMBDA_Mc)
		#else:
		#	return (A, B, BETA, STDA, STDB, STDBETA)

		return A, B, STDB
		"""

	#TODO: averaged Weichert method (Felzer, 2007)

	def get_estimated_mfd(self,
		Mmin, Mmax, dM=0.1,
		method="Weichert",
		Mtype="MW", Mrelation={},
		completeness=None,
		b_val=None,
		verbose=True):
		"""
		Compute a and b values of Gutenberg Richter relation, and return
		as TruncatedGRMFD object.

		:param Mmin:
			Float, minimum magnitude to use for binning
		:param Mmax:
			Float, maximum magnitude to use for binning
		:param dM:
			Float, magnitude interval to use for binning
			(default: 0.1)
		:param method:
			String, computation method, either "Weichert", "Aki", "LSQc", "LSQi",
			"wLSQc" or "wLSQi"
			(default: "Weichert")
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness`
			(default: None)
		:param b_val:
			Float, fixed b value to constrain MLE estimation
			Currently only supported by Weichert method
			(default: None)
		:param verbose:
			Bool, whether some messages should be printed or not
			(default: False)

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
		calcGR_func = {"Weichert": self.calcGR_Weichert,
						"Aki": self.calcGR_Aki,
						"LSQ": self.calcGR_LSQ}[method]
		a, b, stda, stdb = calcGR_func(Mmin=Mmin, Mmax=Mmax, dM=dM, Mtype=Mtype,
								Mrelation=Mrelation, completeness=completeness,
								b_val=b_val, verbose=verbose, **kwargs)
		return TruncatedGRMFD(Mmin, Mmax, dM, a, b, stda, stdb, Mtype)

	def plot_mfd(self,
		Mmin, Mmax, dM=0.2,
		method="Weichert",
		Mtype="MW", Mrelation={},
		completeness=None,
		b_val=None,
		num_sigma=0,
		color_observed="b",
		color_estimated="r",
		plot_completeness_limits=True,
		title=None, lang="en",
		fig_filespec=None, dpi=300,
		verbose=False, **kwargs):
		"""
		Compute GR MFD from observed MFD, and plot result

		:param Mmin:
			Float, minimum magnitude to use for binning
		:param Mmax:
			Float, maximum magnitude to use for binning
		:param dM:
			Float, magnitude interval to use for binning
			(default: 0.1)
		:param method:
			String, computation method, either "Weichert", "Aki", "LSQc", "LSQi",
			"wLSQc" or "wLSQi"
			If None, only observed MFD will be plotted.
			(default: "Weichert")
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness`
			(default: None)
		:param b_val:
			Float, fixed b value to constrain Weichert estimation
			(default: None)
		:param num_sigma:
			Int, number of standard deviations to consider for plotting uncertainty
			(default: 0)
		:param color_observed:
			matplotlib color specification for observed MFD
		:param color_estimated:
			matplotlib color specification for estimated MFD
		:param plot_completeness_limits:
			Bool, whether or not to plot completeness limits
			(default: True)
		:param title:
			String, plot title. If None, title will be automatically generated
			(default: None)
		:param lang:
			String, language of plot axis labels
			(default: "en")
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param dpi:
			Int, image resolution in dots per inch
			(default: 300)
		:param verbose:
			Bool, whether some messages should be printed or not
			(default: False)
		:kwargs:
			additional keyword arguments understood by
			:func:`generic_mpl.plot_xy`

		:return:
			matplotlib Axes instance
		"""
		from hazard.rshalib.mfd import plot_mfds

		mfd_list, labels, colors, styles = [], [], [], []
		cc_catalog = self.subselect_completeness(completeness, Mtype, Mrelation,
												verbose=verbose)
		observed_mfd = cc_catalog.get_incremental_mfd(Mmin, Mmax, dM=dM, Mtype=Mtype,
								Mrelation=Mrelation, completeness=completeness)
		mfd_list.append(observed_mfd)
		label = {"en": "Observed", "nl": "Waargenomen"}[lang]
		labels.append(label)
		colors.append(color_observed)

		styles.append('o')
		if method:
			estimated_mfd = cc_catalog.get_estimated_mfd(Mmin, Mmax, dM=dM, method=method,
							Mtype=Mtype, Mrelation=Mrelation, completeness=completeness,
							b_val=b_val, verbose=verbose)
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
				sigma_mfd1 = cc_catalog.get_estimated_mfd(Mmin, Mmax, dM=dM, method=method,
								Mtype=Mtype, Mrelation=Mrelation, completeness=completeness,
								b_val=b+num_sigma*stdb, verbose=verbose)
				mfd_list.append(sigma_mfd1)
				label = {"en": "Computed", "nl": "Berekend"}[lang]
				label += " $\pm$ %d sigma" % num_sigma
				labels.append(label)
				colors.append(color_estimated)
				styles.append('--')
				sigma_mfd2 = cc_catalog.get_estimated_mfd(Mmin, Mmax, dM=dM, method=method,
								Mtype=Mtype, Mrelation=Mrelation, completeness=completeness,
								b_val=b-num_sigma*stdb, verbose=verbose)
				mfd_list.append(sigma_mfd2)
				labels.append("_nolegend_")
				colors.append(color_estimated)
				styles.append('--')

		if title is None:
			num_events = len(cc_catalog)
			Mmax_obs = cc_catalog.get_Mmax(Mtype, Mrelation)
			title = "%s (%d events, Mmax=%.2f)" % (self.name, num_events, Mmax_obs)
		completeness_limits = {True: completeness, False: None}[plot_completeness_limits]
		end_year = timelib.to_year(self.end_date)

		return plot_mfds(mfd_list, colors=colors, styles=styles, labels=labels,
						completeness=completeness_limits, end_year=end_year,
						title=title, lang=lang,
						fig_filespec=fig_filespec, dpi=dpi, **kwargs)

	## Random catalogs

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
			if eq.errM in (0, None, np.nan):
				if eq.year < 1910:
					errM = 0.5
				elif 1910 <= eq.year < 1985:
					if eq.MS > 0:
						errM = 0.2
					else:
						errM = 0.3
				elif eq.year >= 1985:
					eq.errM = 0.2
			elif eq.errM >= 1.:
				# A lot of earthquakes have errM = 9.9 ???
				errM = 0.3
			else:
				errM = eq.errM

			if eq.errh in (0, None, np.nan):
				if eq.year < 1650:
					errh = 25
				elif 1650 <= eq.year < 1910:
					errh = 15
				elif 1910 <= eq.year < 1930:
					errh = 10
				elif 1930 <= eq.year < 1960:
					errh = 5.
				elif 1960 <= eq.year < 1985:
					errh = 2.5
				elif eq.year >= 1985:
					errh = 1.5
			else:
				errh = eq.errh

			## Convert uncertainty in km to uncertainty in lon, lat
			errlon = errh / ((40075./360.) * np.cos(np.radians(eq.lat)))
			errlat = errh / (40075./360.)

			if eq.errz in (0, None, np.nan):
				errz = 5.
			else:
				errz = eq.errz

			if not eq.ML in (None, np.nan):
				ML[i,:] = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, eq.ML,
													errM, size=num_samples)
			else:
				ML[i,:] = np.nan
			if not eq.MS in (None, np.nan):
				MS[i,:] = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, eq.MS,
													errM, size=num_samples)
			else:
				MS[i,:] = np.nan
			if not eq.MW in (None, np.nan):
				MW[i,:] = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, eq.MW,
													errM, size=num_samples)
			else:
				MW[i,:] = np.nan
			lons[i,:] = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, eq.lon,
													errlon, size=num_samples)
			lats[i,:] = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, eq.lat,
													errlat, size=num_samples)
			depths[i,:] = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma,
								np.nan_to_num(eq.depth), errz, size=num_samples)
		depths = depths.clip(min=0.)

		synthetic_catalogs = []
		for n in range(num_samples):
			eq_list = []
			for i, eq in enumerate(self):
				new_eq = copy.deepcopy(eq)
				new_eq.mag['ML'] = ML[i,n]
				new_eq.mag['MS'] = MS[i,n]
				new_eq.mag['MW'] = MW[i,n]
				new_eq.lon = lons[i,n]
				new_eq.lat = lats[i,n]
				new_eq.depth = depths[i,n]
				eq_list.append(new_eq)
			synthetic_catalogs.append(EQCatalog(eq_list, self.start_date,
									self.end_date, region=self.region,
									default_Mrelations=self.default_Mrelations,
									default_completeness=self.default_completeness))

		return synthetic_catalogs

	## Mmax estimation

	def get_Bayesian_Mmax_pdf(self, prior_model="CEUS_COMP", Mmin_n=4.5,
					b_val=None, dM=0.1, truncation=(5.5, 8.25), Mtype='MW',
					Mrelation={}, completeness=None,
					verbose=True):
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes
			(default: None)
		:param verbose:
			Bool, whether or not to print additional information
			(default: True)

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
				a_val, b_val, stda, stdb = self.calcGR_Weichert(Mmin=completeness.min_mag,
						Mmax=mean_Mmax, dM=dM, Mtype=Mtype, Mrelation=Mrelation,
						completeness=completeness, b_val=b_val, verbose=verbose)
			mfd = cc_catalog.get_incremental_mfd(Mmin=completeness.min_mag,
						Mmax=mean_Mmax, dM=dM, Mtype=Mtype, Mrelation=Mrelation,
						completeness=completeness, verbose=verbose)
		else:
			from hazard.rshalib.mfd import EvenlyDiscretizedMFD
			Mmax_obs = 0.
			n = 0.
			b_val = np.nan
			## Fake MFD
			mfd = EvenlyDiscretizedMFD(Mmin_n, dM, [1.0])

		prior, likelihood, posterior, params = mfd.get_Bayesian_Mmax_pdf(prior_model=prior_model,
				Mmax_obs=Mmax_obs, n=n, Mmin_n=Mmin_n, b_val=b_val, bin_width=dM,
				truncation=truncation, completeness=completeness,
				end_date=self.end_date, verbose=verbose)
		return (prior, likelihood, posterior, params)

	def plot_Bayesian_Mmax_pdf(self, prior_model="CEUS_COMP", Mmin_n=4.5,
						b_val=None, dM=0.1, truncation=(5.5, 8.25), Mtype='MW',
						Mrelation={}, completeness=None,
						num_discretizations=0, title=None, fig_filespec=None,
						verbose=True):
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
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes
			(default: None)
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
			Bool, whether or not to print additional information
			(default: True)
		"""
		prior, likelihood, posterior, params = self.get_Bayesian_Mmax_pdf(prior_model,
					Mmin_n, b_val=b_val, dM=dM, truncation=truncation, Mtype=Mtype,
					Mrelation=Mrelation, completeness=completeness, verbose=verbose)
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

	## Various plots
	def plot_mag_histogram(self,
							Mmin, Mmax, dM=0.5,
							Mtype='MW', Mrelation={}, completeness=None,
							color=None, label='',
							**kwargs):
		"""
		Plot magnitude histogram of earthquake catalog

		:param Mmin:
			float, minimum magnitude to bin
		:param Mmax:
			float, maximum magnitude to bin
		:param dM:
			float, magnitude binning interval
			(default: 0.5)
		:param Mtype:
			str, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} ordered dict, mapping magnitude type ("MW", "MS" or "ML")
			to name of magnitude conversion relation for :param:`Mtype`
			(default: {})
		:param completeness:
			instance of :class:`Completeness` containing initial years of
			completeness and corresponding minimum magnitudes
			(default: None)
		:param color:
			matplotlib color specification
			(default: None, use default color)
		:param label:
			str, label to use for histogram
			(default: '')

		See :func:`plotting.generic_mpl.plot_histogram` for additional
		keyword arguments
		"""
		from plotting.generic_mpl import plot_histogram

		bins_N, bins_mag = self.bin_by_mag(Mmin, Mmax, dM, completeness=completeness,
											Mtype=Mtype, Mrelation=Mrelation,
											include_right_edge=True, verbose=False)

		kwargs['colors'] = [color] if color else []
		kwargs['labels'] = [label] if label else []

		kwargs['xlabel'] = kwargs.get('xlabel', "Magnitude ($M_%s$)" % Mtype[1])
		kwargs['ylabel'] = kwargs.get('ylabel', "Number of events")
		kwargs['title'] = kwargs.get('title', "%s (%d events)" % (self.name, np.sum(bins_N)))

		return plot_histogram([bins_N], bins_mag, data_is_binned=True, **kwargs)

	def plot_CumulativeYearHistogram(self,
		start_year, end_year, dYear,
		Mmin, Mmax,
		Mtype="MW", Mrelation={},
		major_ticks=10, minor_ticks=1,
		completeness_year=None,
		regression_range=[],
		lang="en"):
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
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param major_tick_interval:
			Int, interval in years for major ticks
			(default: 10)
		:param minor_tick_interval:
			Int, interval in years for minor ticks
			(default: 1)
		:param completeness_year:
			Int, year of completeness where arrow should be plotted
			(default: None)
		:param regression_range:
			List, range of years where regression should be computed and plotted
			(default: [])
		:param lang:
			String, language of plot labels
			(default: "en")
		"""
		from matplotlib.patches import FancyArrowPatch
		catalog_start_year = timelib.to_year(self.start_date) // dYear * dYear
		if start_year <= catalog_start_year:
			start_year = catalog_start_year
		bins_N, bins_Years = self.bin_by_year(catalog_start_year, end_year, dYear,
									Mmin, Mmax, Mtype=Mtype, Mrelation=Mrelation)
		bins_N_cumul = np.cumsum(bins_N)
		start_year_index = np.where(bins_Years == start_year)[0][0]
		bins_N = bins_N[start_year_index:]
		bins_Years = bins_Years[start_year_index:]
		bins_N_cumul = bins_N_cumul[start_year_index:]

		pylab.plot(bins_Years, bins_N_cumul, "b", label="_nolegend_")
		pylab.plot(bins_Years, bins_N_cumul, "bo", label="Cumulated number of events")

		## Optionally, plot regression for a particular range
		ymin, ymax = pylab.ylim()
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
			arr = FancyArrowPatch((completeness_year, min(bins_N_cumul)),
							(completeness_year, bins_N_cumul[year_index]),
							arrowstyle='-|>', mutation_scale=30, facecolor='r',
							edgecolor='r', lw=2)
			pylab.plot(xfit, m*xfit+b, 'r--', lw=2, label="Regression")

		pylab.xlabel({"en": "Time (years)", "nl": "Tijd (jaar)"}[lang], fontsize='x-large')
		pylab.ylabel({"en": "Cumulative number of events since",
					"nl": "Gecumuleerd aantal aardbevingen sinds"}[lang]
					+ " %d" % timelib.to_year(self.start_date), fontsize='x-large')
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

	def plot_cumulated_moment(self, start_date=None, end_date=None,
						rel_time_unit=None, Mrelation={}, M0max=None,
						**kwargs):
		"""
		Plot cumulated seismic moment versus time.

		:param start_date:
			datetime spec, datetime where to start the plot
			(default: None)
		:param end_date:
			datetime spec, datetime where to end the plot
			(default: None)
		:param rel_time_unit:
			str, relative time unit ('Y', 'W', 'D', 'h', 'm' or 's')
			(default: None = plot absolute time)
		:param Mrelation:
			OrderedDict or str, magnitude relations to use for conversion
			to seismic moment
			(default: {})
		:param M0max:
			float, maximum moment value in Y axis
			(default: None)

		See :func:`plotting.generic_mpl.plot_xy` for additional
		keyword arguments
		"""
		kwargs.update(locals())
		kwargs.pop('self')
		kwargs.pop('kwargs')

		from .plot import plot_cumulated_moment
		return plot_cumulated_moment([self], ** kwargs)

	def plot_CumulatedM0(self,
		start_date=None, end_date=None, bin_width=10, bin_width_spec="years",
		binned=False, histogram=True,
		Mrelation={}, M0max=None,
		fig_filespec=None):
		"""
		Plot cumulated seismic moment versus time.

		:param start_date:
			Int or instance of :class:`datetime.date` or :class:`np.datetime64`,
			start date or start year
			(default: None)
		:param end_date:
			Int or instance of :class:`datetime.date` or :class:`np.datetime64`,
			end date or end year
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
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
			bins_M0, bins_Dates = self.bin_M0_by_year(timelib.to_year(start_date),
							timelib.to_year(end_date), bin_width, Mrelation=Mrelation)
			#bins_Dates = np.arange(start_date.year, end_date.year+bin_width, bin_width)
			subcatalog = self.subselect(start_date=timelib.to_year(start_date),
										end_date=timelib.to_year(end_date))
			Dates = subcatalog.get_fractional_years()
		elif bin_width_spec.lower()[:3] == "day":
			bins_M0, bins_Dates = self.bin_M0_by_day(timelib.as_np_date(start_date),
						timelib.as_np_date(end_date), bin_width, Mrelation=Mrelation)
			#bins_Dates = np.arange((end_date - start_date).days + 1)
			subcatalog = self.subselect(start_date=start_date, end_date=end_date)
			Dates = timelib.timespan(start_date, self.get_datetimes(), 'D')
		bins_M0_cumul = np.cumsum(bins_M0)
		unbinned_M0 = subcatalog.get_M0(Mrelation=Mrelation)
		M0_cumul = np.cumsum(unbinned_M0)

		## Construct arrays with duplicate points in order to plot horizontal
		## lines between subsequent points
		bins_M0_cumul2 = np.concatenate([[0.], np.repeat(bins_M0_cumul, 2)[:-1]])
		M0_cumul2 = np.concatenate([[0.], np.repeat(M0_cumul, 2)[:-1]])
		bins_Dates2 = np.repeat(bins_Dates, 2)
		Dates2 = np.repeat(Dates, 2)

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
		ymin, ymax = pylab.ylim()
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

	def plot_DateHistogram(self,
		start_date=None, end_date=None, ddate=1, ddate_spec="year",
		mag_limits=[2,3], Mtype="MW", Mrelation={}):
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		"""
		subcatalog = self.subselect(start_date=start_date, end_date=end_date)
		start_date, end_date = subcatalog.start_date, subcatalog.end_date

		if ddate_spec.lower()[:4] == "year":
			bins_Dates = np.arange(timelib.to_year(start_date), timelib.to_year(end_date)+ddate, ddate)
		elif ddate_spec.lower()[:3] == "day":
			#bins_Dates = np.arange((end_date - start_date).days + 1)
			bins_Dates = np.arange(0, int(timelib.timespan(start_date, end_date, 'D')) + 1)
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
				id = np.where(eq.year == bins_Dates)[0][0]
			elif ddate_spec.lower()[:3] == "day":
				id = int(timelib.timespan(start_date, eq.date, 'D'))
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
			ymin, ymax = pylab.ylim()
			pylab.axis((bins_Dates[0], bins_Dates[-1], ymin, ymax))

		pylab.xlabel("Time (%s)" % ddate_spec)
		pylab.show()

	def plot_depth_magnitude(self,
		start_date=None,
		Mtype="MW", Mrelation={},
		remove_undetermined=False,
		title=None,
		fig_filespec="", fig_width=0, dpi=300):
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param remove_zero_depths:
			Boolean, remove earthquakes for which depth equals zero if true
			(default: False)
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

		if remove_zero_depths:
			i = depths.nonzero()
			depths = depths[i]
			magnitudes = magnitudes[i]

		pylab.plot(magnitudes, depths, '.')
		pylab.xlabel("Magnitude (%s)" % Mtype)
		pylab.ylabel("Depth (km)")
		ax = pylab.gca()
		ax.invert_yaxis()
		pylab.grid(True)

		if title is None:
			title = ('Depth-Magnitude function %d-%d, %d events'
				% (timelib.to_year(subcatalog.start_date), timelib.to_year(subcatalog.end_date),
					len(magnitudes)))

		pylab.title(title)

		if fig_filespec:
			default_figsize = pylab.rcParams['figure.figsize']
			#default_dpi = pylab.rcParams['figure.dpi']
			if fig_width:
				fig_width /= 2.54
				dpi = dpi * (fig_width / default_figsize[0])
			pylab.savefig(fig_filespec, dpi=dpi)
			pylab.clf()
		else:
			pylab.show()

	def plot_time_magnitude(self, Mtype="MW", Mrelation={}, rel_time_unit=None,
						Mrange=(None, None), start_date=None, end_date=None,
						marker='o', marker_size=8, edge_color=None, fill_color=None,
						edge_width=0.5, label=None,
						completeness=None, completeness_color='r',
						lang='en', **kwargs):
		"""
		Plot time (X) versus magnitude (Y)

		:param Mtype:
			str, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} ordered dict, mapping magnitude type ("MW", "MS" or "ML")
			to name of magnitude conversion relation
			(default: {})
		:param rel_time_unit:
			str, relative time unit ('Y', 'W', 'D', 'h', 'm' or 's')
			(default: None = plot absolute time)
		:param Mrange:
			(min_mag, max_mag) tuple of floats, magnitude range in Y axis
			(default: (None, None))
		:param start_date:
			datetime spec, start date in X axis:
			- int (year)
			- str (timestamp)
			- datetime.datetime
			- np.datetime64
			(default: None = auto-determine from catalog)
		:param end_date:
			datetme spec, end date in X axis:
			see :param:`start_date` for options
			(default: None = auto-determine from catalog)
		:param marker:
			char, marker symbol
			(default: 'o')
		:param marker_size:
			marker size
			(default: 8)
		:param edge_width:
			float, marker edge width
			(default: 0.5)
		:param edge_color:
			marker edge color
			(default: None, will use default color for :param:`style_sheet`)
		:param fill_color:
			marker fill color
			(default: None, will not apply fill color)
		:param label:
			label for catalog
			(default: '', will not plot label
		:param completeness:
			instance of :class:`eqcatalog.Completeness`,
			catalog completeness to draw as a line over the catalog events
			(default: None)
		:param completeness_color:
			str, color to plot completeness line
			(default: 'r')
		:param lang:
			String, language of plot labels (default: "en")

		See :func:`plotting.generic_mpl.plot_xy` for additional
		keyword arguments
		"""
		kwargs.update(locals())
		kwargs.pop('self')
		kwargs.pop('kwargs')
		kwargs['markers'] = [marker] if marker is not None else None
		kwargs.pop('marker')
		kwargs['marker_sizes'] = [marker_size] if marker_size is not None else None
		kwargs.pop('marker_size')
		kwargs['colors'] = [edge_color] if edge_color is not None else None
		kwargs.pop('edge_color')
		kwargs['fill_colors'] = [fill_color] if fill_color is not None else None
		kwargs.pop('fill_color')
		kwargs['edge_widths'] = [edge_width] if edge_width is not None else None
		kwargs.pop('edge_width')
		kwargs['labels'] = [label] if label is not None else None
		kwargs.pop('label')
		kwargs['x_is_time'] = kwargs.get('x_is_time', True)

		from .plot import plot_time_magnitude
		return plot_time_magnitude([self], **kwargs)

	def plot_magnitude_time(self, Mtype="MW", Mrelation={}, rel_time_unit=None,
						Mrange=(None, None), start_date=None, end_date=None,
						marker='o', marker_size=8, edge_color=None, fill_color=None,
						label=None,
						completeness=None, completeness_color='r',
						lang='en', **kwargs):
		"""
		Plot magnitude versus time

		Identical to :meth:`plot_time_magnitude`, but with axes swapped
		"""
		kwargs.update(locals())
		kwargs.pop('self')
		kwargs.pop('kwargs')
		kwargs['x_is_time'] = kwargs.get('x_is_time', False)
		return self.plot_time_magnitude(**kwargs)

	def plot_HourHistogram(self,
		Mmin=None, Mmax=None,
		Mtype="MW", Mrelation={},
		start_year=None, end_year=None):
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param start_year:
			Int, lower year to bin (default: None)
		:param end_year:
			Int, upper year to bin (default: None)
		"""
		bins_N, bins_Hr = self.bin_by_hour(Mmin=Mmin, Mmax=Mmax, Mtype=Mtype,
					Mrelation=Mrelation, start_year=start_year, end_year=end_year)
		pylab.bar(bins_Hr, bins_N)
		ymin, ymax = pylab.ylim()
		pylab.axis((0, 24, ymin, ymax))
		pylab.xlabel("Hour of day", fontsize='x-large')
		pylab.ylabel("Number of events", fontsize='x-large')
		ax = pylab.gca()
		ax.invert_yaxis()
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('large')

		if not start_year:
			start_year = timelib.to_year(self.start_date)
		if not end_year:
			end_year = timelib.to_year(self.end_date)
		if Mmin is None:
			Mmin = self.get_Mmin()
		if Mmax is None:
			Mmax = self.get_Mmax()
		pylab.title("Hourly Histogram %d - %d, M %.1f - %.1f"
					% (start_year, end_year, Mmin, Mmax))
		pylab.show()

	def plot_depth_histogram(self,
		min_depth=0, max_depth=30, bin_width=2,
		depth_error=None,
		normalized=False,
		Mmin=None, Mmax=None, dM=None,
		Mtype="MW", Mrelation={},
		start_date=None, end_date=None,
		colors=[],
		title=None, legend_location=0,
		fig_filespec="", dpi=300,
		ax=None,
		**kwargs):
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param start_date:
			Int or instance of :class:`datetime.date` or :class:`np.datetime64`,
			lower year or date to bin
			(default: None)
		:param end_date:
			Int or instance of :class:`datetime.date` or :class:`np.datetime64`,
			upper year or date to bin
			(default: None)
		:param colors:
			list of matplotlib color specifications for histogram bars
			(one or more, depending on whether :param:`dM` is set)
			(default: [])
		:param title:
			String, title (None = default title, empty string = no title)
			(default: None)
		:param legend_location:
			Int, matplotlib legend location code
			(default: 0)
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)
		:param ax:
			matplotlib Axes instance
			(default: None)
		"""
		from .plot.plot_catalog import plot_depth_histogram

		catalog = self.subselect(start_date=start_date, end_date=end_date)

		if dM:
			catalogs, labels = [], []
			## Compute depth histogram for each magnitude bin
			assert not None in (Mmin, Mmax)
			_, bins_mag = self.bin_by_mag(Mmin, Mmax, dM=dM, Mtype=Mtype,
										Mrelation=Mrelation)

			for mmin in bins_mag[::-1]:
				mmax = mmin + dM
				subcatalog = self.subselect(Mmin=mmin, Mmax=mmax, Mtype=Mtype,
								Mrelation=Mrelation, include_right_edges=False)
				catalogs.append(subcatalog)
				labels.append('M=[%.1f-%.1f[' % (mmin, mmax))

		else:
			subcatalog = self.subselect(Mmin=Mmin, Mmax=Mmax, Mtype=Mtype,
							Mrelation=Mrelation, include_right_edges=True)
			catalogs = [subcatalog]
			labels = ["_nolegend_"]

		if title is None:
			if Mmin is None:
				Mmin = self.get_Mmin(Mtype=Mtype, Mrelation=Mrelation)
			if Mmax is None:
				Mmax = self.get_Mmax(Mtype=Mtype, Mrelation=Mrelation)
			title = "Depth histogram: M %.1f - %.1f" % (Mmin, Mmax)

		stacked = (dM is not None)
		return plot_depth_histogram(catalogs, labels=labels, stacked=stacked,
							min_depth=min_depth, max_depth=max_depth,
							bin_width=bin_width, depth_error=depth_error,
							normalized=normalized,
							title=title, legend_location=legend_location,
							fig_filespec=fig_filespec, dpi=dpi, ax=ax, **kwargs)

	def plot_Depth_M0_Histogram(self,
		min_depth=0, max_depth=30, bin_width=2,
		depth_error=None,
		Mmin=None, Mmax=None,
		Mrelation={},
		start_year=None, end_year=None,
		color='b',
		title=None,
		log=True,
		fig_filespec="", fig_width=0, dpi=300):
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
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
		bins_M0, bins_depth = self.bin_M0_by_depth(min_depth, max_depth, bin_width,
						depth_error, Mmin, Mmax, Mrelation, start_year, end_year)
		try:
			pylab.barh(bins_depth, bins_M0, height=bin_width, log=log, color=color)
		except:
			## This happens when all bins_M0 values are zero
			pass
		xmin, xmax = pylab.xlim()
		pylab.axis((xmin, xmax, min_depth, max_depth))
		pylab.ylabel("Depth (km)", fontsize='x-large')
		pylab.xlabel("Summed seismic moment (N.m)", fontsize='x-large')
		ax = pylab.gca()
		ax.invert_yaxis()
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('large')
		if title is None:
			if not start_year:
				start_year = timelib.to_year(self.start_date)
			if not end_year:
				end_year = timelib.to_year(self.end_date)
			if Mmin is None:
				Mmin = self.get_Mmin()
			if Mmax is None:
				Mmax = self.get_Mmax()
			title = ("Depth Histogram %d - %d, M %.1f - %.1f"
					% (start_year, end_year, Mmin, Mmax))
		pylab.title(title)

		if fig_filespec:
			default_figsize = pylab.rcParams['figure.figsize']
			#default_dpi = pylab.rcParams['figure.dpi']
			if fig_width:
				fig_width /= 2.54
				dpi = dpi * (fig_width / default_figsize[0])
			pylab.savefig(fig_filespec, dpi=dpi)
			pylab.clf()
		else:
			pylab.show()

	def plot_map(self,
		Mtype="MW", Mrelation={},
		label=None, catalog_style=None,
		marker='o', edge_color=None, fill_color=None, edge_width=0.5,
		marker_size=9, mag_size_inc=4,
		coastline_style={}, country_style={}, river_style=None, continent_style=None,
		source_model=None, sm_label_colname="ShortName",
		sm_style={"line_color": 'k', "line_pattern": '-', "line_width": 2},
		sites=[], site_style={"shape": 's', "fill_color": 'b', "size": 10}, site_legend="",
		circles=[], circle_styles=[],
		projection="merc", region=None, origin=(None, None),
		graticule_interval=(1., 1.), graticule_style={"annot_axes": "SE"},
		resolution="i",
		title=None, legend_style={}, border_style={},
		fig_filespec=None, fig_width=0, dpi=None):
		"""
		Plot map of catalog

		:param Mtype:
			String, magnitude type for magnitude scaling (default: "MW")
		:param Mrelation:
			{str: str} ordered dict, mapping magnitude type ("MW", "MS" or "ML")
			to name of magnitude conversion relation for :param:`Mtype`
			(default: {})
		:param label:
			String, legend label for earthquake epicenters
			(default: "Epicenters")
		:param catalog_style:
			instance of :class:`PointStyle` or dict with subset of
			PointStyle attributes as keys.
			(default: None)
		:param marker:
			matplotlib marker specification, earthquake marker symbol,
			overriding style given by :param:`catalog_style`
			(default: 'o')
		:param edge_color:
			matplotlib color specification, earthquake marker edge color,
			overriding style given by :param:`catalog_style`
			(default: None)
		:param edge_width:
			earthquake marker edge width,
			overriding style given by :param:`catalog_style`
			(default: 0.5)
		:param fill_color:
			matplotlib color specification, earthquake marker fill color,
			overriding style given by :param:`catalog_style`
			(default: None)
		:param marker_size:
			int or float, base (M=3) marker size in points,
			overriding style given by :param:`catalog_style`
			(default: 9)
		:param mag_size_inc:
			int or float, symbol size increment per magnitude relative to M=3
			(default: 4)

		See :func:`plot_catalog.plot_map` for remaining keyword arguments
		"""
		if title is None:
			title = self.name

		kwargs = locals().copy()

		kwargs.pop('self')
		kwargs.pop('kwargs', None)
		kwargs['catalog_styles'] = [catalog_style] if catalog_style is not None else None
		kwargs.pop('catalog_style')
		kwargs['symbols'] = [marker] if marker is not None else None
		kwargs.pop('marker')
		kwargs['edge_colors'] = [edge_color] if edge_color is not None else None
		kwargs.pop('edge_color')
		kwargs['fill_colors'] = [fill_color] if fill_color is not None else None
		kwargs.pop('fill_color')
		kwargs['edge_widths'] = [edge_width] if edge_width is not None else None
		kwargs.pop('edge_width')
		kwargs['symbol_sizes'] = [marker_size] if marker_size is not None else None
		kwargs.pop('marker_size')
		kwargs['labels'] = [label] if label is not None else None
		kwargs.pop('label')

		from .plot import plot_map
		return plot_map([self], **kwargs)

	def to_lbm_layer(self,
		Mtype="MW", Mrelation={},
		label=None, catalog_style={},
		marker='o', edge_color=None, fill_color=None, edge_width=0.5,
		marker_size=9, mag_size_inc=4):
		"""
		Generate layer to be plotted with layeredbasemap

		:param Mtype:
		:param Mrelation:
		:param label
		:param catalog_style:
		:param marker:
		:param edge_color:
		:param fill_color:
		:param edge_width:
		:param marker_size:
		:param mag_size_inc:
			see :meth:`plot_map`

		:return:
			instance of :class:`mapping.layeredbasemap.MapLayer`
		"""
		import mapping.layeredbasemap as lbm

		if isinstance(catalog_style, dict):
			catalog_style = lbm.PointStyle.from_dict(catalog_style)
		if marker:
			catalog_style.shape = marker
		if edge_color:
			catalog_style.line_color = edge_color
		if fill_color:
			catalog_style.fill_color = fill_color
		if edge_width is not None:
			catalog_style.line_width = edge_width
		if marker_size is not None:
			catalog_style.size = marker_size

		values = {}
		if mag_size_inc:
			## Magnitude-dependent size
			min_mag = np.floor(self.get_Mmin(Mtype, Mrelation))
			max_mag = np.ceil(self.get_Mmax(Mtype, Mrelation))
			mags = np.linspace(min_mag, max_mag, min(5, max_mag-min_mag+1))
			sizes = catalog_style.size + (mags - 3) * mag_size_inc
			sizes = sizes.clip(min=1)
			catalog_style.thematic_legend_style = lbm.LegendStyle(title="Magnitude",
												location=3, shadow=True, fancy_box=True,
												label_spacing=0.7)
			values['magnitude'] = self.get_magnitudes(Mtype, Mrelation)
			catalog_style.size = lbm.ThematicStyleGradient(mags, sizes, value_key="magnitude")

		# TODO: color by depth
		#values['depth'] = self.get_depths()
		#colorbar_style = lbm.ColorbarStyle(title="Depth (km)", location="bottom", format="%d")
		#style.fill_color = lbm.ThematicStyleRanges([0,1,10,25,50], ['red', 'orange', 'yellow', 'green'], value_key="depth", colorbar_style=colorbar_style)

		# TODO: color by age
		#values['year'] = self.get_fractional_years()
		#style.fill_color = lbm.ThematicStyleRanges([1350,1910,2050], ['green', (1,1,1,0)], value_key="year")

		point_data = lbm.MultiPointData(self.get_longitudes(), self.get_latitudes(),
										values=values)

		if label is None:
			label = self.name
		layer = lbm.MapLayer(point_data, catalog_style, legend_label=label)

		return layer

	def get_distances_to_line(self, start_pt, end_pt):
		"""
		Compute perpendicular distances to straight line defined by two
		points

		:param start_pt:
			(lon, lat) tuple of floats, start point of line
		:param end_pt:
			(lon, lat) tuple of floats, end point of line

		:return:
			1-D array, perpendicular distance of each eq to line
		"""
		from osgeo import osr
		import mapping.geotools.coordtrans as ct
		import mapping.layeredbasemap as lbm

		source_srs = ct.WGS84
		utm_spec = ct.get_utm_spec(*start_pt)
		target_srs = ct.get_utm_srs(utm_spec)
		ct = osr.CoordinateTransformation(source_srs, target_srs)

		line = lbm.LineData([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]])
		ogr_line = line.to_ogr_geom()
		ogr_line.AssignSpatialReference(source_srs)
		ogr_line.Transform(ct)

		points = lbm.MultiPointData(self.get_longitudes(), self.get_latitudes())
		ogr_points = points.to_ogr_geom()
		ogr_points.AssignSpatialReference(source_srs)
		ogr_points.Transform(ct)

		perpendicular_distances = []
		for i in range(ogr_points.GetGeometryCount()):
			ogr_pt = ogr_points.GetGeometryRef(i)
			d = ogr_pt.Distance(ogr_line)
			perpendicular_distances.append(d)
		perpendicular_distances = np.array(perpendicular_distances) / 1000.

		return perpendicular_distances

	def get_distances_along_line(self, start_pt, end_pt, max_pp_dist=None):
		"""
		Compute distances along straight line defined by two points

		:param start_pt:
			(lon, lat) tuple of floats, start point of line
		:param end_pt:
			(lon, lat) tuple of floats, end point of line
		:max_pp_dist:
			max. distance perpendicular to line to construct perpendicular
			line through start point, with respect to which distances
			are computed
			(default: None, will determine automatically)

		:return:
			1-D array, distance of each eq along line
		"""
		from osgeo import osr
		import mapping.geotools.geodetic as geodetic
		import mapping.geotools.coordtrans as ct
		import mapping.layeredbasemap as lbm

		source_srs = ct.WGS84
		utm_spec = ct.get_utm_spec(*start_pt)
		target_srs = ct.get_utm_srs(utm_spec)
		ct = osr.CoordinateTransformation(source_srs, target_srs)

		lon1, lat1 = start_pt[:2]
		lon2, lat2 = end_pt[:2]
		line_az = geodetic.spherical_azimuth(lon1, lat1, lon2, lat2)

		## Construct perpendicular line through start point
		## Distances will be computed with respect to this line
		pp_line_lons, pp_line_lats = [], []
		if max_pp_dist is None:
			max_pp_dist = self.get_distances_to_line(start_pt, end_pt).max()
		max_pp_dist *= 1000
		for daz in (-90, 90):
			lon, lat = geodetic.spherical_point_at(lon1, lat1, max_pp_dist, line_az + daz)
			pp_line_lons.append(lon)
			pp_line_lats.append(lat)
		pp_line = lbm.LineData(pp_line_lons, pp_line_lats)
		ogr_pp_line = pp_line.to_ogr_geom()
		ogr_pp_line.AssignSpatialReference(source_srs)
		ogr_pp_line.Transform(ct)

		points = lbm.MultiPointData(self.get_longitudes(), self.get_latitudes())
		ogr_points = points.to_ogr_geom()
		ogr_points.AssignSpatialReference(source_srs)
		ogr_points.Transform(ct)

		inline_distances = []
		for i in range(ogr_points.GetGeometryCount()):
			ogr_pt = ogr_points.GetGeometryRef(i)
			d = ogr_pt.Distance(ogr_pp_line)
			inline_distances.append(d)
		inline_distances = np.array(inline_distances) / 1000.

		return inline_distances

	def plot_cross_section(self, start_pt, end_pt, distance,
						marker='o', marker_size=lambda m: 9 + (m - 3) * 3,
						edge_color='b', fill_color='None',
						edge_width=0.5, label=None, Mtype="MW", Mrelation={},
						**kwargs):
		"""
		Plot cross-section along a straight line

		:param start_pt:
			(lon, lat) tuple of floats, start point of cross-section
		:param end_pt:
			(lon, lat) tuple of floats, end point of cross-section
		:param distance:
			float, maximum distance to line to be included in cross-section
		:param marker:
			char, matplotlib marker symbol
			(default: 'o')
		:param marker_size:
			float, array or callable, marker size(s)
			If callable, sizes will be computed in function of magnitude
			(default: lambda m: 9 + (m - 3) * 3)
		:param edge_color:
			matplotlib color spec, marker edge color
			(default: 'b')
		:param fill_color:
			matplotlib color spec, marker fill color
			(default: 'None')
		:param edge_width:
			float, marker edge width
			(default: 0.5)
		:param label:
			str, dataset label
			(default: None)
		:param Mtype:
			str, magnitude type to use for calculating marker size
			(default: 'MW')
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param **kwargs:
			additional keyword arguments to be passed to
			:func:`plotting.generic_mpl.plot_xy
		"""
		from plotting.generic_mpl import plot_xy

		subcat = self.subselect_line(start_pt, end_pt, distance)

		distances = subcat.get_distances_along_line(start_pt, end_pt, max_pp_dist=distance)
		depths = subcat.get_depths()

		xmin = kwargs.pop('xmin', 0)
		yscaling = kwargs.pop('yscaling', '-lin')
		ymin = kwargs.pop('ymin', 0)
		xlabel = kwargs.pop('xlabel', 'Distance (km)')
		ylabel = kwargs.pop('ylabel', 'Depth (km)')

		markers = [marker]
		edge_colors = [edge_color] if edge_color else []
		fill_colors = [fill_color] if fill_color else []
		edge_widths = [edge_width] if edge_width else []
		labels = [label] if label else []

		if callable(marker_size):
			mags = subcat.get_magnitudes(Mtype, Mrelation=Mrelation)
			marker_sizes = marker_size(mags)
			marker_sizes = marker_sizes.clip(min=1)
			marker_sizes[np.isnan(marker_sizes)] = 1
			marker_sizes = [marker_sizes]
		else:
			marker_sizes = [marker_size]

		return plot_xy([(distances, depths)], labels=labels,
						markers=markers, marker_sizes=marker_sizes,
						marker_fill_colors=fill_colors, marker_edge_colors=edge_colors,
						marker_edge_widths=edge_widths,
						linestyles=[''],
						xlabel=xlabel, ylabel=ylabel,
						xmin=xmin, ymin=ymin, yscaling=yscaling,
						**kwargs)

	## Export methods

	def to_multi_point_data(self, combine_datetime=True,
							columns=['ID', 'datetime', 'lon', 'lat', 'depth',
							'ML', 'MS', 'MW', 'intensity_max']):
		"""
		Convert to layeredbasemap MultiPointData

		:param combine_datetime:
			bool, whether or not to combine date and time in one attribute
			(default: True)
		:param columns:
			list of column names to export
			(default: ['ID', 'datetime', 'lon', 'lat', 'depth',
					'ML', 'MS', 'MW', 'intensity_max'])

		:return:
			instance of :class:`mapping.layeredbasemap.MultiPointData`
		"""
		import mapping.layeredbasemap as lbm

		lons = self.get_longitudes()
		lats = self.get_latitudes()
		z = [eq.depth for eq in self]

		values = OrderedDict()
		values['ID'] = self.get_ids()

		if combine_datetime and 'datetime' in columns:
			values['datetime'] = self.get_datetimes()
		else:
			if 'date' in columns or 'datetime' in columns:
				values['date'] = [timelib.to_py_date(eq.date) for eq in self]
			if 'time' in columns or 'datetime' in columns:
				## Convert times to strings to avoid problems with Shapefiles
				values['time'] = [str(eq.time) for eq in self]

		if 'lon' in columns:
			values['lon'] = self.get_longitudes()
		if 'lat' in columns:
			values['lat'] = self.get_latitudes()
		if 'depth' in columns:
			values['depth'] = self.get_depths()

		for Mtype in self.get_Mtypes():
			if Mtype in columns:
				#values[Mtype] = self.get_magnitudes(Mtype=Mtype, Mrelation={})
				values[Mtype] = [getattr(eq, Mtype) for eq in self]

		if 'name' in columns:
			names = [eq.name for eq in self]
			if sum([len(name) for name in names]):
				values['name'] = names

		if 'intensity_max'  in columns:
			intensities = self.get_max_intensities()
			if not ((intensities == 0).all() or np.isnan(intensities).all()):
				values['Imax'] = intensities

		for attrib in ['macro_radius', 'errh', 'errz', 'errt', 'errM']:
			if attrib in columns:
				ar = np.array([getattr(eq, attrib) for eq in self])
				if not ((ar == 0).all() or np.isnan(ar).all()):
					if attrib == 'macro_radius':
						## Shorten attibute name to <= 10 characters
						## to avoid problems with Shapefiles
						attrib = 'Rmacro'
					values[attrib] = ar

		for attrib in ['zone', 'agency', 'event_type']:
			if attrib in columns:
				lst = [getattr(eq, attrib) for eq in self]
				if len(set(lst)) > min(1, len(self)-1):
					values[attrib] = lst

		return lbm.MultiPointData(lons, lats, z=z, values=values)

	def to_geojson(self, combine_datetime=True):
		"""
		Convert to GeoJSON

		:param combine_datetime:
			see :meth:`to_multi_point_data`

		:return:
			dict
		"""
		return self.to_multi_point_data(combine_datetime).to_geojson(as_multi=False)

	def export_gis(self, format, filespec, encoding='latin-1',
					columns=['ID', 'datetime', 'lon', 'lat', 'depth',
					'ML', 'MS', 'MW', 'intensity_max'],
					combine_datetime=True, replace_null_values=None):
		"""
		Export to GIS file

		:param format:
			str, OGR format specification (e.g., 'ESRI Shapefile',
			'MapInfo File', 'GeoJSON', 'MEMORY', ...)
		:param out_filespec:
			str, full path to output file, will also be used as layer name
		:param encoding:
			str, encoding to use for non-ASCII characters
			(default: 'latin-1')
		:param columns:
			list of column names to export
			(default: ['ID', 'datetime', 'lon', 'lat', 'depth',
					'ML', 'MS', 'MW', 'intensity_max'])
		:param combine_datetime:
			bool, whether or not to combine date and time in one attribute
			(default: True)
		:param replace_null_values:
			None or str or scalar, value to replace NULL (None, NaN)
			values with
			(default: None, will not replace NULL values)

		:return:
			instance of :class:`ogr.DataSource` if :param:`format`
			== 'MEMORY', else None
		"""
		if format == 'ESRI Shapefile':
			## Shapefiles do not support datetime fields
			combine_datetime = False
		mpd = self.to_multi_point_data(combine_datetime, columns=columns)
		return mpd.export_gis(format, filespec, encoding=encoding,
							replace_null_values=replace_null_values)

	def export_ZMAP(self, filespec, Mtype="MW", Mrelation={}):
		"""
		Export earthquake list to ZMAP format (ETH Zürich).

		:param filespec:
			String, full path specification of output file
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		"""
		f = open(filespec, "w")
		for eq in self.eq_list:
			M = eq.get_or_convert_mag(Mtype, Mrelation)
			year, month, day, hour, minute, second = timelib.to_time_tuple(eq.datetime)
			f.write("%f  %f  %d  %d  %d  %.1f %.2f %d %d\n"
					% (eq.lon, eq.lat, year, month, day, M, eq.depth, hour, minute))
		f.close()

	def export_csv(self, csv_filespec=None,
					columns=['ID', 'date', 'time', 'lon', 'lat', 'depth',
					'ML', 'MS', 'MW', 'intensity_max'],
					Mtype=None, Mrelation={}):
		"""
		Export earthquake list to a csv file.

		:param csv_filespec:
			String, full path specification of output csv file
			(default: None, will write to standard output)
		:param columns:
			list of column names to export
			(default: ['ID', 'date', 'time', 'lon', 'lat', 'depth',
					'ML', 'MS', 'MW', 'intensity_max'])
		:param Mtype:
			Str, magnitude type, either 'ML', 'MS' or 'MW'.
			If None, magnitudes will not be converted
			(default: None)
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		"""
		if csv_filespec == None:
			f = sys.stdout
		else:
			f = open(csv_filespec, "w")

		column_format_dict = {
			'ID': '%s',
			'date': '%s',
			'time': '%s',
			'datetime': '%s',
			'name': '"%s"',
			'lon': '%.4f',
			'lat': '%.4f',
			'depth': '%.1f',
			'ML': '%.2f',
			'MS': '%.2f',
			'MW': '%.2f',
			'mb': '%.2f',
			'errt': '%.2f',
			'errh': '%.2f',
			'errz': '%.2f',
			'errM': '%.1f',
			'intensity_max': '%s',
			'macro_radius': '%s',
			'zone': '%s',
			'agency': '%s',
			'event_type': '%s'}

		#if Mtype:
		#	f.write('ID,Date,Time,Name,Lon,Lat,Depth,%s,Intensity_max,Macro_radius\n'
		#			% Mtype)
		#else:
		#	f.write('ID,Date,Time,Name,Lon,Lat,Depth,ML,MS,MW,Intensity_max,Macro_radius\n')

		if Mtype:
			Mrelation = Mrelation or self.default_Mrelations.get(Mtype, {})
			catalog = self.copy()
			catalog.convert_magnitudes(Mtype, Mrelation=Mrelation)
			eq_list = catalog.eq_list
		else:
			eq_list = self.eq_list

		f.write(','.join(columns) + '\n')

		for eq in eq_list:
			output_line = ', '.join([column_format_dict.get(col, '%s')
									for col in columns])
			## If 'col' is not an earthquake attribute, try getting it from magnitude dict
			values = [getattr(eq, col, eq.mag.get(col)) for col in columns]
			output_line %= tuple(values)
			f.write(output_line + '\n')
			"""
			date = eq.date
			time = eq.time.isoformat()
			if eq.name != None:
				if isinstance(eq.name, bytes):
					eq_name = eq.name.encode('ascii', 'ignore')
				else:
					eq_name = eq.name
			else:
				eq_name = ""
			if Mtype:
				f.write('%d,%s,%s,"%s",%.3f,%.3f,%.1f,%.2f,%s,%s\n'
					% (eq.ID, date, time, eq_name, eq.lon, eq.lat, eq.depth,
					eq.get_or_convert_mag(Mtype, Mrelation), eq.intensity_max,
					eq.macro_radius))
			else:
				f.write('%d,%s,%s,"%s",%.3f,%.3f,%.1f,%.2f,%.2f,%.2f,%s,%s\n'
					% (eq.ID, date, time, eq_name, eq.lon, eq.lat, eq.depth,
					eq.ML, eq.MS, eq.MW, eq.intensity_max, eq.macro_radius))
			"""
		if csv_filespec != None:
			f.close()

	def export_hypo71(self, h71_filespec=None, Mtype=None, Mrelation={}):
		"""
		Export earthquake catalog to text file in HYPO71-2000 format

		:param h71_filespec:
			str, full path to output file.
			If None, output is printed on screen
			(default: None)
		:param Mtype:
		:param Mrelation:
			see :meth:`get_magnitudes`
		"""
		if h71_filespec == None:
			f = sys.stdout
		else:
			f = open(h71_filespec, "w")

		Mrelation = Mrelation or self.default_Mrelations.get(Mtype, {})
		for eq in self.get_sorted():
			f.write('%s\n' % eq.to_hypo71(Mtype=Mtype, Mrelation=Mrelation))

		if h71_filespec != None:
			f.close()

	def export_kml(self,
		kml_filespec=None,
		folders='time',
		instrumental_start_year=1910,
		color_by_depth=False,
		columns=['ID', 'date', 'time', 'name', 'lon', 'lat', 'depth',
				'ML', 'MS', 'MW', 'intensity_max']):
		"""
		Export earthquake catalog to KML.

		:param kml_filespec:
			String, full path to output KML file. If None, kml is printed
			on screen (default: None)
		:param folders:
			str, how to organize earthquakes in folders:
			- "time": instrumental/decade and historical/century folders
			- "time+recent": + past 24 h / past 2 weekds / past 1 year folders
			- "event_type": folders by event type
			(default: "time")
		:param instrumental_start_year:
			Int, start year of instrumental period (only applies when time_folders
			is True) (default: 1910)
		:param color_by_depth:
			Bool, whether or not to color earthquakes by depth (default: False)
		:param columns:
			list of column names to export
			(default: ['ID', 'date', 'time', 'name', 'lon', 'lat', 'depth',
					'ML', 'MS', 'MW', 'intensity_max'])

		:return:
			str, KML code (if :param:`kml_filespec` is not set)
		"""
		import mapping.kml.mykml as mykml

		kmldoc = mykml.KML()
		start_date = timelib.to_py_date(self.start_date)
		year, month, day = start_date.year, start_date.month, start_date.day
		start_time = datetime.datetime(year, month, day)
		kmldoc.addTimeStamp(start_time)
		current_time = timelib.utcnow()
		current_year = timelib.to_year(current_time)
		eq_years = self.get_years()
		eq_centuries = sorted(set((eq_years // 100) * 100))

		event_types = [eq.event_type for eq in self]
		unique_event_types = set(event_types)

		topfolder = kmldoc.addFolder(self.name, visible=True, open=True)

		if 'time' in folders:
			if folders == 'time+recent':
				recent_folder = kmldoc.addFolder("Recent", visible=True, open=True)
				topfolder.appendChild(recent_folder)
				folder_24h = kmldoc.createFolder("Past 24 hours", visible=True, open=True)
				recent_folder.appendChild(folder_24h)
				folder_2w = kmldoc.createFolder("Past 2 weeks", visible=True, open=True)
				recent_folder.appendChild(folder_2w)
				folder_lastyear = kmldoc.createFolder("Past year", visible=True, open=True)
				recent_folder.appendChild(folder_lastyear)

			inst_folder = kmldoc.addFolder("Instrumental", visible=True, open=True)
			topfolder.appendChild(inst_folder)
			hist_folder = kmldoc.addFolder("Historical", visible=False, open=True)
			topfolder.appendChild(hist_folder)

			decade_folders = {}
			for decade in range(max(instrumental_start_year, eq_years.min()),
								current_year, 10)[::-1]:
				folder_name = "%d - %d" % (decade, min(current_year - 1, decade + 9))
				decade_folder = kmldoc.createFolder(folder_name, visible=True, open=False)
				inst_folder.appendChild(decade_folder)
				decade_folders[decade] = decade_folder

			century_folders = {}
			last_century = ((instrumental_start_year - 1) // 100) * 100
			for century in eq_centuries[::-1]:
				if century <= last_century:
					folder_name = ("%d - %d"
						% (century, min(instrumental_start_year, century + 99)))
					century_folder = kmldoc.createFolder(folder_name, visible=True,
														open=False)
					hist_folder.appendChild(century_folder)
					century_folders[century] = century_folder
		elif folders == 'event_type':
			from .earthquake_types import EARTHQUAKE_TYPES

			et_folders = OrderedDict()
			et_colors = {'ke': (255, 0, 0), 'se': (255, 128, 128),
						'ki': (255, 128, 0), 'si': (255, 178, 102),
						'ls': (255, 255, 255),
						'qb': (0, 255, 0), 'sqb': (128, 255, 128),
						'km': (0, 0, 255), 'sm': (0, 102, 204),
						'kr': (0, 255, 255), 'sr': (128, 255, 255),
						'kn': (128, 0, 255), 'sn': (204, 153, 255),
						'cb': (255, 0, 255), 'scb': (255, 128, 255),
						'kx': (255, 0, 127), 'sx': (255, 153, 204),
						'ex': (102, 0, 51), 'sb': (128, 128, 128),
						'uk': (255, 255, 255), None: (0, 0, 0)}
			for et in ['ke', 'se', 'ki', 'si', 'ls',
						'kr', 'sr', 'km', 'sm', 'qb', 'sqb',
						'kn', 'sn', 'cb', 'scb', 'kx', 'sx', 'ex', 'sb',
						'uk', None]:
				if et in unique_event_types:
					try:
						et_name = EARTHQUAKE_TYPES['EN'][et]
					except:
						et_name = 'Unclassified'
					et_name += ' (%d)' % event_types.count(et)
					et_folder = kmldoc.addFolder(et_name, visible=True, open=False)
					topfolder.appendChild(et_folder)
					et_folders[et] = et_folder

		Mtypes = ('MW', 'MS', 'ML')
		for eq in self:
			if eq.event_type == "ke":
				#if eq.year < instrumental_start_year:
				#	Mtype = "MS"
				#else:
				#	Mtype = "ML"
				try:
					idx = [np.isnan(getattr(eq, Mt)) for Mt in Mtypes].index(False)
				except:
					Mtype = 'M'
				else:
					Mtype = Mtypes[idx]
			else:
				Mtype = "ML"

			if 'time' in folders:
				if eq.year < instrumental_start_year:
					century = (eq.year // 100) * 100
					folder = century_folders[century]
					visible = True
					color = (0, 255, 0)
					#Mtype = "MS"
				else:
					visible = True
					#Mtype = "ML"

					decade = (eq.year // 10) * 10
					folder = decade_folders[decade]
					if eq.year >= 2000:
						color = (192, 0, 192)
					else:
						color = (0, 0, 255)

					if folders == 'time+recent':
						if current_time - eq.datetime <= np.timedelta64(1, 'D'):
							folder = folder_24h
							color = (255, 0, 0)
						elif current_time - eq.datetime <= np.timedelta64(14, 'D'):
							folder = folder_2w
							color = (255, 128, 0)
						elif current_time - eq.datetime <= np.timedelta64(365, 'D'):
							folder = folder_lastyear
							color = (255, 255, 0)

			elif folders == 'event_type':
				folder = et_folders.get(eq.event_type, et_folders.get(None))
				visible = True
				color = et_colors.get(eq.event_type, (0, 0, 0))
				#color = (0, 0, 0)
				#color = (255, 128, 0)

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

			t = timelib.to_py_datetime(eq.datetime)
			#url = 'http://seismologie.oma.be/active.php?LANG=EN&CNT=BE&LEVEL=211&id=%d' % eq.ID
			try:
				hash = eq.get_rob_hash()
			except:
				url = None
			else:
				url = 'http://seismologie.oma.be/en/seismology/earthquakes-in-belgium/%s' % hash
			url = '<a href="' + url + '">ROB web page</a>'

			## Attributes
			values = OrderedDict()
			if 'ID' in columns:
				values['ID'] = eq.ID
			if 'datetime' in columns:
				values['Date & Time'] = eq.datetime
			if 'date' in columns:
				values['Date'] = str(eq.date)
			if 'time' in columns:
				values['Time'] = ("%02d:%02d:%02d"
					% (t.hour, t.minute, int(round(t.second + t.microsecond/1e+6))))
			if 'name' in columns:
				values['Name'] = mykml.xmlstr(eq.name)
			if 'lon' in columns:
				values['Lon'] = eq.lon
			if 'lat' in columns:
				values['Lat'] = eq.lat
			if 'depth' in columns:
				values['Depth'] = eq.depth
			if 'ML' in columns:
				values['ML'] = eq.ML
			if 'MS' in columns:
				values['MS'] = eq.MS
			if 'MW' in columns:
				values['MW'] = eq.MW
			if 'intensity_max' in columns:
				values['Imax'] = eq.intensity_max
			if 'macro_radius' in columns:
				values['Rmacro'] = eq.macro_radius
			for attrib in ['errh', 'errz', 'errt', 'errM', 'zone', 'agency']:
				if attrib in columns:
					values[attrib.capitalize()] = getattr(eq, attrib)
			if 'event_type' in columns or len(unique_event_types) > 1:
				values['Event type'] = eq.event_type
			if eq.year >= instrumental_start_year:
				values[None] = url

			## Replace NaN values
			for key, val in values.items():
				try:
					if np.isnan(val):
						values[key] = '-'
				except:
					pass

			name = "%s=%.1f %s %s"
			name %= (Mtype, getattr(eq, Mtype, '?'), t.isoformat(), mykml.xmlstr(eq.name))
			labelstyle = kmldoc.createLabelStyle(scale=0)
			#iconstyle = kmldoc.createStandardIconStyle(palette="pal2", icon_nr=26, scale=0.75+(values[Mtype]-3.0)*0.15, rgb=color)
			icon_href = "http://kh.google.com:80/flatfile?lf-0-icons/shield3_nh.png"
			iconstyle = kmldoc.createIconStyle(href=icon_href,
								scale=0.75+(values[Mtype]-3.0)*0.15, rgb=color)
			style = kmldoc.createStyle(styles=[labelstyle, iconstyle])
			ts = kmldoc.createTimeSpan(begin=timelib.to_py_datetime(eq.datetime))
			kmldoc.addPointPlacemark(name, eq.lon, eq.lat, folder=folder,
				description=values, style=style, visible=visible, timestamp=ts)

		if kml_filespec:
			kmldoc.write(kml_filespec)
		else:
			return kmldoc.root.toxml()

	def export_VTK(self,
		vtk_filespec,
		proj="lambert1972",
		Mtype="MW", Mrelation={}):
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
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
		import cPickle
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
			db = simpledb.SQLiteDB(sqlite_filespec)
			if table_name in db.list_tables():
				db.drop_table(table_name)
			eq = self.eq_list[-1]
			if isinstance(eq.ID, int):
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
			if 'geom' in dic:
				del dic['geom']
			eq = LocalEarthquake.from_dict(dic)
			eq_list.append(eq)

		return cls(eq_list, name=table_name)

	def plot_poisson_test(self,
		Mmin, interval=100,
		Mtype='MW', Mrelation={},
		completeness=None,
		label=None, bar_color=None, line_color='r', title=None,
		fig_filespec=None, dpi=300, ax=None,
		verbose=True,
		**kwargs):
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
			Int, length of interval (number of days)
			(default: 100)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness` containing initial years of
			completeness and corresponding minimum magnitudes.
			If None, use start year of catalog
			(default: None)
		:param label:
			str, label to use for catalog
			(default: None = use catalog name)
		:param bar_color:
			matplotlib color spec for histogram
			(default: None)
		:param line_color:
			matplotlib color spec for theoretical Poisson curve
			(default: 'r')
		:param title:
			String, plot title. (None = default title, "" = no title)
			(default: None)
		:param fig_filespec:
			String, full path of image to be saved.
			If None (default), histogram is displayed on screen.
		:param dpi:
			int, resolution of output figure in DPI
			(default: 300)
		:param ax:
			matplotlib axes instance
			(default: None)
		:param verbose:
			Bool, whether or not to print additional information
		:param **kwargs:
			additional keyword arguments to be passed to
			:func:`plotting.generic_mpl.plot_ax_frame`

		:return:
			matplotlib axes instance
		"""
		from .plot import plot_poisson_test

		if label is None:
			label = self.name
		bar_colors = None if bar_color is None else [bar_color]
		line_colors = None if line_color is None else [line_color]
		return plot_poisson_test([self], Mmin, interval, Mtype=Mtype,
								Mrelation=Mrelation, completeness=completeness,
								labels=[label], bar_colors=bar_colors,
								line_colors=line_colors, title=title,
								fig_filespec=fig_filespec, dpi=dpi, ax=ax,
								verbose=verbose, **kwargs)

	def plot_3d(self, limits=None, Mtype=None, Mrelation={}):
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
		p = ax.scatter(self.get_longitudes(), self.get_latitudes(), self.get_depths()*-1,
						c=self.get_magnitudes(**kwargs), cmap=plt.cm.jet)
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

	def analyse_completeness_CUVI(self,
		magnitudes,
		start_year, dYear, year1=None, year2=None,
		reg_line=None,
		Mtype="MW", Mrelation={},
		title=None,
		fig_filespec="", dpi=300,
		**kwargs):
		"""
		Analyze catalog completeness with the CUVI method (Mulargia, 1987).

		:param magnitudes:
			List of floats, magnitudes to analyze completeness for.
		:param start_year:
			Int, start year of analysis.
		:param dYear:
			Int, bin interval in years
		:param year1:
			Int, year to plot as completeness year
			(default=None)
		:param year2:
			Int, year to plot as next completeness year
			(default=None)
		:param reg_line:
			Float, magnitude to plot regression line for
			(requires :param:`year1` to be set)
			(default=None)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
			(default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param title:
			str, title of plot
			(default: None, automatic title is used)
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width
			(default: 0)
		:param dpi:
			Int, image resolution in dots per inch
			(default: 300)
		:param kwargs:
			see :func:`plotting.generic_mpl.plot_xy`
		"""
		from plotting.generic_mpl import plot_xy

		#colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']
		max_mag = self.get_Mmax(Mtype=Mtype, Mrelation=Mrelation)
		start_year_index = None
		cat_start_year = timelib.to_year(self.start_date)
		cat_end_year = timelib.to_year(self.end_date)

		datasets = []
		labels = []
		regression_line = None
		for i, magnitude in enumerate(magnitudes):
			bins_N, bins_Years = self.bin_by_year(cat_start_year, cat_end_year+1, dYear,
							magnitude, max_mag, Mtype=Mtype, Mrelation=Mrelation)
			bins_N_cumul = np.cumsum(bins_N)
			if not start_year_index:
				start_year_index = np.abs(bins_Years - start_year).argmin()
			bins_Years = bins_Years[start_year_index:]
			bins_N_cumul = bins_N_cumul[start_year_index:]
			#plt.plot(bins_Years, bins_N_cumul, colors[i%len(colors)], label= '%.1f' % magnitude)
			datasets.append((bins_Years, bins_N_cumul))
			labels.append('M>=%.1f' % magnitude)
			#plt.plot(bins_Years, bins_N_cumul, '%so' % colors[i%len(colors)], label='_nolegend_')
			#datasets.append((bins_Years, bins_N_cumul))
			#labels.append('_nolegend_')

			if reg_line and np.allclose(magnitude, reg_line) and year1 != None:
				index = np.abs(bins_Years - year1).argmin()
				bins_Years = bins_Years[index:]
				bins_N_cumul = bins_N_cumul[index:]
				x = np.array([bins_Years, np.ones_like(bins_Years)])
				m, c = np.linalg.lstsq(x.T, bins_N_cumul)[0]
				regression_line = (bins_Years, m*bins_Years+c)
				#plt.plot(bins_Years, m*bins_Years+c, color='k', linestyle='--', linewidth=5)

		#markers = ['', 'o'] * len(datasets) // 2
		markers = ['o']

		ax = plot_xy(datasets, labels=labels, markers=markers,
					fig_filespec='wait', dpi=dpi, ax=kwargs.pop('ax', None),
					**kwargs)

		## Optional completeness years and regression line

		if year1:
			kwargs['vlines'] = [[year1], None, None]
			kwargs['vline_args'] = {'colors': 'r', 'linestyles': ['-'],
									'linewidth': 2.5}
			if year2:
				kwargs['vlines'][0].append(year2)
				kwargs['vline_args']['linestyles'].append('--')

		if regression_line:
			datasets = [regression_line]
		else:
			datasets = []

		kwargs['xmin'] = kwargs.get('xmin', start_year)
		kwargs['xmax'] = kwargs.get('xmax', cat_end_year)
		kwargs['xgrid'] = kwargs.get('xgrid', 1)
		kwargs['ygrid'] = kwargs.get('ygrid', 1)
		kwargs['xtick_interval'] = kwargs.get('xtick_interval', (None, dYear))
		default_xlabel = 'Time (years)'
		kwargs['xlabel'] = kwargs.get('xlabel', default_xlabel)
		default_ylabel = 'Cumulative number of events since %d' % cat_start_year
		kwargs['ylabel'] = kwargs.get('ylabel', default_ylabel)
		default_title = 'CUVI completeness analysis for magnitudes %.1f - %.1f'
		default_title %= (magnitudes[0], magnitudes[-1])
		kwargs['title'] = title if title is not None else default_title
		kwargs['legend_location'] = kwargs.get('legend_location', 0)

		return plot_xy(datasets, labels=['_nolegend_'], colors=['k'],
				linestyles=['--'], linewidths=[3],
				fig_filespec=fig_filespec, dpi=dpi, ax=ax, **kwargs)


		"""
		minorLocator = MultipleLocator(dYear)
		plt.gca().xaxis.set_minor_locator(minorLocator)
		ymin, ymax = plt.ylim()
		if year1:
			plt.vlines(year1, ymin, ymax, colors='r', linestyles='-', linewidth=5)
		if year2:
			plt.vlines(year2, ymin, ymax, colors='r', linestyles='--', linewidth=5)
		plt.axis((start_year, cat_end_year, ymin, ymax))
		plt.xlabel('Time (years)', fontsize='large')
		plt.ylabel('Cumulative number of events since' + ' %d' % cat_start_year,
					fontsize='large')
		title = title or ('CUVI completeness analysis for magnitudes %.1f - %.1f'
						% (magnitudes[0], magnitudes[-1]))
		plt.title(title, fontsize='x-large')
		plt.legend(loc=0)
		plt.grid()
		if fig_filespec:
			default_figsize = pylab.rcParams['figure.figsize']
			#default_dpi = pylab.rcParams['figure.dpi']
			if fig_width:
				fig_width /= 2.54
				dpi = dpi * (fig_width / default_figsize[0])
			pylab.savefig(fig_filespec, dpi=dpi)
			pylab.clf()
		else:
			pylab.show()
		"""

	def plot_completeness_evaluation(self, completeness, num_cols=2,
									Mrelation={},
									fig_filespec=None, dpi=None,
									multi_params={}, ax_params={}):
		"""
		Evaluate particular completeness

		:param completeness:
			instance of :class:`Completeness`
		:param num_cols:
			int, number of columns
			(default: 2)
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param fig_filespec:
			str or None, full path to output file
			(default: None, will plot on screen)
		:param dpi:
			int, resolution in dots per inch
			(default: None)

		"""
		from string import ascii_lowercase
		from plotting.generic_mpl import (create_multi_plot, show_or_save_plot)

		num_plots = len(completeness)
		num_rows = int(np.ceil(num_plots / float(num_cols)))

		if fig_filespec is None:
			dpi = 90

		labels = ascii_lowercase[:num_plots]
		fig = create_multi_plot(num_cols=num_cols, num_rows=num_rows,
								sharex=False, sharey=False,
								share_xlabel=True, share_ylabel=True,
								xlabel='Time (years)',
								ylabel='Cumulative number of earthquakes',
								labels=labels, label_location=4, ax_size=(7.5, 5.),
								dpi=dpi, **multi_params)

		for i in range(num_plots):
			ax = fig.axes[i]
			year1 = completeness.min_years[i]
			if i < (len(completeness) - 1):
				year2 = completeness.min_years[i+1]
			else:
				year2 = self.end_year
			if i > 0:
				start_year = completeness.min_years[i-1]
			else:
				start_year = self.start_year

			if year1 < 1700:
				dYear = 20
			elif year1 < 1800:
				dYear = 10
			elif year1 < 1900:
				dYear = 5
			elif year1 < 1985:
				dYear = 2
			else:
				dYear = 1

			dM = 0.1
			mag = completeness.min_mags[i]
			magnitudes = mag + np.arange(-2, 3) * dM

			self.analyse_completeness_CUVI(magnitudes, start_year, dYear,
											year1, year2, reg_line=mag,
											Mtype=completeness.Mtype,
											Mrelation=Mrelation, ax=ax,
											xlabel='', ylabel='',
											title='', legend_location=2,
											fig_filespec='wait', **ax_params)

		## Hide unused axes
		if num_plots < num_cols * num_rows:
			for i in range(num_plots, num_cols * num_rows):
				fig.axes[i].axis('off')

		return show_or_save_plot(fig, fig_filespec, dpi=dpi)

	## HMTK wrappers

	def analyse_completeness_Stepp(self, dM=0.1, Mtype="MW", Mrelation={}, dt=5.0, increment_lock=True):
		"""
		Analyze catalog completeness with the Stepp method (1971). The GEM algorithm
		from the OQ hazard modeller's toolkit is used.

		:param dM:
			Float, magnitude bin width (default: 0.1)
		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
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

	def decluster_new(self, method="gardner-knopoff", window_opt="GardnerKnopoff", fs_time_prop=0., time_window=60., Mtype="MW", Mrelation={}):
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
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
				year, month, day, hour, minute, second = timelib.to_time_tuple(eq.datetime)
				data_int.append([
					int(year),
					int(month),
					int(day),
					int(hour),
					int(minute),
				])
				data_flt.append([
					float(second),
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

	def analyse_Mmax(self, method='Cumulative_Moment', num_bootstraps=100, iteration_tolerance=None, maximum_iterations=100, num_samples=20, Mtype="MW", Mrelation={}):
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})

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

	def analyse_recurrence(self, dM=0.1, method="MLE", aM=0., dt=1., Mtype="MW", Mrelation={}, completeness=None):
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
			to magnitude type ("MW", "MS" or "ML")
			(default: {})
		:param completeness:
			instance of :class:`Completeness` (default: None)

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

	def get_hmtk_catalogue(self, Mtype='MW', Mrelation={}):
		"""
		Convert ROB catalog to hmtk catalogue

		:param Mtype:
			String, magnitude type: "ML", "MS" or "MW" (default: "MW")
		:param Mrelation:
			{str: str} dict, mapping name of magnitude conversion relation
			to magnitude type ("MW", "MS" or "ML")
			(default: {})

		:return:
			instance of :class:`hmtk.seismicity.catalogue.Catalogue`
		"""
		from hmtk.seismicity.catalogue import Catalogue
		catalogue = Catalogue()
		keys_flt = ['second', 'longitude', 'latitude', 'depth', 'magnitude']
		keys_int = ['year', 'month', 'day', 'hour', 'minute']
		data_int, data_flt = [], []
		for eq in self:
			year, month, day, hour, minute, second = timelib.to_time_tuple(eq.datetime)
			data_flt.append([
				float(second),
				float(eq.lon),
				float(eq.lat),
				float(eq.depth),
				float(eq.get_or_convert_mag(Mtype=Mtype, Mrelation=Mrelation)),
			])
			data_int.append([
				int(year),
				int(month),
				int(day),
				int(hour),
				int(minute),
			])
		catalogue.load_from_array(keys_flt, np.array(data_flt, dtype=np.float))
		catalogue.load_from_array(keys_int, np.array(data_int, dtype=np.int))
		return catalogue

	def get_hmtk_smoothed_source_model(self, spcx=0.1, spcy=0.1, Mtype='MW', Mrelation={}, completeness=None):
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
		return data


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

	if len(catalog_list):
		catalog0 = catalog_list[0]
		eq_list = catalog0.eq_list[:]
		start_date = catalog0.start_date
		end_date = catalog0.end_date
		default_Mrelations = catalog0.default_Mrelations
		default_completeness = catalog0.default_completeness
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

		return EQCatalog(eq_list, start_date=start_date, end_date=end_date,
						region=region, name=name,
						default_Mrelations=default_Mrelations,
						default_completeness=default_completeness)

	else:
		return EQCatalog([], name=name)


# TODO: this function is now obsolete
def plot_catalogs_magnitude_time(catalogs, symbols=['o'], edge_colors=[], fill_colors=['None'],
								edge_widths=[1], labels=[], symbol_size=50,
								Mtype="MW", Mrelation={}, start_year=None,
								Mrange=(None, None), completeness=None,
								completeness_color="r", vlines=False, grid=True,
								plot_date=False, major_tick_interval=None,
								minor_tick_interval=1, tick_unit=None,
								tick_freq=None, tick_by=None, tick_form=None,
								title=None, lang="en", legend_location=0,
								fig_filespec=None, fig_width=0, dpi=300, ax=None):
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
		to magnitude type ("MW", "MS" or "ML")
		(default: {})
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
			x = [timelib.to_py_datetime(dt) for dt in catalog.get_datetimes()]
		else:
			x = catalog.get_fractional_years()
		ax.scatter(x, y, s=symbol_size, edgecolors=edge_color, label=label, marker=symbol, facecolors=fill_color, linewidth=edge_width)

	## crop X axis to data when using fractional years
	xmin, xmax, ymin, ymax = ax.axis()
	if not plot_date:
		if start_year:
			xmin = start_year
		else:
			xmin = min(timelib.to_year(catalog.start_date) for catalog in catalogs)
		xmax = max(timelib.to_year(catalog.end_date) for catalog in catalogs)+1

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
				maj_fmt = mdates.DateFormatter(tick_form)
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
		xlabel = {"en": "Time (years)",
				"nl": "Tijd (jaar)",
				"fr": u"Temps (années)",
				"nlfr": u"Tijd (jaar) / Temps (années)"}[lang]
	else:
		xlabel = {"en": "Date", "nl": "Datum", "fr": "Date", "nlfr": "Datum / Date"}[lang]
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
		clabel = {"en": "Completeness magnitude",
				"nl": "Compleetheidsmagnitude",
				"fr": u"Magnitude de complétude",
				"nlfr": u"Compleetheid / Complétude"}[lang]
		x, y = completeness.min_years, completeness.min_mags
		x = np.append(x, max([catalog.end_date for catalog in catalogs]).year+1)
		if plot_date:
			x = [datetime.datetime(year, 1, 1) for year in x]
		xmin, xmax, ymin, ymax = ax.axis()
		ax.hlines(y, xmin=x[:-1], xmax=x[1:], colors=completeness_color, label=clabel)
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
		#default_dpi = plt.rcParams['figure.dpi']
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
	from .rob import GIS_ROOT

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

	tab_filespec = os.path.join(GIS_ROOT, "KSB-ORB", os.path.splitext(tabname)[0] + ".TAB")
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
		name = "%s %s - %s" % (zone_name, start_date, end_date)
		if verbose:
			print(query)
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
	# TODO: this import doesn't always work!
	from thirdparty.recipes.dummyclass import DummyClass

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
	from hazard.rshalib.mfd.truncated_gr import alphabetalambda
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

			print("%s" % rec.Name)
			print("1,0,1,1")
			print(" %d" % len(coords))
			for pt in coords:
				print("%f,%f,%.1f" % (pt.x, pt.y, depth))
			print("%.3f,%.3f,0,%.1f,0,%.1f,%.1f"
				% (lambda0, beta, rec.MS_max_evaluated, rec.MS_max_observed, Mc))
			print(" 0")

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
		print("%s - a: %.3f  ->  %.3f" % (zone.name, zone.a, zone.a_new))

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
	completeness = None
	Mtype = "MW"
	Mrange = (1.5, 7.0)
	Freq_range = (1E-4, 10**1.25)
	Mc = 3.5

	## Read catalog from intranet database
	#region = (4.5, 4.65, 50.60, 50.70)
	#start_date = datetime.date(2008,7,12)
	catalog = read_catalogSQL(region=region, start_date=start_date, end_date=end_date)
	#catalog.plot_MagnitudeDate()

	## Bin magnitudes with verbose output
	#bins_N, bins_Mag, bins_Years, num_events, Mmax = catalog.bin_by_mag(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)

	## Plot magnitude histogram
	#catalog.plot_Mhistogram(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness)

	## Plot Magnitude/Frequency diagram
	#catalog.mag_freqs(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)
	#fig_filespec = os.path.join(r"C:\PSHA\MagFreq", "MagFreq " + catalog.name + ".PNG")
	fig_filespec = None
	#catalog.plot_mag_freqs(Mmin, Mmax, dM, discrete=True, Mtype=Mtype, completeness=completeness, verbose=True, Mrange=Mrange, Freq_range=Freq_range, want_exponential=True, fig_filespec=fig_filespec)

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
	#print(completeness)

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
		print(Mmin, "%.1f" % mean_night, "%.1f" % (mean_night*24,))

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
	catalog.plot_mag_freqs(Mmin, Mmax, dM, discrete=False, Mtype=Mtype, completeness=completeness, verbose=True)
	"""

	## Read catalog from pickled file and plot Magnitude/Frequency diagram
	#filespec = r"Test\ROBcatalog.p"
	#f = open(filespec, "r")
	#catalog = cPickle.load(f)
	#f.close()
	#catalog.plot_mag_freqs(Mmin, Mmax, dM, discrete=True, Mtype=Mtype, completeness=completeness, verbose=True, Mrange=Mrange, Freq_range=Freq_range)

	## Report total seismic moment
	#historical_catalog = catalog.subselect(end_date=datetime.date(1909, 12, 31))
	#instrumental_catalog = catalog.subselect(start_date = datetime.date(1910, 1, 1))
	#M0_historical = historical_catalog.get_M0_rate(Mrelation="geller")
	#M0_instrumental = instrumental_catalog.get_M0_rate(Mrelation="hinzen")
	#M0_total = M0_historical + M0_instrumental
	#print("Total seismic moment: %.2E (historical) + %.2E (instrumental) = %.2E N.m" % (M0_historical, M0_instrumental, M0_total))
	#catalog.plot_CumulatedM0(ddate=10, ddate_spec="years", Mrelation=None)

	## Read full catalog from database, calculate a and b values, and store these for later use
	"""
	catalog = read_catalogSQL(region=region, start_date=start_date, end_date=end_date)
	cat = DummyClass()
	cat.a, cat.b, cat.beta, cat.stda, cat.stdb, cat.stdbeta = catalog.calcGR_mle(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)
	bins_N, bins_Mag, bins_Years, num_events, Mmax = catalog.bin_by_mag(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=False)
	cat.num_events = num_events
	"""

	## Read catalog from MapInfo, split according to source zone model, and generate magnitude/frequency plots
	"""
	zone_model = "Leynaud"
	zone_names = []
	catalogs = read_catalogMI(region=region, start_date=start_date, end_date=end_date, zone_model=zone_model, zone_names=zone_names, verbose=True)
	for catalog in catalogs.values():
		print(catalog.name)
		print("  Mmax: %.1f" % catalog.Mminmax()[1])
		if len(catalog) > 0:
			## Plot Mag/Freq for each zone and save figure to file
			dirname = os.path.join(r"C:\PSHA\MagFreq", zone_model)
			fig_filespec = os.path.join(dirname, "MagFreq " + catalog.name + ".PNG")
			try:
				catalog.plot_mag_freqs(Mmin, Mmax, dM, discrete=False, Mtype=Mtype, completeness=completeness, verbose=True, fig_filespec=fig_filespec, Mrange=Mrange, Freq_range=Freq_range, fixed_beta=cat.beta)
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
		print(catalog.name)
		print("  Mmax: %.1f" % catalog.Mminmax()[1])
		if len(catalog) > 0:

			## Just print a and b values for each zone
			BETA = cat.beta
			zone = DummyClass()
			zone.name = catalog.name
			try:
				a, b, r = catalog.calcGR_lsq(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)
				a, b, beta, stda, stdb, stdbeta = catalog.calcGR_mle(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True)
				zone.a, zone.b, zone.beta, zone.stda, zone.stdb, zone.stdbeta = catalog.calcGR_mle(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, verbose=True, beta=BETA)
				zone.num_events = catalog.bin_by_mag(Mmin, Mmax, dM, Mtype=Mtype, completeness=completeness, trim=True, verbose=False)[3]
			except:
				pass
			else:
				zones.append(zone)
			print

	#M=completeness.get_completeness_magnitude(datetime.date.today())
	distribute_avalues(zones, cat, Mc)

	for zone in zones:
		print(zone.name)
		if "Single Large Zone" in zone.name:
			for a in split_avalues(zone.a_new, [0.793, 0.207]):
				alpha, beta, lambda0 = alphabetalambda(a, zone.b, Mc)
				print("%.3f  %.3f" % (lambda0, beta))
		else:
			alpha, beta, lambda0 = alphabetalambda(zone.a_new, zone.b, Mc)
			print("%.3f  %.3f" % (lambda0, beta))
	"""


	## Format zone model info for use in CRISIS
	"""
	zone_model = "Leynaud_updated"
	format_zones_CRISIS(zone_model, fixed_depth=3.5, smooth=False)
	"""


	## Calculate alpha, beta, lambda for different Mc
	"""
	for a in [2.4, 1.5, 1.7, 2.0, 2.3, 1.7, 1.4, 1.4, 1.5]:
		print("%.3f  %.3f" % tuple(alphabetalambda(a, 0.87, 2.0)[1:]))

	print(alphabetalambda(1.4, 0.87, 3.0))
	"""
