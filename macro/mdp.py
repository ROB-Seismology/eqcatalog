"""
Macroseismic data point(s)
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import sys
if sys.version[0] == '3':
	basestring = str

import numpy as np

from .macro_info import MacroseismicInfo, MacroInfoCollection



__all__ = ['MacroseismicDataPoint', 'MDP', 'MDPCollection']


class MacroseismicDataPoint():
	"""
	Macroseismic data point

	:param id:
		int or str, ID of data point in database
	:param id_earth:
		int or str, ID of earthquake in database
	:param Imin:
		int, minimum intensity assigned to this point
	:param Imax:
		int, maximum intensity assigned to this point
	:param imt:
		str, intensity measure type, e.g. 'EMS98', 'MMI', 'CII'
	:param lon:
		float, longitude (in degrees)
	:param lat:
		float, latitude (in degrees)
	:param data_type:
		str, type of macroseismic data, one of:
		- 'internet', 'online' or 'dyfi'
		- 'traditional'
		- 'historical'
		- 'official'
		- 'isoseismal'
	:param id_com:
		int or str, ID of locality in database
	:param id_main:
		int or str, ID of main commune in database
	:param fiability:
		float, reliability (in percent)
	:**kwargs:
		addition keyword arguments
	"""
	def __init__(self, id, id_earth, Imin, Imax, imt, lon, lat,
				data_type, id_com, id_main, fiability, **kwargs):
		self.id = id
		self.id_earth = id_earth
		self.Imin = Imin
		self.Imax = Imax
		self.lon = lon if lon is not None else np.nan
		self.lat = lat if lat is not None else np.nan
		self.data_type = data_type
		self.imt = imt
		self.id_com = id_com
		self.id_main = id_main
		self.fiability = fiability
		for kw, val in kwargs.items():
			setattr(self, kw, val)

	def __repr__(self):
		txt = '<MDP eq#%s:%s | I=%d-%d %s | lon=%.3f, lat=%.3f | %s>'
		txt %= (self.id_earth, self.id, self.Imin, self.Imax, self.imt,
				self.lon, self.lat, self.data_type)
		return txt

	@property
	def Imean(self):
		"""
		:return:
			float, mean intensity
		"""
		return np.nanmean([self.Imin, self.Imax])

	def get_eq(self):
		"""
		Get earthquake from ROB catalog

		:return:
			instance of :class:`LocalEarthquake` or None
		"""
		from ..rob import query_local_eq_catalog_by_id

		try:
			[eq] = query_local_eq_catalog_by_id(self.id_earth)
		except:
			eq = None

		return eq


MDP = MacroseismicDataPoint


class MDPCollection():
	"""
	A collection of macroseismic data points

	Contains various methods to filter, split or aggregate MDPs

	:param mdp_list:
		list with instances of :class:`MacroseismicDataPoint`
	"""
	def __init__(self, mdp_list):
		self.mdp_list = mdp_list

	def __repr__(self):
		txt = '<MDPCollection (n=%d)>' % len(self)
		return txt

	def __iter__(self):
		return self.mdp_list.__iter__()

	def __len__(self):
		return len(self.mdp_list)

	def __getitem__(self, item):
		return self.mdp_list.__getitem__(item)

	def append(self, mdp):
		self.mdp_list.append(mdp)

	def extend(self, mdp_list):
		self.mdp_list.extend(mdp_list)

	@property
	def Imin(self):
		return np.array([mdp.Imin for mdp in self])

	@property
	def Imax(self):
		return np.array([mdp.Imax for mdp in self])

	@property
	def Imean(self):
		return np.array([mdp.Imean for mdp in self])

	def get_longitudes(self):
		return np.array([mdp.lon for mdp in self])

	def get_latitudes(self):
		return np.array([mdp.lat for mdp in self])

	def get_aggregated_intensity(self, Imin_or_max, agg_function):
		"""
		Compute aggregated intensity of MDP collection

		:param Imin_or_max:
			str, either 'min', 'mean' or 'max', to select between minimum,
			mean or maximum intensitiy of each MDP
		:param agg_function:
			str, aggregation function, one of 'min', 'max', 'mean', 'median'

		:return:
			float, aggregated intensity
		"""
		if Imin_or_max == 'min':
			intensities = self.Imin
		elif Imin_or_max == 'max':
			intensities = self.Imax
		elif Imin_or_max == 'mean':
			intensities = self.Imean

		agg_func = getattr(np, 'nan'+agg_function)

		return agg_func(intensities)

	def calc_distances(self, lon, lat, depth=0):
		"""
		Compute distance with respect to a point, optionally at some depth

		:param lon:
			float, longitude of reference point (in degrees)
		:param lat:
			float, latitude of reference point (in degrees)
		:param depth:
			float, depth of reference point (in km)
			(default: 0)

		return:
			1-D float array, distances (in km)
		"""
		from mapping.geotools.geodetic import spherical_distance

		d_epi = spherical_distance(lon, lat, self.get_longitudes(),
									self.get_latitudes())
		d_epi /= 1000

		if depth:
			d_hypo = np.sqrt(d_epi**2 + depth**2)
		else:
			d_hypo = d_epi

		return d_hypo

	def subselect_by_property(self, prop_name, prop_val):
		"""
		Select MDPs based on the value of a particular property

		:param prop_name:
			str, property name
		:param prop_val:
			property value

		:return:
			instance of :class:`MDPCollection`
		"""
		mdp_list = []
		for mdp in self:
			if getattr(mdp, prop_name) == prop_val:
				mdp_list.append(mdp)

		return self.__class__(mdp_list)

	def split_by_property(self, prop_name):
		"""
		Split MDP collection based on values of a particular property

		:param prop_name:
			str, property name

		:return:
			dict, mapping property values to instances of
			:class:`MDPCollection`
		"""
		mdpc_dict = {}
		for mdp in self:
			prop_val = getattr(mdp, prop_name)
			if not prop_val in mdpc_dict:
				mdpc_dict[prop_val] = self.__class__([mdp])
			else:
				mdpc_dict[prop_val].append(mdp)

		return mdpc_dict

	def aggregate_by_property(self, prop_name, Imin_or_max, agg_function,
							min_fiability=80):
		"""
		Aggregate MDPs based on values of a particular property

		:param prop_name:
			str, property name
		:param Imin_or_max:
		:param agg_function:
			see :meth:`get_aggregated_intensity`
		"""
		# TODO: min_fiability
		#mdpc = self.subselect_by_property('fiability')
		mdpc_dict = self.split_by_property(prop_name)
		macro_info_list = []
		for agg_key, mdpc in mdpc_dict.items():
			if len(mdpc):
				mdp0 = mdpc[0]
				id_earth = mdp0.id_earth
				id_com = mdp0.id_com
				intensity = mdpc.get_aggregated_intensity(Imin_or_max, agg_function)
				agg_type = prop_name
				data_type = mdp0.data_type
				num_replies = len(mdpc)
				lon, lat = 0., 0.
				db_ids = [mdp.id for mdp in mdpc]
				macro_info = MacroseismicInfo(id_earth, id_com, intensity, agg_type,
											data_type, num_replies, lon, lat,
											db_ids=db_ids)
				macro_info_list.append(macro_info)

		# TODO: proc_info
		proc_info = {}
		# TODO: geometries
		return MacroInfoCollection(macro_info_list, agg_type, data_type, proc_info)

	def subselect_by_polygon(self, poly_obj):
		"""
		Select MDPs that are situated inside given polygon

		:param poly_obj:
			polygon or closed linestring object (ogr geometry object
			or oqhazlib.geo.polygon.Polygon object)

		:return:
			instance of :class:`MDPCollection`
		"""
		from mapping.geotools.pt_in_polygon import filter_collection_by_polygon

		mdp_list = filter_collection_by_polygon(self, poly_obj)
		return self.__class__(mdp_list)

	def split_by_polygon_set(self, poly_dict):
		"""
		Split MDP collection according to a set of polygons

		:param poly_dict:
			dict, mapping polygon IDs to polygon objects

		:return:
			dict, mapping polygon IDs to instances of
			:class:`MDPCollection`
		"""
		mdpc_dict = {}
		for poly_id, poly_obj in poly_dict.items():
			mdpc_dict[poly_id] = self.subselect_by_polygon(poly_obj)

		return mdpc_dict

	@staticmethod
	def _parse_pt(pt):
		"""
		Parse a point that could be a tuple or a point-like object

		:return:
			(lon, lat, depth) tuple
		"""
		if hasattr(pt, 'lon'):
			lon, lat = pt.lon, pt.lat
			depth = getattr(pt, 'depth', 0)
		elif isinstance(pt, (tuple, list)):
			lon, lat = pt[:2]
			if len(pt) == 3:
				depth = pt[2]
			else:
				detph = 0

		return (lon, lat, depth)

	def subselect_by_distance(self, ref_pt, max_dist, min_dist=0.):
		"""
		Select MDPs that are situated within a given distance range
		with respect to a reference point

		:param ref_pt:
			(lon, lat, [depth]) tuple or instance having lon, lat, [depth]
			properties
		:param max_dist:
			float, maximum distance (in km)
		:param min_dist:
			float, minimum distance (in km)

		:return:
			instance of :class:`MDPCollection`
		"""
		lon, lat, depth = self._parse_pt(ref_pt)
		distances = self.calc_distances(lon, lat, depth)

		is_in_range = (distances >= min_dist) & (distances < max_dist)

		mdp_list = []
		for idx in np.where(is_in_range)[0]:
				mdp_list.append(mdp[idx])

		return self.__class__(mdp_list)

	def split_by_distance(self, ref_pt, distance_interval):
		"""
		Split MDP collection in different distance bins with respect
		to a reference point

		:param ref_pt:
			see :meth:`subselect_by_distance`
		:param distance_interval:
			float, distance interval (in km)

		:return:
			dict, mapping maximum distance to instances of
			:class:`MDPCollection`
		"""
		lon, lat, depth = self._parse_pt(ref_pt)
		distances = self.calc_distances(lon, lat)

		distance_interval = float(distance_interval)
		binned_distances = np.floor(distances / distance_interval) * distance_interval
		binned_distances += distance_interval

		mdpc_dict = {}
		for m, mdp in enumerate(self):
			bin = binned_distances[m]
			if np.isnan(bin):
				bin = None
			if not bin in mdpc_dict:
				mdpc_dict[bin] = self.__class__([mdp])
			else:
				mdpc_dict[bin].append(mdp)

		return mdpc_dict

	def split_by_grid_cells(self, grid_spacing=5, srs='LAMBERT1972'):
		"""
		Split MDP collection according to a regular kilometric grid

		:param grid_spacing:
			grid spacing (in km)
			(default: 5)
		:param srs:
			osr spatial reference system or str, name of known srs
			(default: 'LAMBERT1972')

		:return:
			dict, mapping (center_lon, center_lat) tuples to instances of
			:class:`MacroseismicEnquiryEnsemble`
		"""
		import mapping.geotools.coordtrans as ct

		if isinstance(srs, basestring):
			srs = getattr(ct, srs)

		grid_spacing *= 1000
		lons = self.get_longitudes()
		lats = self.get_latitudes()
		mask = np.isnan(lons)
		lons = np.ma.array(lons, mask=mask)
		lats = np.ma.array(lats, mask=mask)
		X, Y = ct.transform_array_coordinates(ct.WGS84, srs, lons, lats)

		mdpc_dict = {}
		for m, mdp in enumerate(self):
			x, y = X[m], Y[m]
			## Center X, Y
			x_bin = np.floor(x / grid_spacing) * grid_spacing + grid_spacing/2.
			y_bin = np.floor(y / grid_spacing) * grid_spacing + grid_spacing/2.
			## Center longitude and latitude
			[(lon_bin, lat_bin)] = ct.transform_coordinates(srs, ct.WGS84,
															[(x_bin, y_bin)])
			key = (lon_bin, lat_bin)
			if not key in mdpc_dict:
				mdpc_dict[key] = self.__class__([mdp])
			else:
				mdpc_dict[key].append(mdp)

		return mdpc_dict

	def aggregate_by_nothing(self):
		"""
		"""
		pass

	def remove_outliers(self):
		"""
		"""
		pass

	def plot_intensity_vs_distance(self, ref_pt, Imin_or_max, marker='.',
									**kwargs):
		"""
		Plot intensity (Y axis) versus distance (X axis)

		:param ref_pt:
			see :meth:`subselect_by_distance`
		:param Imin_or_max:
			str, either 'min', 'mean' or 'max', to select between minimum,
			mean or maximum intensitiy of each MDP
		:**kwargs:
			additional keyword arguments understood by
			:func:`generic_mpl.plot_xy`

		:return:
			matplotlib Axes instance
		"""
		from plotting.generic_mpl import plot_xy

		lon, lat, depth = self._parse_pt(ref_pt)
		distances = self.calc_distances(lon, lat, depth)
		if Imin_or_max == 'min':
			intensities = self.Imin
		elif Imin_or_max == 'mean':
			intensities = self.Imean
		elif Imin_or_max == 'max':
			intensities = self.Imax

		xlabel = kxargs.pop('xlabel', 'Distance (km)')
		ylabel = kwargs.pop('ylabel', 'Intensity (%s)' % self.mdp_list[0].imt)
		ymin = kwargs.pop('ymin', 1)
		xmin = kwargs.pop('xmin', 0)
		linestyles = ['']
		linewidths = [0]
		markers = [marker]
		return plot_xy([(distances, intensities)], linestyles=linestyles,
						linewidths=linewidths, markers=markers,
						xlabel=xlabel, ylabel=ylabel, xmin=xmin, ymin=ymin,
						**kwargs)

	def plot_histogram(self):
		pass
