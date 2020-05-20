"""
Macroseismic data point(s)
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import operator
import sys
if sys.version[0] == '3':
	basestring = str

import numpy as np

from .macro_info import AggregatedMacroInfo, AggregatedMacroInfoCollection



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

	def round(self, half_unit=False):
		"""
		Round Imin / Imax intensity values

		:param half_unit:
			bool, whether or not to round to half units
		"""
		if half_unit:
			self.Imin = np.round(self.Imin * 2) / 2.
			self.Imax = np.round(self.Imax * 2) / 2.
		else:
			self.Imin = np.round(self.Imin)
			self.Imax = np.round(self.Imax)


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

	def __getitem__(self, spec):
		if isinstance(spec, (int, np.integer, slice)):
			return self.mdp_list.__getitem__(spec)
		elif isinstance(spec, (list, np.ndarray)):
			## spec can contain indexes or bools
			mdp_list = []
			if len(spec):
				idxs = np.arange(len(self))
				idxs = idxs[np.asarray(spec)]
				for idx in idxs:
					mdp_list.append(self.mdp_list[idx])
			return self.__class__(mdp_list)

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

	def get_region(self):
		"""
		Determine bounding region

		:return:
			(lonmin, lonmax, latmin, latmax) tuple of floats
		"""
		lons = self.get_longitudes()
		lats = self.get_latitudes()

		return (np.nanmin(lons), np.nanmax(lons), np.nanmin(lats), np.nanmax(lats))

	def get_intensities(self, Imin_or_max):
		"""
		Get intensities

		:param Imin_or_max:
			str, either 'min', 'mean', 'median' or 'max', to select
			between minimum, mean, median or maximum intensitiy of each MDP

		:return:
			1D float array
		"""
		if Imin_or_max == 'min':
			intensities = self.Imin
		elif Imin_or_max == 'max':
			intensities = self.Imax
		elif Imin_or_max == 'mean':
			intensities = self.Imean

		return intensities

	def Iminmax(self, Imin_or_max):
		"""
		Return minimum and maximum intensity

		:param Imin_or_max:
			see :meth:`get_intensities`
		"""
		intensities = self.get_intensities(Imin_or_max)
		return (np.nanmin(intensities), np.nanmax(intensities))

	def round_intensities(self, half_unit=False):
		"""
		Round Imin / Imax intensity values of individual MDPs

		:param half_unit:
			bool, whether or not to round to half units
		"""
		for mdp in self:
			mdp.round(half_unit=half_unit)

	def get_aggregated_intensity(self, Imin_or_max, agg_function):
		"""
		Compute aggregated intensity of MDP collection

		:param Imin_or_max:
			see :meth:`get_intensities`
		:param agg_function:
			str, aggregation function, one of 'min', 'max', 'mean',
			'median', 'std' or 'PXX' (where XX is percentile level)

		:return:
			float, aggregated intensity
		"""
		from functools import partial

		intensities = self.get_intensities(Imin_or_max)
		if agg_function[0].upper() == 'P':
			perc_level = float(agg_function[1:])
			agg_func = partial(np.nanpercentile, q=perc_level)
		else:
			agg_func = getattr(np, 'nan'+agg_function)

		return agg_func(intensities)

	def remove_outliers(self, Imin_or_max, min_pct=2.5, max_pct=97.5):
		"""
		Remove outliers (with intensity outside of confidence range)
		from collection

		:param Imin_or_max:
			see :meth:`get_intensities`
		:param min_pct:
			float, lower percentile
			(default: 2.5)
		:param max_pct:
			float, upper percentile
			(default: 97.5)

		:return:
			instance of :class:`MDPCollection`
		"""
		intensities = self.get_intensities(Imin_or_max)
		pct0 = np.percentile(intensities, min_pct)
		pct1 = np.percentile(intensities, max_pct)
		within_confidence = (intensities >= pct0) & (intensities <= pct1)
		mdp_list = []
		for idx in np.where(within_confidence)[0]:
			mdp_list.append(self.mdp_list[idx])

		return self.__class__(mdp_list)

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

	def get_prop_values(self, prop_name):
		"""
		Get values corresponding to given property

		:param prop_name:
			str, property name

		:return:
			1D array
		"""
		return np.array([getattr(mdp, prop_name) for mdp in self])

	def get_unique_prop_values(self, prop_name):
		"""
		Get unique values corresponding to given property

		:pram prop_name:
			str, property name

		:return:
			1D array
		"""
		return np.unique(self.get_prop_values(prop_name))

	def subselect_by_property(self, prop_name, prop_val, comparator=operator.eq):
		"""
		Select MDPs based on the value of a particular property

		:param prop_name:
			str, property name
		:param prop_val:
			property value
		:param comparator:
			comparison function
			(default: operator.eq)

		:return:
			instance of :class:`MDPCollection`
		"""
		mdp_list = []
		for mdp in self:
			if comparator(getattr(mdp, prop_name), prop_val):
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
							min_fiability=80, min_num_mdp=1):
		"""
		Aggregate MDPs based on values of a particular property

		:param prop_name:
			str, property name
		:param Imin_or_max:
		:param agg_function:
			see :meth:`get_aggregated_intensity`
		:param min_fiability:
			int, minimum fiability
			(default: 80)
		:param min_num_mdp:
			int, minimum number of MDPs needed to define an aggregate
			(default: 1)

		:return:
			instance of :class:`AggregatedMacroInfoCollection`
		"""
		from ..rob import get_communes

		if prop_name == 'id_main':
			communes = get_communes(main_communes=True)
			commune_coords = {rec['id']: (rec['longitude'], rec['latitude'])
							for rec in communes}

		mdpc = self.subselect_by_property('fiability', min_fiability, operator.ge)
		mdpc_dict = mdpc.split_by_property(prop_name)
		macro_info_list = []
		for agg_key, mdpc in mdpc_dict.items():
			if len(mdpc) >= min_num_mdp:
				unique_id_earths = mdpc.get_unique_prop_values('id_earth')
				id_earth = unique_id_earths[0] if len(unique_id_earths) == 1 else None
				unique_id_coms = mdpc.get_unique_prop_values('id_com')
				id_com = unique_id_coms[0] if len(unique_id_coms) == 1 else None
				intensity = mdpc.get_aggregated_intensity(Imin_or_max, agg_function)
				agg_type = prop_name
				unique_data_types = mdpc.get_unique_prop_values('data_type')
				data_type = unique_data_types[0] if len(unique_data_types) == 1 else 'mixed'
				num_replies = len(mdpc)
				if prop_name == 'id_com':
					mdp0 = mdpc[0]
					lon, lat = mdp0.lon, mdp0.lat
				elif prop_name == 'id_main':
					id_main = mdpc[0].id_main
					lon, lat = commune_coords.get(id_main, (0., 0.))
					id_com = id_main
				else:
					lon, lat = 0., 0.
				db_ids = [mdp.id for mdp in mdpc]
				macro_info = AggregatedMacroInfo(id_earth, id_com, intensity, agg_type,
											data_type, num_replies, lon, lat,
											db_ids=db_ids, geom_key_val=agg_key)
				macro_info_list.append(macro_info)

		proc_info = dict(agg_method=agg_function, min_fiability=min_fiability,
					min_or_max=Imin_or_max)
		return AggregatedMacroInfoCollection(macro_info_list, agg_type, data_type,
											proc_info=proc_info)

	def aggregate_by_nothing(self, Imin_or_max, min_fiability=80):
		"""
		Turn MDP collection in aggregated macro information, with
		each MDP corresponding to an aggregate

		:param Imin_or_max:
			see :meth:`get_intensities`
		:param min_fiability:
			int, minimum fiability
			(default: 80)

		:return:
			instance of :class:`AggregatedMacroInfoCollection`
		"""
		mdpc = self.subselect_by_property('fiability', min_fiability, operator.ge)
		intensities = self.get_intensities(Imin_or_max)
		macro_info_list = []
		for m, mdp in enumerate(mdpc):
			id_earth = mdp.id_earth
			id_com = mdp.id_com
			intensity = intensities[m]
			agg_type = ''
			data_type = mdp.data_type
			num_replies = 1
			lon, lat = mdp.lon, mdp.lat
			db_ids = [mdp.id]
			macro_info = AggregatedMacroseismicInfo(id_earth, id_com, intensity,
												agg_type, data_type, num_replies,
												lon, lat, db_ids=db_ids)
			macro_info_list.append(macro_info)

		proc_info = dict(agg_method=agg_function, min_fiability=min_fiability,
					min_or_max=Imin_or_max)
		return AggregatedMacroInfoCollection(macro_info_list, agg_type, data_type,
											proc_info=proc_info)

	def subselect_by_polygon(self, poly_obj):
		"""
		Select MDPs that are situated inside given polygon

		:param poly_obj:
			polygon or closed linestring object (ogr geometry object
			or oqhazlib.geo.polygon.Polygon object)

		:return:
			(mpds_inside, mdps_outside) tuple:
			instances of :class:`MDPCollection`
		"""
		from mapping.geotools.pt_in_polygon import filter_points_by_polygon

		lons, lats = self.get_longitudes(), self.get_latitudes()
		idxs_inside, idxs_outside = filter_points_by_polygon(lons, lats, poly_obj)
		return (self.__getitem__(idxs_inside), self.__getitem__(idxs_outside))

	def split_by_polygon_data(self, poly_data, value_key):
		"""
		Split MDP collection according to a set of polygons

		:param poly_data:
			instance of :class:`layeredbasemap.MultiPolygonData`
			or list of instances of :class:`osgeo.ogr.Geometry`
			or str, full path to GIS file containing polygon data
		:param value_key:
			str, key in values dict of :param:`poly_data` that should
			be used to link MDP collections to polygons
			If None, use sequential number

		:return:
			dict, mapping polygon IDs to instances of
			:class:`MDPCollection`
		"""
		import mapping.layeredbasemap as lbm

		if isinstance(poly_data, basestring):
			gis_data = lbm.GisData(poly_data)
			_, _, poly_data = gis_data.get_data()

		if value_key is not None:
			if len(poly_data) != len(np.unique(poly_data.values[value_key])):
				print("Warning: Polygon data values not unique for key %s!"
						% value_key)

		mdpc_dict = {}
		mdps_outside = self
		for p, poly_obj in enumerate(poly_data):
			try:
				poly_id = poly_obj.value.get(value_key, p)
			except:
				poly_id = p
			mdps_inside, mdps_outside = mdps_outside.subselect_by_polygon(poly_obj)
			mdpc_dict[poly_id] = mdps_inside

		return mdpc_dict

	def aggregate_by_polygon_data(self, poly_data, value_key,
								Imin_or_max, agg_function,
								min_fiability=80, min_num_mdp=3,
								include_unmatched_polygons=True):
		"""
		Aggregate MDP collection according to a set of polygons

		:param poly_data:
		:param value_key:
			see :meth:`split_by_polygon_data`
		:param Imin_or_max:
		:param agg_function:
			see :meth:`get_aggregated_intensity`
		:param min_fiability:
			int, minimum fiability
			(default: 80)
		:param min_num_mdp:
			int, minimum number of MDPs needed to define an aggregate
			(default: 3)
		:param include_unmatched_polygons:
			bool, whether or not unmatched polygons should be included
			in the result (their intensity will be set to nan!)
			(default: True)

		:return:
			instance of :class:`AggregatedMacroInfoCollection`
		"""
		import mapping.layeredbasemap as lbm

		if isinstance(poly_data, basestring):
			gis_data = lbm.GisData(poly_data)
			_, _, poly_data = gis_data.get_data()

		mdpc = self.subselect_by_property('fiability', min_fiability, operator.ge)
		mdpc_dict = mdpc.split_by_polygon_data(poly_data, value_key)
		macro_info_list = []
		polygon_list = []
		for geom_key_val, mdpc in mdpc_dict.items():
			poly_idx = poly_data.values[value_key].index(geom_key_val)
			poly_obj = poly_data[poly_idx]

			unique_id_earths = mdpc.get_unique_prop_values('id_earth')
			id_earth = unique_id_earths[0] if len(unique_id_earths) == 1 else None
			unique_id_coms = mdpc.get_unique_prop_values('id_com')
			id_com = unique_id_coms[0] if len(unique_id_coms) == 1 else None
			if len(mdpc) >= min_num_mdp:
				intensity = mdpc.get_aggregated_intensity(Imin_or_max, agg_function)
			else:
				intensity = np.nan
			agg_type = 'polygon'
			unique_data_types = mdpc.get_unique_prop_values('data_type')
			data_type = unique_data_types[0] if len(unique_data_types) == 1 else 'mixed'
			num_replies = len(mdpc)
			centroid = poly_obj.get_centroid()
			lon, lat = centroid.lon, centroid.lat
			db_ids = [mdp.id for mdp in mdpc]
			if len(mdpc) >= min_num_mdp or include_unmatched_polygons:
				macro_info = AggregatedMacroInfo(id_earth, id_com, intensity, agg_type,
											data_type, num_replies, lon, lat,
											db_ids=db_ids, geom_key_val=geom_key_val)
				macro_info_list.append(macro_info)
				polygon_list.append(poly_obj)

		macro_geoms = lbm.MultiPolygonData.from_polygons(polygon_list)

		proc_info = dict(agg_method=agg_function, min_fiability=min_fiability,
					min_or_max=Imin_or_max)
		return AggregatedMacroInfoCollection(macro_info_list, 'polygon', data_type,
									macro_geoms=macro_geoms, geom_key=value_key,
									proc_info=proc_info)

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

	def subselect_by_region(self, region):
		"""
		Select part of collection situated inside given geographic extent

		:param region:
			(W, E, S, N) tuple

		:return:
			instance of :class:`MDPCollection`
		"""
		lonmin, lonmax, latmin, latmax = region
		longitudes, latitudes = self.get_longitudes(), self.get_latitudes()
		lon_idxs = (lonmin <= longitudes) & (longitudes <= lonmax)
		lat_idxs = (latmin <= latitudes) & (latitudes <= latmax)
		return self.__getitem__(lon_idxs & lat_idxs)

	def subselect_by_distance(self, ref_pt, max_dist, min_dist=0.):
		"""
		Select MDPs that are situated within a given distance range
		with respect to a reference point

		:param ref_pt:
			(lon, lat, [depth]) tuple or instance having lon, lat, [depth]
			properties. Set depth of ref_pt to zero for epicentral distance
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

		return self.__getitem__(is_in_range)

	def split_by_distance(self, ref_pt, distance_interval):
		"""
		Split MDP collection in different distance bins with respect
		to a reference point

		:param ref_pt:
			see :meth:`subselect_by_distance`
		:param distance_interval:
			float, distance interval (in km)
			or 1D array, distance bin edges

		:return:
			dict, mapping maximum distance to instances of
			:class:`MDPCollection`
		"""
		lon, lat, depth = self._parse_pt(ref_pt)
		distances = self.calc_distances(lon, lat, depth)

		if np.isscalar(distance_interval):
			distance_interval = float(distance_interval)
			binned_distances = np.floor(distances / distance_interval) * distance_interval
			binned_distances += distance_interval / 2.
		else:
			distance_bin_edges = np.asarray(distance_interval)
			bin_idxs = np.digitize(distances, distance_bin_edges)
			bin_idxs -= 1
			distance_bin_centers = (distance_bin_edges[:-1]
									+ np.diff(distance_bin_edges)/2.)
			binned_distances = [distance_bin_centers[i] for i in bin_idxs]
			binned_distances = np.array(binned_distances)

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

	def aggregate_by_distance(self, ref_pt, distance_interval, Imin_or_max,
							agg_function, min_fiability=80, min_num_mdp=3,
							create_polygons=True):
		"""
		Aggregate MDPs by distance with respect to a reference point

		:param ref_pt:
		:param distance_interval:
			see :meth:`split_by_distance`
		:param Imin_or_max:
		:param agg_function:
			see :meth:`get_aggregated_intensity`
		:param min_fiability:
			int, minimum fiability
			(default: 80)
		:param min_num_mdp:
			int, minimum number of MDPs needed to define an aggregate
			(default: 3)
		:param create_polygons:
			bool, whether or not to create polygon objects necessary
			for plotting on a map
			(default: True)

		:return:
			instance of :class:`AggregatedMacroInfoCollection`
		"""
		from mapping.geotools.geodetic import spherical_point_at
		import mapping.layeredbasemap as lbm

		mdpc = self.subselect_by_property('fiability', min_fiability, operator.ge)
		mdpc_dict = mdpc.split_by_grid_cells(grid_spacing, srs=srs)
		macro_info_list = []
		if create_polygons:
			polygon_list = []
			azimuths = np.linspace(0, 360, 361)
		geom_key = 'max_radius'
		agg_type = 'distance'
		for max_radius, mdpc in self.split_by_distance(ref_pt, distance_interval):
			if len(mdpc) >= min_num_mdp:
				unique_id_earths = mdpc.get_unique_prop_values('id_earth')
				id_earth = unique_id_earths[0] if len(unique_id_earths) == 1 else None
				unique_id_coms = mdpc.get_unique_prop_values('id_com')
				id_com = unique_id_coms[0] if len(unique_id_coms) == 1 else None
				intensity = mdpc.get_aggregated_intensity(Imin_or_max, agg_function)
				unique_data_types = mdpc.get_unique_prop_values('data_type')
				data_type = unique_data_types[0] if len(unique_data_types) == 1 else 'mixed'
				num_replies = len(mdpc)
				lon, lat, _ = self._parse_pt(ref_pt)
				db_ids = [mdp.id for mdp in mdpc]
				macro_info = AggregatedMacroInfo(id_earth, id_com, intensity, agg_type,
											data_type, num_replies, lon, lat,
											db_ids=db_ids, geom_key_val=max_radius)
				macro_info_list.append(macro_info)

				## Create polygon
				if create_polygons:
					lons, lats = spherical_point_at(lon, lat, max_radius, azimuths)
					min_radius = max_radius - distance_interval
					if min_radius:
						interior_lons, interior_lats = spherical_point_at(lon, lat,
																min_radius, azimuths)
						interior_lons, interior_lats = [interior_lons], [interior_lats]
					else:
						interior_lons, interior_lats = None, None

					value = {'min_radius': min_radius, 'max_radius': max_radius}
					poly_obj = lbm.PolygonData(lons, lats, interior_lons=interior_lons,
												interior_lats=interior_lats,
												value=value)
					polygon_list.append(poly_obj)

		if create_polygons:
			macro_geoms = lbm.MultiPolygonData.from_polygons(polygon_list)
		else:
			macro_geoms = None

		proc_info = dict(agg_method=agg_function, min_fiability=min_fiability,
					min_or_max=Imin_or_max)
		return AggregatedMacroInfoCollection(macro_info_list, agg_type, data_type,
									macro_geoms=macro_geoms, geom_key=geom_key,
									proc_info=proc_info)

	def split_by_grid_cells(self, grid_spacing, srs='LAMBERT1972'):
		"""
		Split MDP collection according to a regular kilometric grid

		:param grid_spacing:
			grid spacing (in km)
		:param srs:
			osr spatial reference system or str, name of known srs
			(default: 'LAMBERT1972')

		:return:
			dict, mapping (center_lon, center_lat) tuples to instances of
			:class:`MDPCollection`
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

	def aggregate_by_grid_cells(self, grid_spacing, Imin_or_max, agg_function,
								srs='LAMBERT1972', min_fiability=80,
								min_num_mdp=3):
		"""
		Aggregate MDPs in grid cells

		:param grid_spacing:
			see :meth:`split_gy_grid_cells`
		:param Imin_or_max:
		:param agg_function:
			see :meth:`get_aggregated_intensity`
		:param srs:
			see :meth:`split_by_grid_cells`
		:param min_fiability:
			int, minimum fiability
			(default: 80)
		:param min_num_mdp:
			int, minimum number of MDPs needed to define an aggregate
			(default: 3)

		:return:
			instance of :class:`AggregatedMacroInfoCollection`
		"""
		mdpc = self.subselect_by_property('fiability', min_fiability, operator.ge)
		mdpc_dict = mdpc.split_by_grid_cells(grid_spacing, srs=srs)
		macro_info_list = []
		agg_type = 'grid_%d' % grid_spacing
		for grid_key, mdpc in mdpc_dict.items():
			if len(mdpc) >= min_num_mdp:
				unique_id_earths = mdpc.get_unique_prop_values('id_earth')
				id_earth = unique_id_earths[0] if len(unique_id_earths) == 1 else None
				unique_id_coms = mdpc.get_unique_prop_values('id_com')
				id_com = unique_id_coms[0] if len(unique_id_coms) == 1 else None
				intensity = mdpc.get_aggregated_intensity(Imin_or_max, agg_function)
				unique_data_types = mdpc.get_unique_prop_values('data_type')
				data_type = unique_data_types[0] if len(unique_data_types) == 1 else 'mixed'
				num_replies = len(mdpc)
				lon, lat = grid_key
				db_ids = [mdp.id for mdp in mdpc]
				macro_info = AggregatedMacroInfo(id_earth, id_com, intensity,
												agg_type, data_type, num_replies,
												lon, lat, db_ids=db_ids)
				macro_info_list.append(macro_info)

		proc_info = dict(agg_method=agg_function, min_fiability=min_fiability,
					min_or_max=Imin_or_max)
		return AggregatedMacroInfoCollection(macro_info_list, agg_type, data_type,
											proc_info=proc_info)

	def estimate_intensity_attenuation(self, ref_pt, independent_var='distance',
									Imin_or_max='mean', bin_interval='default',
									bin_func='mean', bin_min_num_mdp=3,
									polyfit_degree=3):
		"""
		Estimate intensity-distance relation based on binning
		followed by polynomial fitting

		:param ref_pt:
			reference point for distance calculation.
			(lon, lat, [depth]) tuple or instance having lon, lat, [depth]
			properties. Set depth of ref_pt to zero for epicentral distance
		:param independent_var:
			str, name of independent variable, either 'distance'
			or 'intensity'
			(default: 'intensity')
		:param Imin_or_max:
			see :meth:`get_intensities`
		:param bin_interval:
			float, interval for binning independent variable before
			polynomial fit
			(default: 'default' = 10 km for distance, 0.5 for intensity)
		:param bin_func':
			str, aggregation function to apply to bins before fitting,
			either 'mean' or 'median'
			(default: 'mean')
		:param bin_min_num_mdp:
			int, minimum number of MDPs in a bin
			(default: 3)
		:param polyfit_degree:
			int, degree of the fitting polynomial
			see :func:`np.polyfit`
			A negative value is interpreted as the width (number of bins)
			for a running average
			If zero, polynomial fitting or smoothing will be skipped
			(default: 3)

		:return:
			tuple of 4 arrays:
			- independent variable
			- mean or median of the dependent variable
			- mean - sigma or 32nd percentile of the dependent variable
			- mean + sigma or 68th percentile of the dependent variable
		"""
		# TODO: support bin_func = 'median'
		# TODO (in plot): oqhazlib IPE relations

		from scipy.ndimage.filters import uniform_filter1d

		if independent_var == 'distance':
			if bin_interval == 'default':
				bin_interval = 5.
			mdpc_dict = self.split_by_distance(ref_pt, bin_interval)
			distances = np.sort(mdpc_dict.keys())
			num_mdp = np.array([len(mdpc_dict[d]) for d in distances])
			distances = distances[num_mdp >= bin_min_num_mdp]
			Imean = [mdpc_dict[d].get_aggregated_intensity(Imin_or_max, 'mean')
					for d in distances]
			Isigma = [mdpc_dict[d].get_aggregated_intensity(Imin_or_max, 'std')
					for d in distances]
			if polyfit_degree > 0:
				Imean_fit = np.polyfit(distances, Imean, polyfit_degree)
				Isigma_fit = np.polyfit(distances, Isigma, polyfit_degree)
				Imean = np.poly1d(Imean_fit)(distances)
				Isigma = np.poly1d(Isigma_fit)(distances)
			elif polyfit_degree < 0:
				## Running average
				window_len = int(abs(polyfit_degree))
				#Imean = np.convolve(np.ones(window_len)/window_len, Imean, "same")
				#Isigma = np.convolve(np.ones(window_len)/window_len, Isigma, "same")
				Imean = uniform_filter1d(Imean, size=window_len)
				Isigma = uniform_filter1d(Isigma, size=window_len)

			return (distances, Imean, Imean - Isigma, Imean + Isigma)

		elif independent_var == 'intensity':
			if bin_interval == 'default':
				bin_interval = 0.5
			distances = self.calc_distances(ref_pt.lon, ref_pt.lat, ref_pt.depth)
			intensities = self.get_intensities(Imin_or_max)
			Imax = np.ceil(intensities.max()) + bin_interval
			Imin = max(0, 1 - bin_interval / 2.)
			intensity_bins = np.arange(Imin, Imax, bin_interval)
			idxs = np.digitize(intensities, intensity_bins)
			idxs -= 1
			dmean, dsigma = [], []
			for i in range(len(intensity_bins) - 1):
				subidxs = (idxs == i)
				if np.sum(subidxs) >= bin_min_num_mdp:
					dmean.append(np.nanmean(distances[subidxs]))
					dsigma.append(np.nanstd(distances[subidxs]))
				else:
					dmean.append(np.nan)
					dsigma.append(np.nan)
			intensities = intensity_bins[:-1] + np.diff(intensity_bins) / 2.
			dmean, dsigma = np.array(dmean), np.array(dsigma)
			nan_idxs = np.isnan(dmean)
			if nan_idxs.any():
				dmean[nan_idxs] = np.interp(intensities[nan_idxs],
											intensities[~nan_idxs],
											dmean[~nan_idxs])
				dsigma[nan_idxs] = np.interp(intensities[nan_idxs],
											intensities[~nan_idxs],
											dsigma[~nan_idxs])

			if polyfit_degree > 0:
				dmean_fit = np.polyfit(intensities, dmean, polyfit_degree)
				dsigma_fit = np.polyfit(intensities, dsigma, polyfit_degree)
				dmean = np.poly1d(dmean_fit)(intensities)
				dsigma = np.poly1d(dsigma_fit)(intensities)
			elif polyfit_degree < 0:
				## Running average
				window_len = int(abs(polyfit_degree))
				dmean = uniform_filter1d(dmean, size=window_len)
				dsigma = uniform_filter1d(dsigma, size=window_len)

			return (intensities, dmean, dmean - dsigma, dmean + dsigma)

	def plot_intensity_vs_distance(self, ref_pt, Imin_or_max, marker='.',
									marker_size=8, azimuth_cmap=None, **kwargs):
		"""
		Plot intensity (Y axis) versus distance (X axis)

		:param ref_pt:
			see :meth:`subselect_by_distance`
		:param Imin_or_max:
			see :meth:`get_intensities`
		:**kwargs:
			additional keyword arguments understood by
			:func:`generic_mpl.plot_xy`

		:return:
			matplotlib Axes instance
		"""
		from plotting.generic_mpl import plot_xy
		from mapping.geotools.geodetic import spherical_azimuth

		lon, lat, depth = self._parse_pt(ref_pt)
		distances = self.calc_distances(lon, lat, depth)
		intensities = self.get_intensities(Imin_or_max)
		if azimuth_cmap:
			from matplotlib.colors import Normalize
			if isinstance(azimuth_cmap, basestring):
				from matplotlib.cm import get_cmap
				cmap = get_cmap(azimuth_cmap)
			norm = Normalize(0, 360)
			azimuths = spherical_azimuth(lon, lat,
									self.get_longitudes(), self.get_latitudes())
			marker_fill_colors = [cmap(norm(azimuths))]
		else:
			marker_fill_colors = []

		xlabel = kwargs.pop('xlabel', 'Distance (km)')
		ylabel = kwargs.pop('ylabel', 'Intensity (%s)' % self.mdp_list[0].imt)
		ymin = kwargs.pop('ymin', 1)
		xmin = kwargs.pop('xmin', 0)
		linestyles = ['']
		linewidths = [0]
		markers = [marker]
		marker_sizes = [marker_size]

		return plot_xy([(distances, intensities)], linestyles=linestyles,
						linewidths=linewidths, markers=markers,
						marker_sizes=marker_sizes,
						marker_fill_colors=marker_fill_colors, fill_colors=[None],
						xlabel=xlabel, ylabel=ylabel, xmin=xmin, ymin=ymin,
						**kwargs)

	def plot_histogram(self, Imin_or_max, color='usgs', label='', **kwargs):
		"""
		Plot intensity histogram

		:param Imin_or_max:
			see :meth:`get_intensities`
		:param color:
			matplotlib color specification (uniform color for all bars)
			or list of colors (one for each bar)
			or matplotlib color map or string (colormap name)
			(default: 'usgs')
		:param label:
			str, legend label
			(default: '')
		:**kwargs:
			additional keyword arguments understood by
			:func:`generic_mpl.plot_histogram`

		:return:
			matplotlib Axes instance
		"""
		from plotting.generic_mpl import plot_histogram

		if color in ('usgs', 'rob'):
			import mapping.layeredbasemap as lbm
			colors = lbm.cm.get_cmap('macroseismic', color)
		elif isinstance(color, (basestring, list, np.ndarray)):
			colors = color
		else:
			colors = [color]

		xmin = kwargs.pop('xmin', 0)

		intensities = self.get_intensities(Imin_or_max)
		if color == 'rob':
			Imax = 7
			xmax = kwargs.pop('xmax', Imax + 0.5)
		elif color == 'usgs':
			Imax = 9
			xmax = kwargs.pop('xmax', Imax + 0.5)
		else:
			xmax = kwargs.pop('xmax', 12.5)
			Imax = np.floor(xmax)
		bins = np.arange(1, Imax + 2)
		xticks = kwargs.pop('xticks', bins)

		labels = kwargs.pop('label', [label])
		xlabel = kwargs.pop('xlabel', 'Intensity (%s)' % self.mdp_list[0].imt)

		return plot_histogram([intensities], bins, colors=colors, labels=labels,
							xmin=xmin, xmax=xmax, xlabel=xlabel, xticks=xticks,
							**kwargs)

	def plot_distance_histogram(self, ref_pt, bin_width=10, Imin_or_max='mean',
								label='', **kwargs):
		"""
		Plot distance histogram

		:param ref_pt:
			Reference point for distance calculations
			see :meth:`subselect_by_distance`
		:param bin_width:
			float, distance bin width (in km)
			(default: 10)
		:param Imin_or_max:
			str, 'min', 'max' or 'mean'
			If specified, separate histograms are drawn for different
			intensity classes. Note that this will only work if
			intensities are rounded
			(default: 'mean')
		:param label:
			str, legend label
			(default: '')
		:**kwargs:
			additional keyword arguments understood by
			:func:`generic_mpl.plot_histogram`

		:return:
			matplotlib Axes instance

		"""
		from plotting.generic_mpl import plot_histogram

		lon, lat, depth = self._parse_pt(ref_pt)
		distances = self.calc_distances(lon, lat, depth)

		if Imin_or_max:
			datasets, labels = [], []
			mdpc_dict = self.split_by_property('I' + Imin_or_max)
			for I in sorted(mdpc_dict.keys()):
				mdpc = mdpc_dict[I]
				distances = mdpc.calc_distances(lon, lat, depth)
				datasets.append(distances)
				labels.append('I=%s' % I)
		else:
			datasets = [distances]
			labels = kwargs.pop('label', [label])

		bin_width = float(bin_width)
		dmax = np.floor(distances.max() / bin_width) * bin_width + bin_width
		distance_bins = np.arange(0, dmax, bin_width)

		xlabel = kwargs.pop('xlabel', 'Distance (km)')

		return plot_histogram(datasets, distance_bins, labels=labels,
								xlabel=xlabel, **kwargs)
