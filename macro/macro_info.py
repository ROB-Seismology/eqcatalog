# -*- coding: iso-Latin-1 -*-
"""
Aggregated macroseismic information
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int

try:
	## Python 2
	basestring
	PY2 = True
except:
	## Python 3
	PY2 = False
	basestring = str

import os
from collections import OrderedDict

import numpy as np


__all__ = ["AggregatedMacroInfo", "AggregatedMacroInfoCollection",
			"aggregate_online_macro_info",
			"aggregate_traditional_macro_info",
			"aggregate_official_macro_info",
			"aggregate_historical_macro_info"]


class AggregatedMacroInfo():
	"""
	Container class to hold information of aggregated records retrieved
	from the historical, official or internet macroseismic enquiry database,
	and used for plotting maps.

	:param id_earth:
		int, ID of earthquake in ROB catalog
		or 'all'
	:param id_com:
		int, ID of commune in ROB database
	:param intensity:
		int or float, macroseismic intensity
	:param imt:
		str, intensity measure type, e.g. 'EMS98', 'MMI', 'CII'
	:param agg_type:
		str, type of aggregation, one of:
		- 'id_com' or 'commune'
		- 'id_main' or 'main commune'
		- 'grid_X' (where X is grid spacing in km)
		- None or ''
	:param data_type:
		str, type of enquiry, one of:
		- 'internet', 'online' or 'dyfi'
		- 'traditional'
		- 'historical'
		- 'official'
		- 'isoseismal'
	:param num_mdps:
		int, number of MDPs in aggregate
		(default: 1)
	:param lon:
		float, longitude
		(default: 0)
	:param lat:
		float, latitude
		(default: 0)
	:param residual:
		float, residual between :param:`intensity` and another
		intensity value
		(default: 0)
	:param db_ids:
		list of ints, IDs of database records represented in aggregate
	:param geom_key_val:
		int or str, value corresponding to geometry key in
		AggregatedMacroInfoCollection geometries
		(default: None)
	"""
	def __init__(self, id_earth, id_com, intensity, imt, agg_type, data_type,
				num_mdps=1, lon=0, lat=0, residual=0, db_ids=[], geom_key_val=None):
		self.id_earth = id_earth
		self.id_com = id_com
		self.intensity = intensity
		self.imt = imt
		self.agg_type = agg_type
		self.data_type = data_type
		self.num_replies = self.num_mdps = num_mdps
		self.lon = lon
		self.lat = lat
		self.residual = residual
		self.db_ids = db_ids
		self.geom_key_val = geom_key_val

	def __repr__(self):
		txt = '<AggregatedMacroInfo | by %s | I=%.1f (%s)| %s>'
		txt %= (self.agg_type, self.intensity, self.imt, self.data_type)
		return txt

	@property
	def I(self):
		return self.intensity

	def get_eq(self):
		"""
		Fetch earthquake from ROB database

		:return:
			instance of :class:`eqcatalog.LocalEarthquake`
		"""
		from ..rob.seismodb import query_local_eq_catalog_by_id

		if isinstance(self.id_earth, (int, np.integer, str)):
			[eq] = query_local_eq_catalog_by_id(self.id_earth)
			return eq

	def get_online_enquiries(self, min_fiability=80, verbose=False):
		"""
		Fetch macroseismic enquiry records from the database, based on
		either db_ids or, if this is empty, id_earth

		:param min_fiability:
			int, minimum fiability (ignored if db_ids is not empty)
		:param verbose:
			bool, whether or not to print useful information

		:return:
			instance of :class:`eqcatalog.macro.MacroseismicEnquiryEnsemble`
		"""
		from ..rob.seismodb import query_online_macro_catalog

		if self.data_type in ('traditional', 'official', 'historical',
							'isoseismal'):
			return

		if self.db_ids:
			ensemble = query_online_macro_catalog(web_ids=self.db_ids, verbose=verbose)
		else:
			ensemble = query_online_macro_catalog(self.id_earth, id_com=self.id_com,
								min_fiability=min_fiability, verbose=verbose)

		return ensemble

	def get_historical_texts(self, verbose=False):
		"""
		Fetch historical texts corresponding to :prop:`id_earth`
		and :prop:`id_com` from database

		:return:
			list of dicts
		"""
		from ..rob.seismodb import query_historical_texts

		return query_historical_texts(self.id_earth, self.id_com)

	def to_mdp(self):
		"""
		Convert to macroseismic data point

		:return:
			instance of :class:`eqcatalog.macro.MDP`
		"""
		from .mdp import MDP

		mdp = MDP(self.db_ids, self.id_earth, self.I, self.I, self.imt,
				self.lon, self.lat, self.data_type, self.id_com,
				self.id_com, 100)

		return mdp


class AggregatedMacroInfoCollection():
	"""
	Class representing a collection of aggregated macroseismic records

	:param macro_infos:
		list with instances of :class:`AggregatedMacroInfo`
	:param agg_type:
		str, aggregation type
	:param data_type:
		str, type of enquiry, one of:
		- 'internet', 'online' or 'dyfi'
		- 'traditional'
		- 'historical'
		- 'official'
		- 'isoseismal'
	:param macro_geoms:
		instance of :class:`layeredbasemap.base.MultiData`,
		geometries corresponding to aggregated macroseismic records
		(required if :param:`agg_type` not in ('id_com', 'commune',
		'id_main', 'main commune', 'grid_X', 'distance', '' or None)
		(default: None)
	:param geom_key:
		str, name of geometry key used to link macroseismic records to
		geometries in :param:`macro_geoms`
		(default: '')
	:proc_info:
		dict, containing processing parameters
		(default: {})
	"""
	def __init__(self, macro_infos, agg_type, data_type,
				macro_geoms=None, geom_key='',
				proc_info={}):
		#assert len(set([mi.imt for mi in macro_infos])) <= 1
		self.macro_infos = macro_infos
		self.agg_type = agg_type
		if not data_type and len(macro_infos):
			data_type = macro_infos[0].data_type
		self.data_type = data_type
		if (agg_type not in ('id_com', 'commune', 'id_main', 'main commune',
								'distance', '', None) and agg_type[:4] != 'grid'):
			assert macro_geoms is not None
		self.macro_geoms = macro_geoms
		if macro_geoms is not None:
			assert geom_key
		elif agg_type in ('id_com', 'commune', 'id_main', 'main commune'):
			geom_key = 'ID_ROB'
			for macro_info in self.macro_infos:
				if macro_info.geom_key_val is None:
					macro_info.geom_key_val = macro_info.id_com
		self.geom_key = geom_key
		self.proc_info = proc_info

	def __len__(self):
		return len(self.macro_infos)

	def __iter__(self):
		return self.macro_infos.__iter__()

	def __getitem__(self, spec):
		if isinstance(spec, (int, np.integer, slice)):
			return self.macro_infos.__getitem__(spec)
		elif isinstance(spec, (list, np.ndarray)):
			## spec can contain indexes or bools
			mi_list = []
			if len(spec):
				idxs = np.arange(len(self))
				idxs = idxs[np.asarray(spec)]
				for idx in idxs:
					mi_list.append(self.macro_infos[idx])
			return self.__class__(mi_list, self.agg_type, self.data_type,
										macro_geoms=self.macro_geoms,
										geom_key=self.geom_key, proc_info=self.proc_info)

	def __repr__(self):
		txt = '<AggregatedMacroInfoCollection | by %s | n=%d | %s>'
		txt %= (self.agg_type, len(self), self.data_type)
		return txt

	@property
	def longitudes(self):
		return np.array([rec.lon if rec.lon is not None else np.nan for rec in self])

	@property
	def latitudes(self):
		return np.array([rec.lat if rec.lat is not None else np.nan for rec in self])

	@property
	def intensities(self):
		return np.array([rec.I for rec in self])

	@property
	def imt(self):
		imts = sorted(set([rec.imt for rec in self]))
		return ' / '.join(imts)

	@property
	def residuals(self):
		return np.array([rec.residual for rec in self])

	@property
	def id_earth(self):
		id_earths = set([rec.id_earth for rec in self])
		if len(id_earths) == 1:
			return id_earths.pop()
		else:
			return sorted(id_earths)

	@property
	def num_replies(self):
		return np.sum([rec.num_replies for rec in self])

	def Iminmax(self):
		intensities = self.intensities
		return (np.nanmin(intensities), np.nanmax(intensities))

	def copy(self):
		"""
		Copy macroinfo collection
		Note: :prop:`macro_geoms` will not be copied!

		:return:
			instance of :class:`AggregatedMacroInfoCollection`
		"""
		from copy import deepcopy

		macro_infos = deepcopy(self.macro_infos)

		return self.__class__(macro_infos, self.agg_type, self.data_type,
							macro_geoms=self.macro_geoms, geom_key=self.geom_key,
							proc_info=self.proc_info.copy())

	def get_eq_catalog(self):
		"""
		Fetch catalog of earthquakes linked to this AggregatedMacroInfoCollection

		:return:
			instance of :class:`EQCatalog`
		"""
		from ..rob.seismodb import query_local_eq_catalog_by_id

		return query_local_eq_catalog_by_id(id_earth=self.id_earth)

	def get_epicentral_distances(self):
		"""
		Compute epicentral distances of macroseismic data points

		:return:
			1-D array, epicentral distances (in km)
		"""
		eq = self.get_eq_catalog()[0]
		return eq.epicentral_distance((self.longitudes, self.latitudes))

	def get_hypocentral_distances(self):
		"""
		Compute hypocentral distances of macroseismic data points

		:return:
			1-D array, hypocentral distances (in km)
		"""
		eq = self.get_eq_catalog()[0]
		return eq.hypocentral_distance((self.longitudes, self.latitudes))

	def to_commune_info_dict(self):
		"""
		Convert to dictionary mapping commune IDs to instances of
		:class:`AggregatedMacroInfo`
		"""
		com_info_dict = {}
		for rec in self:
			com_info_dict[rec.id_com] = rec
		return com_info_dict

	def get_commune_ids(self):
		"""
		Get list of commune IDs
		"""
		return [rec.id_com for rec in self]

	def get_commune_info(self, id_com):
		"""
		Get info corresponding to particular commune

		:param id_com:
			int, commune ID

		:return:
			instance of :class:`AggregatedMacroInfo`
		"""
		commune_ids = self.get_commune_ids()
		try:
			idx = commune_ids.index(id_com)
		except:
			return None
		else:
			return self.macro_infos[idx]

	def calc_residual_intensity(self, other_macro_info):
		"""
		Compute residual intensity with respect to other collection
		as self.intensities - other_macro_info.intensities

		:param other_macro_info:
			instance of :class:`AggregatedMacroInfoCollection`

		:return:
			None, 'residual' attribute of instances of
			:class:`AggregatedMacroInfo` in collection are modified in place
		"""
		assert self.imt == other_macro_info.imt

		for i in range(len(self)):
			self.macro_infos[i].residual = 0.
			id_com = self.macro_infos[i].id_com
			I = self.macro_infos[i].intensity
			mi = other_macro_info.get_commune_info(id_com)
			if mi:
				self.macro_infos[i].residual = I - mi.I

	def get_region(self, percentile_width=100):
		"""
		Return geographic extent of collection

		:param percentile_width:
			float, difference between upper and lower percentile
			(range 0 - 100) to include in calculation
			(default: 100 = include all points)

		:return:
			(W, E, S, N) tuple
		"""
		dp = 100. - percentile_width
		percentiles = [0 + dp/2., 100 - dp/2.]
		lons, lats = [], []
		for rec in self:
			if not None in (rec.lon, rec.lat):
				lons.extend([rec.lon] * rec.num_mdps)
				lats.extend([rec.lat] * rec.num_mdps)
		lonmin, lonmax = np.percentile(lons, percentiles)
		latmin, latmax = np.percentile(lats, percentiles)
		return (lonmin, lonmax, latmin, latmax)

	def get_geometries(self, polygons_as_points=False):
		"""
		Transform aggregated macroseismic information to layeredbasemap
		geometries.

		:param polygons_as_points:
			bool, whether to represent polygonal aggregates as points (True)
			or as polygons (False)
			(default: False)

		:return:
			instance of :class:`lbm.MultiPolygonData`
			or :class:`lbm.MultiPointData`
		"""
		import mapping.layeredbasemap as lbm

		from ..rob.seismo_gis import get_communes_gis_file

		if len(self) == 0:
			return []

		## Determine aggregation type
		aggregate_by = self.agg_type
		if aggregate_by == 'commune':
			aggregate_by = 'id_com'
		elif aggregate_by == 'main commune':
			aggregate_by = 'id_main'
		elif not aggregate_by:
			polygons_as_points = True
		elif aggregate_by[:4] == 'grid':
			if '_' in aggregate_by:
				_, grid_spacing = aggregate_by.split('_')
				grid_spacing = float(grid_spacing) * 1000
			else:
				grid_spacing = 5000.
			aggregate_by = 'grid'

		## Construct attribute dictionary
		values = OrderedDict()
		attributes = ['id_earth', 'intensity', 'num_mdps']
		residuals = [rec.residual for rec in self]
		if not np.allclose(residuals, 0):
			attributes += ['residual']
		attributes += ['agg_type', 'data_type']
		if aggregate_by == 'grid' or polygons_as_points or self.macro_geoms:
			## Intensity attributes already matched with polygons
			if aggregate_by in ('id_main', 'id_com'):
				attributes += ['id_com']
			elif self.macro_geoms is None:
				attributes += ['lon', 'lat']
			for attrib in attributes:
				values[attrib] = [getattr(rec, attrib) for rec in self]
		else:
			## Intensity attributes need to be joined with GIS data
			for attrib in attributes:
				values[attrib] = {'key': self.geom_key, 'values':
							{rec.geom_key_val: getattr(rec, attrib) for rec in self}}

		## Select GIS file with commune polygons in function of aggregation type
		#if polygons_as_points:
		#	gis_filename = "Bel_villages_points.TAB"
		gis_filespec = ''
		if not polygons_as_points:
			if aggregate_by == 'id_main':
				gis_filespec = get_communes_gis_file('BE', main_communes=True)
			elif aggregate_by == 'id_com':
				gis_filespec = get_communes_gis_file('BE', main_communes=False)

		## Grid
		if aggregate_by == 'grid':
			if polygons_as_points:
				macro_geoms = lbm.MultiPointData(self.longitudes, self.latitudes,
													values=values)
			else:
				import mapping.geotools.coordtrans as ct
				X_center, Y_center = ct.transform_array_coordinates(ct.WGS84,
									ct.LAMBERT1972, self.longitudes, self.latitudes)
				X_left = X_center - grid_spacing / 2.
				X_right = X_left + grid_spacing
				Y_bottom = Y_center - grid_spacing / 2.
				Y_top = Y_bottom + grid_spacing
				all_lons, all_lats = [], []
				for i in range(len(self)):
					X = [X_left[i], X_right[i], X_right[i], X_left[i], X_left[i]]
					Y = [Y_bottom[i], Y_bottom[i], Y_top[i], Y_top[i], Y_bottom[i]]
					lons, lats = ct.transform_array_coordinates(ct.LAMBERT1972, ct.WGS84, X, Y)
					all_lons.append(lons)
					all_lats.append(lats)
				macro_geoms = lbm.MultiPolygonData(all_lons, all_lats, values=values)

		## Points
		elif polygons_as_points:
			lons = self.longitudes
			lats = self.latitudes
			macro_geoms = lbm.MultiPointData(lons, lats, values=values)

		## Commune polygons
		elif gis_filespec:
			gis_data = lbm.GisData(gis_filespec, joined_attributes=values)
			_, _, polygon_data = gis_data.get_data()
			macro_geoms = polygon_data

		else:
			macro_geoms = self.macro_geoms
			macro_geoms.values.update(values)

		return macro_geoms

	to_lbm_data = get_geometries

	def to_geojson(self, polygons_as_points=False):
		"""
		Convert to GeoJSON.

		:param polygons_as_points:
			see :meth:`get_geometries`

		:return:
			dict
		"""
		multi_data = self.get_geometries(polygons_as_points=polygons_as_points)
		return multi_data.to_geojson()

	def to_mdp_collection(self, skip_missing_pt_locations=True):
		"""
		Convert to MDP collection

		:param skip_missing_pt_locations:
			bool, whether or not records with missing point locations
			should be skipped
			(default: True)

		:return:
			instance of :class:`eqcatalog.macro.MDPCollection`
		"""
		from .mdp import MDPCollection

		mdp_list = []
		for rec in self:
			if rec.lon is not None or skip_missing_pt_locations is False:
				mdp_list.append(rec.to_mdp())

		return MDPCollection(mdp_list)

	def export_gis(self, format, filespec, encoding='latin-1',
					polygons_as_points=False):
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
		:param polygons_as_points:
			see :meth:`get_geometries`

		:return:
			instance of :class:`ogr.DataSource` if :param:`format`
			== 'MEMORY', else None
		"""
		multi_data = self.get_geometries(polygons_as_points=polygons_as_points)
		return multi_data.export_gis(format, filespec, encoding=encoding)

	def plot_map(self, region=(2, 7, 49.25, 51.75), projection="merc",
				graticule_interval=(1, 1), plot_info="intensity",
				int_conversion="round", symbol_style=None, line_style="default",
				thematic_num_replies=False, thematic_classes=None, interpolate_grid={},
				cmap="rob", color_gradient="discontinuous", event_style="default",
				country_style="default", city_style="default",
				admin_source='gadm', admin_level="province", admin_style="default",
				colorbar_style="default", radii=[], plot_pie=None,
				title="", fig_filespec=None, ax=None, copyright=u"© ROB", text_box={},
				dpi="default", border_width=0.2, verbose=True):
		"""
		Plot macroseismic map

		see :func:`eqcatalog.plot.plot_macro_map.plot_macroseismic_map`
		"""
		kwargs = locals().copy()
		kwargs.pop('self')
		from ..plot.plot_macro_map import plot_macroseismic_map

		return plot_macroseismic_map(self, **kwargs)

	def export_geotiff(self, out_filespec, region=(2, 7, 49.25, 51.75),
				projection="merc", plot_info="intensity", int_conversion="round",
				symbol_style=None, line_style="default", thematic_num_replies=False,
				interpolate_grid={}, cmap="rob", color_gradient="discontinuous",
				colorbar_style=None, copyright="", dpi=120, hide_border=True):
		"""
		Export to GeoTIFF map

		:param out_filespec:
			str, full path to output TIF file
		:param region:
		:param projection:
		:param plot_info:
		:param int_conversion:
		:param symbol_style:
		:param line_style:
		:param thematic_num_replies:
		:param interpolate_grid:
		:param cmap:
		:param color_gradient:
		:param colorbar_style:
		:param copyright:
			see :func:`eqcatalog.plot.plot_macro_map.plot_macroseismic_map`
		:param dpi:
			int, image resolution in dots per inch
			(default: 120)
		:param hide_border:
			bool, whether or not to hide map border
			(default: True)
		"""
		graticule_interval = ()
		event_style = None
		admin_level = ''
		radii = []
		plot_pie = None
		title = ""
		fig_filespec = "hold"
		ax = None

		map = self.plot_map(region=region, projection=projection,
				graticule_interval=graticule_interval, plot_info=plot_info,
				int_conversion=int_conversion, symbol_style=symbol_style,
				line_style=line_style, thematic_num_replies=thematic_num_replies,
				interpolate_grid=interpolate_grid, cmap=cmap,
				color_gradient=color_gradient, event_style=event_style,
				admin_level=admin_level, colorbar_style=colorbar_style,
				radii=radii, plot_pie=plot_pie, title=title, fig_filespec=fig_filespec,
				ax=ax, copyright=copyright, verbose=False)
		if hide_border:
			map.border_style.line_color = 'w'
			map.border_style.line_width = 0
		map.export_geotiff(out_filespec, dpi=dpi)

	def interpolate_grid(self, num_cells, region=(None, None, None, None),
						prop='intensity', interpolation_method='cubic',
						interpolation_params={}):
		"""
		Interpolate intensity grid

		:param num_cells:
			Integer or tuple, number of grid cells in lon and lat direction
		:param extent:
			(lonmin, lonmax, latmin, latmax) tuple of floats
			(default: (None, None, None, None)
		:param prop:
			str, name of property to interpolate, either 'intensity',
			'residual' or 'num_replies'
			(default: 'intensity')
		:param interpolation_method:
			str, interpolation method supported by griddata (either
			"linear", "nearest" or "cubic")
			(default: "cubic")
		:param interpolation_params:
			dict, containing names and values of additional interpolation
			parameters
			(default: {})

		:return:
			instance of :class:`mapping.layeredbasemap.MeshGridData`
		"""
		from mapping.layeredbasemap import UnstructuredGridData

		if prop == 'intensity':
			values = self.intensities
			unit = 'Intensity'
		elif prop == 'residual':
			values = np.array([rec.residual for rec in self])
			unit = 'Intensity difference'
		elif prop in ('num_mdps', 'num_replies'):
			values = np.array([rec.num_mdps for rec in self])
			unit = ''
		else:
			print("Unknown property %s" % prop)
		unstructured_grid = UnstructuredGridData(self.longitudes, self.latitudes,
												values, unit=unit)
		grid = unstructured_grid.to_mesh_grid_data(num_cells, extent=region,
									interpolation_method=interpolation_method,
									**interpolation_params)
		return grid

	def interpolate_isoseismals(self, intensity_levels=None,
								interpolation_method='cubic', as_lines=True):
		"""
		Interpolate intensity contours

		:param intensity_levels:
			list of floats, intensity contour levels
			(default: None, will auto-determine integer intensity
			levels between minimum and maximum)
		:param interpolation_method:
			see :meth:`interpolate_grid`
		:param as_lines:
			bool, whether isoseismals should be returned as lines
			(True) or polygonal areas (False)

		:return:
			list with instances of :class:`mapping.layeredbasemap.MultiLineData`
		"""
		if intensity_levels is None:
			intensities = self.intensities
			Imin = np.floor(intensities.min())
			Imax = np.floor(intensities.max())
			intensity_levels = np.arange(Imin, Imax)
		grid_resolution = 200
		grid = self.interpolate_grid(grid_resolution,
									interpolation_method=interpolation_method)
		if as_lines:
			isoseismals = grid.extract_contour_lines(intensity_levels)
		else:
			isoseismals = grid.extract_contour_intervals(intensity_levels)
		return isoseismals

	def get_proc_info_text(self):
		"""
		Construct text summarizing processing parameters

		:return:
			str
		"""
		text = "Aggregation by: %s" % self.agg_type
		if (self.agg_type or self.data_type != 'isoseismal'
			or (self.data_type in ('traditional', 'historical', 'official')
			and self.agg_type in ('id_main', 'main commune'))):
			text += "\nAgg. method: %s" % self.proc_info.get('agg_method')

		if self.data_type in ('internet', 'online', 'dyfi'):
			text += ("\nMin. replies / fiability: %d / %d"
				% (self.proc_info['min_replies'], self.proc_info['min_fiability']))
			text += "\nFilter floors: %s" % str(self.proc_info.get('filter_floors'))
			text += "\nFix records: %s" % self.proc_info.get('fix_records')
			include_other_felt = self.proc_info.get('include_other_felt')
			include_heavy_appliance = self.proc_info.get('include_heavy_appliance')
			if include_other_felt and not include_heavy_appliance:
				cws_calculation = 'DYFI'
			elif include_heavy_appliance and not include_other_felt:
				cws_calculation = 'ROB'
			else:
				cws_calculation = ('of=%s /ha=%s'
					% (include_other_felt, include_heavy_appliance))
			text += '\nCWS calculation: %s' % cws_calculation
			if self.proc_info['agg_method'][:4] == "mean":
				text += ("\nRemove outliers: %s"
					% str(self.proc_info['remove_outliers']))
		elif self.data_type in ('traditional', 'official', 'historical'):
			text += "\nMin. fiability: %d" % self.proc_info['min_fiability']
			text += "\nImin_or_max: %s" % self.proc_info['Imin_or_max']

		return text


def aggregate_online_macro_info(id_earth, min_replies=3, query_info="cii",
				min_fiability=80, filter_floors=(0, 4), aggregate_by="commune",
				agg_method='mean', fix_records=True,
				include_other_felt=True, include_heavy_appliance=False,
				remove_outliers=(2.5, 97.5), verbose=False):
	"""
	Obtain aggregated internet macroseismic information for given earthquake

	:param id_earth:
		int or str, ID of earthquake in ROB database
	:param min_replies:
		int, minimum number of replies to use for aggregating macroseismic data
	:param query_info:
		str, info to query from the database, either 'cii', 'cdi' or
		'num_replies'
		(default: 'cii')
	:param min_fiability:
		int, minimum fiability of enquiries to include in plot
		(default: 80)
	:param filter_floors:
		(lower_floor, upper_floor) tuple, floors outside this range
			(basement floors and upper floors) are filtered out
			(default: (0, 4))
	:param aggregate_by:
		str, type of aggregation, specifying how macroseismic data should
		be aggregated in the map, one of:
		- 'id_com' or 'commune'
		- 'id_main' or 'main commune'
		- 'grid_X' (where X is grid spacing in km)
		- None or '' (= no aggregation, i.e. all replies are
		plotted individually (potentially on top of each other)
		(default: 'commune')
	:param agg_method:
		str, how to aggregate individual enquiries,
		either 'mean' (= ROB practice) or 'aggregated' (= DYFI practice)
		(default: 'mean')
	:param fix_records:
		bool, whether or not to fix various issues (see :meth:`fix_all`)
		(default: True)
	:param include_other_felt:
		bool, whether or not to take into acoount the replies to the
		question "Did others nearby feel the earthquake ?"
		(default: True)
	:param include_heavy_appliance:
		bool, whether or not to take heavy_appliance into account
		as well (not standard, but occurs with ROB forms)
		(default: False)
	:param remove_outliers:
		(min_pct, max_pct) tuple, percentile range to use.
		Only applies if :param:`agg_method` = 'mean'
		and if :param:`agg_info` = 'cii'
		(default: 2.5, 97.5)
	:param verbose:
		bool, if True the query string will be echoed to standard output

	:return:
		instance of :class:`AggregatedMacroInfoCollection`
	"""
	from ..rob import query_local_eq_catalog_by_id

	query_info = query_info.lower()
	if query_info in ('num_mdps', 'num_replies'):
		min_replies = 1

	if aggregate_by == 'commune':
		aggregate_by = 'id_com'
	elif aggregate_by == 'main commune':
		aggregate_by = 'id_main'
	elif not aggregate_by:
		min_replies = 1

	## Retrieve macroseismic information from database
	[eq] = query_local_eq_catalog_by_id(id_earth, verbose=verbose)
	dyfi_ensemble = eq.get_online_macro_enquiries(min_fiability, verbose=verbose)

	## Aggregate
	macro_info_coll = dyfi_ensemble.aggregate(aggregate_by, min_replies,
							agg_info=query_info, min_fiability=min_fiability,
							filter_floors=filter_floors, agg_method=agg_method,
							fix_records=fix_records, include_other_felt=include_other_felt,
							include_heavy_appliance=include_heavy_appliance,
							remove_outliers=remove_outliers)
	macro_info_coll.proc_info['fix_records'] = fix_records

	return macro_info_coll


## Obsolete
def _get_aggregated_intensity(macro_recs, Imin_or_max, agg_function):
	"""
	:param agg_function:
		str, aggregation function, one of 'min', 'max', 'mean', 'median'
	"""
	if Imin_or_max == 'min':
		intensities = np.array([rec['Imin'] for rec in macro_recs])
	elif Imin_or_max == 'max':
		intensities = np.array([rec['Imax'] for rec in macro_recs])
	elif Imin_or_max == 'mean':
		intensities = np.array([np.mean([rec['Imin'], rec['Imax']])
								for rec in macro_recs])

	agg_func = getattr(np, 'nan'+agg_function)
	return agg_func(intensities)


def aggregate_traditional_macro_info(id_earth, id_com=None, data_type='', min_fiability=80,
				Imin_or_max='max', aggregate_by="commune", agg_method="mean",
				verbose=False):
	"""
	Obtain aggregated traditional macroseismic information for given earthquake

	:param id_earth:
		int or str, ID of earthquake in ROB database
	:param id_com:
		int, commune ID
		or list of ints
		(default: None)
	:param data_type:
		str, type of macroseismic data: '', 'official' or 'historical'
		(default: '')
	:param min_fiability:
		int, minimum fiability of macroseismic record
		(default: 80)
	:param Imin_or_max:
		str, one of 'min', 'mean' or 'max' to select between
		intensity_min and intensity_max values in database
		(default: 'max')
	:param aggregate_by:
		str, type of aggregation, specifying how macroseismic data should
		be aggregated in the map, one of:
		- 'id_com' or 'commune'
		- 'id_main' or 'main commune'
		(default: 'commune')
	:param agg_method:
		str, aggregation function to use if :param:`aggregate_by`
		is 'main commune', one of "min(imum)", "max(imum)", "median'
		or "average"/"mean"
		(default: "mean")

	:return:
		instance of :class:`AggregatedMacroInfoCollection`
	"""
	from ..rob.seismodb import (query_traditional_macro_catalog,
								query_seismodb_table_generic)

	if aggregate_by in ('main commune', 'id_main'):
		group_by_main_commune = True
		prop_name = 'id_main'
	else:
		group_by_main_commune = False
		prop_name = 'id_com'

	## Retrieve macroseismic information from database
	#macro_info_coll = query_traditional_macro_catalog(id_earth, data_type=data_type,
	#		min_or_max=min_or_max, group_by_main_commune=group_by_main_commune,
	#		agg_method=agg_method, min_fiability=min_fiability, verbose=verbose)
	mdp_coll = query_traditional_macro_catalog(id_earth, id_com=id_com,
			data_type=data_type, group_by_main_commune=group_by_main_commune,
			min_fiability=min_fiability, verbose=verbose)
	mdp_coll = mdp_coll.remove_empty_locations()

	if agg_method == 'average':
		agg_function = 'mean'
	elif agg_method in ('minimum', 'maximum'):
		agg_function = agg_method[:3]
	else:
		agg_function = agg_method

	# TODO: aggregate by grid?

	#proc_info = dict(agg_method=agg_method, min_fiability=min_fiability,
	#				min_or_max=min_or_max)
	#macro_info_coll = AggregatedMacroInfoCollection(macro_infos, aggregate_by,
	#											data_type, proc_info=proc_info)
	macro_info_coll = mdp_coll.aggregate_by_property(prop_name, Imin_or_max,
													agg_function, min_fiability)

	return macro_info_coll


def aggregate_official_macro_info(id_earth, id_com=None, min_fiability=80,
				Imin_or_max='max', aggregate_by="commune", agg_method="mean"):
	"""
	Obtain aggregated official macroseismic information for given earthquake
	This is a wrapper for :func:`aggregate_traditional_macro_info`
	"""
	kwargs = locals().copy()
	return aggregate_official_macro_info(data_type='official', **kwargs)


def aggregate_historical_macro_info(id_earth, id_com=None, min_fiability=80,
				Imin_or_max='max', aggregate_by="commune", agg_method="mean"):
	"""
	Obtain aggregated historical macroseismic information for given earthquake
	This is a wrapper for :func:`aggregate_traditional_macro_info`
	"""
	kwargs = locals().copy()
	return aggregate_official_macro_info(data_type='historical', **kwargs)
