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


from ..rob import SEISMOGIS_ROOT
GIS_FOLDER = os.path.join(SEISMOGIS_ROOT, "collections", "Bel_administrative_ROB", "TAB")
#GIS_FOLDER = "D:\\seismo-gis\\collections\\Bel_administrative_ROB\\TAB"


__all__ = ["MacroseismicInfo", "MacroInfoCollection",
			"get_aggregated_info_web", "get_aggregated_info_official"]


class MacroseismicInfo():
	"""
	Container class to hold information of (aggregated) records retrieved
	from the official or internet macroseismic enquiry database, and
	used for plotting maps.

	:param id_earth:
		int, ID of earthquake in ROB catalog
		or 'all'
	:param id_com:
		int, ID of commune in ROB database
	:param intensity:
		int or float, macroseismic intensity
	:param agg_type:
		str, type of aggregation, one of:
		- 'id_com' or 'commune'
		- 'id_main' or 'main commune'
		- 'grid_X' (where X is grid spacing in km)
		- None or ''
	:param enq_type:
		str, type of enquirey, one of:
		- 'internet' or 'online'
		- 'official'
	:param num_replies:
		int, number of replies in aggregate
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
	"""
	def __init__(self, id_earth, id_com, intensity, agg_type, enq_type, num_replies=1,
				lon=0, lat=0, residual=0, db_ids=[]):
		self.id_earth = id_earth
		self.id_com = id_com
		self.intensity = intensity
		self.agg_type = agg_type
		self.enq_type = enq_type
		self.num_replies = num_replies
		self.lon = lon
		self.lat = lat
		self.residual = residual
		self.db_ids = db_ids

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

		if isinstance(self.id_earth, (int, str)):
			[eq] = query_local_eq_catalog_by_id(self.id_earth)
			return eq

	def get_web_enquiries(self, min_fiability=20, verbose=False):
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
		from ..rob.seismodb import query_web_macro_enquiries

		if self.enq_type == 'official':
			return

		if self.db_ids:
			ensemble = query_web_macro_enquiries(web_ids=self.db_ids, verbose=verbose)
		else:
			ensemble = query_web_macro_enquiries(self.id_earth, id_com=self.id_com,
								min_fiability=min_fiability, verbose=verbose)

		return ensemble


class MacroInfoCollection():
	"""
	Class representing a collection of aggregated macroseismic records

	:param macro_infos:
		list with instances of :class:`MacroseismicInfo`
	:param agg_type:
		str, aggregation type
	:param enq_type:
		str, type of enquirey, one of:
		- 'internet' or 'online'
		- 'official'
	:proc_info:
		dict, containing processing parameters
		(default: {})
	"""
	def __init__(self, macro_infos, agg_type, enq_type, proc_info={}):
		self.macro_infos = macro_infos
		self.agg_type = agg_type
		if not enq_type:
			enq_type = macro_infos[0].enq_type
		self.enq_type = enq_type
		self.proc_info = proc_info

	def __len__(self):
		return len(self.macro_infos)

	def __iter__(self):
		return self.macro_infos.__iter__()

	def __getitem__(self, item):
		return self.macro_infos.__getitem__(item)

	@property
	def longitudes(self):
		return np.array([rec.lon for rec in self])

	@property
	def latitudes(self):
		return np.array([rec.lat for rec in self])

	@property
	def intensities(self):
		return np.array([rec.I for rec in self])

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

	def to_commune_info_dict(self):
		"""
		Convert to dictionary mapping commune IDs to instances of
		:class:`MacroseismicInfo`
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
			instance of :class:`MacroseismicInfo`
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
			instance of :class:`MacroInfoCollection`
		:return:
			None, 'residual' attribute of instances of :class:`MacroseismicInfo`
			in collection are modified in place
		"""
		for id_com in self.get_commune_ids():
			mi = other_macro_info.get_commune_info(id_com)
			if mi:
				self.residual = self.I - mi.I

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
			lons.extend([rec.lon] * rec.num_replies)
			lats.extend([rec.lat] * rec.num_replies)
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
		attributes = ['id_earth', 'intensity', 'num_replies']
		residuals = [rec.residual for rec in self]
		if not np.allclose(residuals, 0):
			attributes += ['residual']
		attributes += ['agg_type', 'enq_type']
		if aggregate_by == 'grid' or polygons_as_points:
			if aggregate_by in ('id_main', 'id_com'):
				attributes += ['id_com']
			attributes += ['lon', 'lat']
			for attrib in attributes:
				values[attrib] = [getattr(rec, attrib) for rec in self]
		else:
			join_key = "ID_ROB"
			for attrib in attributes:
				values[attrib] = {'key': join_key, 'values':
							{rec.id_com: getattr(rec, attrib) for rec in self}}

		## Select GIS file with commune polygons in function of aggregation type
		#if polygons_as_points:
		#	gis_filename = "Bel_villages_points.TAB"
		gis_filename = ""
		if not polygons_as_points:
			if aggregate_by == 'id_main':
				gis_filename = "Bel_villages_polygons.TAB"
			elif aggregate_by == 'id_com':
				gis_filename = "Bel_communes_avant_fusion.TAB"
		if gis_filename:
			gis_filespec = os.path.join(GIS_FOLDER, gis_filename)
			#gis_filespec = "http://seishaz.oma.be:8080/geoserver/rob/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=rob:bel_villages_polygons&outputFormat=application%2Fjson"

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
		else:
			gis_data = lbm.GisData(gis_filespec, joined_attributes=values)
			_, _, polygon_data = gis_data.get_data()
			macro_geoms = polygon_data

		return macro_geoms

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
				thematic_num_replies=False, interpolate_grid={},
				cmap="rob", color_gradient="discontinuous", event_style="default",
				admin_level="province", admin_style="default", colorbar_style="default",
				radii=[], plot_pie=None, title="", fig_filespec=None,
				ax=None, copyright=u"© ROB", text_box={}, dpi="default",
				verbose=True):
		"""
		Plot macroseismic map

		see :func:`eqcatalog.plot.plot_macro_map.plot_macroseismic_map`
		"""
		from ..plot.plot_macro_map import plot_macroseismic_map

		return plot_macroseismic_map(self, region=region, projection=projection,
				graticule_interval=graticule_interval, plot_info=plot_info,
				int_conversion=int_conversion, symbol_style=symbol_style,
				line_style=line_style, thematic_num_replies=thematic_num_replies,
				interpolate_grid=interpolate_grid, cmap=cmap,
				color_gradient=color_gradient, event_style=event_style,
				admin_level=admin_level, admin_style=admin_style,
				colorbar_style=colorbar_style, radii=radii, plot_pie=plot_pie,
				title=title, fig_filespec=fig_filespec, ax=ax,
				copyright=copyright, text_box=text_box, dpi=dpi,
				verbose=verbose)

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
		elif prop == 'num_replies':
			values = np.array([rec.num_replies for rec in self])
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
		text += "\nAgg. method: %s" % self.proc_info['agg_method']

		if self.enq_type in ('internet', 'online'):
			text += ("\nMin. replies / fiability: %d / %d"
				% (self.proc_info['min_replies'], self.proc_info['min_fiability']))
			text += "\nFilter floors: %s" % str(self.proc_info['filter_floors'])
			text += "\nFix records: %s" % self.proc_info['fix_records']
			include_other_felt = self.proc_info['include_other_felt']
			include_heavy_appliance = self.proc_info['include_heavy_appliance']
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
		else:
			text += "\nMin. fiability: %d" % self.proc_info['min_fiability']
			text += "\nImin_or_max%s" % self.proc_info['min_or_max']

		return text


def get_aggregated_info_web(id_earth, min_replies=3, query_info="cii",
				min_fiability=20, filter_floors=(0, 4), aggregate_by="commune",
				agg_method='mean', fix_records=True,
				include_other_felt=True, include_heavy_appliance=False,
				remove_outliers=(2.5, 97.5)):
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
		(default: 20)
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

	:return:
		instance of :class:`MacroInfoCollection`
	"""
	from ..rob import query_local_eq_catalog_by_id, query_web_macro_enquiries

	query_info = query_info.lower()
	if query_info == 'num_replies':
		min_replies = 1

	if aggregate_by == 'commune':
		aggregate_by = 'id_com'
	elif aggregate_by == 'main commune':
		aggregate_by = 'id_main'
	elif not aggregate_by:
		min_replies = 1

	## Retrieve macroseismic information from database
	[eq] = query_local_eq_catalog_by_id(id_earth)
	dyfi_ensemble = eq.get_macroseismic_enquiries(min_fiability)

	## Aggregate
	macro_info_coll = dyfi_ensemble.get_aggregated_info(aggregate_by, min_replies,
							agg_info=query_info, min_fiability=min_fiability,
							filter_floors=filter_floors, agg_method=agg_method,
							fix_records=fix_records, include_other_felt=include_other_felt,
							include_heavy_appliance=include_heavy_appliance,
							remove_outliers=remove_outliers)

	return macro_info_coll


def get_aggregated_info_official(id_earth, min_fiability=20, min_or_max='max',
				aggregate_by="commune", agg_method="average", min_val=1):
	"""
	Obtain aggregated official macroseismic information for given earthquake

	:param id_earth:
		int or str, ID of earthquake in ROB database
	:param min_fiability:
		int, minimum fiability of macroseismic record
		(default: 20)
	:param min_or_max:
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
		is 'main commune', one of "minimum", "maximum" or "average"/"mean"
		(default: "average")
	:param min_val:
		float, minimum intensity to return
		(default: 1, avoids getting NULL results)

	:return:
		instance of :class:`MacroInfoCollection`
	"""
	from ..rob import query_local_eq_catalog_by_id

	if aggregate_by in ('main commune', 'id_main'):
		group_by_main_village = True
	else:
		group_by_main_village = False

	## Retrieve macroseismic information from database
	[eq] = query_local_eq_catalog_by_id(id_earth)

	macro_info_coll = eq.get_macroseismic_data_aggregated_official(min_or_max=min_or_max,
			min_val=min_val, group_by_main_village=group_by_main_village,
			agg_method=agg_method, min_fiability=min_fiability)

	## Remove records without location
	for i in range(len(macro_info_coll)-1, -1, -1):
		macro_info = macro_info_coll[i]
		if macro_info.lon is None:
			print("Commune #%s has no location!" % macro_info.id_com)
			macro_info_coll.macro_infos.pop(i)

	return macro_info_coll
