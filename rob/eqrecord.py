"""
ROB-specific extensions of eqrecord classes
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

from ..eqrecord import LocalEarthquake


__all__ = ["DEFAULT_MRELATIONS", "ROBLocalEarthquake"]


DEFAULT_MRELATIONS = {
	'ML': {},
	'MS': {"ML": "Ambraseys1985"},
	'MW': OrderedDict([("MS", "Geller1976"), ("ML", "Ahorner1983")])
}


class ROBLocalEarthquake(LocalEarthquake):

	@property
	def id_earth(self):
		return self.ID

	def get_MW(self, Mrelation='default'):
		if Mrelation == 'default':
			Mrelation = DEFAULT_MRELATIONS['MW']
		return super(ROBLocalEarthquake, self).get_MW(Mrelation)
	get_MW.__doc__ = LocalEarthquake.get_MW.__doc__

	def get_MS(self, Mrelation='default'):
		if Mrelation == 'default':
			Mrelation = DEFAULT_MRELATIONS['MS']
		return super(ROBLocalEarthquake, self).get_MS(Mrelation)
	get_MS.__doc__ = LocalEarthquake.get_MS.__doc__

	def get_or_convert_mag(self, Mtype, Mrelation='default'):
		if Mrelation == 'default':
			Mrelation = DEFAULT_MRELATIONS.get(Mtype, {})
		return super(ROBLocalEarthquake, self).get_or_convert_mag(Mtype, Mrelation)
	get_or_convert_mag.__doc__ = LocalEarthquake.get_or_convert_mag.__doc__

	def get_rob_hash(self):
		"""
		Generate hash from earthquake ID as used on ROB website

		:return:
			string, hash
		"""
		import hashids
		salt = "8dffaf6e-fb3a-11e5-86aa-5e5517507c66"
		min_length = 9
		alphabet = "abcdefghijklmnopqrstuvwxyz1234567890"
		hi = hashids.Hashids(salt=salt, min_length=min_length, alphabet=alphabet)
		hash = hi.encode(self.ID)
		return hash

	def get_aggregated_online_macro_info(self, min_replies=3, query_info="cii",
					min_fiability=80, filter_floors=(0,4),
					aggregate_by='commune', agg_method="mean", fix_records=True,
					include_other_felt=True, include_heavy_appliance=False,
					remove_outliers=(2.5, 97.5), verbose=False):
		"""
		Get online macroseismic information (possibly aggregated by
		main commune) for this earthquake

		:param min_replies:
			int, minimum number of replies
			(default: 3)
		:param query_info:
			str, either "cii", "cdi" or "mi"
			(default: "cii")
		:param min_fiability:
			float, minimum fiability of enquiry
			(default: 80)
		:param filter_floors:
				(min_floor, max_floor) tuple, floors outside this range
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
			either 'mean' (= ROB practice) or 'dyfi' (= DYFI practice)
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
		kwargs = locals().copy()
		kwargs.pop('self')
		#from .seismodb import query_online_macro_catalog_aggregated
		#return query_online_macro_catalog_aggregated(self.ID, min_replies=min_replies,
		#			query_info=query_info, min_val=min_val, min_fiability=min_fiability,
		#			group_by_main_commune=group_by_main_commune, filter_floors=filter_floors,
		#			agg_method=agg_method, verbose=verbose)
		from ..macro import aggregate_online_macro_info
		return aggregate_online_macro_info(self.ID, **kwargs)

	def get_aggregated_traditional_macro_info(self, id_com=None, data_type='',
			min_or_max='max', min_fiability=80, aggregate_by="commune",
			agg_method="mean", verbose=False):
		"""
		Get traditional (historical / official) macroseismic information
		(possibly aggregated by main commune) for this earthquake

		:param id_com:
			int, commune ID
			or list of ints
			(default: None)
		:param data_type:
			str, type of macroseismic data: '', 'official' or 'historical'
			(default: '')
		:param min_or_max:
			str, one of 'min', 'mean' or 'max' to select between
			intensity_min and intensity_max values in database
			(default: 'max')
		:param min_fiability:
			float, minimum fiability of enquiry
			(default: 80)
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
		:param verbose:
			Bool, if True the query string will be echoed to standard output

		:return:
			instance of :class:`AggregatedMacroInfoCollection`
		"""
		kwargs = locals().copy()
		kwargs.pop('self')
		#from .seismodb import query_traditional_macro_catalog_aggregated
		#return query_traditional_macro_catalog_aggregated(self.ID, **kwargs)
		from ..macro import aggregate_traditional_macro_info
		return aggregate_traditional_macro_info(self.ID, **kwargs)

	def get_aggregated_official_macro_info(self, id_com=None, min_or_max='max',
			aggregate_by="commune", agg_method="mean", min_fiability=80,
			verbose=False):
		"""
		Get official macroseismic information (possibly aggregated by
		main commune) for this earthquake.
		This is a wrapper for :meth:`get_aggregated_traditional_macro_info`
		"""
		kwargs = locals().copy()
		kwargs.pop('self')
		return self.get_aggregated_traditional_macro_info(data_type='official', **kwargs)

	def get_aggregated_historical_macro_info(self, id_com=None, min_or_max='max',
			aggregate_by="commune", agg_method="mean", min_fiability=80,
			verbose=False):
		"""
		Get historical macroseismic information (possibly aggregated by
		main commune) for this earthquake.
		This is a wrapper for :meth:`get_aggregated_traditional_macro_info`
		"""
		kwargs = locals().copy()
		kwargs.pop('self')
		return self.get_aggregated_traditional_macro_info(data_type='historical', **kwargs)

	def get_traditional_macroseismic_info(self, id_com=None, data_type='',
									group_by_main_commune=False, min_fiability=80,
									verbose=False, errf=None):
		"""
		Get traditional macroseismic information for this earthquake

		:param id_com:
			int, commune ID
			or list of ints
			(default: None)
		:param data_type:
			str, type of macroseismic data: '', 'official' or 'historical'
			(default: '')
		:param group_by_main_commune:
			bool, whether or not to group the results by main village
			(default: False)
		:param min_fiability:
			float, minimum fiability of enquiry
			(default: 80)
		:param verbose:
			Bool, if True the query string will be echoed to standard output
		:param errf:
			File object, where to print errors

		:return:
			instance of :class:`eqcatalog.macro.MDPCollection`
		"""
		kwargs = locals().copy()
		kwargs.pop('self')
		from .seismodb import query_traditional_macro_catalog
		return query_traditional_macro_catalog(self.ID, **kwargs)

	def get_official_macroseismic_info(self, id_com=None,
								group_by_main_commune=False, min_fiability=80,
								verbose=False, errf=None):
		"""
		Get official macroseismic information for this earthquake
		This is a wrapper for :meth:`get_traditional_macroseismic_info`
		"""
		kwargs = locals().copy()
		kwargs.pop('self')
		return self.get_traditional_macroseismic_info(data_type='official', **kwargs)

	def get_historical_macroseismic_info(self, id_com=None,
								group_by_main_commune=False, min_fiability=80,
								verbose=False, errf=None):
		"""
		Get historical macroseismic information for this earthquake
		This is a wrapper for :meth:`get_traditional_macroseismic_info`
		"""
		kwargs = locals().copy()
		kwargs.pop('self')
		return self.get_traditional_macroseismic_info(data_type='historical', **kwargs)

	def get_isoseismal_macroseismic_info(self, group_by_main_commune=False,
										as_points=True):
		"""
		Get isoseismal macroseismic information for this earthquake

		:param main_communes:
			bool, whether or not to use main communes
			(default: False)
		:param as_points:
			bool, whether  points (True) or polygons (False) should
			be used for testing whether communes are within a particular
			isoseismal
			(default: True)

		:return:
			dict, mapping commune IDs to intensities
		"""
		from ..macro.isoseismal import get_commune_intensities_from_isoseismals
		return get_commune_intensities_from_isoseismals(self.ID,
						main_communes=group_by_main_commune, as_points=as_points)

	def get_online_macro_enquiries(self, min_fiability=80, min_location_quality=6,
									verbose=False):
		"""
		Get all online macroseismic enquiries for this earthquake

		:param min_fiability:
			float, minimum fiability of enquiry
			(default: 80)
		:param min_location_quality:
			int, minimum quality of location to read from web_location table
			(default: 6)
		:param verbose:
			bool, if True the query string will be echoed to standard output
			(default: False)

		:return:
			instance of :class:`MacroseismicEnquiryEnsemble`
		"""
		from .seismodb import query_online_macro_catalog
		ensemble = query_online_macro_catalog(self.ID, min_fiability=min_fiability,
										min_location_quality=min_location_quality,
										verbose=verbose)
		return ensemble

	def get_num_online_macro_enquiries(self, min_fiability=80):
		"""
		Count number of online macroseismic enquiries for this earthquake

		:param min_fiability:
			float, minimum fiability of enquiries
			(default: 80)

		:return:
			int, number of enquiries
		"""
		from .seismodb import get_num_online_macro_enquiries
		[num_enquiries] = get_num_online_macro_enquiries([self.ID], min_fiability)
		return num_enquiries

	def get_num_official_macro_enquiries(self):
		"""
		Count number of official macroseismic enquiries  for this earthquake

		:return:
			int, number of enquiries
		"""
		from .seismodb import get_num_official_enquiries
		[num_enquiries] = get_num_official_enquiries([self.ID])
		return num_enquiries

	def get_Imax_online(self, min_replies=3, min_fiability=80.0, filter_floors=(0,4),
					aggregate_by='commune', agg_method="mean", fix_records=True,
					include_other_felt=True, include_heavy_appliance=False,
					remove_outliers=(2.5, 97.5), verbose=False):
		"""
		Report maximum intensity from online enquiry in any commune
		for this earthquake

		:param min_replies:
		:param min_fiability:
		:param filter_floors:
		:param aggregate_by:
		:param agg_method:
		:param fix_records:
		:param include_other_felt:
		:param include_heavy_appliance:
		:param remove_outliers:
		:param verbose:
			see :meth:`get_aggregated_online_macro_info`

		:return:
			float, Imax
		"""
		kwargs = locals().copy()
		kwargs.pop('self')
		macro_info_col = self.get_aggregated_online_macro_info(query_info="cii",
																**kwargs)
		if len(macro_info_col):
			_, Imax = macro_info_col.Iminmax()
		else:
			Imax = 0

		# TODO: add recalc argument and use get_aggregated_info method of dyfi
		#dyfi = self.get_macroseismic_enquiries(min_fiability)
		#Imax = 0
		#if len(dyfi):
		#	for id_com, com_ensemble in dyfi.aggregate_by_commune().items():
		#		if len(com_ensemble) >= min_replies:
		#			I = com_ensemble.calc_mean_cii(filter_floors=filter_floors,
		#					include_other_felt=include_other_felt,
		#					include_heavy_appliance=include_heavy_appliance,
		#					remove_outliers=remove_outliers)
		#			if I and I > Imax:
		#				Imax = I
		return Imax

	def get_Imax_traditional(self, data_type='', min_or_max='max',
			min_fiability=80, aggregate_by="commune", agg_method="mean",
			verbose=False):
		"""
		Report maximum traditional intensity in any commune for this earthquake.

		:param data_type:
		:param min_or_max:
		:param min_fiability:
		:param aggregate_by:
		:param agg_method:
		:param verbose:
			see :meth:`get_aggregated_traditional_macro_info`

		:return:
			float, Imax
		"""
		kwargs = locals().copy()
		kwargs.pop('self')
		macro_info_col = self.get_aggregated_traditional_macro_info(**kwargs)
		if len(macro_info_col):
			_, Imax = macro_info_col.Iminmax()
		else:
			Imax = 0
		return Imax

	def get_Imax_official(self, min_or_max='max', min_fiability=80,
				aggregate_by="commune", agg_method="mean", verbose=False):
		"""
		Report maximum official intensity in any commune for this earthquake.
		This is a wrapper for :meth:`get_Imax_traditional`
		"""
		kwargs = locals().copy()
		kwargs.pop('self')
		return self.get_Imax_traditional(data_type='official', **kwargs)

	def get_Imax_historical(self, min_or_max='max', min_fiability=80,
				aggregate_by="commune", agg_method="mean", verbose=False):
		"""
		Report maximum historical intensity in any commune for this earthquake.
		This is a wrapper for :meth:`get_Imax_traditional`
		"""
		kwargs = locals().copy()
		kwargs.pop('self')
		return self.get_Imax_traditional(data_type='historical', **kwargs)

	def get_Imax_isoseismal(self):
		"""
		Report maximum isoseismal intensity for this earthquake
		"""
		import numpy as np
		from ..macro.isoseismal import read_isoseismals

		pg_data = read_isoseismals(self.ID, filled=True)
		if pg_data:
			return np.max(pg_data.values['Intensity'])
		else:
			return 0

	def get_Imax(self, min_fiability=80,
				min_or_max='mean',
				min_replies=3, filter_floors=(0, 4),
				agg_method_online='mean', fix_records=True, include_other_felt=True,
				include_heavy_appliance=False, remove_outliers=(2.5, 97.5),
				by_main_commune=False, agg_subcommunes='mean',
				verbose=False):
		"""
		Report maximum intensity (online or traditional) in any commune
		for this earthquake

		See :meth:`get_Imax_online` and :meth:`get_Imax_traditional`

		:return:
			float, Imax
		"""
		## Note: same min_fiability for traditional and online
		aggregate_by = {True: 'main commune', False: 'commune'}[by_main_commune]
		Imax_traditional = self.get_Imax_traditional(min_fiability=min_fiability,
								min_or_max=min_or_max, aggregate_by=by_main_commune,
								agg_method=agg_subcommunes, verbose=verbose)
		Imax_online = self.get_Imax_online(min_replies=min_replies,
							min_fiability=min_fiability, filter_floors=filter_floors,
							aggregate_by=aggregate_by, agg_method=agg_method_online,
							fix_records=fix_records, include_other_felt=include_other_felt,
							include_heavy_appliance=include_heavy_appliance,
							remove_outliers=remove_outliers)

		return max(Imax_traditional, Imax_online)

	def get_focal_mechanism(self, verbose=False):
		"""
		Get focal mechanism for this earthquake

		:return:
			instance of :class:`FocMecRecord`
		"""
		from .seismodb import query_focal_mechanisms
		try:
			return query_focal_mechanisms(id_earth=self.ID, verbose=verbose)[0]
		except IndexError:
			return None

	def get_phase_picks(self, station_code=None, network=None, verbose=False):
		"""
		Get phase picks for this earthquake

		:param station_code:
			str, station code
			(default: None)
		:param network:
			str, network code
			(default: None)
		:param verbose:
			bool, if True the query string will be echoed to standard output
			(default: False)

		:return:
			dict, mapping station codes to dicts, mapping phase names
			to PhasePick objects
			or dict, mapping phase names to PhasePick objects
			if :param:`station_code` is not None
		"""
		import datetime
		from .seismodb import query_phase_picks
		from robspy import UTCDateTime
		from robspy.phase_pick import PhasePick

		recs = query_phase_picks(self.ID, station_code=station_code,
								network=network, verbose=verbose)
		picks = {}
		for rec in recs:
			phase_name = rec['name']
			dt = rec['datetime'] + datetime.timedelta(microseconds=rec['hund'] * 1E+4)
			dt = UTCDateTime(dt)
			component = rec['comp'] if rec['comp'] != 'V' else 'Z'
			## Note: network may sometimes contain extra spaces...
			pick = PhasePick(phase_name, dt, self.ID, rec['station_code'],
							component, rec['movement'], rec['id_mesure_t'],
							rec['include_in_loc'], rec['amplitude'],
							rec['periode'], rec['magnitude'], rec['mag_type'],
							rec['distance'], station_network=rec['network'].strip())
			_station_code = rec['station_code']
			if station_code:
				## Trim record station code to length of given station code
				_station_code = _station_code[:len(station_code)]
			if not _station_code in picks:
				picks[_station_code] = {}
			picks[_station_code][rec['name']] = pick

		if station_code:
			return picks.get(station_code, {})
		else:
			return picks

	def get_theoretical_phase_arrivals(self, station_code, velocity_model="iasp91"):
		"""
		:param station_code:
			str, 3- or 4-character code of seismic station
		:param velocity_model:
			str, name of velocity model understood by obspy.taup,
			e.g., "iasp91", "ak135"
			(default: "iasp91")

		:return:
			dict, mapping phase names to obspy Arrival objects
		"""
		import mapping.geotools.geodetic as geodetic
		from obspy.taup import TauPyModel
		from .seismodb import get_station_coordinates

		stat_lon, stat_lat, stat_alt = get_station_coordinates(station_code, include_z=True)
		dist_deg = geodetic.spherical_distance(self.lon, self.lat, stat_lon, stat_lat)
		## Station depth must not be negative
		stat_depth = max(0, -stat_alt / 1000.)

		m = TauPyModel(model="ak135")
		arrivals = m.get_travel_times(distance_in_degree=dist_deg,
			source_depth_in_km=self.depth, receiver_depth_in_km=stat_depth)
		return {arr.name: arr for arr in arrivals}

	def calc_MLbg(self, distance_metric='hypocentral', verbose=False):
		"""
		Compute local magnitude (MLbg) from phase pick amplitudes

		:param distance_metric:
			str, distance metric to use, either 'epi[central]' or
			'hypo[central]'
			(default: 'hypocentral')
		:param verbose:
			bool, whether or not to print magnitude determinations
			for individual stations
			(default: False)

		:return:
			(ML_mean, ML_sigma) tuple of floats
		"""
		import numpy as np
		from robspy.rob import get_stations, calc_MLbg

		phase_picks = self.get_phase_picks()
		ML_estimates = []
		for stat_code in sorted(phase_picks.keys()):
			stat_phase_picks = phase_picks[stat_code]
			if 'S' in stat_phase_picks:
				Spick = stat_phase_picks['S']
				if Spick.amplitude:
					station = get_stations(stat_code)[0]
					station_coords = station.get_coordinates()
					lon = station_coords['longitude']
					lat = station_coords['latitude']
					z = -station_coords['elevation'] / 1000.
					if distance_metric[:4] == 'hypo':
						distance = self.hypocentral_distance((lon, lat, z))
					elif distance_metric[:3] == 'epi':
						distance = self.epicentral_distance((lon, lat, z))
					MLbg = calc_MLbg(Spick.amplitude/1000., distance)
					ML_estimates.append(MLbg)
					if verbose:
						msg = "%s:\tA=%.1f nm\tD=%.0f km\tML=%.2f"
						msg %= (stat_code, Spick.amplitude, distance, MLbg)
						print(msg)

		ML_mean, ML_sigma = np.mean(ML_estimates), np.std(ML_estimates)
		if verbose:
			msg = "AVG:\tML=%.2f +/- %.2f"
			msg %= (ML_mean, ML_sigma)
			print(msg)

		return (ML_mean, ML_sigma)
