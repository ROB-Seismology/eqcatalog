"""
Determine max. intensity reported in all Belgian communes
"""

from __future__ import absolute_import, division, print_function#, unicode_literals
from builtins import int


import numpy as np

from ..rob import seismodb
from .macro_info import MacroseismicInfo, MacroInfoCollection


__all__ = ["get_eq_intensities_for_commune_web",
			"get_eq_intensities_for_commune_official",
			"get_isoseismal_intensities_for_all_communes",
			"get_all_commune_intensities",
			"get_imax_by_commune"]


#TODO: count / plot number of exceedances of particular intensity

def get_eq_intensities_for_commune_web(id_com, as_main_commune=False,
				min_replies=3, min_fiability=20, filter_floors=(0, 4),
				agg_method='mean', fix_records=True,
				include_other_felt=True, include_heavy_appliance=False,
				remove_outliers=(2.5, 97.5)):
	"""
	Get list of all internet intensities due to known earthquakes
	for a given commune

	:param id_com:
		int, ID of commune in database
	:param as_main_commune:
		bool, whether or not to group subcommunes belonging to a
		main commune
		(default: False)
	:param min_replies:
		int, minimum number of replies
		(default: 3)
	:param min_fiability:
		int, minimum fiability of enquiry
		(default: 20)
	:param filter_floors:
		(min_floor, max_floor) tuple, floors outside this range
		(basement floors and upper floors) are filtered out
		(default: False)
	:param agg_method:
		str, how to aggregate individual enquiries in a subcommune,
		either 'mean' (= ROB practice) or 'aggregated' (= DYFI practice)
		(default: 'mean')
	:param fix_records:
		bool, whether or not to fix various issues (see :meth:`fix_all`)
		(default: True)
	:param include_other_felt:
		bool, whether or not to include the replies to the question
		"Did others nearby feel the earthquake ?"
		(default: False)
	:param include_heavy_appliance:
		bool, whether or not to take heavy_appliance into account
		as well (not standard, but occurs with ROB forms)
		(default: False)
	:param remove_outliers:
		(min_pct, max_pct) tuple, percentile range to use
		Only applies if :param:`agg_method` = 'aggregated'
		(default: 2.5, 97.5)

	:return:
		dict mapping earthquake IDs to intensities
	"""
	if as_main_commune:
		subcommunes = seismodb.get_subcommunes(id_com)
		if len(subcommunes) == 0:
			raise Exception("Commune #%d is not a main commune" % id_com)
		## Use zip_code for query, as it is likely more reliable than id_com
		## (should work for enquiries where commune assignment has failed)
		zip_code = [r['code_p'] for r in subcommunes]
		id_com = None
	else:
		zip_code = None

	eq_intensities = {}
	dyfi = seismodb.query_web_macro_enquiries('ke', id_com=id_com,
						zip_code=zip_code, min_fiability=min_fiability)
	if fix_records and len(dyfi):
		dyfi = dyfi.fix_all()
	if len(dyfi):
		all_eq_ids = dyfi.get_eq_ids()
		unique_eq_ids = np.unique(all_eq_ids)
		## Only keep enquiries for earthquakes since 2002 Alsdorf earthquake
		unique_eq_ids = unique_eq_ids[unique_eq_ids >= 1306]
		for id_earth in unique_eq_ids:
			eq_dyfi = dyfi[all_eq_ids == id_earth]
			if filter_floors:
				eq_dyfi = eq_dyfi.filter_floors(*filter_floors)
			if len(eq_dyfi) >= min_replies:
				if agg_method == 'aggregated':
					I = eq_dyfi.calc_cii(filter_floors=False,
						include_other_felt=include_other_felt,
						include_heavy_appliance=include_heavy_appliance)
				else:
					I = eq_dyfi.calc_mean_cii(filter_floors=False,
						include_other_felt=include_other_felt,
						include_heavy_appliance=include_heavy_appliance,
						remove_outliers=remove_outliers)
				eq_intensities[id_earth] = I

	return eq_intensities


def get_eq_intensities_for_commune_official(id_com, as_main_commune=False,
							min_or_max='mean', min_fiability=20):
	"""
	Get list of all official intensities due to known earthquakes
	for a given commune

	:param id_com:
		int, ID of commune in database
	:param as_main_commune:
		bool, whether or not to group subcommunes belonging to a
		main commune
		(default: False)
	:param min_or_max:
		str, one of 'min', 'mean' or 'max' to select between
		intensity_min and intensity_max values in database
		(default: 'mean')
	:param min_fiability:
		int, minimum fiability of enquiry
		(default: 20)

	:return:
		dict mapping earthquake IDs to intensities
		(if :param:`as_main_commune` is False)
		or to lists of intensities (if :param:`as_main_commune` is True)
	"""
	if as_main_commune:
		subcommunes = seismodb.get_subcommunes(id_com)
		if len(subcommunes) == 0:
			raise Exception("Commune #%d is not a main commune" % id_com)
		id_com_str = ','.join(['%d' % sc['id'] for sc in subcommunes])
	else:
		id_com_str = id_com

	table_clause = 'macro_detail'
	where_clause = 'id_com IN (%s) AND fiability >= %d'
	where_clause %= (id_com_str, min_fiability)
	macro_recs = seismodb.query_seismodb_table(table_clause, where_clause=where_clause)
	eq_intensities = {}
	for mrec in macro_recs:
		id_earth = mrec['id_earth']
		Imin, Imax = mrec['intensity_min'], mrec['intensity_max']
		## Do not take into account Imin/Imax = 13 values
		if Imin == 13:
			continue
		elif Imax == 13:
			Imax = Imin

		I = {'min': Imin, 'max': Imax, 'mean': np.mean([Imin, Imax])}[min_or_max]
		if id_earth in eq_intensities:
			eq_intensities[id_earth].append(I)
		else:
			eq_intensities[id_earth] = [I]

	return eq_intensities


def get_isoseismal_intensities_for_all_communes(as_main_commune=False,
												as_points=True):
	"""
	Get intensities for all communes from available isoseismals

	:param as_main_commune:
		bool, whether or not to group subcommunes belonging to a
		main commune
		(default: False)
	:param as_points:
		bool, whether  points (True) or polygons (False) should
		be used for testing whether communes are within a particular
		isoseismal
		(default: True)

	:return:
		dict, mapping commune IDs to dicts, in turn mapping
		earthquake IDs to intensities
	"""
	from .isoseismal import (get_commune_intensities_from_isoseismals,
							get_available_isoseismals)
	commune_eq_intensities = {}
	for eq_id in get_available_isoseismals():
		comm_intensities = get_commune_intensities_from_isoseismals(eq_id,
							main_communes=as_main_commune, as_points=as_points)
		for id_com, I in comm_intensities.items():
			if not id_com in commune_eq_intensities:
				commune_eq_intensities[id_com] = {eq_id: I}
			else:
				commune_eq_intensities[id_com][eq_id] = I
	return commune_eq_intensities


def parse_enq_type(enq_type):
	"""
	Parse enquiry type specification

	:param enq_type:
		list or str

	:return:
		list
	"""
	if isinstance(enq_type, list):
		enq_types = enq_type
	elif isinstance(enq_type, str):
		if enq_type == 'all':
			enq_types = ['internet', 'official', 'isoseismal']
		else:
			enq_types = enq_type.split('+')
		for e in range(len(enq_types)):
			if enq_types[e] in ('online', 'dyfi', 'web'):
				enq_types[e] = 'internet'

	return enq_types


def get_all_commune_intensities(enq_type='all',
				min_fiability=20,
				min_or_max='mean',
				min_replies=3, filter_floors=(0, 4),
				agg_method_web='mean', fix_records=True, include_other_felt=True,
				include_heavy_appliance=False, remove_outliers=(2.5, 97.5),
				by_main_commune=False, agg_subcommunes='mean',
				verbose=False):
	"""
	Get intensities in all communes by all earthquakes

	:param enq_type:
		str, one of "internet", "official", "isoseismal",
		"internet+official", "official+isoseismal" or "all"
		(default: 'all')
	:param min_fiability:
		int, minimum fiability of internet or official enquiry
		(default: 20)
	:param min_or_max:
		see :func:`get_eq_intensities_for_commune_official`
	:param min_replies:
	:param filter_floors:
	:param agg_method_web:
	:param fix_records:
	:param include_other_felt:
	:param include_heavy_appliance:
	:param remove_outliers:
		see :func:`get_eq_intensities_for_commune_web`
	:param by_main_commune:
		bool, whether or not to aggregate communes by main commune
		(default: False)
	:param agg_subcommunes:
		str, name of numpy function for aggregation of historical
		data into subcommunes if :param:`by_main_commune` is True
		One of 'mean', 'minimum', 'maximum'
		(default: 'mean')
	:param verbose:
		bool, whether or not to print progress information
		(default: False)

	:return:
		dict, mapping commune IDs to dicts, in turn mapping
		earthquake IDs to intensities
	"""
	agg_func = getattr(np, agg_subcommunes.lower())

	## Fetch communes from database
	comm_recs = seismodb.get_communes('BE', main_communes=by_main_commune)

	enq_types = parse_enq_type(enq_type)

	if 'isoseismal' in enq_types:
		comm_iso_intensities = get_isoseismal_intensities_for_all_communes(
											by_main_commune, as_points=False)
		#if 'official' in enq_types:
		#	comm_iso_intensities.pop(509)
		#	comm_iso_intensities.pop(987)

	commune_eq_intensities = {}
	for rec in comm_recs:
		id_com = rec['id']
		commune_eq_intensities[id_com] = {}

		## Isoseismals
		if 'isoseismal' in enq_types:
			eq_intensities = comm_iso_intensities.get(id_com, {})
			commune_eq_intensities[id_com] = eq_intensities

		## Official / Historical macroseismic information
		if 'official' in enq_types:
			eq_intensities = get_eq_intensities_for_commune_official(id_com,
							as_main_commune=by_main_commune, min_or_max=min_or_max,
							min_fiability=min_fiability)
			for id_earth in eq_intensities.keys():
				I = agg_func(eq_intensities[id_earth])
				## Isoseismal intensities will be overwritten
				commune_eq_intensities[id_com][id_earth] = I

		## Online enquiries
		if 'internet' in enq_types:
			eq_intensities = get_eq_intensities_for_commune_web(id_com,
				as_main_commune=by_main_commune, min_replies=min_replies,
				min_fiability=min_fiability, filter_floors=filter_floors,
				agg_method=agg_method_web, fix_records=fix_records,
				include_other_felt=include_other_felt,
				include_heavy_appliance=include_heavy_appliance,
				remove_outliers=remove_outliers)
			for id_earth in eq_intensities.keys():
				I = eq_intensities[id_earth]
				## If there is already an intensity value for this
				## earthquake and commune, take the maximum
				I = max(I, commune_eq_intensities[id_com].get(id_earth, 0))
				commune_eq_intensities[id_com][id_earth] = I

	return commune_eq_intensities


def get_imax_by_commune(enq_type='all',
				min_fiability=20,
				min_or_max='mean',
				min_replies=3, filter_floors=(0, 4),
				agg_method_web='mean', fix_records=True, include_other_felt=True,
				include_heavy_appliance=False, remove_outliers=(2.5, 97.5),
				by_main_commune=False, agg_subcommunes='mean',
				verbose=False):
	"""
	Determine historical Imax for every commune in Belgium

	:param enq_type:
		str, one of "internet", "official", "isoseismal",
		"internet+official", "official+isoseismal" or "all"
		(default: 'all')
	:param min_fiability:
		int, minimum fiability of internet or official enquiry
		(default: 20)
	:param min_or_max:
		see :func:`get_eq_intensities_for_commune_official`
	:param min_replies:
	:param filter_floors:
	:param agg_method_web:
	:param fix_records:
	:param include_other_felt:
	:param include_heavy_appliance:
	:param remove_outliers:
		see :func:`get_eq_intensities_for_commune_web`
	:param by_main_commune:
		bool, whether or not to aggregate communes by main commune
		(default: False)
	:param agg_subcommunes:
		str, name of numpy function for aggregation of historical
		data into subcommunes if :param:`by_main_commune` is True
		One of 'mean', 'minimum', 'maximum'
		(default: 'mean')
	:param verbose:
		bool, whether or not to print progress information
		(default: False)

	:return:
		instance of :class:`MacroInfoCollection`
	"""
	## Fetch communes from database
	comm_recs = seismodb.get_communes('BE', main_communes=by_main_commune)

	## Fetch all intensities
	commune_eq_intensities = get_all_commune_intensities(enq_type=enq_type,
				min_fiability=min_fiability, min_or_max=min_or_max,
				min_replies=min_replies, filter_floors=filter_floors,
				agg_method_web=agg_method_web, fix_records=fix_records,
				include_other_felt=include_other_felt,
				include_heavy_appliance=include_heavy_appliance,
				remove_outliers=remove_outliers, by_main_commune=by_main_commune,
				agg_subcommunes=agg_subcommunes, verbose=verbose)

	agg_func = getattr(np, agg_subcommunes.lower())

	enq_types = parse_enq_type(enq_type)
	enq_type = '+'.join(enq_types)

	macro_infos = []
	for rec in comm_recs:
		id_com = rec['id']
		lon, lat = rec['longitude'], rec['latitude']
		name = rec['name']

		eq_ids = commune_eq_intensities[id_com].keys()
		intensities = commune_eq_intensities[id_com].values()

		num_replies = len(eq_ids)
		if num_replies:
			Imax = np.max(intensities)
		else:
			Imax = 0

		## Construct MacroseismicInfo
		if Imax > 0:
			id_earth = sorted(eq_ids)
			agg_type = 'id_main' if by_main_commune else 'id_com'
			db_ids = []
			macro_info = MacroseismicInfo(id_earth, id_com, Imax, agg_type,
										enq_type, num_replies, lon, lat, db_ids)
			macro_infos.append(macro_info)

			if verbose:
				msg = '%d (%s): I=%d (n=%d)'
				msg %= (id_com, name, Imax, num_replies)
				print(msg)

		proc_info = dict(min_fiability=min_fiability)
		if 'internet' in enq_types:
			proc_info['agg_method'] = agg_method_web
			proc_info['min_replies'] = min_replies
			proc_info['filter_floors'] = filter_floors
			proc_info['fix_records'] = fix_records
			proc_info['include_other_felt'] = include_other_felt
			proc_info['include_heavy_appliance'] = include_heavy_appliance
			proc_info['remove_outliers'] = remove_outliers
		if 'official' in enq_types:
			## Note that agg_method of DYFI data will be overwritten,
			## but historical data are more important for this type of map
			proc_info['agg_method'] = agg_func
			proc_info['min_or_max'] = min_or_max
		macro_info_coll = MacroInfoCollection(macro_infos, agg_type, enq_type,
											proc_info=proc_info)

	return macro_info_coll


def get_num_exceedances_by_commune(Imin, enq_type='all',
				min_fiability=20,
				min_or_max='mean',
				min_replies=3, filter_floors=(0, 4),
				agg_method_web='mean', fix_records=True, include_other_felt=True,
				include_heavy_appliance=False, remove_outliers=(2.5, 97.5),
				by_main_commune=False, agg_subcommunes='mean',
				verbose=False):
	"""
	Determine number of exceedances of a particular intensity level
	in every commune in Belgium
	"""
	pass
