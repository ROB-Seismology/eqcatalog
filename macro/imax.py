"""
Determine max. intensity reported in all Belgian communes
"""

from __future__ import absolute_import, division, print_function#, unicode_literals
from builtins import int


import numpy as np

import ..rob.seismodb as seismodb
from .macro_info import MacroseismicInfo, MacroInfoCollection


__all__ = ["get_eq_intensities_for_commune_web",
			"get_eq_intensities_for_commune_official",
			"get_eq_intensities_for_commune"]


def get_eq_intensities_for_commune_web(id_com, as_main_commune=False, min_replies=3,
				min_fiability=20, filter_floors=(0, 4), include_other_felt=True,
				include_heavy_appliance=False, remove_outliers=(2.5, 97.5),
				recalc=False):
	"""
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
	if len(dyfi):
		all_eq_ids = dyfi.get_eq_ids()
		unique_eq_ids = np.unique(all_eq_ids)
		for id_earth in unique_eq_ids:
			eq_dyfi = dyfi[all_eq_ids == id_earth]
			if filter_floors:
				eq_dyfi = eq_dyfi.filter_floors(*filter_floors)
			if len(eq_dyfi) >= min_replies:
				if recalc:
					# TODO: fix_all
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
							min_or_max='mean', min_replies=3, min_fiability=20):
	"""
	:return:
		dict mapping earthquake IDs to lists of intensities
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


# TODO: agg_subcommune_func
def get_Imax_by_commune(enq_type='all', min_or_max='mean', min_replies=3,
				min_fiability=20, filter_floors=(0, 4), include_other_felt=True,
				include_heavy_appliance=False, remove_outliers=(2.5, 97.5),
				by_main_commune=False, agg_subcommunes='mean',
				recalc_web=False, verbose=False):
	"""
	Determine historical Imax for every commune in Belgium

	:param enq_type:
		str, one of "internet", "official", "all"
		(default: 'all')
	:param min_or_max:
		see :func:`seismodb.query_official_macro_catalog`
	:param min_replies:
	:param min_fiability:
	:param filter_floors:
	:param include_other_felt:
	:param include_heavy_appliance:
	:param remove_outliers:
		see :func:`seismodb.query_web_macro_enquiries`
	:param by_main_commune:
		bool, whether or not to aggregate communes by main commune
		(default: False)
	:param agg_subcommunes:
		str, name of numpy function for aggregation of subcommunes if
		:param:`by_main_commune` is True.
		One of 'mean', 'minimum', 'maximum'
		(default: 'mean')
	:param verbose:
		bool, whether or not to print progress information
		(default: False)

	:return:
		dict mapping commune IDs to instances of :class:`MacroseismicInfo`
	"""
	agg_func = getattr(np, agg_subcommunes.lower())

	## Fetch communes from database
	table_clause = 'communes'
	where_clause = 'country = "BE"'
	if by_main_commune:
		where_clause += ' AND id = id_main'
	comm_recs = seismodb.query_seismodb_table(table_clause, where_clause=where_clause)

	#comm_macro_dict = {}
	macro_infos = []
	for rec in comm_recs:
		#if rec['id'] != 6:
		#	continue
		id_com = rec['id']
		lon, lat = rec['longitude'], rec['latitude']
		name = rec['name']

		## Online enquiries
		Imax_web = 0
		num_replies_web = 0
		eq_ids_web = []
		if enq_type in ('all', 'internet', 'online'):
			eq_intensities = get_eq_intensities_for_commune_web(id_com,
				as_main_commune=by_main_commune, min_replies=min_replies,
				min_fiability=min_fiability, filter_floors=filter_floors,
				include_other_felt=include_other_felt,
				include_heavy_appliance=include_heavy_appliance,
				remove_outliers=remove_outliers, recalc=recalc_web)
			eq_ids_web = eq_intensities.keys()
			for id_earth in eq_ids_web:
				I = eq_intensities[id_earth]
				num_replies_web += 1
				if I and I > Imax_web:
					Imax_web = I

		## Official / Historical macroseismic information
		Imax_official = 0
		num_replies_official = 0
		eq_ids_official = []
		if enq_type in ('all', 'official'):
			eq_intensities = get_eq_intensities_for_commune_official(id_com,
							as_main_commune=by_main_commune, min_or_max=min_or_max,
							min_replies=min_replies, min_fiability=min_fiability)
			eq_ids_official = eq_intensities.keys()
			for id_earth in eq_ids_official:
				I = agg_func(eq_intensities[id_earth])
				num_replies_official += 1
				if I > Imax_official and I < 13:
					Imax_official = I

		Imax = max(Imax_web, Imax_official)

		## Construct MacroseismicInfo
		if Imax > 0:
			id_earth = {'web': sorted(eq_ids_web), 'official': sorted(eq_ids_official)}
			agg_type = 'id_main' if by_main_commune else 'id_com'
			num_replies = num_replies_web + num_replies_official
			db_ids = []
			macro_info = MacroseismicInfo(id_earth, id_com, Imax, agg_type,
										enq_type, num_replies, lon, lat, db_ids)
			#comm_macro_dict[id_com] = macro_info
			macro_infos.append(macro_info)

			if verbose:
				msg = '%d (%s): Iweb=%d (n=%d) - Ioff=%d (n=%d)'
				msg %= (id_com, name, Imax_web, len(eq_ids_web),
						Imax_official, len(eq_ids_official))
				print(msg)

		macro_info_coll = MacroInfoCollection(macro_infos, agg_type, enq_type)

	#return comm_macro_dict
	return macro_info_coll


if __name__ == "__main__":
	import os
	from eqcatalog.plot import plot_macroseismic_map

	#print(get_eq_intensities_for_commune_official(6, min_or_max='min', as_main_commune=False))
	#print(get_eq_intensities_for_commune_official(6, min_or_max='max', as_main_commune=False))
	#print(get_eq_intensities_for_commune_web(6, as_main_commune=False, include_other_felt=False))
	#exit()

	enq_type = 'official'
	#enq_type = 'online'
	by_main_commune = True
	macro_info_coll = get_Imax_by_commune(enq_type=enq_type, include_other_felt=False,
										by_main_commune=by_main_commune, verbose=False)
	print(sum(macro.num_replies for macro in macro_info_coll))
	print([macro.I for macro in macro_info_coll])
	#for id_com in comm_macro_dict:
	#	macro_info = comm_macro_dict[id_com]
	#	print("%d: Imax=%d (n=%d)" % (id_com, macro_info.I, macro_info.num_replies))

	region = (2, 7, 49.25, 51.75)
	projection = "merc"
	graticule_interval = (2, 1)
	title = "Maximum intensity by commune (%s)" % enq_type
	fig_folder = "C:\\Temp"
	if by_main_commune:
		fig_filename = "Imax_by_main_commune_%s.PNG"
	else:
		fig_filename = "Imax_by_commune_%s.PNG"
	fig_filename %= enq_type
	#fig_filespec = os.path.join(fig_folder, fig_filename)
	fig_filespec = None


	macro_info_coll.plot_map(region=region, projection=projection,
					graticule_interval=graticule_interval,
					event_style=None, cmap="usgs", title=title,
					fig_filespec=fig_filespec)

	#print(macro_info_coll.to_geojson())

	#gis_file = os.path.splitext(fig_filespec)[0] + ".TAB"
	#macro_info_coll.export_gis('MapInfo File', gis_file)

	#geotiff_file = os.path.splitext(fig_filespec)[0] + ".TIF"
	#macro_info_coll.export_geotiff(geotiff_file)
