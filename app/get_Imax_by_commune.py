"""
Determine max. intensity reported in all Belgian communes
"""

import numpy as np

import eqcatalog.rob.seismodb as seismodb
from eqcatalog.macrorecord import MacroseismicInfo


def get_Imax_by_commune(enq_type='all', min_or_max='max', min_replies=3,
				min_fiability=20, filter_floors=(0, 4), include_other_felt=True,
				include_heavy_appliance=False, remove_outliers=(2.5, 97.5),
				verbose=False):
	"""
	Determine historical Imax for every commune in Belgium

	:param enq_type:
		str, one of "internet", "official", "all"
		(default: all)
	:param min_or_max:
		see :func:`seismodb.query_official_macro_catalog`
	:param min_replies:
	:param min_fiability:
	:param filter_floors:
	:param include_other_felt:
	:param include_heavy_appliance:
	:param remove_outliers:
		see :func:`seismodb.query_web_macro_enquiries`
	:param verbose:
		bool, whether or not to print progress information
		(default: False)

	:return:
		dict mapping commune IDs to instances of :class:`MacroseismicInfo`
	"""
	## Fetch communes from database
	table_clause = 'communes'
	where_clause = 'country = "BE"'
	comm_recs = seismodb.query_seismodb_table(table_clause, where_clause=where_clause)
	comm_macro_dict = {}

	for rec in comm_recs:
		id_com = rec['id']
		lon, lat = rec['longitude'], rec['latitude']
		name = rec['name']

		## Online enquiries
		Imax_web = 0
		eq_ids_web = []
		if enq_type in ('all', 'internet', 'online'):
			dyfi = seismodb.query_web_macro_enquiries('all', id_com,
													min_fiability=min_fiability)
			if len(dyfi):
				all_eq_ids = np.array(dyfi.get_prop_values('id_earth'))
				eq_ids_web = np.unique(all_eq_ids)
				for eq_id in eq_ids_web:
					eq_dyfi = dyfi[all_eq_ids == eq_id]
					if len(eq_dyfi) >= min_replies:
						I = eq_dyfi.calc_mean_cii(filter_floors=filter_floors,
								include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance,
								remove_outliers=remove_outliers)
						if I and I > Imax_web:
							Imax_web = I

		## Official / Historical macroseismic information
		Imax_official = 0
		eq_ids_official = []
		if enq_type in ('all', 'official'):
			table_clause = 'macro_detail'
			where_clause = 'id_com = %d AND fiability >= %d' % (id_com, min_fiability)
			macro_recs = seismodb.query_seismodb_table(table_clause, where_clause=where_clause)
			eq_ids_official = sorted([mrec['id_earth'] for mrec in macro_recs])
			for mrec in macro_recs:
				I = {'min': mrec['intensity_min'],
				'max': mrec['intensity_max'],
				'mean': np.mean([mrec['intensity_min'], mrec['intensity_max']])}[min_or_max]
				if I > Imax_official:
					Imax_official = I

		Imax = max(Imax_web, Imax_official)

		## Construct MacroseismicInfo
		if Imax > 0:
			id_earth = {'web': sorted(eq_ids_web), 'official': sorted(eq_ids_official)}
			agg_type = 'id_com'
			enq_type = 'internet / official'
			num_replies = len(eq_ids_web) + len(eq_ids_official)
			db_ids = []
			macro_info = MacroseismicInfo(id_earth, id_com, Imax, agg_type,
										enq_type, num_replies, lon, lat, db_ids)
			comm_macro_dict[id_com] = macro_info

		if verbose:
			msg = '%d (%s): Iweb=%d (n=%d) - Ioff=%d (n=%d)'
			msg %= (id_com, name, Imax_web, len(eq_ids_web),
					Imax_official, len(eq_ids_official))
			print(msg)

	return comm_macro_dict


if __name__ == "__main__":
	comm_macro_dict = get_Imax_by_commune(verbose=True, include_other_felt=False)
	for id_com in comm_macro_dict:
		macro_info = comm_macro_dict[id_com]
		print("%d: Imax=%d (n=%d)" % (id_com, macro_info.I, macro_info.num_replies))
