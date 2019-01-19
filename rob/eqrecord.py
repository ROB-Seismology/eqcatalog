"""
ROB-specific extensions of eqrecord classes
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from ..eqrecord import LocalEarthquake


class ROBLocalEarthquake(LocalEarthquake):

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

	def get_macroseismic_data_aggregated_web(self, min_replies=3, query_info="cii",
					min_val=1, min_fiability=20.0, group_by_main_village=False,
					filter_floors=False, agg_function="", verbose=False):
		from .seismodb import query_web_macro_catalog
		return query_web_macro_catalog(self.ID, min_replies=min_replies,
					query_info=query_info, min_val=min_val, min_fiability=min_fiability,
					group_by_main_village=group_by_main_village, filter_floors=filter_floors,
					agg_function=agg_function, verbose=verbose)

	def get_macroseismic_data_aggregated_official(self, Imax=True, min_val=1,
			group_by_main_village=False, agg_function="maximum", verbose=False):
		from .seismodb import query_official_macro_catalog
		return query_official_macro_catalog(self.ID, Imax=Imax, min_val=min_val,
			group_by_main_village=group_by_main_village, agg_function=agg_function,
			verbose=verbose)

	def get_macroseismic_enquiries(self, min_fiability=20, verbose=False):
		from .seismodb import query_web_macro_enquiries
		ensemble = query_web_macro_enquiries(self.ID, min_fiability=min_fiability,
											verbose=verbose)
		return ensemble

	def get_focal_mechanism(self, verbose=False):
		from .seismodb import query_focal_mechanisms
		try:
			return query_focal_mechanisms(id_earth=self.ID, verbose=verbose)[0]
		except IndexError:
			return None
