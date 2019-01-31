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
		"""
		Get online macroseismic information (possibly aggregated by
		main commune) for this earthquake

		:param min_replies:
			int, minimum number of replies
			(default: 3)
		:param query_info:
			str, either "cii", "cdi" or "mi"
			(default: "cii")
		:param min_val:
			float, minimum intensity to return
			(default: 1)
		:param min_fiability:
			float, minimum fiability of enquiry
			(default: 20.)
		:param group_by_main_village:
			bool, whether or not to aggregate the results by main village
			(default: False)
		:param filter_floors:
				(min_floor, max_floor) tuple, floors outside this range
				(basement floors and upper floors) are filtered out
				(default: False)
		:param agg_function:
			str, aggregation function to use, one of "minimum", "maximum" or
			"average". If :param:`group_by_main_village` is False, aggregation
			applies to the enquiries within a given (sub)commune.
			(default: "average")
		:param verbose:
			Bool, if True the query string will be echoed to standard output

		:return:
			dict mapping commune IDs to instances of :class:`MacroseismicInfo`
		"""
		from .seismodb import query_web_macro_catalog
		return query_web_macro_catalog(self.ID, min_replies=min_replies,
					query_info=query_info, min_val=min_val, min_fiability=min_fiability,
					group_by_main_village=group_by_main_village, filter_floors=filter_floors,
					agg_function=agg_function, verbose=verbose)

	def get_macroseismic_data_aggregated_official(self, min_or_max='max', min_val=1,
			group_by_main_village=False, agg_function="maximum", min_fiability=20,
			verbose=False):
		"""
		Get official macroseismic information (possibly aggregated by
		main commune) for this earthquake

		:param min_or_max:
			str, one of 'min', 'mean' or 'max' to select between
			intensity_min and intensity_max values in database
			(default: 'max')
		:param min_val:
			float, minimum intensity to return
			(default: 1)
		:param group_by_main_village:
			bool, whether or not to aggregate the results by main village
			(default: False)
		:param agg_function:
			str, aggregation function to use if :param:`group_by_main_village`
			is True, one of "minimum", "maximum" or "average"
			(default: "maximum")
		:param min_fiability:
			float, minimum fiability of enquiry
			(default: 20.)
		:param verbose:
			Bool, if True the query string will be echoed to standard output

		:return:
			dict mapping commune IDs to instances of :class:`MacroseismicInfo`
		"""
		from .seismodb import query_official_macro_catalog
		return query_official_macro_catalog(self.ID, min_or_max=min_or_max, min_val=min_val,
			group_by_main_village=group_by_main_village, agg_function=agg_function,
			min_fiability=min_fiability, verbose=verbose)

	def get_macroseismic_enquiries(self, min_fiability=20, verbose=False):
		"""
		Get all macroseismic enquiries for this earthquake

		:param min_fiability:
			float, minimum fiability of enquiry
			(default: 20.)
		:param verbose:
			bool, if True the query string will be echoed to standard output
			(default: False)

		:return:
			instance of :class:`MacroseismicEnquiryEnsemble`
		"""
		from .seismodb import query_web_macro_enquiries
		ensemble = query_web_macro_enquiries(self.ID, min_fiability=min_fiability,
											verbose=verbose)
		return ensemble

	def get_Imax_web(self, min_replies=3, min_fiability=20, filter_floors=(0, 4),
					include_other_felt=True, include_heavy_appliance=False,
					remove_outliers=(2.5, 97.5)):
		"""
		Report maximum intensity from online enquiry in any commune
		for this earthquake

		:param min_replies:
		:param min_fiability:
		:param filter_floors:
		:param include_other_felt:
		:param include_heavy_appliance:
		:param remove_outliers:
			see :meth:`get_macroseismic_data_aggregated_web`

		:return:
			float, Imax
		"""
		dyfi = self.get_macroseismic_enquiries(min_fiability)
		Imax = 0
		if len(dyfi):
			for id_com, com_ensemble in dyfi.aggregate_by_commune().items():
				if len(com_ensemble) >= min_replies:
					I = com_ensemble.calc_mean_cii(filter_floors=filter_floors,
							include_other_felt=include_other_felt,
							include_heavy_appliance=include_heavy_appliance,
							remove_outliers=remove_outliers)
					if I and I > Imax:
						Imax = I
		return Imax

	def get_Imax_official(self, min_or_max='max', min_fiability=20):
		"""
		Report maximum intensity in any commune for this earthquake.

		:param min_or_max:
		:param min_fiability:
			see :meth:`get_macroseismic_data_official`

		:return:
			float, Imax
		"""
		Imax = 0
		macro_recs = self.get_macroseismic_data_aggregated_official(min_or_max=min_or_max,
													min_val=0, min_fiability=min_fiability)
		for id_com, macro_info in macro_recs.items():
			if macro_info.I > Imax:
				Imax = macro_info.I
		return Imax

	def get_Imax(self, min_or_max='max', min_replies=3, min_fiability=20,
				filter_floors=(0, 4), include_other_felt=True,
				include_heavy_appliance=False, remove_outliers=(2.5, 97.5)):
		"""
		Report maximum intensity (online or official) in any commune
		for this earthquake

		:return:
			float, Imax
		"""
		## Note: same min_fiability for official and web
		Imax_official = self.get_Imax_official(min_or_max=min_or_max,
												min_fiability=min_fiability)
		Imax_web = self.get_Imax_web(min_replies, min_fiability, filter_floors,
					include_other_felt, include_heavy_appliance, remove_outliers)
		return max(Imax_official, Imax_web)

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
