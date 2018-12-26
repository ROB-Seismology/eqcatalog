# -*- coding: iso-Latin-1 -*-

"""
Functionality (e.g., MFD balancing) for composite earthquake catalogs
consisting of a number of subcatalogs
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
from .rob.completeness import DEFAULT_COMPLETENESS
from .eqcatalog import EQCatalog



class CompositeEQCatalog:
	"""
	Class representing a catalog that has been split into a number
	of non-overlapping subcatalogs (e.g., split according to different
	source zones).

	:param zone_catalogs:
		Dict, with zource zone ID's as keys and lists of instances of
		:class:`EQCatalog`, non-overlapping subcatalogs corresponding to
		different source zones, as values
	:param source_model_name:
		Str, name of source model (will be used to read additional info from
		GIS table, if necessary)
	:param Mtype:
		String, magnitude type: "MW", "MS" or "ML" (default: "MW")
	:param Mrelation":
		{str: str} dict, mapping name of magnitude conversion relation
		to magnitude type ("MW", "MS" or "ML") (default: None, will
		select the default relation for the given Mtype)
	:param completeness:
		instance of :class:`Completeness` (default: Completeness_MW_201303a)
	:param min_mag:
		Float, minimum magnitude of sampled MFD's. Note that lower completenes
		magnitude will be used to compute MFD's. (default: 4.0)
	:param mfd_bin_width:
		Float, bin width of sampled MFD's (default: 0.1)
	:param master_MFD:
		instance of :class:`TruncatedGRMFD`, MFD of master catalog (default: None)
	:param zone_MFDs:
		list of instances of :class:´TruncatedGRMFD`, MFDs of subcatalogs
		(default: [])
	"""
	# TODO: modify to make it work with master and zone MFD's without catalogs
	# TODO: modify to make sure parameters for _get_zone_Mmaxes are consistent
	def __init__(self, zone_catalogs, source_model_name, Mtype="MW", Mrelation=None,
				completeness=DEFAULT_COMPLETENESS, min_mag=4.0, mfd_bin_width=0.1,
				master_MFD=None, zone_MFDs=[]):
		self.zone_catalogs = zone_catalogs
		self.source_model_name = source_model_name
		self.Mtype = Mtype
		self.Mrelation = Mrelation
		self.completeness = completeness
		self.min_mag = min_mag
		self.mfd_bin_width = mfd_bin_width
		self.master_MFD = master_MFD
		self.zone_MFDs = zone_MFDs
		self.master_catalog = self.construct_master_catalog()

	def construct_master_catalog(self):
		"""
		Construct master catalog from zone catalogs

		:return:
			instance of :class:`EQCatalog`
		"""
		eq_list = []
		for zone_catalog in self.zone_catalogs.values():
			eq_list.extend(zone_catalog.eq_list)
		start_date = zone_catalog.start_date
		end_date = zone_catalog.end_date
		master_catalog = EQCatalog(eq_list, start_date=start_date, end_date=end_date)
		return master_catalog

	def _get_zone_Mmaxes(self, prior_model_category="CEUS", use_posterior=True):
		"""
		Determine Mmax for each zone catalog as median value of EPRI pdf

		:param prior_model_category:
			str, category of prior model to consider, either "EPRI" or "CEUS"
			(default: "CEUS")
		:param use_posterior:
			bool, whether to consider the posterior (True) or the prior (False)
			Mmax pdf (default: True)

		:return:
			Dict, mapping zone ids (str) to Mmax values (float)
		"""
		zone_catalogs = self.zone_catalogs
		max_mags = dict.fromkeys(zone_catalogs.keys())
		for zone_id, catalog in zone_catalogs.items():
			# TODO: find better mechanism to discern between extended and non-extended
			# (maybe using GIS table)
			if zone_id in ("RVG", "RVRS", "BGRVRS"):
				if prior_model_category == "CEUS":
					prior_model = "CEUS_MESE"
					max_mag = 7.35
				else:
					prior_model = "EPRI_extended"
					max_mag = 6.4
			else:
				if prior_model_category == "CEUS":
					prior_model = "CEUS_NMESE"
					max_mag = 6.7
				else:
					prior_model = "CEUS_NMESE"
					max_mag = 6.3
			if self.zone_MFDs:
				b_val = self.zone_MFDs[zone_id].b_val
			else:
				b_val = None
			if use_posterior:
				prior, likelihood, posterior, params = catalog.get_Bayesian_Mmax_pdf(
						prior_model=prior_model, Mmin_n=4.5, b_val=b_val,
						dM=self.mfd_bin_width, Mtype=self.Mtype, Mrelation=self.Mrelation,
						completeness=self.completeness, verbose=False)
				max_mag = posterior.get_percentiles([50])[0]
			max_mag = np.ceil(max_mag / self.mfd_bin_width) * self.mfd_bin_width
			max_mags[zone_id] = max_mag
		return max_mags

	def _get_zone_areas(self):
		"""
		Determine surface area for each zone in the source model

		:return:
			Dict, mapping zone id's (str) to surface areas (float)
		"""
		import osr
		from mapping.geotools.coordtrans import WGS84, get_utm_spec, get_utm_srs
		from .rob.source_models import read_source_model

		zone_areas = dict.fromkeys(self.zone_catalogs.keys())
		model_data = read_source_model(self.source_model_name, verbose=False)
		for zone_id, zone_data in model_data.items():
			zone_poly = zone_data['obj']
			centroid = zone_poly.Centroid()
			utm_spec = get_utm_spec(centroid.GetX(), centroid.GetY())
			utm_srs = get_utm_srs(utm_spec)
			coordTrans = osr.CoordinateTransformation(WGS84, utm_srs)
			zone_poly.Transform(coordTrans)
			zone_areas[zone_id] = zone_poly.GetArea() / 1E6
		return zone_areas

	def _compute_MFD(self, catalog, Mmax, method="Weichert", b_val=None):
		"""
		Generic method to compute MFD of a catalog using Weichert method
		with parameters (Mtype, Mrelation, completeness) stored as class
		properties

		:param catalog:
			instance of :class:`EQCatalog`
		:param Mmax:
			float, maximum magnitude
		:param method:
			str, calculation method, either "Weichert", "Aki", "LSQc" or "LSQi"
			(default: "Weichert")
		:param b_val:
			float, imposed b value (default: None = unconstrained)

		:return:
			instance of :class:`TruncatedGRMFD`
		"""
		mfd_bin_width = self.mfd_bin_width
		Mtype, Mrelation, completeness = self.Mtype, self.Mrelation, self.completeness
		min_mag = completeness.min_mag
		MFD = catalog.get_estimated_MFD(min_mag, Mmax, mfd_bin_width, method=method,
								b_val=b_val, Mtype=Mtype, Mrelation=Mrelation,
								completeness=completeness, verbose=False)
		MFD.min_mag = self.min_mag
		return MFD

	def _get_min_SCR_zone_MFDs(self, b_val=None):
		"""
		Determine minimum MFD (SCR) for each zone.
		This is a wrapper; depending on Mtype, either :meth:`_get_Fenton_zone_MFDs`
		(MS) or :meth:`_get_Johnston_zone_MFDs` (MW) will be called.

		:param b_val:
			float, imposed b value (default: None = use Fenton's b value)

		:return:
			Dict, mapping zone id's (str) to instances of :class:`TruncatedGRMFD`
		"""
		if self.Mtype == "MS":
			return self._get_Fenton_zone_MFDs(b_val=b_val)
		elif self.Mtype == "MW":
			return self._get_Johnston_zone_MFDs(b_val=b_val)

	def _get_Fenton_zone_MFDs(self, b_val=None):
		"""
		Determine minimum MFD for each zone according to Fenton et al. (2006)

		:param b_val:
			float, imposed b value (default: None = use Fenton's b value)

		:return:
			Dict, mapping zone id's (str) to instances of :class:`TruncatedGRMFD`
		"""
		import hazard.rshalib.mfd.mfd as mfd

		zone_Mmaxes = self._get_zone_Mmaxes(prior_model_category="CEUS", use_posterior=True)
		zone_areas = self._get_zone_areas()
		zone_MFDs = dict.fromkeys(self.zone_catalogs.keys())
		for zone_id in self.zone_catalogs.keys():
			zone_Mmax = zone_Mmaxes[zone_id]
			zone_area = zone_areas[zone_id]
			zone_MFD = mfd.TruncatedGRMFD.construct_FentonEtAl2006MFD(self.min_mag,
										zone_Mmax, self.mfd_bin_width, zone_area)
			if b_val != None:
				b = b_val
				stda, stdb = 0, 0
				lamda, M = zone_MFD.occurrence_rates[0], self.min_mag
				a_val = (mfd.a_from_lambda(lamda, M, b)
						+ mfd.get_a_separation(b, self.mfd_bin_width))
				zone_MFD = mfd.TruncatedGRMFD(self.min_mag, zone_Mmax,
									self.mfd_bin_width, a_val, b, stda, stdb)
			zone_MFDs[zone_id] = zone_MFD
		return zone_MFDs

	def _get_Johnston_zone_MFDs(self, b_val=None):
		"""
		Determine minimum MFD for each zone according to Johnston (1994)

		:param b_val:
			float, imposed b value (default: None = use Johnston's b value)

		:return:
			Dict, mapping zone id's (str) to instances of :class:`TruncatedGRMFD`
		"""
		import hazard.rshalib.mfd.mfd as mfd

		zone_Mmaxes = self._get_zone_Mmaxes(prior_model_category="CEUS", use_posterior=True)
		zone_areas = self._get_zone_areas()
		zone_MFDs = dict.fromkeys(self.zone_catalogs.keys())
		for zone_id in self.zone_catalogs.keys():
			zone_Mmax = zone_Mmaxes[zone_id]
			zone_area = zone_areas[zone_id]
			zone_MFD = mfd.TruncatedGRMFD.construct_Johnston1994MFD(self.min_mag,
										zone_Mmax, self.mfd_bin_width, zone_area)
			if b_val != None:
				b = b_val
				stda, stdb = 0, 0
				lamda, M = zone_MFD.occurrence_rates[0], self.min_mag
				a_val = (mfd.a_from_lambda(lamda, M, b)
						+ mfd.get_a_separation(b, self.mfd_bin_width))
				zone_MFD = mfd.TruncatedGRMFD(self.min_mag, zone_Mmax,
									self.mfd_bin_width, a_val, b, stda, stdb)
			zone_MFDs[zone_id] = zone_MFD
		return zone_MFDs

	def _compute_zone_MFDs(self, method="Weichert", b_val=None, num_sigma=0):
		"""
		Compute MFD for each zone using same imposed b value
		If MFD cannot be computed, a "minimum" MFD, corresponding
		to the average for SCR will be determined.

		:param method:
			str, calculation method, either "Weichert", "Aki", "LSQc" or "LSQi"
			(default: "Weichert")
		:param b_val:
			float, imposed b value (default: None = unconstrained)
		:param num_sigma:
			float, number of standard deviations. If not zero,
			mean + num_sigma stdevs and mean - num_sigma stddevs MFD's
			will be computed as well

		:return:
			Dict, mapping zone id's (str) to instances of :class:`TruncatedGRMFD`
			or (if num_sigma > 0) to lists of instances of :class:`TruncatedGRMFD`
		"""
		import hazard.rshalib.mfd.mfd as mfd

		zone_Mmaxes = self._get_zone_Mmaxes(prior_model_category="CEUS", use_posterior=True)
		zone_min_SCR_MFDs = self._get_min_SCR_zone_MFDs()
		zone_MFDs = dict.fromkeys(self.zone_catalogs.keys())
		for zone_id, zone_catalog in self.zone_catalogs.items():
			zone_Mmax = zone_Mmaxes[zone_id]
			try:
				zone_MFD = self._compute_MFD(zone_catalog, zone_Mmax, method=method,
											b_val=b_val)
				zone_MFD.Weichert = True
			except ValueError:
				## Note: it is critical that this doesn't fail for any one zone,
				## so, fall back to minimum MFD for SCR, based on area
				zone_MFD = zone_min_SCR_MFDs[zone_id]
				zone_MFD.Weichert = False
			else:
				## If computed MFD is below min SCR MFD, select the latter
				mags, rates = zone_MFD.get_center_magnitudes(), zone_MFD.occurrence_rates
				mags_scr, rates_scr = (zone_min_SCR_MFDs[zone_id].get_center_magnitudes(),
									zone_min_SCR_MFDs[zone_id].occurrence_rates)
				M = 5.5
				if rates_scr[mags_scr > M][0] > rates[mags > M][0]:
					zone_MFD = zone_min_SCR_MFDs[zone_id]
					zone_MFD.Weichert = False
			zone_MFDs[zone_id] = zone_MFD

			if num_sigma > 0:
				zone_MFDs[zone_id] = [zone_MFD]
				b_val1 = zone_MFD.b_val + zone_MFD.b_sigma * num_sigma
				b_val2 = zone_MFD.b_val - zone_MFD.b_sigma * num_sigma
				if zone_MFD.Weichert:
					MFD_sigma1 = self._compute_MFD(zone_catalog, zone_Mmax,
												method=method, b_val=b_val1)
					MFD_sigma2 = self._compute_MFD(zone_catalog, zone_Mmax,
												method=method, b_val=b_val2)
				else:
					## If median MFD could not be computed
					lamda, M = zone_MFD.occurrence_rates[0], self.min_mag
					a_val1 = (mfd.a_from_lambda(lamda, M, b_val1) +
							mfd.get_a_separation(b_val1, self.mfd_bin_width))
					MFD_sigma1 = mfd.TruncatedGRMFD(self.min_mag, zone_Mmax,
											self.mfd_bin_width, a_val1, b_val1)
					a_val2 = (mfd.a_from_lambda(lamda, M, b_val2) +
							mfd.get_a_separation(b_val2, self.mfd_bin_width))
					MFD_sigma2 = mfd.TruncatedGRMFD(self.min_mag, zone_Mmax,
											self.mfd_bin_width, a_val2, b_val2)
				zone_MFDs[zone_id].append(MFD_sigma1)
				zone_MFDs[zone_id].append(MFD_sigma2)

		return zone_MFDs

	def _compute_master_MFD(self, method="Weichert", num_sigma=0):
		"""
		Compute MFD of master catalog

		:param method:
			str, calculation method, either "Weichert", "Aki", "LSQc" or "LSQi"
			(default: "Weichert")
		:param num_sigma:
			float, number of standard deviations. If not zero,
			mean + num_sigma stdevs and mean - num_sigma stddevs MFD's
			will be computed as well

		:return:
			instances of :class:`TruncatedGRMFD` (if num_sigma == 0)
			or list of instances of :class:`TruncatedGRMFD` (if num_sigma > 0)
		"""
		zone_Mmaxes = self._get_zone_Mmaxes(prior_model_category="CEUS", use_posterior=True)
		overall_Mmax = max(zone_Mmaxes.values())
		master_catalog = self.master_catalog
		master_MFD = self._compute_MFD(master_catalog, overall_Mmax,
										method=method, b_val=None)
		if num_sigma > 0:
			b_val1 = master_MFD.b_val + num_sigma * master_MFD.b_sigma
			master_MFD1 = self._compute_MFD(master_catalog, overall_Mmax,
											method=method, b_val=b_val1)
			b_val2 = master_MFD.b_val - num_sigma * master_MFD.b_sigma
			master_MFD2 = self._compute_MFD(master_catalog, overall_Mmax,
											method=method, b_val=b_val2)
			return [master_MFD, master_MFD1, master_MFD2]
		else:
			return master_MFD

	def _compute_summed_MFD(self, method="Weichert", b_val=None, num_sigma=0):
		"""
		Compute summed MFD of zone catalogs, where MFD of each zone catalog
		is computed using the same b value

		:param method:
			str, calculation method, either "Weichert", "Aki", "LSQc" or "LSQi"
			(default: "Weichert")
		:param b_val:
			float, imposed b value. If None, the b value of the master catalog
			MFD will be used (default: None)
		:param num_sigma:
			float, number of standard deviations. If not zero,
			mean + num_sigma stdevs and mean - num_sigma stddevs MFD's
			will be computed as well (again, using b_val +/- b_sigma
			of master catalog MFD)

		:return:
			instances of :class:`TruncatedGRMFD` (if num_sigma == 0)
			or list of instances of :class:`TruncatedGRMFD` (if num_sigma > 0)
		"""
		import hazard.rshalib.mfd.mfd as mfd

		if num_sigma > 0:
			master_MFD, master_MFD1, master_MFD2 = self._compute_master_MFD(method=method,
															num_sigma=num_sigma)
		else:
			master_MFD = self._compute_master_MFD(method=method)
		zone_MFDs = self._compute_zone_MFDs(method=method, b_val=master_MFD.b_val)
		summed_MFD = mfd.sum_MFDs(zone_MFDs.values())
		if num_sigma > 0:
			zone_MFDs1 = self._compute_zone_MFDs(method=method, b_val=master_MFD1.b_val)
			summed_MFD1 = mfd.sum_MFDs(zone_MFDs1.values())
			zone_MFDs2 = self._compute_zone_MFDs(method=method, b_val=master_MFD2.b_val)
			summed_MFD2 = mfd.sum_MFDs(zone_MFDs2.values())
			return [summed_MFD, summed_MFD1, summed_MFD2]
		else:
			return summed_MFD

	def balance_MFD_by_moment_rate(self, num_samples, mr_num_sigma=1,
						max_test_mag=None, b_num_sigma=2, use_master=False):
		"""
		Balance MFD's of zone catalogs by moment rate.
		First, calculate moment rate range corresponding to b_val +/-
		mr_num_sigma of master catalog. Then, for each subcatalog, do
		Monte Carlo sampling of b_val within +/- b_num_sigma bounds,
		compute corresponding a_val. Finally, sum total moment rate of
		all subcatalogs, and check if it falls within the moment rate
		range of the master catalog.

		:param num_samples:
			Int, number of MFD samples to generate
		:param mr_num_sigma:
			Float, number of standard deviations on b value of master catalog
			to determine moment rate range (default: 1)
		:param b_num_sigma:
			Float, number of standard deviations on b value of zone catalogs
			for Monte Carlo sampling (default: 2)
		:param max_test_mag:
			Float, maximum magnitude to test if summed moment rate is
			below upper range of master catalog (default: None, will take
			a default value depending on use_master)
		:param use_master:
			Bool, whether master catalog (True) or summed catalog (False)
			should be used to constrain frequencies

		:return:
			Dict, with zone IDs as keys and a list of num_samples MFD's as
			values
		"""
		import scipy.stats
		import hazard.rshalib.mfd.mfd as mfd

		## Determine Mmax of each zone catalog, and overall Mmax
		zone_Mmaxes = self._get_zone_Mmaxes(prior_model_category="CEUS", use_posterior=True)
		if max_test_mag is None:
			if use_master:
				max_test_mag = min(zone_Mmaxes.values()) - self.mfd_bin_width
			else:
				max_test_mag = max(zone_Mmaxes.values()) - self.mfd_bin_width
		max_test_mag_index = (int(round((max_test_mag - self.min_mag)
								/ self.mfd_bin_width)) + 1)
		print("Maximum magnitude to test: %.1f (i=%d)"
			% (max_test_mag, max_test_mag_index))

		## Determine moment rate range of master catalog
		if use_master:
			master_MFD, master_MFD1, master_MFD2 = self._compute_master_MFD(
														num_sigma=mr_num_sigma)
			#master_MFD1.max_mag = max_test_mag
			master_MFD2.max_mag = max_test_mag
		else:
			master_MFD, master_MFD1, master_MFD2 = self._compute_summed_MFD(
														num_sigma=mr_num_sigma)
			min_mag = master_MFD1.min_mag
			#master_MFD1 = mfd.EvenlyDiscretizedMFD(min_mag, self.mfd_bin_width,
			# 				master_MFD1.occurrence_rates[:max_test_mag_index])
			master_MFD2 = mfd.EvenlyDiscretizedMFD(min_mag, self.mfd_bin_width,
							master_MFD2.occurrence_rates[:max_test_mag_index])
		master_moment_rate_range = np.zeros(2, 'd')
		master_moment_rate_range[0] = master_MFD1._get_total_moment_rate()
		master_moment_rate_range[1] = master_MFD2._get_total_moment_rate()
		print("Moment rate range: %E - %E N.m"
			% (master_moment_rate_range[0], master_moment_rate_range[1]))

		## Determine unconstrained MFD for each zone catalog
		if not self.zone_MFDs:
			zone_MFDs = self._compute_zone_MFDs()
		else:
			zone_MFDs = self.zone_MFDs

		## Monte Carlo sampling
		zone_catalogs = self.zone_catalogs
		MFD_container = dict.fromkeys(zone_catalogs.keys())
		num_passed, num_rejected, num_failed = 0, 0, 0
		num_iterations = 0
		while num_passed < num_samples:
			if num_iterations % 10 == 0:
				print("%05d  (passed: %05d; rejected: %05d; failed: %05d)"
					% (num_iterations, num_passed, num_rejected, num_failed))
			failed = False
			temp_MFD_container = dict.fromkeys(zone_catalogs.keys())

			## Draw random b value for each zone
			for zone_id, zone_catalog in zone_catalogs.items():
				zone_Mmax = zone_Mmaxes[zone_id]
				zone_MFD = zone_MFDs[zone_id]
				## Monte Carlo sampling from truncated normal distribution
				mu, sigma = zone_MFD.b_val, zone_MFD.b_sigma
				b_val = scipy.stats.truncnorm.rvs(-b_num_sigma, b_num_sigma, mu, sigma)
				if zone_MFD.Weichert:
					try:
						MFD = self._compute_MFD(zone_catalog, zone_Mmax, b_val=b_val)
					except ValueError:
						failed = True
						num_failed += 1
						break
					else:
						if not np.isinf(MFD.a_val):
							temp_MFD_container[zone_id] = MFD
						else:
							temp_MFD_container[zone_id] = zone_MFD
				else:
					## Do not recompute if mean MFD is min SCR MFD
					lamda, M = zone_MFD.occurrence_rates[0], self.min_mag
					a_val = (mfd.a_from_lambda(lamda, M, b_val)
							+ mfd.get_a_separation(b_val, self.mfd_bin_width))
					MFD = mfd.TruncatedGRMFD(self.min_mag, zone_Mmax,
											self.mfd_bin_width, a_val, b_val)
					temp_MFD_container[zone_id] = MFD

			## Check if summed moment rate lies within master moment rate range
			if not failed:
				zone_mfds = temp_MFD_container.values()
				summed_moment_rate = np.sum([mfd._get_total_moment_rate()
											for mfd in zone_mfds])
				if (master_moment_rate_range[0] <= summed_moment_rate
					<= master_moment_rate_range[1]):
					for zone_id in zone_catalogs.keys():
						if num_passed == 0:
							MFD_container[zone_id] = [temp_MFD_container[zone_id]]
						else:
							MFD_container[zone_id].append(temp_MFD_container[zone_id])
					num_passed += 1
				else:
					num_rejected += 1

			num_iterations += 1
		print("%05d  (passed: %05d; rejected: %05d; failed: %05d)"
			% (num_iterations, num_passed, num_rejected, num_failed))

		return MFD_container

	def balance_MFD_by_frequency(self, num_samples, num_sigma=2, max_test_mag=None,
								use_master=False, random_seed=None):
		"""
		Balance MFD's of zone catalogs by frequency.
		First, calculate frequency range corresponding to b_val +/-
		num_sigma of master catalog. Then, for each subcatalog, do
		Monte Carlo sampling of b_val within +/- num_sigma bounds,
		compute corresponding a_val. Finally, sum frequency of
		all subcatalogs (up to max_test_mag), and check if it falls
		within the frequency range of the master catalog.

		:param num_samples:
			Int, number of MFD samples to generate
		:param num_sigma:
			Float, number of standard deviations on b value of master
			catalog (to determine bounds) and of zone catalogs (for
			Monte Carlo sampling) (default: 2)
		:param max_test_mag:
			Float, maximum magnitude to test if summed frequency is
			below upper range of master catalog (default: None, will take
			a default value depending on use_master)
		:param use_master:
			Bool, whether master catalog (True) or summed catalog (False)
			should be used to constrain frequencies (default: False)
		:param random_seed:
			None or int, seed to initialize internal state of random number
			generator (default: None, will seed from current time)

		:return:
			Dict, with zone IDs as keys and a list of num_samples MFD's as
			values
		"""
		import scipy.stats
		import hazard.rshalib.mfd.mfd as mfd

		## Determine Mmax of each zone catalog, and overall Mmax
		zone_Mmaxes = self._get_zone_Mmaxes(prior_model_category="CEUS", use_posterior=True)
		if max_test_mag is None:
			if use_master:
				max_test_mag = min(zone_Mmaxes.values()) - self.mfd_bin_width
			else:
				max_test_mag = max(zone_Mmaxes.values()) - self.mfd_bin_width
		max_test_mag_index = int(round((max_test_mag - self.min_mag)
								 / self.mfd_bin_width)) + 1
		print("Maximum magnitude to test: %.1f (i=%d)"
			% (max_test_mag, max_test_mag_index))

		## Determine frequency range of master catalog
		if use_master:
			master_MFD, master_MFD1, master_MFD2 = self._compute_master_MFD(
															num_sigma=num_sigma)
		else:
			master_MFD, master_MFD1, master_MFD2 = self._compute_summed_MFD(
															num_sigma=num_sigma)
		master_frequency_range = np.zeros((2, len(master_MFD)), 'd')
		master_frequency_range[0] = master_MFD1.get_cumulative_rates()
		master_frequency_range[1] = master_MFD2.get_cumulative_rates()
		#print(master_frequency_range)

		## Determine unconstrained MFD for each zone catalog
		if not self.zone_MFDs:
			zone_MFDs = self._compute_zone_MFDs()
		else:
			zone_MFDs = self.zone_MFDs

		## Monte Carlo sampling
		np.random.seed(seed=random_seed)
		zone_catalogs = self.zone_catalogs
		MFD_container = dict.fromkeys(zone_catalogs.keys())
		num_passed, num_rejected, num_failed = 0, 0, 0
		num_iterations = 0
		while num_passed < num_samples:
			if num_iterations % 10 == 0:
				print("%05d  (passed: %05d; rejected: %05d; failed: %05d)"
					% (num_iterations, num_passed, num_rejected, num_failed))
			failed = False
			temp_MFD_container = dict.fromkeys(zone_catalogs.keys())

			## Draw random b value for each zone
			for zone_id, zone_catalog in zone_catalogs.items():
				zone_Mmax = zone_Mmaxes[zone_id]
				zone_MFD = zone_MFDs[zone_id]
				## Monte Carlo sampling from truncated normal distribution
				mu, sigma = zone_MFD.b_val, zone_MFD.b_sigma
				b_val = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, mu, sigma)
				if zone_MFD.Weichert:
					try:
						MFD = self._compute_MFD(zone_catalog, zone_Mmax, b_val=b_val)
					except ValueError:
						failed = True
						num_failed += 1
						break
					else:
						if not np.isinf(MFD.a_val):
							temp_MFD_container[zone_id] = MFD
						else:
							temp_MFD_container[zone_id] = zone_MFD
				else:
					## Do not recompute if mean MFD is min SCR MFD
					lamda, M = zone_MFD.occurrence_rates[0], self.min_mag
					a_val = (mfd.a_from_lambda(lamda, M, b_val)
							+ mfd.get_a_separation(b_val, self.mfd_bin_width))
					MFD = mfd.TruncatedGRMFD(self.min_mag, zone_Mmax,
											self.mfd_bin_width, a_val, b_val)
					temp_MFD_container[zone_id] = MFD

			## Check if summed frequencies lie within master frequency range
			if not failed:
				zone_mfds = temp_MFD_container.values()
				summed_frequency_range = np.zeros(len(master_MFD), 'd')
				for MFD in zone_mfds:
					summed_frequency_range[:len(MFD)] += MFD.get_cumulative_rates()
				if ((master_frequency_range[0] <= summed_frequency_range).all()
					and (summed_frequency_range[:max_test_mag_index]
						<= master_frequency_range[1, :max_test_mag_index]).all()):
					#print(master_frequency_range[0, max_test_mag_index],
					# 		summed_frequency_range[max_test_mag_index],
					# 		master_frequency_range[1, max_test_mag_index])
					for zone_id in zone_catalogs.keys():
						if num_passed == 0:
							MFD_container[zone_id] = [temp_MFD_container[zone_id]]
						else:
							MFD_container[zone_id].append(temp_MFD_container[zone_id])
					num_passed += 1
				else:
					num_rejected += 1

			num_iterations += 1
		print("%05d  (passed: %05d; rejected: %05d; failed: %05d)"
			% (num_iterations, num_passed, num_rejected, num_failed))

		return MFD_container

	def balance_MFD_by_fixed_b_value(self, num_samples, num_sigma=2, random_seed=None):
		"""
		Balance MFD's of zone catalogs by Monte Carlo sampling of b value
		of master catalog MFD, and computing zone MFD's with this fixed
		b value

		:param num_samples:
			Int, number of MFD samples to generate
		:param num_sigma:
			Float, number of standard deviations on b value of master
			catalog (to determine bounds) and of zone catalogs (for
			Monte Carlo sampling) (default: 2)
		:param random_seed:
			None or int, seed to initialize internal state of random number
			generator (default: None, will seed from current time)

		:return:
			Dict, with zone IDs as keys and a list of num_samples MFD's as
			values
		"""
		import scipy.stats

		if not self.master_MFD:
			master_MFD = self._compute_master_MFD()
		else:
			master_MFD = self.master_MFD
		MFD_container = dict.fromkeys(self.zone_catalogs.keys())

		## Monte Carlo sampling from truncated normal distribution
		np.random.seed(seed=random_seed)
		mu, sigma = master_MFD.b_val, master_MFD.b_sigma
		for i in range(num_samples):
			b_val = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, mu, sigma)
			zone_MFDs = self._compute_zone_MFDs(b_val=b_val, num_sigma=0)
			zone_min_SCR_MFDs = self._get_min_SCR_zone_MFDs(b_val=b_val)
			for zone_id in self.zone_catalogs.keys():
				zone_MFD = zone_MFDs[zone_id]
				if np.isinf(zone_MFD.a_val):
					zone_MFD = zone_min_SCR_MFDs[zone_id]
				if i == 0:
					MFD_container[zone_id] = [zone_MFD]
				else:
					MFD_container[zone_id].append(zone_MFD)

		return MFD_container

	def sample_MFD_unconstrained(self, num_samples, num_sigma=2, random_seed=None):
		"""
		Perform unconstrained sampling on b value of zone catalogs

		:param num_samples:
			Int, number of MFD samples to generate
		:param num_sigma:
			Float, number of standard deviations on b value of zone catalogs
			for Monte Carlo sampling (default: 2)
		:param random_seed:
			None or int, seed to initialize internal state of random number
			generator (default: None, will seed from current time)

		:return:
			Dict, with zone IDs as keys and a list of num_samples MFD's as
			values
		"""
		import scipy.stats
		import hazard.rshalib.mfd.mfd as mfd

		zone_catalogs = self.zone_catalogs
		zone_Mmaxes = self._get_zone_Mmaxes(prior_model_category="CEUS", use_posterior=True)

		## Determine unconstrained MFD for each zone catalog
		if not self.zone_MFDs:
			zone_MFDs = self._compute_zone_MFDs()
		else:
			zone_MFDs = self.zone_MFDs

		## Monte Carlo sampling
		np.random.seed(seed=random_seed)
		MFD_container = dict.fromkeys(zone_catalogs.keys())
		num_passed, num_failed = 0, 0
		while num_passed < num_samples:
			failed = False
			temp_MFD_container = dict.fromkeys(zone_catalogs.keys())

			## Draw random b value for each zone
			for zone_id, zone_catalog in zone_catalogs.items():
				zone_Mmax = zone_Mmaxes[zone_id]
				zone_MFD = zone_MFDs[zone_id]
				## Monte Carlo sampling from truncated normal distribution
				mu, sigma = zone_MFD.b_val, zone_MFD.b_sigma
				b_val = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, mu, sigma)
				if zone_MFD.Weichert:
					try:
						MFD = self._compute_MFD(zone_catalog, zone_Mmax, b_val=b_val)
					except ValueError:
						failed = True
						num_failed += 1
						break
					else:
						if not np.isinf(MFD.a_val):
							temp_MFD_container[zone_id] = MFD
						else:
							temp_MFD_container[zone_id] = zone_MFD
				else:
					## Do not recompute if mean MFD is min SCR MFD
					lamda, M = zone_MFD.occurrence_rates[0], self.min_mag
					a_val = (mfd.a_from_lambda(lamda, M, b_val)
							 + mfd.get_a_separation(b_val, self.mfd_bin_width))
					MFD = mfd.TruncatedGRMFD(self.min_mag, zone_Mmax,
											self.mfd_bin_width, a_val, b_val)
					temp_MFD_container[zone_id] = MFD

			if not failed:
				for zone_id in zone_catalogs.keys():
					if num_passed == 0:
						MFD_container[zone_id] = [temp_MFD_container[zone_id]]
					else:
						MFD_container[zone_id].append(temp_MFD_container[zone_id])
				num_passed += 1

		return MFD_container
