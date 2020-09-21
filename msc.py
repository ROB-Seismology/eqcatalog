"""
Magnitude Scale Conversion (MSC) module

Author: Bart Vleminckx
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np

from .moment import log_moment_to_mag



class MSCE(object):
	"""
	Base class for Magnitude Scale Conversion Equation
	"""
	def __call__(self, val):
		return self.get_mean(val)


class MSCE_ML_MW(MSCE):
	"""
	Base class for ML -> MW MSCE's
	"""
	_FROM = 'ML'
	_TO = 'MW'


class MSCE_MS_MW(MSCE):
	"""
	Base class for MS -> MW MSCE's
	"""
	_FROM = 'MS'
	_TO = 'MW'


class MSCE_ML_MS(MSCE):
	"""
	Base class for ML -> MS MSCE's
	"""


class Ahorner1983(MSCE_ML_MW):
	"""
	Conversion ML -> log seismic moment
	Published in:
	Ahorner, L. (1983). Historical seismicity and present-day microearthquake
	activity of the Rhenish Massif, Central Europe. In K. Fuchs, K. von Gehlen,
	H. Maelzer, H. Murawski, and A. Semmel (Eds.), Plateau Uplift (pp. 198-221).
	Berlin, Heidelberg: Springer-Verlag.

	Used by in Leynaud et al. (2000)
	Validity range: ML 0.9 - 5.7
	Region: Rhenish Massif
	"""
	def get_mean(self, ML):
		"""
		Return mean MW value(s)

		:param ML:
			scalar, list or array: local magnitude(s)

		:return:
			moment magnitude(s), same type as input
		"""
		log_Mo_dyncm = 17.4 + 1.1 * ML
		MW = log_moment_to_mag(log_Mo_dyncm, unit='dyn.cm')
		return MW

	def get_sigma(self, ML):
		"""
		Return standard deviation of MW value

		:param ML:
			scalar, list or array: local magnitude(s)

		:return:
			standard deviation(s), same type as input
		"""
		sigma_log_Mo = 0.21
		sigma = (2. / 3.) * sigma_log_Mo * np.ones_like(ML, dtype='float')
		return sigma


class Ambraseys1985(MSCE_ML_MS):
	"""
	Conversion ML -> MS
	Published in:
	Ambraseys, N.N., 1985, Magnitude assessment of northwestern European
	earthquakes: Earthquake Engineering and Structural Dynamics, v. 13,
	p. 307-320.

	Validity range: ML 1.5 - 5.7 (judging from their Fig. 6)
	Region: NW Europe
	"""
	def get_mean(self, ML):
		MS = 0.09 + 0.93 * ML
		return MS

	def get_sigma(self, ML):
		sigma = 0.3 * np.ones_like(ML, dtype='float')
		return sigma


class Benouar1994(MSCE_ML_MS):
	"""
	Conversion ML -> MS
	Published in:
	Benouar, D., 1994, Materials for the investigation of the seismicity of
	Algeria and adjacent regions during the twentieth century: Annali Di
	Geofisica XXXVII:4

	Validity range: Not known
	Region: Algeria
	"""
	def get_mean(self, ML):
		MS = 1.40 + 0.76 * ML
		return MS

	def get_sigma(self, ML=None):
		return None


class AmbraseysFree1997(MSCE_MS_MW):
	"""
	Conversion MS -> log seismic moment
	Published in:
	Ambraseys, N. N., & Free, M. W. (1997). Surface-wave magnitude calibration
	for European region earthquakes. Journal of Earthquake Engineering,
	1(1), 1-22.

	Validity range: MS 3.4 - 7.9 (judging from their Fig. 15)
	Region: Europe, events < 30 km depth
	"""
	def get_mean(self, MS):
		## Events shallower than 30 km
		log_Mo_dyncm = 23.123 - 0.505 * MS + 0.140 * MS**2
		## Regression not taking depth into account
		#log_Mo_dyncm = 24.401 - 0.895 * MS + 0.170 * MS**2
		MW = log_moment_to_mag(log_Mo_dyncm, unit='dyn.cm')
		return MW

	def get_sigma(self, MS):
		sigma = (2. / 3.) * sigma_log_Mo * np.ones_like(MS, dtype='float')
		return sigma


class BungumEtAl2003NCEurope(MSCE_MS_MW):
	"""
	Conversion MS -> MW
	Published in:
	Bungum, H., Lindholm, C. D., & Dahle, A. (2003). Long-period ground-
	motions for large European earthquakes, 1905 - 1992, and comparisons
	with stochastic predictions. Journal of Seismology, 7, 377-396.

	Similar to equivalence between MS and MW on world-wide scale found
	by Ekstrom and Dziewonski (1988).

	Region: north and central Europe (north of N 47.5)
	Validity range: MS 5.0 - 7.8 (judging from their Fig. 4)
	Standard deviation not given.
	"""
	def get_mean(self, MS):
		MW = MS * 1.
		return MW

	def get_sigma(self, MS=None):
		return None


class BungumEtAl2003SEurope(MSCE_MS_MW):
	"""
	Conversion MS -> MW
	Published in:
	Bungum, H., Lindholm, C. D., & Dahle, A. (2003). Long-period ground-
	motions for large European earthquakes, 1905 - 1992, and comparisons
	with stochastic predictions. Journal of Seismology, 7, 377-396.

	Region: southern Europe (south of N 47.5)
	Validity range: MS 4.25 - 7.5 (judging from their Fig. 5)
	Standard deviation not given.
	"""
	def get_mean(self, MS):
		## southern Europe
		input_type = type(MS)
		MS = np.asarray(MS)
		MW = np.zeros_like(MS, dtype='float')
		MW[MS >= 5.4] = 0.796 * MS[MS >= 5.4] + 1.280
		MW[MS < 5.4] = 0.585 * MS[MS < 5.4] + 2.422
		if input_type != np.ndarray:
			MW = input_type(MW)
		return MW

	def get_sigma(self, MS=None):
		return None


class Camelbeeck1985(MSCE_ML_MW):
	"""
	Conversion MS -> log seismic moment
	Published in:
	Camelbeeck, T. (1985). Recent seismicity in Hainaut - Scaling laws from the
	seismological stations in Belgium and Luxemburg. P. Melchior (ed.), Seismic
	Activity in Western Europe, 109-126. D. Reidel Publishing Company.

	Validity range: ML 2.6 - 4.4 (judging from their Table 1)
	Region: Hainaut, Belgium
	Standard deviation not given
	"""
	def get_mean(self, ML):
		log_Mo_dyncm = 18.22 + 0.99 * ML
		MW = log_moment_to_mag(log_Mo_dyncm, unit='dyn.cm')
		return MW

	def get_sigma(self, ML=None):
		return None


class Geller1976(MSCE_MS_MW):
	"""
	Conversion MS -> log seismic moment (assuming a stress drop of 50 bars)
	Published in:
	Geller, R. J. (1976). Scaling relations for earthquake source parameters
	and magnitudes. Bulletin of the Seismological Society of America,
	66(5), 1501-1523.

	Validity range: MS 5.5 (judging from their Fig. 7) - 8.22
	Region: global
	Standard deviation not given
	"""
	def get_mean(self, MS):
		input_type = type(MS)
		MS = np.asarray(MS)
		log_Mo_dyncm = np.zeros_like(MS, dtype='float')
		log_Mo_dyncm[MS < 6.76] = MS[MS < 6.76] + 18.89
		log_Mo_dyncm[MS >= 6.76] = (3. / 2.) * MS[MS >= 6.76] + 15.51
		log_Mo_dyncm[MS >= 8.12] = 3. * MS[MS >= 8.12] + 3.33
		log_Mo_dyncm[MS >= 8.22] = np.NaN
		MW = log_moment_to_mag(log_Mo_dyncm, unit='dyn.cm')
		if input_type != np.ndarray:
			MW = input_type(MW)
		return MW

	def get_sigma(self, MS=None):
		return None


class Goutbeek2008(MSCE_ML_MW):
	"""
	"""
	def get_mean(self, ML):
		MW = 0.843 * ML + 0.1954
		return MW

	def get_sigma(self, ML=None):
		return None


class GruenthalWahlstrom2003(MSCE_ML_MW):
	"""
	Conversion ML -> MW, used in Gruenthal catalogue 2003
	Published in:
	Gruenthal, G., & Wahlstrom, R. (2003). An earthquake catalogue for
	central, northern and northwestern Europe based on Mw magnitudes.
	GeoForschunszentrum Potsdam.

	Validity range: ML 1 - 6
	Region: central Europe
	Uncertainties defined on constants, but no standard deviation on MW
	"""
	def get_mean(self, ML):
		MW = 0.67 + 0.56 * ML + 0.046 * ML**2
		return MW

	def get_sigma(self, ML=None):
		return None


class GruenthalEtAl2009(MSCE_ML_MW):
	"""
	Conversion ML -> MW, used in CENEC catalog
	Published in:
	Gruenthal, G., Wahlstrom, R., & Stromeyer, D. (2009). The unified
	catalogue of earthquakes in central, northern and northwestern Europe
	(CENEC) - updated and expanded to the last millennium. Journal of
	Seismology, 13(4), 517-541. doi:10.1007/s10950-008-9144-9

	Validity range: ML 3.5 and up (but used for ML >= 1.0)
	Region: Europe
	"""
	def get_mean(self, ML):
		MW = 0.0376 * ML**2 + 0.646 * ML + 0.53
		return MW

	def get_sigma(self, ML):
		var = (0.97 * ML**4 - 12.4 * ML**3 + 58.4 * ML**2 - 120 * ML + 921) * 1E-4
		return np.sqrt(var)


class ISC_GEM2013(MSCE_MS_MW):
	"""
	Conversion MS -> MW
	By ISC-GEM Global Instrumental Earthquake Catalogue (1900-2009)
	Validity range: MS 3.6 - 8.0
	Region: global
	Standard deviation not given (maybe in publication?)
	"""
	def get_mean(self, MS):
		input_type = type(MS)
		MS = np.asarray(MS)
		MW = np.zeros_like(MS, dtype='float')
		MW[MS >= 3.6] = np.exp(-0.222 + 0.233 * MS[MS >= 3.6]) + 2.863
		MW[MS < 3.6] = np.NaN
		if input_type != np.ndarray:
			MW = input_type(MW)
		return MW

	def get_sigma(self, MS=None):
		return None


class OkalRomanowicz1994(MSCE_MS_MW):
	"""
	Okal and Romanowicz (1994), cited by Bungum et al. (2003)

	Region: global
	Standard deviation not known
	"""
	def get_mean(self, MS):
		input_type = type(MS)
		MS = np.asarray(MS)
		MW = np.zeros_like(MS, dtype='float')
		MW[MS < 6.7] = (2./3) * MS[MS < 6.7] + 2.24
		MW[MS >= 6.7] = MS[MS >= 6.7]
		if input_type != np.ndarray:
			MW = input_type(MW)
		return MW

	def get_sigma(self, MS=None):
		return None


class ReamerHinzen2004L(MSCE_ML_MW):
	"""
	Conversion ML(Bensberg) -> log seismic moment, linear regression
	Published in:
	Reamer, S. K., & Hinzen, K. G. (2004). An earthquake catalog for the
	Northern Rhine area, Central Europe (1975-2002). Seismological
	Research Letters, 75(6), 713-725.

	Validity range: ML 2 - 6
	Region: northern Rhine area
	Standard deviation not given
	"""
	def get_mean(self, ML):
		log_Mo = 1.083 * ML + 10.215
		MW = log_moment_to_mag(log_Mo)
		return MW

	def get_sigma(self, ML=None):
		return None


class ReamerHinzen2004Q(MSCE_ML_MW):
	"""
	Conversion ML(Bensberg) -> log seismic moment, quadratic regression
	Published in:
	Reamer, S. K., & Hinzen, K. G. (2004). An earthquake catalog for the
	Northern Rhine area, Central Europe (1975-2002). Seismological
	Research Letters, 75(6), 713-725.

	Validity range: ML 2 - 6
	Region: northern Rhine area
	Standard deviation not given
	"""
	def get_mean(self, ML):
		log_Mo = 0.049 * (ML**2) + 0.703 * ML + 10.897
		MW = log_moment_to_mag(log_Mo)
		return MW

	def get_sigma(self, ML=None):
		return None


class Scordilis2006(MSCE_MS_MW):
	"""
	Conversion MS -> MW
	Published in:
	Scordilis, E. M. (2006). Empirical global relations converting Ms and
	mb to mement magnitude. Journal of Seismology, 10(2), 225-236.
	doi:10.1007/s10950-006-9012-4

	Validity range: MS 3.0 - 8.2
	Region: global
	"""
	def get_mean(self, MS):
		input_type = type(MS)
		MS = np.asarray(MS)
		MW = np.zeros_like(MS, dtype='float')
		MW[MS < 6.2] = 0.67 * MS[MS < 6.2] + 2.07
		MW[MS >= 6.2] = 0.99 * MS[MS >= 6.2] + 0.08
		MW[MS < 3.0] = np.NaN
		MW[MS > 8.2] = np.NaN
		if input_type != np.ndarray:
			MW = input_type(MW)
		return MW

	def get_sigma(self, MS):
		input_type = type(MS)
		MS = np.asarray(MS)
		sigma = np.zeros_like(MS, dtype='float')
		sigma[MS < 6.2] = 0.17
		sigma[MS >= 6.2] = 0.20
		sigma[MS < 3.0] = np.NaN
		sigma[MS > 8.2] = np.NaN
		if input_type != np.ndarray:
			sigma = input_type(sigma)
		return sigma


class Scordilis2006mb(MSCE):
	"""
	Conversion mb -> MW

	mb range: 3.5 - 6.2
	Region: global
	"""
	def get_mean(self, mb):
		if np.isscalar(mb):
			if 3.5 <= mb <= 6.2:
				MW = 0.85 * mb + 1.03
			else:
				MW = np.nan
		else:
			MW = 0.85 * mb + 1.03
			MW[(mb < 3.5) | (mb > 6.2)] = np.nan
		return MW

	def get_sigma(self, mb):
		if np.isscalar(mb):
			if 3.5 <= mb <= 6.2:
				sigma = 0.29
			else:
				sigma = np.nan
		else:
			sigma = np.zeros_like(mb, dtype='float') + 0.29
			sigma[(mb < 3.5) | (mb > 6.2)] = np.nan
		return sigma


class Utsu2002(MSCE_MS_MW):
	"""
	Conversion MS -> MW
	Cited by Gruenthal et al. (2009), and used in CENEC catalog
	Published in:

	Validity range: up to MS=7
	Standard deviation not given (maybe in original publication)
	"""
	def get_mean(self, MS):
		MW = 10.85 - np.sqrt(73.74 - 8.38 * MS)
		return MW

	def get_sigma(self, MS=None):
		return None


class IdentityMSC(MSCE):
	"""
	One-to-one conversion of magntiudes
	"""
	def get_mean(self, other_mag):
		return other_mag

	def get_sigma(self, other_mag=None):
		return None


class Suckale2016MS_USGS(MSCE_MS_MW):
	"""
	Regression between MS and MW based on the USGS/NEIC catalog
	(for Vanuatu)

	From: Suckale J., Gruenthal, G., Regnier, M. & Bosse, C., 2016,
		Probabilistic seismic hazard assessment for Vanuatu,
		Technical Report STR 05/16, GFZ
	"""
	def get_mean(self, MS):
		MW = 1.269 * MS - 1.0436
		return MW

	def get_sigma(self, MS=None):
		return None

class Suckale2016mb_USGS(MSCE):
	"""
	Regression between mb and MW based on the USGS/NEIC catalog
	(for Vanuatu)

	From: Suckale J., Gruenthal, G., Regnier, M. & Bosse, C., 2016,
		Probabilistic seismic hazard assessment for Vanuatu,
		Technical Report STR 05/16, GFZ
	"""
	def get_mean(self, mb):
		MW = 0.7813 * mb + 1.5175
		return MW

	def get_sigma(self, mb=None):
		return None


class KadiriogluKartal2016(MSCE):
	"""
	Regression between Md and MW

	From:
	Kadirioglu, F.T. & Kartal R.F., 2016, The new empirical magnitude
	conversion relations using an improved earthquake catalogue for
	Turkey and its near vicinity (1900-2012), Turkish Journal of Earth
	Sciences, 25: 300-310, doi: 0.3906/yer-1511-7

	Md range: 3.5 - 7.4
	Region: Turkey
	"""
	def get_mean(self, Md):
		MW = 0.7947 * Md + 1.342
		return MW

	def get_sigma(self, Md=None):
		return None


def get_available_Mrelations(msce_type='MSCE'):
	"""
	Function to get all magnitude scaling relations

	:param msce_type:
		str, MSCE type
		(default: 'MSCE')

	:return:
		ordered dict, mapping relation names to instances of :class:`MSCE`
	"""
	import sys, inspect
	from collections import OrderedDict

	this_mod = sys.modules[__name__]
	msce_class = getattr(this_mod, msce_type)

	def is_msce(member):
		r_val = False
		subclass = getattr
		if inspect.isclass(member):
			if issubclass(member, msce_class):
				return True
		return r_val

	Mrelations = inspect.getmembers(this_mod, is_msce)
	#Mrelations = [(name.replace('Window', ''), val) for (name, val) in windows]
	Mrelations = OrderedDict(Mrelations)
	return Mrelations
