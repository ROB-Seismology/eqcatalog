"""
Magnitude Scale Conversion Equation (MSCE) module
"""


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MultipleLocator

from eqcatalog import read_catalogSQL


datasets = [
	{'name': 'roermond_aftershocks', 'ML': [2.4, 2.5, 2.5, 3.4, 2.6, 3.0, 3.2, 2.7, 2.7, 2.2, 2.0, 2.3, 2.8, 2.9, 1.9, 2.0, 2.3, 2.5], 'MW': [2.5, 2.6, 2.5, 3.2, 2.5, 2.9, 3.0, 2.6, 2.8, 2.3, 2.2, 2.5, 2.8, 2.8, 2.2, 2.2, 2.4, 2.5]},
	{'name': 'liege', 'ML': 5.0, 'MS': 4.4, 'MW': 4.8},
	{'name': '??', 'ML': 4.4, 'MS': 4.0, 'MW': 4.4},
	{'name': '??', 'ML': 4.5, 'MS': 4.2, 'MW': 4.5},
]


class MSCE(object):
	"""
	"""
	pass


class MSCE_ML_MW(MSCE):
	"""
	"""
	_FROM = 'ML'
	_TO = 'MW'


class MSCE_MS_MW(MSCE):
	"""
	"""
	_FROM = 'MS'
	_TO = 'MW'


class Ahorner1983(MSCE_ML_MW):
	"""
	Conversion ML -> log seismic moment
	Published in:
	Ahorner, L. (1983). Historical seismicity and present-day microearthquake
	activity of the Rhenish Massif, Central Europe. In K. Fuchs, K. von Gehlen,
	H. Maelzer, H. Murawski, and A. Semmel (Eds.), Plateau Uplift (pp. 198-221).
	Berlin, Heidelberg: Springer-Verlag.

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
		input_type = type(ML)
		ML = np.array(ML, copy=False, ndmin=1)
		log_Mo_dyncm = 17.4 + 1.1 * ML
		MW = (2. / 3.) * log_Mo_dyncm - 10.73
		if input_type != np.ndarray:
			MW = input_type(MW)
		return MW

	def get_sigma(self, ML):
		"""
		Return standard deviation of MW value

		:param ML:
			scalar, list or array: local magnitude(s)

		:return:
			standard deviation(s), same type as input
		"""
		input_type = type(ML)
		sigma_log_Mo = 0.21
		sigma = (2. / 3.) * sigma_log_Mo * np.ones_like(ML)
		if input_type != np.ndarray:
			sigma = input_type(sigma)
		return sigma



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
		input_type = type(MS)
		MS = np.array(MS, copy=False, ndmin=1)
		## Events shallower than 30 km
		log_Mo_dyncm = 23.123 - 0.505 * MS + 0.140 * MS**2
		## Regression not taking deth into account
		#log_Mo_dyncm = 24.401 - 0.895 * MS + 0.170 * MS**2
		MW = (2. / 3.) * log_Mo_dyncm - 10.73
		if input_type != np.ndarray:
			MW = input_type(MW)
		return MW

	def get_sigma(self, MS):
		input_type = type(MS)
		sigma_log_Mo = 0.225
		sigma = (2. / 3.) * sigma_log_Mo * np.ones_like(MS)
		if input_type != np.ndarray:
			sigma = input_type(sigma)
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
		MS = np.array(MS, copy=False, ndmin=1)
		MW = np.zeros_like(MS)
		MW[MS >= 5.4] = 0.796 * MS[MS >= 5.4] + 1.280
		MW[MS < 5.4] = 0.585 * MS[MS < 5.4] + 2.422
		if input_type != np.ndarray:
			MW = input_type(MW)
		return MW

	def get_sigma(self, MS=None):
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
		MS = np.array(MS, copy=False, ndmin=1)
		log_Mo_dyncm = np.zeros_like(MS)
		log_Mo_dyncm[MS < 6.76] = MS[MS < 6.76] + 18.89
		log_Mo_dyncm[MS >= 6.76] = (3. / 2.) * MS[MS >= 6.76] + 15.51
		log_Mo_dyncm[MS >= 8.12] = 3. * MS[MS >= 8.12] + 3.33
		log_Mo_dyncm[MS >= 8.22] = np.NaN
		MW = (2. / 3.) * log_Mo_dyncm - 10.73
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


class Gruenthal2003(MSCE_ML_MW):
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
		MS = np.array(MS, copy=False, ndmin=1)
		MW = np.zeros_like(MS)
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
		MS = np.array(MS, copy=False, ndmin=1)
		MW = np.zeros_like(MS)
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
		MW = (2. / 3.) * log_Mo - 6.06
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
		MW = (2. / 3.) * log_Mo - 6.06
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
		MS = np.array(MS, copy=False, ndmin=1)
		MW = np.zeros_like(MS)
		MW[MS < 6.2] = 0.67 * MS[MS < 6.2] + 2.07
		MW[MS >= 6.2] = 0.99 * MS[MS >= 6.2] + 0.08
		MW[MS < 3.0] = np.NaN
		MW[MS > 8.2] = np.NaN
		if input_type != np.ndarray:
			MW = input_type(MW)
		return MW

	def get_sigma(self, MS):
		input_type = type(MS)
		MS = np.array(MS, copy=False, ndmin=1)
		sigma = np.zeros_like(MS)
		sigma[MS < 6.2] = 0.17
		sigma[MS >= 6.2] = 0.20
		sigma[MS < 3.0] = np.NaN
		sigma[MS > 8.2] = np.NaN
		if input_type != np.ndarray:
			sigma = input_type(sigma)
		return sigma


#class Utsu2002(MSCE_MS_MW):
#	"""
#	Conversion MS -> MW
#	Cited by Gruenthal et al. (2009), and used in CENEC catalog
#	Published in:

#	Validity range: up to MS=7
#	Standard deviation not given (maybe in original publication)
#	"""
#	def get_mean(self, MS):
#		MW = 10.85 - np.sqrt(73.74 - 8.38 * MS)
#		return MW

#	def get_sigma(self, MS=None):
#		return None



def plot(msces, datasets=[], Mmin=1, Mmax=7.1, dM=0.1, fig_filespec=None, dpi=None, fig_width=None):
	"""
	"""
	mags = np.arange(Mmin, Mmax, dM)

	if issubclass(msces, MSCE):
		msces = msces.__subclasses__()

	## plot MSCEs
	for msce in msces:
		plt.plot(mags, msce().get_mean(mags), label=msce.__name__)

	## plot data from catalog
	ec = read_catalogSQL()
	for e in ec:
		if getattr(e, msce._FROM) and getattr(e, msce._TO):
			plt.scatter(getattr(e, msce._FROM), getattr(e, msce._TO))
	
	## plot additional data
	for dataset in datasets:
		if msce._FROM in dataset and msce._TO in dataset:
			plt.scatter(dataset[msce._FROM], dataset[msce._TO])

	## plot 1:1 relation
	plt.plot(mags, mags, color='0.75', label='1:1 relation')

	plt.axis((mags[0]-1, mags[-1]+1, mags[0]-1, mags[-1]+1))

	plt.grid()
	plt.legend(loc=2)

	majorLocator = MultipleLocator(1.)
	minorLocator = MultipleLocator(dM)
	ax = plt.gca()
	ax.xaxis.set_major_locator(majorLocator)
	ax.xaxis.set_minor_locator(minorLocator)
	ax.yaxis.set_major_locator(majorLocator)
	ax.yaxis.set_minor_locator(minorLocator)

	plt.xlabel(msce._FROM)
	plt.ylabel(msce._TO)

	plt.title('%s to %s magnitude scale conversion relations' % (msce._FROM, msce._TO))

	plt.tight_layout()

	if fig_filespec:
		default_figsize = plt.rcParams['figure.figsize']
		default_dpi = plt.rcParams['figure.dpi']
		if fig_width:
			fig_width /= 2.54
			dpi = dpi * (fig_width / default_figsize[0])
		plt.savefig(fig_filespec, dpi=dpi)
	else:
		plt.show()


if __name__ == '__main__':
	"""
	"""

#	fig_filespec = r'D:\Temp\fig.png'
	fig_filespec = None

#	plot(MSCE_ML_MW, datasets=[roermond_aftershocks], fig_filespec=fig_filespec)

#	thierry_data = ([4.4, 4.5], [4.4, 4.5])



	plot(MSCE_MS_MW, datasets=datasets, fig_filespec=fig_filespec)
