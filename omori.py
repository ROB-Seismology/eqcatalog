"""
Afterschocks
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import pylab



class OmoriLaw(object):
	"""
	Implements Modified Omori law, representing the decay of aftershock
	activity with time.

	:param K:
		float, productivity parameter (depends on mainshock magnitude)
	:param c:
		float (time unit), small positive constant,
		usually between 0.003 and 0.3 days
	:param p:
		float, power-law coefficient, usually between 0.9 and 1.5
	:param Mm:
		float, mainshock magnitude
	:param Mc:
		float, cutoff or completeness magnitude, lower magnitude above
		which aftershocks are counted (and are assumed to be complete)
	:param time_unit:
		time unit in which Omori law is expressed: 'Y', 'W', 'D', 'h', 'm' or 's'
		(default: 'D' = days)

	Although Mm and Mc are not necessary to compute aftershock decay,
	they are characteristic of the aftershock sequence.
	Note that modifying Mm or Mc will not change the Omori law.
	"""
	def __init__(self, K, c, p, Mc, Mm, time_unit='D'):
		# TODO: time unit?
		assert K >= 0
		assert c > 0
		assert p > 0

		self.K = float(K)
		self.c = float(c)
		self.p = float(p)
		self.Mm = float(Mm)
		self.Mc = float(Mc)

		assert time_unit in ('Y', 'W', 'D', 'h', 'm', 's')
		self.time_unit = time_unit

	@property
	def delta_M(self):
		return self.Mm - self.Mc

	@property
	def time_delta(self):
		return np.timedelta64(1, self.time_unit)

	def get_aftershock_rate(self, delta_t):
		"""
		Compute rate of aftershocks at particular time since mainshock

		:param delta_t:
			float or float array, time interval since mainshock

		:return:
			float or float array
		"""
		K, c, p = self.K, self.c, self.p
		return K * (delta_t + c)**-p

	def get_num_aftershocks(self, delta_t2, delta_t1=0):
		"""
		Compute number of aftershocks in a particular time interval

		:param delta_t2:
			float, end of time interval since mainshock
		:param delta_t1:
			float, start of time interval since mainschock
			(default: 0)

		:return:
			float, number of aftershocks
		"""
		K, c, p = self.K, self.c, self.p
		if p == 1:
			N = K * (np.log(delta_t2/c + 1) - np.log(delta_t1/c + 1))
		else:
			N = K * ((delta_t1 + c)**(1-p) - (delta_t2 + c)**(1-p)) / (p - 1)
		return N

	def get_time_delta_for_n_aftershocks(self, N):
		"""
		Compute time interval since mainshock corresponding to a
		particular number of aftershocks.
		This is the inverse of :meth:`get_num_aftershocks`

		:param N:
			float, number of earthquakes since mainshock

		:return:
			float, time interval
		"""
		K, c, p = self.K, self.c, self.p
		if p == 1:
			delta_t = c * (np.exp(N / K) - 1)
		else:
			delta_t = ((N / K) * (1-p) + c**(1-p))**(1./(1-p)) - c
		return delta_t

	def get_prob_n_aftershocks(self, n, delta_t2, delta_t1=0):
		"""
		Compute probability of n aftershocks in a particular time interval,
		assuming a Poisson process.

		:param n:
			int, number of aftershocks
		:param delta_t2:
		:param delta_t1:
			see :meth:`get_num_aftershocks`

		:return:
			float, probability
		"""
		try:
			from scipy.special import factorial
		except ImportError:
			## Older scipy versions
			from scipy.misc import factorial

		N = self.get_num_aftershocks(delta_t2, delta_t1)
		return (np.exp(-N) * N**n) / factorial(n)

	def get_prob_one_or_more_aftershocks(self, delta_t2, delta_t1=0):
		"""
		Compute probability of at least one aftershock in a particular
		time interval, assuming a Poisson process.

		:param delta_t2:
		:param delta_t1:
			see :meth:`get_num_aftershocks`

		:return:
			float, probability
		"""
		N = self.get_num_aftershocks(delta_t2, delta_t1)
		return 1 - np.exp(-N)

	def get_random_num_aftershocks(self, delta_t2, delta_t1=0, size=None,
									random_seed=None):
		"""
		Randomly sample number of aftershocks in a particular time
		interval, assuming a Poisson process.

		:param delta_t2:
		:param delta_t1:
			see :meth:`get_num_aftershocks`

		:return:
			int array
		"""
		np.random.seed(random_seed)
		N = self.get_num_aftershocks(delta_t2, delta_t1)
		return np.random.poisson(N, size=size)

	def get_aftershock_duration(self, mu):
		"""
		Compute aftershock duration, this is the time necessary for
		the aftershock rate to decrease to the background level

		:param mu:
			float, background rate for M >= Mc (in same time unit)

		:return:
			float, aftershock duration (in same time unit)
		"""
		K, c, p = self.K, self.c, self.p
		return (mu / K)**(-1./p) - c

	def get_interaction_time(self, prob, delta_t):
		"""
		Compute time interval necessary to wait in order to be prob
		confident of observing the next event in the sequence at
		time interval delta_t since the mainshock

		:param prob:
			float, probability
		:param delta_t:
			float, time interval since mainshock

		:return:
			float, interaction time (interval)
		"""
		K, c, p = self.K, self.c, self.p
		return -np.log(1 - prob) * (delta_t + c)**p / K

	def determine_K(self, N, delta_t2, delta_t1=0):
		"""
		Determine K from number of earthquakes in a time interval,
		assuming c and p are known

		:param N:
			int, total number of earthquakes in time interval
		:param delta_t2:
			float, end of time interval since mainshock
		:param delta_t1:
			float, start of time interval since mainschock
			(default: 0)

		:return:
			float, K value
		"""
		p, c = self.p, self.c
		return (N * (p - 1)) / ((delta_t1 + c)**(1-p) - (delta_t2 + c)**(1-p))

	@classmethod
	def fit_rate(cls, delta_t, n, initial_guess=(1, 0.01, 1.2),
				bounds=((0, 0.001, 0.1), (np.inf, 10, 10))):
		"""
		Fit K, c and p parameters of Omori law based on aftershock rates

		:param delta_t:
			1D array, elapsed times since mainshock
		:param n:
			1D array, aftershock rates
		:param initial_guess:
			(K, c, p) tuple of floats
			(default: (1, 0.01, 1.2))
		:param bounds:
			((Kmin, cmin, pmin) (Kmax, cmax, pmax)) tuple of floats
			(default: ((0, 0.001, 0.1), (np.inf, 10, 10)))

		:return:
			(popt, pcov, perr) tuple of arrays
			- popt: fitted values of K, c, p
			- pcov: covariance matrix
			- perr: standard deviations for K, c, p
		"""
		from scipy.optimize import curve_fit

		def omori_rate(delta_t, K, c, p):
			return K * (delta_t + c)**-p

		popt, pcov = curve_fit(omori_rate, delta_t, n, p0=initial_guess,
							bounds=bounds)
		perr = np.sqrt(np.diag(pcov))
		return (popt, pcov, perr)

	@classmethod
	def fit_cumulative(cls, delta_t, N, initial_guess=(1, 0.01, 1.2),
				bounds=((0, 0.001, 0.1), (np.inf, 10, 10))):
		"""
		Fit K, c and p parameters of Omori law based on cumulative number of
		aftershocks

		:param delta_t:
			1D array, elapsed times since mainshock
		:param N:
			1D array, cumulative numbers of aftershocks
		:param initial_guess:
			(K, c, p) tuple of floats
			(default: (1, 0.01, 1.2))
		:param bounds:
			((Kmin, cmin, pmin) (Kmax, cmax, pmax)) tuple of floats
			(default: ((0, 0.001, 0.1), (np.inf, 10, 10)))

		:return:
			(popt, pcov, perr) tuple of arrays
			- popt: fitted values of K, c, p
			- pcov: covariance matrix
			- perr: standard deviations for K, c, p
		"""
		from scipy.optimize import curve_fit

		def omori_cumulative(delta_t, K, c, p):
			if np.allclose(p, 1):
				N = K * (np.log(delta_t/c + 1) - np.log(1/c + 1))
			else:
				N = K * (c**(1-p) - (delta_t + c)**(1-p)) / (p - 1)
			return N

		popt, pcov = curve_fit(omori_cumulative, delta_t, N, p0=initial_guess,
							bounds=bounds)
		perr = np.sqrt(np.diag(pcov))
		return (popt, pcov, perr)

	@classmethod
	def fit_mle(cls, delta_t, initial_guess=(0.01, 1.2),
				bounds=((1E-5, 0.1), (10, 10)), Ts=0.,
				minimize_method='Nelder-Mead'):
		"""
		Fit Omori law c and p parameters using MLE estimation,
		and determine corresponding K parameter
		based on equations 11 and 12 in Utsu et al. (1995)

		:param delta_t:
			1D array, elapsed times since mainshock
		:param initial_guess:
		:param bounds:
		:param Ts:
		:param minimize_method:
			see :func:`estimate_omori_params`

		:return:
			(K, c, p) tuple of floats
		"""
		return estimate_omori_params(delta_t, initial_guess=initial_guess, Ts=Ts,
										bounds=bounds, minimize_method=minimize_method)

	def plot_rate(self, delta_t, observed_delta_t=None, observed_rate=None,
					label='', color='b', linestyle='-', linewidth=1, marker='',
					**kwargs):
		"""
		Plot aftershock rate versus elapsed time

		:param delta_t:
			1D array, elapsed times to plot as X values
		:param observed_delta_t:
			1D array, elapsed times of observed aftershocks
			(default: None)
		:param observed_rate:
			1D array, rates of observed aftershocks
			corrsponding to :param:`observed_delta_t`
			(default: None)
		:param label:
			str, label to use for plot of predicted values
			(default: '')
		:param color:
			matplotlib color spec, color to use for plot of predicted values
			(default: 'b')
		:param linestyle:
			matplotlib linestyle spec, linestyle to use for plot of
			predicted values
			(default: '')
		:param linewidth:
			float, linewidth to use for plot of predicted values
			(default: 1)
		:param marker:
			str, matplotlib marker spec, marker symbol to use for plot of
			predicted values
			(default: '')
		:kwargs:
			additional keyword arguments understood by :func:`generic_mpl.plot_xy`

		:return:
			matplotlib Axes instance
		"""
		from plotting.generic_mpl import plot_xy

		as_rate = self.get_aftershock_rate(delta_t)
		datasets = [(delta_t, as_rate)]
		labels = [label]
		colors = [color]
		linestyles = [linestyle]
		markers = [marker]

		if observed_delta_t is not None and observed_rate is not None:
			datasets.append((observed_delta_t, observed_rate))
			if label in (None, ''):
				labels = ['Omori law']
			labels.append('Observed')
			colors.append('r')
			linestyles.append('')
			markers.append('x')

		if not 'xlabel' in kwargs:
			kwargs['xlabel'] = 'Time since mainshock (%s)' % self.time_unit
		if not 'ylabel' in kwargs:
			kwargs['ylabel'] = 'Aftershock rate (1/%s)' % self.time_unit
		if not 'xmin' in kwargs:
			kwargs['xmin'] = 0
		if not 'ymin' in kwargs:
			kwargs['ymin'] = 0
		if not 'xgrid' in kwargs:
			kwargs['xgrid'] = 1
		if not 'ygrid' in kwargs:
			kwargs['ygrid'] = 1

		return plot_xy(datasets, labels=labels, colors=colors,
						linestyles=linestyles, markers=markers, **kwargs)

	def plot_cumulative(self, delta_t, observed_delta_t=None, observed_N=None,
							label='', color='b', linestyle='-', linewidth=1,
							marker='', **kwargs):
		"""
		Plot cumulative number of aftershocks versus elapsed time

		:param delta_t:
			1D array, elapsed times to plot as X values
		:param observed_delta_t:
			1D array, elapsed times of observed aftershocks
			(default: None)
		:param observed_N:
			1D array, cumulative numbers of observed aftershocks
			corrsponding to :param:`observed_delta_t`
			(default: None)
		:param label:
			str, label to use for plot of predicted values
			(default: '')
		:param color:
			matplotlib color spec, color to use for plot of predicted values
			(default: 'b')
		:param linestyle:
			matplotlib linestyle spec, linestyle to use for plot of
			predicted values
			(default: '')
		:param linewidth:
			float, linewidth to use for plot of predicted values
			(default: 1)
		:param marker:
			str, matplotlib marker spec, marker symbol to use for plot of
			predicted values
			(default: '')
		:kwargs:
			additional keyword arguments understood by :func:`generic_mpl.plot_xy`

		:return:
			matplotlib Axes instance
		"""
		from plotting.generic_mpl import plot_xy

		num_as = self.get_num_aftershocks(delta_t)
		datasets = [(delta_t, num_as)]
		labels = [label]
		colors = [color]
		linestyles = [linestyle]
		markers = [marker]
		if observed_delta_t is not None:
			if observed_N is None:
				observed_N = np.arange(len(observed_delta_t)) + 1
			datasets.append((observed_delta_t, observed_N))
			if label in (None, ''):
				labels = ['Omori law']
			labels.append('Observed')
			colors.append('r')
			linestyles.append('')
			markers.append('x')

		if not 'xlabel' in kwargs:
			kwargs['xlabel'] = 'Time since mainshock (%s)' % self.time_unit
		if not 'ylabel' in kwargs:
			kwargs['ylabel'] = 'Number of aftershocks'
		if not 'xmin' in kwargs:
			kwargs['xmin'] = 0
		if not 'ymin' in kwargs:
			kwargs['ymin'] = 0
		if not 'xgrid' in kwargs:
			kwargs['xgrid'] = 1
		if not 'ygrid' in kwargs:
			kwargs['ygrid'] = 1

		return plot_xy(datasets, labels=labels, colors=colors,
						linestyles=linestyles, markers=markers, **kwargs)

	def to_gr_omori_law(self, b):
		"""
		Convert to Gutenberg-Richter Omori law

		:param b:
			float, b-value of Gutenberg-Richter relation

		:return:
			instance of :class:`Base10GROmoriLaw`
		"""
		A = np.log10(self.K) - b * (self.Mm - self.Mc)

		return Base10GROmoriLaw(A, self.c, self.p, b, self.Mm, self.Mc,
									time_unit=self.time_unit)


class GROmoriLaw(OmoriLaw):
	"""
	Base class for Gutenberg-Richter Omori law, where aftershock
	magnitudes follow a Gutenberg-Richter distribution.
	"""
	def get_random_magnitudes(self, n, Mmax=None, random_seed=None):
		"""
		Generate random aftershock magnitudes

		:param n:
			int, length of aftershock sequence
		:param Mmax:
			float, maximum magnitude
			(default: None, will use mainshock magnitude in :prop:`Mm`)
		:param random_seed:
			int, seed for random number generator
			(default: None)

		:return:
			float array, aftershock magnitudes
		"""
		np.random.seed(random_seed)
		rnd = np.random.random(n)
		beta = self.get_beta()
		Mc = self.Mc
		if Mmax is None:
			Mmax = self.Mm
		assert Mmax > Mc
		return (-1./beta) * np.log(np.exp(-beta*Mc)
							- (1 - rnd) * (np.exp(-beta*Mc) - np.exp(-beta*Mmax)))

	def get_random_time_deltas(self, duration, random_seed=None):
		"""
		Generate random aftershock times with respect to mainshock at t=0

		:param duration:
			float, duration of aftershock sequence (in same time units
			as Omori law)
		:param random_seed:
			int, seed for random number generator
			(default: None)

		:return:
			float array, aftershock times (in same time units as Omori law)
		"""
		np.random.seed(random_seed)
		tail_size = self.get_num_aftershocks(duration)
		num_expected = np.random.poisson(tail_size)
		n_values = tail_size * (1 - np.random.random(num_expected))
		return np.sort(self.get_time_delta_for_n_aftershocks(n_values))

	def gen_aftershock_sequence(self, duration, etas=True, Mmax=None,
									random_seed=None):
		"""
		Generate random aftershock sequence

		:param duration:
			float, duration of aftershock sequence (in same time units
			as Omori law)
		:param etas:
			bool, whether or not to apply ETAS model, in which each
			aftershock incites its own aftershock sequence
			(self-exciting process)
			(default: True)
		:param Mmax:
			float, maximum magnitude
			(default: None, will use mainshock magnitude in :prop:`Mm`)
		:param random_seed:
			int, seed for random number generator
			(default: None)

		:return:
			(delta_time, magnitude, index, parent_index) tuples
		"""
		np.random.seed(random_seed)
		current, parent = 0, -1
		mainshock = (0, self.Mm, current, parent)
		as_sequence = [mainshock]
		#yield mainshock
		for t_parent, m_parent, parent, _ in as_sequence:
			self.Mm = m_parent
			as_duration = duration - t_parent
			time_deltas = t_parent + self.get_random_time_deltas(as_duration)
			num_expected = len(time_deltas)
			magnitudes = self.get_random_magnitudes(num_expected)
			for i in range(num_expected):
				current += 1
				aftershock = (time_deltas[i], magnitudes[i], current, parent)
				yield aftershock
				if etas:
					as_sequence.append(aftershock)
		## Set mainshock magnitude back to original value
		self.Mm = mainshock[1]

	def gen_aftershock_catalog(self, duration, mainshock=None, etas=True,
									Mmax=None, Mtype='MW', random_seed=None):
		"""
		Generate random aftershock catalog

		:param duration:
		:param etas:
		:param random_seed:
			see :meth:`gen_aftershock_sequence`
		:param mainshock:
			instance of :class:`eqcatalog.LocalEarthquake`
			mainshock (used for origin time and location)
		:param Mtype:
			str, magnitude type to consider for random aftershock magnitudes

		:return:
			instance of :class:`eqcatalog.EQCatalog`
		"""
		from .time import (as_np_datetime, as_np_timedelta, as_np_date,
								to_py_time, convert_time_unit)
		from .eqrecord import LocalEarthquake
		from .eqcatalog import EQCatalog

		if mainshock:
			origin_time = mainshock.datetime
		else:
			origin_time = as_np_datetime('now')

		eq_list = []
		for item in self.gen_aftershock_sequence(duration=duration, etas=etas,
													Mmax=Mmax, random_seed=random_seed):
			time_delta, mag, index, parent_index = item
			time_delta = convert_time_unit(time_delta, self.time_unit, 'ms')
			time_delta = as_np_timedelta(int(round(time_delta)), 'ms')
			date_time = origin_time + time_delta
			date = as_np_date(date_time)
			time = to_py_time(date_time)
			# TODO: it may be possible to generate random location
			# in radius of max. interaction distance around mainshock location
			lon = getattr(mainshock, 'lon', np.nan)
			lat = getattr(mainshock, 'lat', np.nan)
			depth = getattr(mainshock, 'depth', np.nan)
			name = 'AS #%d (parent: %d)' % (index, parent_index)
			eq = LocalEarthquake(index, date, time, lon, lat, depth, {Mtype: mag},
										name=name)
			eq_list.append(eq)

		catalog = EQCatalog(eq_list)
		return catalog


class ExpGROmoriLaw(GROmoriLaw):
	"""
	Omori law where K depends on difference between mainshock and
	cutoff magnitude, and on efficiency parameter alpha, which in fact
	corresponds to the beta value (= b * ln(10)) of the aftershock
	sequence in the exponential notation of the Gutenberg-Richte relation

	:param A:
		float, productivity parameter
	:param c:
	:param p:
		see :class:`OmoriLaw`
	:param alpha:
		float, efficiency of a shock with a certain magnitude
		to generate its aftershock activity,
		ranges mostly from 0.2 to 3.0,
		lower values typically represent swarm-type activity,
		higher values represent typical mainshock-aftershock activity
	:param Mm:
	:param Mc:
	:param time_unit:
		see :class:`OmoriLaw`

	Note that, in contrast to :class:`OmoriLaw`, Mm and/or Mc can be
	changed to adjust the Omori law!
	"""
	def __init__(self, A, c, p, alpha, Mm, Mc, time_unit='D'):
		self.A = A
		self.c = c
		self.p = p
		self.alpha = alpha
		self.Mm = Mm
		self.Mc = Mc
		self.time_unit = time_unit

	@property
	def K(self):
		return self.A * np.exp(self.alpha * self.delta_M)

	def get_b(self):
		"""
		Get Gutenberg-Richter b-value corresponding to alpha
		"""
		return self.alpha * np.log10(np.e)

	def get_beta(self):
		"""
		Get Gutenberg-Richter beta value (= alpha)
		"""
		return self.alpha

	def to_base10(self):
		"""
		Convert to :class:`Base10GROmoriLaw`
		"""
		A = np.log10(self.A)
		b = self.get_b()
		return Base10GROmoriLaw(A, self.c, self.p, b, self.Mm, self.Mc,
									self.time_unit)


class Base10GROmoriLaw(GROmoriLaw):
	"""
	Omori law where K depends on difference between mainshock and
	cutoff magnitude, as well as on a constant A and the b-value
	in the base-10 notation of the Gutenberg-Richter relation.

	:param A:
		float, aftershock productivity (independent of Mm and Mc)
	:param c:
	:param p:
		see :class:`OmoriLaw`
	:param b:
		float, Gutenberg-Richter b-value (of the aftershock sequence)
	:param Mm:
	:param Mc:
	:param time_unit:
		see :class:`OmoriLaw`

	Note that, in contrast to :class:`OmoriLaw`, Mm and/or Mc can be
	changed to adjust the Omori law!
	"""
	def __init__(self, A, c, p, b, Mm, Mc, time_unit='D'):
		self.A = A
		self.c = c
		self.p = p
		self.b = b
		self.Mm = Mm
		self.Mc = Mc
		self.time_unit = time_unit

	@property
	def K(self):
		return 10**(self.A + self.b*self.delta_M)

	def get_alpha(self):
		"""
		Get alpha for equivalent exponential GROmoriLaw
		"""
		return self.b * np.log(10)

	def get_beta(self):
		"""
		Get Gutenberg-Richter beta value (= alpha)
		"""
		return self.get_alpha()

	def determine_A(self, K, Mc):
		"""
		Determine A from K and cutoff magnitude

		:param K:
			float, Omori K parameter
		:param Mc:
			float, cutoff magnitude

		:return:
			float, magnitude-independent aftershock productivity A
		"""
		return np.log10(K) - self.b * (self.Mm - Mc)

	def determine_K_for_Mc(self, new_Mc):
		"""
		Detemine K value for different cutoff magnitude

		:param new_Mc:
			float, new cutoff magnitude
		"""
		return 10 ** (np.log10(self.K) - self.b * (new_Mc - self.Mc))

	def to_exp(self):
		"""
		Convert to :class:`ExpGROmoriLaw`
		"""
		A = 10**self.A
		alpha = self.get_alpha()
		return ExpGROmoriLaw(A, self.c, self.p, alpha, self.Mm, self.Mc,
								self.time_unit)


class Reasenberg1985OmoriLaw(Base10GROmoriLaw):
	"""
	Version of Base10GROmoriLaw with A and b constants for California
	(Reasenberg, 1985)

	:param c:
	:param p:
	:param Mm:
	:param Mc:
	:param time_unit:
		see :class:`Base10GROmoriLaw`
	"""
	def __init__(self, c, p, Mm, Mc, time_unit='D'):
		A = -2./3
		b = 2./3
		super(Reasenberg1985OmoriLaw, self).__init__(A, c, p, b, Mm, Mc,
																self.time_unit)



def estimate_omori_params(as_time_deltas, initial_guess=(0.01, 1.2),
				bounds=((1E-5, 0.1), (10, 10)), Ts=0.,
				minimize_method='Nelder-Mead', verbose=False):
	"""
	MLE estimation of Omori c and p parameters by simultaneous solving
	equations 11 and 12 in Utsu et al. (1995)
	K is determined from c and p using eq. 12 in Utsu et al. (1995)

	See also:
	https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-9fbff27ea12f

	:param as_time_deltas:
		float array, fractional time differences (arbitrary time unit)
		between aftershocks and mainshock
	:param initial_guess:
		(c, p) tuple: initial estimate for c and p
	:param bounds:
		((cmin, cmax), (pmin, pmax)): bounds for c and p
		May be ignored by some minimization methods
	:param Ts:
		float, start timedelta in same time unit as :param:`as_time_deltas`
		If None, as_time_deltas[0] will be used
		(default: 0.)
	:param minimize_method:
		str, method name understood by :func:`scipy.optimize.minimize`
		(default: 'Nelder-Mead')
	:param verbose:
		bool, whether or not intermediate solutions of Utsu equations
		should be reported
		(default: False)

	:return:
		(K, c, p) tuple of floats
	"""
	from scipy.optimize import minimize

	N = len(as_time_deltas)
	if Ts is None:
		Ts = as_time_deltas[0]
	Te = as_time_deltas[-1]

	def minimize_cp(cp, args):
		c, p = cp
		Tsc_term, Tec_term = (Ts+c)**(1-p), (Te+c)**(1-p)
		denominator = Tsc_term - Tec_term

		eq1 = (np.sum(np.log(as_time_deltas + c))
				- N/(p-1.)
				- N * ((np.log(Ts+c) * Tsc_term - np.log(Te+c) * Tec_term)
				/ denominator))

		eq2 = (p * np.sum(1./(as_time_deltas + c))
				- (N * (p-1) * ((Ts+c)**-p - (Te+c)**-p)) / denominator)

		if verbose:
			print(eq1, eq2)
		return np.sum(eq1**2 + eq2**2)

	x0 = np.array(initial_guess)
	result = minimize(minimize_cp, x0, bounds=bounds, args=[],
						method=minimize_method)
	if result.success:
		c, p = result.x
		K = N * (p-1) / ((Ts+c)**(1-p) - (Te+c)**(1-p))
		return (K, c, p)
	else:
		print(result.message)


def calc_smoothed_rates(elapsed_times, n=3):
	"""
	Compute smoothed aftershock rates from elapsed times.
	The smoothed rates are computed as n divided by the
	elapsed time to the center of the time window corresponding
	to each subsequent set of n earthquakes

	:param elapsed_times:
		1D array, elapsed times since mainshock
	:param n:
		int, smoothing factor
		(default: 3)

	:return:
		(delta_t, as_rate) tuple of arrays:
			- delta_t: elapsed times
			- as_rate: aftershock rates
	"""
	## Compute time intervals for n aftershocks
	elapsed_times = elapsed_times[::n]
	time_intervals = np.diff(elapsed_times)
	delta_t = elapsed_times[:-1] + time_intervals / 2.
	as_rate = n / delta_t

	return (delta_t, as_rate)



if __name__ == "__main__":
	K, c, p = 10., 0.01, 1.
	Mm, Mc = 6.5, 2.0
	mol = OmoriLaw(K, c, p, Mm, Mc)
	dt1 = 5
	dt2 = 10
	N = mol.get_num_aftershocks(dt2)
	print(N)
