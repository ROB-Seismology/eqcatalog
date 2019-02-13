"""
Afterschocks
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import pylab



class OmoriLaw():
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
	:param Mc:
		float, cutoff magnitude, lower magnitude above which aftershocks
		are counted
	"""
	def __init__(self, K, c, p, Mc, time_delta=np.timedelta64(1, 'D')):
		# TODO: time unit?
		assert K >= 0
		assert c > 0
		assert p > 0

		self.K = K
		self.c = c
		self.p = p
		self.Mc = Mc

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

	def get_prob_one_or_more_aftershocks(self, delta_t2, delta_t1=0):
		"""
		Compute probability of at least one aftershock in a particular
		time interval

		:param delta_t2:
		:param delta_t1:
			see :meth:`get_num_aftershocks`

		:return:
			float, probability
		"""
		N = self.get_num_aftershocks(delta_t2, delta_t1)
		return 1 - np.exp(-N)

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
		confident of observing the next event in the sequance at
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
		from scipy.optimize import curve_fit

		def omori_rate(delta_t, K, c, p):
			return K * (delta_t + c)**-p

		popt, pcov = curve_fit(omori_rate, delta_t, n, p0=initial_guess)
		#					bounds=bounds)
		perr = np.sqrt(np.diag(pcov))
		return (popt, perr)

	@classmethod
	def fit_cumulative(cls, delta_t, N, initial_guess=(1, 0.01, 1.2),
				bounds=((0, 0.001, 0.1), (np.inf, 10, 10))):
		"""
		Note: delta_t not necessarily binned
		"""
		from scipy.optimize import curve_fit

		def omori_cumulative(delta_t, K, c, p):
			if np.allclose(p, 1):
				N = K * (np.log(delta_t/c + 1) - np.log(1/c + 1))
			else:
				N = K * (c**(1-p) - (delta_t + c)**(1-p)) / (p - 1)
			return N

		popt, pcov = curve_fit(omori_cumulative, delta_t, N, p0=initial_guess)
		#					bounds=bounds)
		perr = np.sqrt(np.diag(pcov))
		return (popt, perr)

	def plot_rate(self, delta_t):
		as_rate = self.get_aftershock_rate(delta_t)
		pylab.plot(delta_t, as_rate)
		pylab.xlabel('Time since mainshock')
		pylab.ylabel('Aftershock rate')
		pylab.show()

	def plot_cumulative(self, delta_t):
		num_as = self.get_num_aftershocks(delta_t)
		pylab.plot(delta_t, num_as)
		pylab.xlabel('Time since mainshock')
		pylab.ylabel('Number of aftershocks')
		pylab.show()


class EtasOmoriLaw(OmoriLaw):
	"""
	Omori law where K depends on difference between mainshock and
	cutoff magnitude, and on efficiency parameter alpha

	:param Mm:
		float, mainshock magnitude
	:param Mc:
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
	"""
	def __init__(self, Mm, Mc, A, c, p, alpha):
		self.Mm = Mm
		self.Mc = Mc
		self.A = K
		self.c = c
		self.p = p
		self.alpha = alpha

	@property
	def delta_M(self):
		return self.Mm - self.Mc

	@property
	def K(self):
		return self.A * np.exp(self.alpha * self.delta_M)


class ReasenbergOmoriLaw(OmoriLaw):
	"""
	Omori law where K depends on difference between mainshock and
	cutoff magnitude according to the relation in Reasenberg (1985)

	:param Mm:
		float, mainshock magnitude
	:param Mc:
	:param c:
	:param p:
		see :class:`OmoriLaw`
	"""
	def __init__(self, Mm, Mc, c, p):
		self.Mm = Mm
		self.Mc = Mc
		self.c = c
		self.p = p

	@property
	def delta_M(self):
		return self.Mm - self.Mc

	@property
	def K(self):
		return 10**(2 * (self.delta_M - 1.) / 3.)

	def get_etas_rate(self, alpha, delta_t):
		"""
		:param alpha:
			float, efficiency of a shock with a certain magnitude
			to generate its aftershock activity,
			ranges mostly from 0.2 to 3.0,
			lower values typically represent swarm-type activity,
			higher values represent typical mainshock-aftershock activity
		:param delta_t:
			float or float array, time interval since mainshock

		:return:
			float or float array
		"""
		return np.exp(alpha * self.delta_M) * self.get_aftershock_rate(delta_t)

	def get_etas_num_aftershocks(self, alpha, delta_t2, delta_t1=0):
		return np.exp(alpha * self.delta_M) * self.get_num_aftershocks(delta_t2, delta_t1)


class GROmoriLaw(ReasenbergOmoriLaw):
	"""
	Omori law where K depends on difference between mainshock and
	cutoff magnitude, as well as on Gutenberg-Richter b-value and
	a constant A

	:param Mm:
	:param Mc:
	:param c:
	:param p:
		see :class:`ReasenbergOmoriLaw`
	:param A:
		float, constant
	:param b:
		float, Gutenberg-Richter b-value
	"""
	def __init__(self, Mm, Mc, c, p, A, b):
		super(GROmoriLaw, self).__init__(Mm, Mc, c, p)
		self.A = A
		self.b = b

	@property
	def K(self):
		return 10**(A + b*self.delta_M)



if __name__ == "__main__":
	K, c, p = 10., 0., 1.
	Mc = 2.0
	mol = OmoriLaw(K, c, p, Mc)
	dt1 = 5
	dt2 = 10
	N = mol.get_num_aftershocks(dt2)
	print(N)
