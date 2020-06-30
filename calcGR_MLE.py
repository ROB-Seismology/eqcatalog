"""
Maximum-likelihood estimation of seismic activity rate + uncertainty
according to Stromeyer & Gruenthal (2015)
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.optimize import minimize_scalar



__all__ = ['estimate_gr_params', 'estimate_gr_params_multi',
			'calc_gr_sigma',
			'construct_mfd_at_epsilon', 'construct_mfd_pmf',
			'discretize_normal_distribution']


def estimate_gr_params(ni, Mi, dMi, completeness, end_date, precise=False,
						log10=False, prior_b=1., prior_weight=0.):
	"""
	Estimate Gutenberg-Richter parameters and their associated
	uncertainties using maximum-likelihood estimation, following
	the equations in Stromeyer & Gruenthal (2015)

	:param ni:
		1D array, number of observed earthquakes in each magnitude bin
		Note:
		- NOT rates and NOT corrected for completeness!
		- may contain NaN values for bins with unknown number of
		  earthquakes
	:param Mi:
		1D array, magnitude bin centers
	:param dMi:
		float or 1D array, magnitude bin width
	:param completeness:
		instance of :class:`eqcatalog.Completeness`, catalog completeness
	:param end_date:
		int or datetime.date or datetime.datetime, end date to consider
		for seismicity rate calculation
	:param precise:
		bool, whether to use precise formulation (True) or approximation
		(False).
		Note that if there is any magnitude bin width larger than 0.5,
		the precise formulation will always be used
		(default: False)
	:param log10:
		bool, whether to return parameters for the Gutenberg-Richter
		relation in log10 notation (i.e., a and b values) or in
		the natural logarithm notation (i.e., alpha and beta values)
		(default: False)
	:param prior_b:
		float, prior estimate of the b-value (used to stabilize
		the GR estimation in regions with limited seismicity
		cf. Veneziano & Van Dyck, 1985; Johnston, 1994).
		Note: log10-notation regardless of :param:`log10` !!
		(default: 1.)
	:param prior_weight:
		float, weighting parameter of the prior b-value, can be
		considered equivalent to the inverse of its variance
		(Johnston, 1994)
		(default: 0.)

	:return:
		(a/alpha, b/beta, cov) tuple
		- a/alpha: float, a or alpha value
		- b/beta: float, b or beta value
		- cov: 2-D matrix [2,2], covariance matrix

	Note: alpha != a * ln(10)
	but exp(alpha) / beta = exp(a * ln(10))
	or a = ln(exp(alpha) / beta) / ln(10)
	"""
	I = len(ni)

	assert len(Mi) == I

	if np.isscalar(dMi):
		dMi = np.array([dMi] * I)
	assert len(dMi) == I
	## dM is half-bin size in paper!
	## Note: do not use /= to avoid modifying original array
	dMi = dMi / 2.

	ti = completeness.get_completeness_timespans(Mi, end_date)
	N = np.nansum(ni)

	beta0 = prior_b * np.log(10)
	W0 = prior_weight

	idxs = ~np.isnan(ni)

	if (dMi > 0.25).any():
		precise = True

	## Determine beta
	if not precise:
		## Small magnitude bins
		sum_niMi = np.nansum(ni * Mi)

		def minimize_func(beta):
			## Eq. 12
			exp_term = np.exp(-beta * Mi[idxs])
			common_term = ti[idxs] * dMi[idxs] * exp_term
			est_sum_niMi = (N * np.sum(Mi[idxs] * common_term)
							/ np.sum(common_term))
			penalty_term = W0 * (beta - beta0)

			#print(beta, est_sum_niMi)
			return np.abs(sum_niMi + penalty_term - est_sum_niMi)

	else:
		## "Finite" magnitude bins
		def minimize_func(beta):
			## Eq. 15
			sum_niMi = np.sum(ni[idxs] * (Mi[idxs] - dMi[idxs]
										/ np.tanh(beta * dMi[idxs])))

			exp_term = np.exp(-beta * Mi[idxs])
			sinh_term = np.sinh(beta * dMi[idxs])
			coth_term = 1. / np.tanh(beta * dMi[idxs])
			nom = (np.sum(ti[idxs] * sinh_term
						* (Mi[idxs] - dMi[idxs] * coth_term) * exp_term))
			denom = np.sum(ti[idxs] * sinh_term * exp_term)
			est_sum_niMi = N * nom / denom
			penalty_term = W0 * (beta - beta0)

			return np.abs(sum_niMi + penalty_term - est_sum_niMi)

	result = minimize_scalar(minimize_func, bounds=(0, 2*np.log(10)),
							method='bounded')

	if not result.success:
		print(result.message)
		return

	beta = result.x

	## Determine alpha
	exp_term = np.exp(-beta * Mi[idxs])
	if not precise:
		## Eq. 11
		alpha = np.log(N / np.sum(2 * ti[idxs] * dMi[idxs] * exp_term))
	else:
		## Eq. 16
		nom = beta * N
		denom = np.sum(2 * ti[idxs] * np.sinh(beta * dMi[idxs]) * exp_term)
		alpha = np.log(nom / denom)

	## determine covariance
	exp_term = np.exp(alpha - beta * Mi[idxs])
	cov_alpha = float(N)
	if not precise:
		## Eq. 14
		#cov_alpha = 2 * np.sum(ti[idxs] * dMi[idxs] * exp_term)
		cov_beta = W0 + 2 * np.sum(Mi[idxs]**2 * ti[idxs] * dMi[idxs] * exp_term)
		cov_alpha_beta = -2 * np.sum(Mi[idxs] * ti[idxs] * dMi[idxs] * exp_term)
	else:
		## Eq. 17
		sinh_term = np.sinh(beta * dMi[idxs]) / beta
		coth_term = 1. / np.tanh(beta * dMi[idxs])
		#cov_alpha = 2 * np.sum(ti[idxs] * sinh_term * exp_term)
		cov_beta = W0 + 2 * np.sum(ti[idxs] * sinh_term
							* (Mi[idxs] - dMi[idxs] * coth_term)**2 * exp_term)
		cov_alpha_beta = -2 * np.sum(ti[idxs] * sinh_term
								* (Mi[idxs] - dMi[idxs] * coth_term) * exp_term)

	## Eq. 13
	cov = 1. / np.mat([[cov_alpha, cov_alpha_beta],
						[cov_alpha_beta, cov_beta]])

	if log10:
		#a = alpha / np.log(10)

		## From calcGR_Weichert
		#SUMEXP = np.sum(np.exp(-beta * Mi))
		#SUMTEX = np.sum(ti * np.exp(-beta * Mi))
		#FNGTMO = N * SUMEXP / SUMTEX
		#a = np.log10(FNGTMO * np.exp(beta * (Mi[0]-dMi[0])))
		a = np.log(np.exp(alpha) / beta) / np.log(10)
		b = beta / np.log(10)
		## Note: covariance is probably not correct for log10=True !
		## (because relation between a and alpha is more complex),
		## but I have verified that it is approximately correct
		## Error propagation for alpha corresponding to a * ln(10)
		## is almost indistinguishable from sigma_alpha
		#sigma_alpha, sigma_beta = cov[0, 0], cov[1, 1]
		#sigma_alpha2 = np.sqrt((sigma_beta / beta)**2 + sigma_alpha**2)
		#cov[0, 0] = sigma_alpha2
		cov /= np.log(10)
		return (a, b, cov)

	else:
		return alpha, beta, cov


def estimate_gr_params_multi(Mi, dMi, nij, completeness, end_date,
							precise=False):
	"""
	Estimate Gutenberg-Richter parameters and associated uncertainties
	for a set of low-seismicity zones that are assumed to have a
	common beta value

	:param nij:
		2D array [num_zones, num_mag_bins]
		number of observed earthquakes in each bin and each zone
	:param Mi:
	:param dMi:
	:param completeness:
	:param end_date:
	:param precise:
		see :func:`estimate_gr_params`

	Note: it is assumed that magnitude bins and completeness
	are the same for all zones!

	:return:
		(alpha_values, beta, cov_list):
		- alpha_values: list of alpha values for each zone
		- beta: float, common beta value
		- cov_list: list of 2D matrices, covariance matrices for each zone
	"""
	J, I = nij.shape

	assert len(Mi) == I

	if np.isscalar(dMi):
		dMi = np.array([dMi] * I)
	assert len(dMi) == I
	## dM is half-bin size in paper!
	## Note: do not use /= to avoid modifying original array
	dMi = dMi / 2.

	ti = completeness.get_completeness_timespans(Mi, end_date)

	if (dMi > 0.25).any():
		precise = True

	## Determine common beta value
	if not precise:
		## Small magnitude bins
		sum_nM = np.sum([np.nansum(nij[j] * Mi) for j in range(J)])

		def minimize_func(beta):
			## Eq. B3
			est_sum_nM = 0
			exp_term = np.exp(-beta * Mi)
			for j in range(J):
				ni = nij[j]
				idxs = np.isnan(ni)
				Nj = np.nansum(ni)
				nom = np.sum(ti[idxs] * dMi[idxs] * Mi[idxs] * exp_term)
				denom = np.sum(ti[idxs] * dMi[idxs] * exp_term)
				est_sum_nM += (Nj * nom / denom)

			return np.abs(sum_niMi - est_sum_niMi)

	else:
		## "Finite" magnitude bins
		def minimize_func(beta):
			## Eq. B8
			sum_nM = 0
			est_sum_nM = 0
			for j in range(J):
				ni = nij[j]
				idxs = np.isnan(ni)
				coth_term = 1./tanh(beta * dMi[idxs])
				sum_nM += np.sum(ni[idxs] * (Mi[idxs] - dMi[idxs] * coth_term))

				Nj = np.nansum(ni)
				sinh_term = np.sinh(beta * dMi[idxs])
				exp_term = np.exp(-beta * Mi[idxs])
				nom = np.sum(ti[idxs] * sinh_term
							* (Mi[idxs] - dMi[idxs] * coth_term) * exp_term)
				denom = np.sum(ti[idxs] * sinh_term * exp_term)
				est_sum_nM += (Nj * nom / denom)

			return np.abs(sum_niMi - est_sum_niMi)

	result = minimize_scalar(minimize_func, bounds=(0, 2*np.log(10)),
							method='bounded')

	if not result.success:
		print(result.message)
		return

	beta = result.x

	## Determine alpha values
	## Eq. B4
	alpha_values = []
	for j in range(J):
		ni = nij[j]
		idxs = np.isnan(ni)
		Nj = np.nansum(ni)
		if not precise:
			denom = np.sum(2 * ti[idxs] * dMi[idxs] * np.exp(-beta * Mi[idxs]))
			alpha = np.log(Nj / denom)

		else:
			exp_term = np.exp(-beta * Mi[idxs])
			denom = np.sum(2 * ti[idxs] * np.sinh(beta * dMi[idxs]) * exp_term)
			alpha = np.log(beta * Nj / denom)
		alpha_values.append(alpha)

	## Determine covariance matrices
	## Eq. B6
	cov_list = []
	cov_beta = 0.
	for j in range(J):
		ni = nij[j]
		idxs = np.isnan(ni)
		Nj = np.nansum(ni)
		cov_alpha = float(Nj)
		exp_term = np.exp(alpha_values[j] - beta * Mi[idxs])
		if not precise:
			cov_beta += np.sum(Mi[idxs]**2 * ti[idxs] * dMi[idxs] * exp_term)
			cov_alpha_beta = -2 * np.sum(Mi[idxs] * ti[idxs] * dMi[idxs] * exp_term)
		else:
			sinh_term = np.sinh(beta * dMi[idxs]) / beta
			coth_term = 1. / tanh(beta * dMi[idxs])
			cov_beta += np.sum(ti[idxs] * sinh_term
								* (Mi[idxs] - dMi[idxs] * coth_term)**2 * exp_term)
			cov_alpha_beta = -2 * np.sum(ti[idxs] * sinh_term
								* (Mi[idxs] - dMi[idxs] * coth_term) * exp_term)
		cov = np.mat([[cov_alpha, cov_alpha_beta],
					[cov_alpha_beta, 0.]])
		cov_list.append(cov)
	cov_beta *= 2
	for j in range(J):
		## Add missing cov_beta value
		cov_list[j][1,1] = cov_beta
		## And inverse matrix!
		cov_list[j] = 1./cov_list[j]

	return (alpha_values, beta, cov_list)


def calc_gr_sigma(Mi, cov):
	"""
	Compute magnitude-dependent standard deviation on the GR relation
	(alpha - beta * M or a - b * M) based on the covariance matrix

	:param Mi:
		1D array, magnitudes
	:param cov:
		2D matrix, covariance matrix

	:return:
		1D array, standard deviations corresponding to :param:`Mi`
	"""
	## Eq. 18
	ar_m = np.mat([np.ones_like(Mi), -Mi])
	var_m = (ar_m.T * cov * ar_m).diagonal()
	sigma_m = np.sqrt(np.array(var_m)[0])

	return sigma_m


## Note: the following function could be integrated in rshalib.mfd.TruncatedGRMFD,
def construct_mfd_at_epsilon(alpha, beta, cov, epsilon, Mmin, Mmax, dM,
							precise=True, log10=False):
	"""
	Construct magnitude-frequency distribution (MFD) corresponding to a
	particular epsilon value

	:param alpha:
		float, alpha or a value
	:param beta:
		float, beta or b value
	:param cov:
		2D matrix, covariance matrix
	:param epsilon:
		float, epsilon value (= number of standard deviations
		above/below the mean)
	:param Mmin:
		float, minimum magnitude of the MFD (left edge of first bin)
	:param Mmax:
		float, maximum magnitude of the MFD (right edge of last bin)
	:param dM:
		float, fixed magnitude bin width of the MFD
	:param precise:
		bool, whether to use precise formulation (True) or approximation
		(False) of the MFD frequencies.
		Only applies if :param:`log10` is false
		(default: False)
	:param log10:
		bool, whether :param:`alpha` and :param:`beta` of the
		Gutenberg-Richter relation correspond to log10 notation
		(i.e., a and b values) or to the natural logarithm notation
		(i.e., alpha and beta values)
		(default: False)

	:return:
		instance of :class:`rshalib.mfd.EvenlyDiscretizedMfd`
		or (if :param:`epsilon` = 0 and :param:`log10` is True)
		instance of :class:`rshalib.mfd.TruncatedGRMFD`
	"""
	from hazard.rshalib.mfd import (TruncatedGRMFD, EvenlyDiscretizedMFD)

	if epsilon == 0 and log10 is True:
		#Mmin = Mi[0] - dM/2.
		#Mmax = Mi[-1] + dM/2.
		a_val, b_val = alpha, beta
		mfd = TruncatedGRMFD(Mmin, Mmax, dM, a_val, b_val)
	else:
		num_bins = np.round((Mmax - Mmin) / dM)
		Mi = Mmin + np.arange(num_bins) * dM + dM/2.
		sigma_m = calc_gr_sigma(Mi, cov)
		if log10:
			a_val, b_val = alpha, beta
			Ndisc = (10**(a_val - b_val * (Mi - dM/2.) + sigma_m * epsilon)
					-10**(a_val - b_val * (Mi + dM/2.) + sigma_m * epsilon))
		else:
			Ndisc = 2 * np.exp(alpha - beta * Mi + sigma_m * epsilon)
			if not precise:
				Ndisc *= (dM / 2.)
			else:
				## Note: probably need to add uncertainty on beta
				Ndisc *= (np.sinh(beta * dM / 2.) / beta)

		Mmin = Mi[0]
		mfd = EvenlyDiscretizedMFD(Mmin, dM, Ndisc)

	return mfd


def discretize_normal_distribution(k):
	"""
	Determine epsilon values and corresponding weights that represent
	an optimal approximation of the standard normal distribution
	according to the approved procedure by Miller & Rice (1983) based
	on Gaussian quadrature.

	A discretization with k sampling points can match at least
	the first (2k - 1) moments exactly

	:param k:
		int, number of sampling points, either 1, 3, 4 or 5

	:return:
		instance of :class:`rshalib.NumericPMF`, with:
		- values: 1D array, values of the normalized variable z
		- weights: 1D array, corresponding weights
	"""
	from hazard.rshalib.pmf import NumericPMF

	if k == 1:
		epsilons = np.array([0.])
		weights = np.array([1])
	## Optimal sampling point positions according to Miller & Rice (1983)
	elif k == 3:
		SQRT3 = np.sqrt(3)
		epsilons = np.array([-SQRT3, 0., SQRT3])
		weights = np.array([1., 4., 1.]) / 6
	elif k == 4:
		SQRT6 = np.sqrt(6)
		epsilons = np.array([-np.sqrt(3 + SQRT6), -np.sqrt(3 - SQRT6),
							np.sqrt(3 - SQRT6), np.sqrt(3 + SQRT6)])
		weights = np.array([(3 - SQRT6), (3 + SQRT6),
							(3 + SQRT6), (3 - SQRT6)]) / 12.
	elif k == 5:
		SQRT10 = np.sqrt(10)
		epsilons = np.array([-np.sqrt(5 + SQRT10), -np.sqrt(5 - SQRT10),
							0., np.sqrt(5 - SQRT10), np.sqrt(5 + SQRT10)])
		weights = np.array([7 - 2*SQRT10, 7 + 2*SQRT10, 32.,
							7 + 2*SQRT10, 7 - 2*SQRT10]) / 60.
	else:
		raise Exception("%d number of discretizations not supported!" % k)

	#epsilon_pmf = NumericPMF(zip(epsilons, weights))
	#return epsilon_pmf

	return (epsilons, weights)


def construct_mfd_pmf(alpha, beta, cov, Mmin, Mmax_pmf, dM,
					num_discretizations, precise=True):
	"""
	Construct a probability mass function of MFDs optimally sampling
	the uncertainty on the activity rates represented by the
	covariance matrix

	Note that this is not entirely correct, because the GR parameters
	estimated using :func:`estimate_gr_params` depend on Mmax
	(empty bins beyond bin with largest observed magnitude),
	but this dependency is only slight and could be ignored.

	:param alpha:
		float, alpha value of the GR relation
	:param beta:
		float, beta value of the GR relation
	:param cov:
		2D matrix [2,2], covariance matrix
	:param Mmin:
		float, minimum magnitude of the MFD (left edge of first bin)
	:param Mmax_pmf
		float, maximum magnitude of the MFD (right edge of last bin)
		or instance of :class:`rshalib.pmf.MmaxPMF`, probability
		mass function of Mmax values
	:param dM:
		float, fixed magnitude bin width of the MFD
	:param num_discretizations:
		int, number of sampling points of the uncertainty,
		either 1, 3, 4 or 5
	:param precise:
		bool, whether to use precise formulation (True) or approximation
		of the MFD frequencies (False).
		(default: False)

	:return:
		(mfd_list, weights) tuple
		- mfd_list: list with instances of
		  :class:`rshalib.mfd.EvenlyDiscretizedMFD`
		- weights: 1D array
	"""
	from hazard.rshalib.pmf import MmaxPMF, MFDPMF

	if np.isscalar(Mmax_pmf):
		Mmax_pmf = MmaxPMF([Mmax_pmf], [1])

	epsilons, eps_weights = discretize_normal_distribution(num_discretizations)

	mfd_list, weights = [], []
	for Mmax, mmax_weight in Mmax_pmf:
		for epsilon, eps_weight in zip(epsilons, eps_weights):
			weight = float(mmax_weight) * eps_weight
			mfd = construct_mfd_at_epsilon(alpha, beta, cov, epsilon,
											Mmin, Mmax, dM,
											precise=precise, log10=False)
			mfd_list.append(mfd)
			weights.append(weight)

	weights = np.array(weights)

	return (mfd_list, weights)



if __name__ == "__main__":
	pass
