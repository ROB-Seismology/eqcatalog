"""
Maximum-likelihood estimation of seismic activity rate + uncertainty
according to Stromeyer & Gruenthal (2015)
"""

from __future__ import absolute_import, division, print_function

import numpy as np



__all__ = ['estimate_gr_params', 'estimate_gr_params_multi',
			'estimate_gr_params_minimize', 'estimate_gr_params_multi_minimize',
			'estimate_gr_params_curvefit', 'calc_gr_sigma',
			'construct_mfd_at_epsilon', 'construct_mfd_pmf',
			'discretize_normal_distribution']


## Upper bound on b-value applied in bound-constrained minimization
MAX_B_VAL = 2.0



def estimate_gr_params(ni, Mi, dMi, completeness, end_date, precise=False,
						log10=False, prior_b=1., prior_weight=0.):
	"""
	Estimate Gutenberg-Richter parameters and their associated
	uncertainties using maximum-likelihood estimation, following
	the equations in Stromeyer & Gruenthal (2015)

	An important property of MLE is that it always results in a MFD where the
	cumulative number of events is equal to the observed cumulative number
	(i.e., the cumulative MFD always goes through the 1st data point, which
	acts like a kind of hinge point)!

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
		considered equivalent to the inverse of the maximum allowed variance
		(Johnston, 1994). Note that low prior_weights may result in huge
		standard deviations on the b-value, but setting it to a value corresponding
		to an acceptable standard deviation (e.g., 4, equivalent to variance of
		0.25 and standard devation of 0.5) will result in a b-value very close
		to the prior, so use with caution!
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
	from scipy.optimize import minimize_scalar

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

	result = minimize_scalar(minimize_func, bounds=(0, MAX_B_VAL*np.log(10)),
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
		a = np.log(np.exp(alpha) / beta) / np.log(10)
		b = beta / np.log(10)
		## Note: covariance_a_b is probably not entirely correct for log10=True !
		## but I have verified that it is approximately correct
		sigma_alpha, sigma_beta = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
		sign = np.sign(cov[0, 1])
		sigma_alpha_beta = np.sqrt(np.abs(cov[0, 1]))
		sigma_b = sigma_beta / np.log(10)
		sigma_a = np.sqrt((sigma_beta / beta)**2 + sigma_alpha**2)
		sigma_a_b = sigma_alpha_beta / np.log(10)
		cov[0, 0] = sigma_a**2
		cov[1, 1] = sigma_b**2
		cov[0, 1] = cov[1, 0] = sign * sigma_a_b**2
		return (a, b, cov)

	else:
		return alpha, beta, cov


def estimate_gr_params_multi(nij, Mi, dMi, completeness, end_date,
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
	from scipy.optimize import minimize_scalar

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
		sum_niMi = np.sum([np.nansum(nij[j] * Mi) for j in range(J)])

		def minimize_func(beta):
			## Eq. B3
			est_sum_niMi = 0
			exp_term = np.exp(-beta * Mi)
			for j in range(J):
				ni = nij[j]
				idxs = ~np.isnan(ni)
				Nj = np.nansum(ni)
				nom = np.sum(ti[idxs] * dMi[idxs] * Mi[idxs] * exp_term[idxs])
				denom = np.sum(ti[idxs] * dMi[idxs] * exp_term[idxs])
				est_sum_niMi += (Nj * nom / denom)

			return np.abs(sum_niMi - est_sum_niMi)

	else:
		## "Finite" magnitude bins
		def minimize_func(beta):
			## Eq. B8
			sum_niMi = 0
			est_sum_niMi = 0
			for j in range(J):
				ni = nij[j]
				idxs = ~np.isnan(ni)
				coth_term = 1./np.tanh(beta * dMi[idxs])
				sum_niMi += np.sum(ni[idxs] * (Mi[idxs] - dMi[idxs] * coth_term))

				Nj = np.nansum(ni)
				sinh_term = np.sinh(beta * dMi[idxs])
				exp_term = np.exp(-beta * Mi[idxs])
				nom = np.sum(ti[idxs] * sinh_term
							* (Mi[idxs] - dMi[idxs] * coth_term) * exp_term)
				denom = np.sum(ti[idxs] * sinh_term * exp_term)
				est_sum_niMi += (Nj * nom / denom)

			return np.abs(sum_niMi - est_sum_niMi)

	result = minimize_scalar(minimize_func, bounds=(0, MAX_B_VAL*np.log(10)),
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
		idxs = ~np.isnan(ni)
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
		idxs = ~np.isnan(ni)
		Nj = np.nansum(ni)
		cov_alpha = float(Nj)
		exp_term = np.exp(alpha_values[j] - beta * Mi[idxs])
		if not precise:
			cov_beta += np.sum(Mi[idxs]**2 * ti[idxs] * dMi[idxs] * exp_term)
			cov_alpha_beta = -2 * np.sum(Mi[idxs] * ti[idxs] * dMi[idxs] * exp_term)
		else:
			sinh_term = np.sinh(beta * dMi[idxs]) / beta
			coth_term = 1. / np.tanh(beta * dMi[idxs])
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


def estimate_gr_params_minimize(ni, Mi, dMi, completeness, end_date,
										prior_b=None, max_b_var=0.001):
	"""
	Maximum-likelihood estimation of Gutenberg-Richter parameters by minimizing
	the negative log-likelihood function directly (instead of the derived formulas)

	This essentially gives the same results as :func:`estimate_gr_params`,
	but with slightly larger uncertainties

	Notes:
		- finite bins not yet supported!

	:param ni:
	:param Mi:
	:param dMi:
	:param completeness:
	:param end_date:
	:param prior_b:
		see :func:`estimate_gr_params`
	:param max_b_var:
		float, maximum variation allowed for b value if :param:`prior_b` is set
		(default: 0.001)

	:return:
		(a/alpha, b/beta, cov) tuple
		- a/alpha: float, a or alpha value
		- b/beta: float, b or beta value
		- cov: 2-D matrix [2,2], covariance matrix
	"""
	from scipy.optimize import minimize, minimize_scalar

	try:
		from scipy.special import factorial
	except ImportError:
		## Older scipy versions
		from scipy.misc import factorial

	I = len(ni)

	assert len(Mi) == I

	if np.isscalar(dMi):
		dMi = np.array([dMi] * I)
	assert len(dMi) == I
	## dM is half-bin size in paper!
	## Note: do not use /= to avoid modifying original array
	dMi = dMi / 2.

	ti = completeness.get_completeness_timespans(Mi, end_date)
	idxs = ~np.isnan(ni)

	if prior_b:
		## Use max_b_var to define bounds on b(eta) value
		b_bounds = (prior_b-max_b_var, prior_b+max_b_var)
	else:
		prior_b = 1.0
		b_bounds = (-np.inf, np.inf)
	prior_beta = prior_b * np.log(10)
	beta_bounds = [val * np.log(10) for val in b_bounds]

	# TODO: finite bins

	def minimize_func(x):
		## Function returning negative log-likelihood, which should be minimized
		## Eq. 7
		alpha, beta = x
		abM_term = alpha - beta * Mi[idxs]
		tdM_term = 2 * ti[idxs] * dMi[idxs]
		log_likelihood = np.sum(ni[idxs] * (abM_term + np.log(tdM_term))
									- np.log(factorial(ni[idxs]))
									- tdM_term * np.exp(abM_term))
		return -log_likelihood

	## Initial guess
	beta0 = np.log(10)
	N = np.nansum(ni)
	exp_term = np.exp(-beta0 * Mi[idxs])
	alpha0 = np.log(N / np.sum(2 * ti[idxs] * dMi[idxs] * exp_term))
	x0 = np.array([alpha0, beta0])

	result = minimize(minimize_func, x0, options={'ftol': 1E-9},
							bounds=[(None, None), beta_bounds])

	if not result.success:
		print(result.message)
		return

	# Compute covariance matrix from minimization result
	# See: https://stackoverflow.com/questions/43593592/errors-to-fit-parameters-of-scipy-optimize
	#ftol = 2.220446049250313e-09
	cov = result.hess_inv.todense() #* max(1, abs(result.fun)) * ftol

	return (result.x[0], result.x[1], cov)


def estimate_gr_params_multi_minimize(nij, Mi, dMi, completeness, end_date):
	"""
	'minimize' version of :func:`estimate_gr_params_multi`, similar to
	:func:`estimate_gr_params_minimize`.
	This function is not yet finished and should not be used!

	Notes:
		- log10 case not yet supported!
		- finite bins not yet supported!

	:param nij:
	:param Mi:
	:param dMi:
	:param completeness:
	:param end_date:
		see :func:`estimate_gr_params_multi`
	"""
	from scipy.optimize import minimize

	try:
		from scipy.special import factorial
	except ImportError:
		## Older scipy versions
		from scipy.misc import factorial

	J, I = nij.shape

	assert len(Mi) == I

	if np.isscalar(dMi):
		dMi = np.array([dMi] * I)
	assert len(dMi) == I
	## dM is half-bin size in paper!
	## Note: do not use /= to avoid modifying original array
	dMi = dMi / 2.

	ti = completeness.get_completeness_timespans(Mi, end_date)

	## Determine common beta value (Note: dMi * 2 !!)
	ni = np.nansum(nij, axis=0)
	#alpha0, beta0, cov0
	result = estimate_gr_params_minimize(ni, Mi, dMi*2, completeness, end_date)
	alpha0, beta0 = result.x

	## Initial guess for zone alpha values
	alphas0 = np.array([np.log(np.exp(alpha0) / J)] * J)
	#alphas0, beta0, covs = estimate_gr_params_multi(nij, Mi, dMi, completeness,
	#															end_date)

	def constrain_alphas(x):
		## Constraint function to ensure that summed activity equals total activity
		## Should return zero
		# TODO: take into account variance on alpha0?
		alphas = x[1:]
		return np.log(np.sum(np.exp(alphas))) - alpha0

	def minimize_func(x):
		## Function returning negative log-likelihood, which should be minimized
		## Eq. 7, summed over all zones
		beta, alphas = x[0], x[1:]
		log_likelihood = 0
		for j in range(J):
			ni = nij[j]
			idxs = ~np.isnan(ni)
			abM_term = alphas[j] - beta * Mi[idxs]
			tdM_term = 2 * ti[idxs] * dMi[idxs]
			log_likelihood += np.sum(ni[idxs] * (abM_term + np.log(tdM_term))
										- np.log(factorial(ni[idxs]))
										- tdM_term * np.exp(abM_term))
		return -log_likelihood

	# TODO: finite bins

	x0 = np.hstack([[beta0], alphas0])
	# TODO: allow variance on beta value?
	result = minimize(minimize_func, x0, method='trust-constr',
							bounds=[(beta0-1, beta0+1)] + [(None, None)] * J,
							constraints={"fun": constrain_alphas, "type": "eq"})

	if not result.success:
		print(result.message)
		return

	cov = result.hess_inv.todense() #* max(1, abs(result.fun)) * ftol

	return (result.x[0], result.x[1], cov)


def estimate_gr_params_curvefit(ni, Mi, dMi, completeness, end_date,
										log10=False, incremental=False,
										prior_b=None, max_b_var=0.001):
	"""
	Estimate Gutenberg-Richter parameters by directly fitting the incremental
	or cumulative relation (log_e or log_10), minimizing the sum of squared residuals.

	Note that, in contrast to MLE, earthquake numbers and timespans are
	collapsed into rates.

	:param ni:
	:param Mi:
	:param dMi:
	:param completeness:
	:param end_date:
	:param log10:
	:param prior_b:
		see :func:`estimate_gr_params`
	:param incremental:
		bool, whether to fit the incremental (True) or the cumulative (False)
		rates. Note that uncertainties are considerably larger when fitting
		incremental rates!
		(default: False)
	:param max_b_var:
		float, maximum variation allowed for b value if :param:`prior_b` is set
		(default: 0.001)

	:return:
		(a/alpha, b/beta, cov) tuple
		- a/alpha: float, a or alpha value
		- b/beta: float, b or beta value
		- cov: 2-D matrix [2,2], covariance matrix
	"""
	from scipy.optimize import curve_fit

	I = len(ni)

	assert len(Mi) == I

	if np.isscalar(dMi):
		dMi = np.array([dMi] * I)
	assert len(dMi) == I
	## dM is half-bin size in paper!
	## Note: do not use /= to avoid modifying original array
	dMi = dMi / 2.

	ti = completeness.get_completeness_timespans(Mi, end_date)
	idxs = ~np.isnan(ni)

	if not incremental:
		Mmin = Mi[0] - dMi[0]
		Mmax = Mi[-1] + dMi[-1]

	Mi = Mi[idxs]
	dMi = dMi[idxs]

	inc_rates = ni[idxs] / ti[idxs]
	if not incremental:
		cumul_rates = np.cumsum(inc_rates[::-1])[::-1]

	if prior_b:
		b_bounds = (prior_b-max_b_var, prior_b+max_b_var)
	else:
		prior_b = 1.0
		b_bounds = (-np.inf, np.inf)

	if not log10:
		prior_beta = prior_b * np.log(10)
		beta_bounds = [val * np.log(10) for val in b_bounds]

	a0 = np.log10(ni[0]) + prior_b * Mi[0]

	if log10 is False:
		alpha0 = np.log(prior_beta * np.exp(a0 * np.log(10)))
		initial_guess = (alpha0, prior_beta)
		bounds = ((-np.inf, beta_bounds[0]), (np.inf, beta_bounds[1]))

		def inc_gr_rate(M, alpha, beta):
			## Eq. 4
			dnu = 2 * np.exp(alpha - beta * M) * np.sinh(beta * dMi) / beta
			return dnu

		def cumul_gr_rate(M, alpha, beta):
			M = M - dMi
			lamda0 = np.exp(alpha) / beta / np.exp(beta * Mmin)
			Mmax_term = np.exp(-beta * Mmax)
			lamda = (lamda0
					* (np.exp(-beta * M) - Mmax_term) / (np.exp(-beta * Mmin) - Mmax_term))
			return lamda

	else:
		initial_guess = (a0, prior_b)
		bounds = ((-np.inf, b_bounds[0]), (np.inf, b_bounds[1]))

		def inc_gr_rate(M, a, b):
			a_inc = a + np.log10(10**(b*dMi) - 10**(-b*dMi))
			return 10**(a_inc - b*M)

		def cumul_gr_rate(M, a, b):
			M = M - dMi
			return 10**(a - b*M) - 10**(a - b*Mmax)

	if incremental:
		popt, pcov = curve_fit(inc_gr_rate, Mi, inc_rates, p0=initial_guess,
									bounds=bounds, method='trf')
	else:
		popt, pcov = curve_fit(cumul_gr_rate, Mi, cumul_rates, p0=initial_guess,
									bounds=bounds, method=('trf'))

	#perr = np.sqrt(np.diag(pcov))

	return (popt[0], popt[1], pcov)


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
def construct_mfd_at_epsilon(a_or_alpha, b_or_beta, cov, epsilon, Mmin, Mmax, dM,
							precise=True, log10=False):
	"""
	Construct magnitude-frequency distribution (MFD) corresponding to a
	particular epsilon value

	:param a_or_alpha:
		float, alpha or a value
	:param b_or_beta:
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
		bool, whether :param:`a_or_alpha` and :param:`b_or_beta` of the
		Gutenberg-Richter relation (and :param:`cov`) correspond to
		log10 notation (i.e., a and b values) or to the natural logarithm
		notation (i.e., alpha and beta values)
		(default: False)

	:return:
		instance of :class:`rshalib.mfd.EvenlyDiscretizedMfd`
		or (if :param:`epsilon` = 0 and :param:`log10` is True)
		instance of :class:`rshalib.mfd.TruncatedGRMFD`
	"""
	from hazard.rshalib.mfd import (TruncatedGRMFD, NatLogTruncatedGRMFD,
											EvenlyDiscretizedMFD)

	if epsilon == 0:
		#Mmin = Mi[0] - dM/2.
		#Mmax = Mi[-1] + dM/2.
		if log10:
			a_val, b_val = a_or_alpha, b_or_beta
			mfd = TruncatedGRMFD(Mmin, Mmax, dM, a_val, b_val)
		else:
			alpha, beta = a_or_alpha, b_or_beta
			mfd = NatLogTruncatedGRMFD(Mmin, Mmax, dM, alpha, beta)
	else:
		num_bins = np.round((Mmax - Mmin) / dM)
		Mi = Mmin + np.arange(num_bins) * dM + dM/2.
		sigma_m = calc_gr_sigma(Mi, cov)
		if log10:
			a_val, b_val = a_or_alpha, b_or_beta
			Mi1 = Mi - dM / 2.
			Mi2 = Mi + dM / 2.
			sigma_m1 = calc_gr_sigma(Mi1, cov)
			sigma_m2 = calc_gr_sigma(Mi2, cov)
			Ndisc = (10**(a_val - b_val * Mi1 + sigma_m1 * epsilon)
					- 10**(a_val - b_val * Mi2 + sigma_m2 * epsilon))
		else:
			alpha, beta = a_or_alpha, b_or_beta
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


def construct_mfd_pmf(a_or_alpha, b_or_beta, cov, Mmin, Mmax_pmf, dM,
					num_discretizations, precise=True, log10=False):
	"""
	Construct a probability mass function of MFDs optimally sampling
	the uncertainty on the activity rates represented by the
	covariance matrix

	Note that this is not entirely correct, because the GR parameters
	estimated using :func:`estimate_gr_params` depend on Mmax
	(empty bins beyond bin with largest observed magnitude),
	but this dependency is only slight and could be ignored.

	:param a_or_alpha:
		float, alpha or a value of the GR relation
	:param b_or_beta:
		float, beta or b value of the GR relation
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
	:param log10:
		bool, whether :param:`a_or_alpha` and :param:`b_or_beta` of the
		Gutenberg-Richter relation (and :param:`cov`) correspond to
		log10 notation (i.e., a and b values) or to the natural logarithm
		notation (i.e., alpha and beta values)
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
			mfd = construct_mfd_at_epsilon(a_or_alpha, b_or_beta, cov, epsilon,
											Mmin, Mmax, dM,
											precise=precise, log10=log10)
			mfd_list.append(mfd)
			weights.append(weight)

	weights = np.array(weights)

	return (mfd_list, weights)



if __name__ == "__main__":
	pass
