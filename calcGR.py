"""
Implementation of different methods to compute Gutenberg-Richter MFDs
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
from scipy import stats



def calcGR_Weichert(magnitudes, bins_N, completeness, end_date, b_val=None, verbose=False):
	"""
	Calculate a and b values of Gutenberg-Richter relation using maximum likelihood
	estimation for variable observation periods for different magnitude increments.
	Adapted from calB.m and calBfixe.m Matlab modules written by Philippe Rosset
	(ROB, 2004), which is based on the method by Weichert, 1980
	(BSSA, 70, Nr 4, 1337-1346).

	:param magnitudes:
		numpy float array, left edges of magnitude bins up to Mmax
	:param bins_N:
		numpy array, number of earthquakes in each magnitude bin
	:param completeness:
		instance of :class:`Completeness`
	:param end_date:
		datetime.date or Int, end date with respect to which observation periods
		will be determined
	:param b_val:
		Float, fixed b value to constrain MLE estimation (default: None)
	:param verbose:
		Bool, whether some messages should be printed or not (default: False)

	:return:
		Tuple (a, b, stda, stdb)
		- a: a value
		- b: b value
		- stda: standard deviation of a value
		- stdb: standard deviation of b value

	Note:
	This regression depends on the Mmax specified, as empty magnitude bins
	are taken into account. It is therefore important to specify Mmax as
	the evaluated Mmax for the specific region or source.
	"""
	obs_periods = completeness.get_completeness_timespans(magnitudes, end_date)

	dM = magnitudes[1] - magnitudes[0]
	## Avoid side effects in calling function
	magnitudes = magnitudes.copy() + dM

	if not b_val:
		## Initial trial value
		BETA = 1.5
	else:
		## Fixed beta
		BETA = b_val * np.log(10)
	BETL = 0
	while(np.abs(BETA-BETL)) >= 0.0001:
		#print(BETA)

		SNM = 0.0
		NKOUNT = 0.0
		STMEX = 0.0
		SUMTEX = 0.0
		STM2X = 0.0
		SUMEXP = 0.0

		for k in range(len(bins_N)):
			SNM += bins_N[k] * magnitudes[k]
			NKOUNT += bins_N[k]
			TJEXP = obs_periods[k] * np.exp(-BETA * magnitudes[k])
			TMEXP = TJEXP * magnitudes[k]
			SUMEXP += np.exp(-BETA * magnitudes[k])
			STMEX += TMEXP
			SUMTEX += TJEXP
			STM2X += magnitudes[k] * TMEXP

		try:
			DLDB = STMEX / SUMTEX
		except:
			break
		else:
			D2LDB2 = NKOUNT * (DLDB*DLDB - STM2X/SUMTEX)
			DLDB = DLDB * NKOUNT - SNM
			BETL = BETA
			if not b_val:
				BETA -= DLDB/D2LDB2

	B = BETA / np.log(10)
	if not b_val:
		STDBETA = np.sqrt(-1.0/D2LDB2)
		STDB = STDBETA / np.log(10)
	else:
		STDB = 0
		STDBETA = 0
	FNGTMO = NKOUNT * SUMEXP / SUMTEX
	FN0 = FNGTMO * np.exp(BETA * (magnitudes[0] - dM/2.0))
	FLGN0 = np.log10(FN0)
	A = FLGN0
	STDFN0 = FN0 / np.sqrt(NKOUNT)
	## Applying error propogation for base-10 logarithm
	## See: http://chemwiki.ucdavis.edu/Analytical_Chemistry/Quantifying_Nature/Significant_Digits/Propagation_of_Error
	STDA = 0.434 * (STDFN0 / FN0)
	## Note: the following formula in Philippe Rosset's program is equivalent
	#A = np.log10(FNGTMO) + B * (magnitudes[0] - dM/2.0)
	## This is also equivalent to:
	#A = np.log10(FNGTMO * np.exp(-BETA * (0. - (magnitudes[0] - (dM/2.0)))))

	if verbose:
		FN5 = FNGTMO * np.exp(-BETA * (5. - (magnitudes[0] - dM/2.0)))
		STDFN5 = FN5 / np.sqrt(NKOUNT)
		print("Maximum-likelihood estimation (Weichert)")
		print("BETA=%.3f +/- %.3f; B=%.3f +/- %.3f" % (BETA, STDBETA, B, STDB))
		print("Total number of events: %d" % NKOUNT)
		print("LOG(annual rate above M0): %.3f" % FLGN0)
		print("Annual rate above M5: %.3f +/- %.3f" % (FN5, STDFN5))
		print("Annual rate above M0: %.3f +/- %.3f" % (FN0, STDFN0))

	## Other parameters computed in Philippe Rosset's version
	#STDA = np.sqrt((magnitudes[0]-dM/2.0)**2 * STDB**2 - (STDFNGTMO**2 / ((np.log(10)**2 * np.exp(2*(A+B*(magnitudes[0]-dM/2.0))*np.log(10))))))
	#STDA = np.sqrt(abs(A)/NKOUNT)
	#ALPHA = FNGTMO * np.exp(-BETA * (magnitudes[0] - dM/2.0))
	#STDALPHA = ALPHA / np.sqrt(NKOUNT)

	return A, B, STDA, STDB


def calcGR_LSQ(magnitudes, occurrence_rates, b_val=None, weights=None, verbose=False):
	"""
	Calculate a and b values of Gutenberg-Richter relation using a linear regression
	(least-squares).

	:param magnitudes:
		numpy float array, left edges of magnitude bins up to Mmax
	:param occurrence_rates:
		numpy float array, occurrence rates (cumulative or incremental)
		corresponding to magnitude bins
	:param b_val:
		Float, fixed b value to constrain LSQ estimation (default: None)
	:param weights:
		weights to be applied to occurrence rates
		(default: None)
	:param verbose:
		Bool, whether some messages should be printed or not (default: False)

	:return:
		Tuple (a, b, a_sigma, b_sigma)
		- a: a value (intercept)
		- b: b value (slope, taken positive)
		- a_sigma: standard deviation of a value
		- b_sigma: standard deviation of b value
	"""
	## Do not consider magnitudes with zero occurrence rates
	idxs = np.where(occurrence_rates > 0)
	occurrence_rates = occurrence_rates[idxs]
	if not weights is None:
		weights = np.asarray(weights)[idxs]
	log_occurrence_rates = np.log10(occurrence_rates)
	magnitudes = magnitudes[idxs]

	if len(magnitudes) == 0:
		return (np.nan, np.nan, 0, 0)

	if not b_val:
		if not weights is None:
			b_val, a_val = np.polyfit(magnitudes, log_occurrence_rates, 1, w=weights)
		else:
			b_val, a_val, r, ttprob, stderr = stats.linregress(magnitudes, log_occurrence_rates)
		## In earlier versions of linregress, stderr was the standard error
		## of the estimate (see), in newer versions, it is the standard error
		## of the slope (b_sigma). To be on the safe side, we ignore it, and
		## compute a_sigma and b_sigma manually.
		b_val = -b_val
		n = len(magnitudes)
		y = log_occurrence_rates
		y_estimate = a_val - b_val * magnitudes
		residual = y - y_estimate
		see = residual.std(ddof=2)

		## Standard deviation on slope and intercept
		## (from: http://mail.scipy.org/pipermail/scipy-user/2008-May/016777.html)
		mx = np.mean(magnitudes)
		sx2 = np.sum((magnitudes - mx)**2)
		a_sigma = see * np.sqrt(1./n + mx*mx/sx2)
		b_sigma = see * np.sqrt(1./sx2)
		## Formula for a_sigma from http://www.chem.mtu.edu/~fmorriso/cm3215/UncertaintySlopeInterceptOfLeastSquaresFit.pdf
		#a_sigma = np.sqrt((see**2 * np.sum(magnitudes**2)) / (n * sx2))
	else:
		## Regression line always goes through mean x and y
		mean_mag = np.mean(magnitudes)
		mean_log_rate = np.mean(log_occurrence_rates)
		a_val = mean_log_rate + b_val * mean_mag
		a_sigma, b_sigma = 0, 0
		r = np.nan

	if verbose:
		print("Linear regression: a=%.3f, b=%.3f (r**2=%.2f)" % (a_val, b_val, r**2))

	return (a_val, b_val, a_sigma, b_sigma)
