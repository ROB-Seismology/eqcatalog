"""
Test GR estimation following Stromeyer & Gruenthal (2015)
"""

import datetime
import numpy as np
import hazard.rshalib as rshalib
import eqcatalog
from eqcatalog.calcGR_MLE import (estimate_gr_params, calc_gr_sigma,
									construct_mfd_at_epsilon)



cat = eqcatalog.read_named_catalog('ROB')
completeness = eqcatalog.rob.Completeness_MW_201303a_ext
Mtype = 'MW'
Mrelation = cat.default_Mrelations[Mtype]
end_date = datetime.date(2019, 12, 31)

## Decluster
dc_method = eqcatalog.declustering.LinkedWindowMethod()
dc_window = eqcatalog.declustering.Gruenthal2009Window()
dc_result = dc_method.analyze_clusters(cat, dc_window, Mrelation)
dc_cat = dc_result.get_declustered_catalog()

## Apply completenes
cc_cat = dc_cat.subselect_completeness(completeness, Mtype=Mtype,
										Mrelation=Mrelation)

dM = 0.1
Mmin = completeness.min_mag
Mmax = 6.4
Mmax_obs = cc_cat.Mminmax(Mtype, Mrelation)[1]
ni, Mi = cc_cat.bin_by_mag(Mmin, Mmax, dM, Mtype=Mtype, Mrelation=Mrelation,
							completeness=None)
Mi += dM / 2.
#ni = ni.astype('d')
#ni[Mi > Mmax_obs] = np.nan

precise = False
alpha, beta, cov = estimate_gr_params(ni, Mi, dM, completeness, end_date,
									precise=precise, log10=False)
a, b, cov10 = estimate_gr_params(ni, Mi, dM, completeness, end_date,
									precise=precise, log10=True)

## Values from paper
#alpha = 5.2254
#beta = 1.7951
#cov = np.mat([[0.099, 0.0312], [0.0312, 0.0105]])

#a = alpha / np.log(10)
b = beta / np.log(10)
print('alpha', alpha)
print('beta', beta)
print('a: ', a)
print('b: ', b)
print('cov: ', cov)

## Not correct
print(np.exp(alpha - beta * Mmin))
## Correct
print(np.exp(a * np.log(10) - beta * Mmin))
print(10**(a - b * Mmin))


## Uncertainty on alpha - beta * M
sigma_m = calc_gr_sigma(Mi, cov)
print('sigma_m: ', sigma_m)

## Compare MFDs at epsilon for log10 / loge
epsilon = -1.5
mfd1 = construct_mfd_at_epsilon(alpha, beta, cov, epsilon, Mmin, Mmax, dM, log10=False)
mfd2 = construct_mfd_at_epsilon(a, b, cov10, epsilon, Mmin, Mmax, dM, log10=True)
rshalib.mfd.plot_mfds([mfd1, mfd2], labels=['loge', 'log10'])


Ni = np.exp(alpha - beta * Mi)
sigma_Ni = sigma_m * Ni
#print('Ni: ', Ni)
#print('sigma_Ni: ', sigma_Ni)

ti = completeness.get_completeness_timespans(Mi, end_date)

## Plot MFD
import pylab

## Observed
pylab.semilogy(Mi, ni/ti, 'o')
pylab.semilogy(Mi-dM/2., np.cumsum((ni/ti)[::-1])[::-1], 'o')

## Estimated
#pylab.semilogy(Mi, Ni * dM)
for z in (-np.sqrt(3), 0, np.sqrt(3)):
#for z in (0,):
	#Ndisc = 2 * Ni + sigma_Ni * np.abs(dM) * z
	Ndisc = 2 * np.exp(alpha - beta * Mi + sigma_m * z)
	#Ndisc = 2 * 10**(a - b * Mi + (sigma_m / np.log(10)) * z)
	if precise:
		Ndisc *= (dM / 2.)
	else:
		## Note: probably need to add uncertainty on beta
		Ndisc *= (np.sinh(beta * dM / 2.) / beta)
	pylab.semilogy(Mi, Ndisc)
pylab.semilogy(Mi-dM/2., np.cumsum((Ni * dM)[::-1])[::-1])
mfd = rshalib.mfd.TruncatedGRMFD(Mmin, Mmax, dM, a, b)
#mfd = rshalib.mfd.EvenlyDiscretizedMFD(Mmin + dM/2., dM, Ni * dM)
#tmfd = mfd.to_truncated_gr_mfd(None, None, 'LSQi')
#print(tmfd.a_val)
pylab.semilogy(Mi-dM/2., mfd.get_cumulative_rates())
pylab.show()
