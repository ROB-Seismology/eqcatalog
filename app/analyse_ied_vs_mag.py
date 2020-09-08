"""
Analyse inter-event_distance as a function of magnitude
"""

import numpy as np
import scipy.optimize
from plotting.generic_mpl import plot_xy
import eqcatalog


cat = eqcatalog.read_named_catalog('ROB')
dc_window = eqcatalog.declustering.Gruenthal2009Window()
dc_method = eqcatalog.declustering.LinkedWindowMethod()
dc_result = dc_method.analyze_clusters(cat, dc_window, cat.default_Mrelations['MW'])
dc_cat = dc_result.get_declustered_catalog()
cc_cat = dc_cat.subselect_completeness()

dM = 1.
curves, labels = [], []
for n in range(1, 4):
	mag_bins, ied_min = cc_cat.calc_min_ied_by_mag_bin(dM, n)
	curves.append((mag_bins[:-1], ied_min))
	labels.append('n=%d' % n)

ied_func = lambda M, c1, c2: c1 * np.exp(c2 * M)

popt, pcov = scipy.optimize.curve_fit(ied_func, mag_bins[:-2], ied_min[:-1])
c1, c2 = popt
print('c1 = %f, c2 = %f' % (c1, c2))

curves.append((mag_bins, ied_func(mag_bins, c1, c2)))
labels.append('Fit')
plot_xy(curves, labels=labels, xlabel='Magnitude', ylabel='Min. IED (km)',
		yscaling='log')
