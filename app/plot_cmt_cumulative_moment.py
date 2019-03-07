"""
Plot cumulative seismic moment in Harvard CMT catalog
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import pylab
import eqcatalog.harvard_cmt as cmt
from eqcatalog.time_functions import fractional_year


db_file = "D:\\seismo-gis\\collections\\Harvard_CMT\\SQLite\\HarvardCMT.sqlite"
catalog = cmt.HarvardCMTCatalog(db_file)

Mmax_values = [6.0, 7.0, 8.0, 9.5]
#Mmax_values = [6.0, 7.0]
colors = ['g', 'b', 'm', 'r']
start_year, end_year = 1976, 2018

for Mmax, color in zip(Mmax_values, colors):
	recs = list(catalog.get_records(start_date=start_year, end_date=end_year, Mmax=Mmax))
	print("Mmax=%.1f: n=%d" % (Mmax, len(recs)))
	frac_years = [fractional_year(rec.hypo_date) for rec in recs]
	moments = [rec.get_moment(unit='N.m') for rec in recs]
	cumul_moments = np.cumsum(moments)
	frac_years = np.repeat(frac_years, 2)
	cumul_moments = np.concatenate([[0.], np.repeat(cumul_moments, 2)[:-1]])
	pylab.plot(frac_years, cumul_moments, color, label="Mmax=%.1f" % Mmax)

pylab.xlabel("Year")
pylab.ylabel("Cumulative seismic moment (N.m)")
pylab.xlim(start_year, end_year)
pylab.grid()
pylab.legend(loc=2)
pylab.title("Harvard CMT Catalog")
pylab.show()
