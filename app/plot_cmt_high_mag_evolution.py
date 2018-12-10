# -*- coding: iso-Latin-1 -*-

"""
Check if N(M>7) has increased in equatorial regions
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import pylab
import eqcatalog.harvard_cmt as cmt


db_file = "D:\\seismo-gis\\collections\\Harvard_CMT\\SQLite\HarvardCMT.sqlite"
catalog = cmt.HarvardCMTCatalog(db_file)

Mmin_values = [7.0, 7.5, 8.0]
colors = ['g', 'b', 'm', 'r']
start_year, end_year = 1976, 2018
region = (-180, 180, -40, 40)

for m, Mmin in enumerate(Mmin_values):
	recs = catalog.get_records(start_date=start_year, end_date=end_year,
								region=region, Mmin=Mmin)
	years = [rec.hypo_date.year for rec in recs]
	num_events = np.bincount(years)[start_year:]
	#print(len(num_events))
	mean_val = num_events[:-2].mean()
	year_bins = range(start_year, end_year+1)
	#print(len(year_bins))
	pylab.plot(year_bins, num_events, colors[m], label="Mmin=%.1f" % Mmin)
	pylab.hlines(mean_val, start_year, end_year, colors=colors[m], linestyles="dashed",
				label="_nolegend_")

pylab.xlabel("Year")
pylab.ylabel("Number of events")
pylab.xlim(start_year, end_year)
pylab.legend(loc=0)
pylab.title(u"Harvard CMT Catalog (%s°S - %s°N)" % (region[2:]))
pylab.show()
