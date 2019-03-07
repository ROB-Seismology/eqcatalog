"""
"""

import os
import numpy as np
import pylab
import eqcatalog



out_folder = r"D:\Earthquake Reports\20180525"

id_earth = 6625

[eq] = eqcatalog.seismodb.query_ROB_LocalEQCatalogByID(id_earth)
ensemble = eq.get_macroseismic_enquiries(min_fiability=20)
print("Found %d enquiries for ID %d" % (len(ensemble), id_earth))


for prop in ["asleep", "noise", "duration"]:
	print(prop)
	ar = getattr(ensemble, prop)
	#print np.unique(ar)
	#print(ensemble.get_histogram(prop))
	#print(ar[:200])
	#exit()
	#fig_filespec = os.path.join(out_folder, "%d_%s.png" % (id_earth, prop))
	fig_filespec = None
	ensemble.plot_histogram(prop, fig_filespec=fig_filespec)
