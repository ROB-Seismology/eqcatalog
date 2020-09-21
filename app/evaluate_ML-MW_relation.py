"""
Evaluate validity of ML->MW relations
"""

import os
import numpy as np
from prettytable import PrettyTable
from plotting.generic_mpl import plot_xy
import eqcatalog


## Read catalog
## Note: only 2 post-1985 earthquakes with ML and MW: Roermond and Alsdorf!
cat = eqcatalog.rob.query_local_eq_catalog()
cat = cat.subselect_Mtype(['ML','MW'])
cat.default_Mrelations = {}
ML_db = cat.get_magnitudes('ML', Mrelation={})
MW_db = cat.get_magnitudes('MW', Mrelation={})
id_cat = cat.get_ids()

## Read MW from SCK_Balmatt
csv_file = r'E:\Home\_kris\Projects\2019 - SCK_Balmatt\MW_all_events.csv'
ar = np.loadtxt(csv_file, skiprows=1, delimiter=',', usecols=[0,2,3,4,5])
ar = ar.T
id_ss, ML_ss, ML_err, MW_ss, MW_err = ar
id_ss = id_ss.astype('int')

common_ids = []
for id_earth in id_ss:
	if id_earth in id_cat:
		common_ids.append(id_earth)
print('Common_ids: %s' % common_ids)

ML_MW_relations = eqcatalog.msc.get_available_Mrelations('MSCE_ML_MW').keys()

## Compute RMS errors
tab = PrettyTable(['Relation', 'Database', 'SCK_Balmatt', 'All'])

all_ML = np.hstack([ML_db, ML_ss])
all_MW = np.hstack([MW_db, MW_ss])

MLmin = 1.6
for Mrelation_name in ML_MW_relations:
	Mrelation = getattr(eqcatalog.msc, Mrelation_name)()
	row = [Mrelation_name]
	for (ML, MW) in [(ML_db, MW_db), (ML_ss, MW_ss), (all_ML, all_MW)]:
		idxs = ML>=MLmin
		ML = ML[idxs]
		MW = MW[idxs]

		MWc = Mrelation(ML)
		RMS = np.sqrt(np.mean((MW - MWc)**2))
		row.append(RMS)
	tab.add_row(row)
tab.float_format = '.3'
print(tab)


## Plot
datasets = [(ML_db, MW_db), (ML_ss, MW_ss)]
labels = ['Database', 'SCK_Balmatt']
linestyles = ['', '']
markers = ['o', 'o']

ML = np.linspace(1.0, 6.0, 51)
for Mrelation_name in ML_MW_relations:
	Mrelation = getattr(eqcatalog.msc, Mrelation_name)()
	MW = Mrelation(ML)
	datasets.append((ML, MW))
	linestyles.append('-')
	markers.append('')
	labels.append(Mrelation_name)

colors = ['r', 'b', 'r', 'c', 'y', 'm', 'purple', 'b', 'k']
plot_xy(datasets, labels=labels, linestyles=linestyles, markers=markers,
		colors=colors, xlabel='ML', ylabel='MW', xgrid=True, ygrid=True)
