"""
Evaluate validity of MS->MW relations
"""

import os
import numpy as np
from prettytable import PrettyTable
from plotting.generic_mpl import plot_xy
import eqcatalog


## Read catalog
## Note: only 2 post-1985 earthquakes with ML and MW: Roermond and Alsdorf!
cat = eqcatalog.rob.query_local_eq_catalog()
cat = cat.subselect_Mtype(['MS','MW'])
cat.default_Mrelations = {}
MS_db = cat.get_magnitudes('MS', Mrelation={})
MW_db = cat.get_magnitudes('MW', Mrelation={})
id_cat = cat.get_ids()


MS_MW_relations = eqcatalog.msc.get_available_Mrelations('MSCE_MS_MW').keys()

## Compute RMS errors
tab = PrettyTable(['Relation', 'RMS'])

for Mrelation_name in MS_MW_relations:
	Mrelation = getattr(eqcatalog.msc, Mrelation_name)()
	MWc = Mrelation(MS_db)
	RMS = np.sqrt(np.nanmean((MW_db - MWc)**2))
	row = [Mrelation_name, RMS]
	tab.add_row(row)
tab.float_format = '.3'
print(tab)


## Plot
datasets = [(MS_db, MW_db)]
labels = ['Database']
linestyles = ['']
markers = ['o']

MS = np.linspace(1.0, 6.0, 51)
for Mrelation_name in MS_MW_relations:
	Mrelation = getattr(eqcatalog.msc, Mrelation_name)()
	MW = Mrelation(MS)
	datasets.append((MS, MW))
	linestyles.append('-')
	markers.append('')
	labels.append(Mrelation_name)

colors = ['r', 'b', 'r', 'c', 'y', 'm', 'purple', 'b', 'k']
plot_xy(datasets, labels=labels, linestyles=linestyles, markers=markers,
		colors=colors, xlabel='MS', ylabel='MW', xgrid=True, ygrid=True)
