"""
Count all enquiries from Germany since 17/06/2010
"""

import numpy as np
import eqcatalog


dyfi = eqcatalog.rob.query_online_macro_catalog(id_earth='all')
dyfi_de = dyfi.subselect_by_property('country', ['DE'])
dyfi_de_2010 = dyfi_de[dyfi_de.get_datetimes() >= np.datetime64('2010-06-17')]
print(len(dyfi_de_2010))

id_earths = np.unique(dyfi_de_2010.get_prop_values('id_earth'))
_, counts = dyfi_de_2010.bincount('id_earth', bins=id_earths)
#for id, count in zip(id_earths, counts):
#	print(id, count)
dyfi_de_2010_cat = eqcatalog.rob.query_local_eq_catalog_by_id(list(id_earths))
print(len(id_earths), len(dyfi_de_2010_cat))

#print(dyfi_de_2010.subselect_by_property('id_earth', [2135]).recs[0])
#exit()

print('Time,\tML,\tnDYFI,\tlon,\tlat,\tREGION')
for i in range(len(id_earths)):
	eq = dyfi_de_2010_cat[i]
	assert id_earths[i] == eq.ID
	msg = "%s,\t%.1f,\t%d,\t%.3f,\t%.3f,\t%s"
	msg %= (eq.datetime, eq.ML, counts[i], eq.lon, eq.lat, eq.name)
	print(msg)
