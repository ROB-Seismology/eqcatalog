"""
Export catalog for Gruenthal's EMEC catalog
"""

import os
import numpy as np
import eqcatalog


out_folder = r"C:\Temp"

end_date = 2018
region = (1, 8, 49, 52)

## Gruenthal
"""
start_date = 1985
Mmin = 1.0
#event_type = 'ke'
event_type = 'ki'
"""

## Musson
"""
start_date = 1910
Mmin = 3.0
event_type = 'ke'
"""

## Tractebel
"""
start_date = 1350
region = (0, 8, 49, 52)
event_type = 'ke'
Mmin = 1.8
Mtype = 'MW'
Mrelation = {}
"""

## David Baumont
"""
start_date = 1350
region = (0, 8, 49, 52)
event_type = 'ke'
Mmin = 1.0
Mtype = 'MW'
Mrelation = {}
"""


## FUGRO
"""
start_date = 1350
region = (1, 8, 49, 52)
event_type = 'ke'
Mmin = 1.53	# results in minimum ML of 1.0
Mtype = 'MW'
Mrelation = {}
"""

## VPO (tectonic)
out_folder = r"E:\Home\_kris\Projects\2019 - VPO Balmatt"
start_date = '1350-01-01'
end_date = '2019-10-31'
#region = (1.25, 8.75, 49.15, 53.3)
region = (1, 8, 49, 52)
event_type = 'ke'
Mmin = None
Mtype = 'MW'
Mrelation = {}
catalog_name = 'ROB tectonic earthquake catalog'

## VPO (all instrumental)
start_date = '1910-01-01'
event_type = 'ke,se,ki,si,qb,sqb,km,sm,kr,sr,cb,scb,kx,sx'
catalog_name = 'ROB instrumental event catalog'



raw_cat = eqcatalog.rob.query_local_eq_catalog(start_date=start_date, end_date=end_date,
											region=region, Mmin=-1, event_type=event_type,
											verbose=True)
## Add converted moment magnitude column for subselecting
moment_mags = raw_cat.get_magnitudes(Mtype, Mrelation)
for i in range(len(raw_cat)):
	raw_cat[i].set_mag('MWc', moment_mags[i])
if Mmin:
	raw_cat = raw_cat.subselect(Mmin=Mmin, Mtype='MWc', Mrelation=eqcatalog.msc.IdentityMSC())

## Remove Mwc column again
#for eq in raw_cat:
#	del eq.mag['MWc']

raw_cat.print_info()

cat = raw_cat.get_sorted()
cat.name = catalog_name
#cat1 = raw_cat.subselect(Mmin=Mmin, Mtype='ML', Mrelation={})
#cat2 = raw_cat.subselect(Mmin=Mmin, Mtype='MS', Mrelation={})
#cat = cat1 + cat2
#print(len(cat), len(cat1), len(cat2))
#cat1[:10].print_list()
#cat2.print_list()

## Export as CSV
if ',' in event_type:
	event_type = 'all'
csv_file = os.path.join(out_folder, "ROB_catalog_%s_%s-%s.csv" % (event_type,
						start_date.replace('-', ''), end_date.replace('-', '')))
columns = ['ID', 'date', 'time', 'name', 'lon', 'lat', 'depth',
			'ML', 'MS', 'MW', 'intensity_max',
			'errt', 'errh', 'errz', 'errM']
cat.export_csv(csv_file, columns=columns)

## Export as Shapefile
gis_file = os.path.splitext(csv_file)[0] + '.SHP'
columns[1] = 'datetime'
del columns[2]
cat.export_gis('ESRI Shapefile', gis_file, columns=columns, replace_null_values=-9999)

## Export as KML
kml_file = os.path.splitext(csv_file)[0] + '.KML'
if event_type == 'all':
	cat.export_kml(kml_file, folders='event_type', color_by_depth=False, columns=columns + ['event_type'])
else:
	cat.export_kml(kml_file, folders='time', color_by_depth=False, columns=columns)

#cat.plot_map(Mtype='MWc', Mrelation=eqcatalog.msc.IdentityMSC())
