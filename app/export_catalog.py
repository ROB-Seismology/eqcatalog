"""
Export catalog for Gruenthal's EMEC catalog
"""

import os
import numpy as np
import eqcatalog


out_folder = r"C:\Temp"

end_date = '2019-12-31'
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
start_date = '1350-01-01'
end_date = '2020-06-30'
region = (0, 8, 49, 52)
event_type = 'ke'
Mmin = 1.8
Mtype = 'MW'
Mrelation = {}
catalog_name = 'ROB Catalog %s - %s'
catalog_name %= (start_date.split('-')[0], end_date.split('-')[0])
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
"""
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
"""

## Q-con (seismicity study Flanders)
"""
start_date = '1350-01-01'
end_date = '2021-01-01'
region = (1, 8, 49, 52)
event_type = 'all'
Mmin = None
catalog_name = 'ROB Catalog %s - %s'
catalog_name %= (start_date.split('-')[0], end_date.split('-')[0])
"""

## UAachen
"""
start_date = '1985-01-01'
end_date = '2021-03-01'
region = (5.5, 6.5, 50.5, 51.0)
event_type = 'ke'
Mmin = 0.0
Mtype='ML'
Mrelation = {}
catalog_name = 'ROB Catalog %s - %s'
catalog_name %= (start_date.split('-')[0], end_date.split('-')[0])
"""

## ULiège
"""
start_date = '1350-01-01'
end_date = '2021-03-01'
region = (3.5, 8., 49.5, 52.)
event_type = 'ke'
Mmin = None
catalog_name = 'ROB Catalog %s - %s'
catalog_name %= (start_date.split('-')[0], end_date.split('-')[0])
"""


## KNMI
start_date = '1350-01-01'
end_date = '2021-04-01'
region = (1, 9, 49, 53)
event_type = 'ke,ki,se,si'
Mmin = None
catalog_name = 'ROB Catalog %s - %s'
catalog_name %= (start_date.split('-')[0], end_date.split('-')[0])


raw_cat = eqcatalog.rob.query_local_eq_catalog(start_date=start_date, end_date=end_date,
											region=region, Mmin=None, event_type=event_type,
											verbose=True)
## Add converted moment magnitude column for subselecting
if Mmin is not None:
	if Mtype == 'MW':
		moment_mags = raw_cat.get_magnitudes(Mtype, Mrelation)
		for i in range(len(raw_cat)):
			raw_cat[i].set_mag('MWc', moment_mags[i])
		raw_cat = raw_cat.subselect(Mmin=Mmin, Mtype='MWc', Mrelation=eqcatalog.msc.IdentityMSC())
	else:
		raw_cat = raw_cat.subselect(Mmin=Mmin, Mtype=Mtype, Mrelation=Mrelation)

	## Remove Mwc column again
	#for eq in raw_cat:
	#	del eq.mag['MWc']

cat = raw_cat.get_sorted()
cat.name = catalog_name
cat.print_info()
#cat1 = raw_cat.subselect(Mmin=Mmin, Mtype='ML', Mrelation={})
#cat2 = raw_cat.subselect(Mmin=Mmin, Mtype='MS', Mrelation={})
#cat = cat1 + cat2
#print(len(cat), len(cat1), len(cat2))
#cat1[:10].print_list()
#cat2.print_list()

## Export as CSV
#if ',' in event_type:
#	event_type = 'all'
csv_file = os.path.join(out_folder, "ROB_catalog_%s_%s-%s.csv" % (event_type,
						start_date.replace('-', ''), end_date.replace('-', '')))
columns = ['ID', 'date', 'time', 'name', 'lon', 'lat', 'depth',
			'ML', 'MS', 'MW', 'intensity_max',
			'errt', 'errh', 'errz', 'errM', 'event_type']
cat.export_csv(csv_file, columns=columns)

## Export as Shapefile
gis_file = os.path.splitext(csv_file)[0] + '.SHP'
columns[1] = 'datetime'
del columns[2]
#cat.export_gis('ESRI Shapefile', gis_file, columns=columns, replace_null_values=-9999)
exit()

## Export as KML
kml_file = os.path.splitext(csv_file)[0] + '.KML'
if event_type == 'all':
	cat.export_kml(kml_file, folders='event_type', color_by_depth=False, columns=columns + ['event_type'])
else:
	cat.export_kml(kml_file, folders='time', color_by_depth=False, columns=columns)

#cat.plot_map(Mtype='MWc', Mrelation=eqcatalog.msc.IdentityMSC())
