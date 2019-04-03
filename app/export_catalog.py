"""
Export catalog for Gruenthal's EMEC catalog
"""

import eqcatalog

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
start_date = 1350
region = (0, 8, 49, 52)
event_type = 'ke'
Mmin = 1.0
Mtype = 'MW'
Mrelation = {}


raw_cat = eqcatalog.rob.query_local_eq_catalog(start_date=start_date, end_date=end_date,
											region=region, Mmin=0, event_type=event_type)

## Add converted moment magnitude column for subselecting
moment_mags = raw_cat.get_magnitudes(Mtype, Mrelation)
for i in range(len(raw_cat)):
	raw_cat[i].set_mag('MWc', moment_mags[i])
raw_cat = raw_cat.subselect(Mmin=Mmin, Mtype='MWc', Mrelation={})

## Remove Mwc column again
#for eq in raw_cat:
#	del eq.mag['MWc']

raw_cat.print_info()
cat = raw_cat.get_sorted()
#cat1 = raw_cat.subselect(Mmin=Mmin, Mtype='ML', Mrelation={})
#cat2 = raw_cat.subselect(Mmin=Mmin, Mtype='MS', Mrelation={})
#cat = cat1 + cat2
#print(len(cat), len(cat1), len(cat2))
#cat1[:10].print_list()
#cat2.print_list()

csv_file = "C:\\Temp\\ROB_catalog_%s_%d-%d.csv" % (event_type, start_date, end_date)
columns = ['ID', 'date', 'time', 'name', 'lon', 'lat', 'depth',
			'ML', 'MS', 'MW', 'MWc', 'intensity_max',
			'errt', 'errh', 'errz', 'errM']
cat.export_csv(csv_file, columns=columns)

cat.plot_map()
