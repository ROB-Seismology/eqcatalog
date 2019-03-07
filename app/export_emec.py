"""
Export catalog for Gruenthal's EMEC catalog
"""

import eqcatalog

end_date = 2018
region = (1, 8, 49, 52)

## Gruenthal
#start_date = 1985
#Mmin = 1.0
#event_type = 'ke'
#event_type = 'ki'

## Musson
start_date = 1910
Mmin = 3.0
event_type = 'ke'

raw_cat = eqcatalog.rob.query_local_eq_catalog(start_date=start_date, end_date=end_date,
											region=region, Mmin=Mmin, event_type=event_type)
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
			'ML', 'MS', 'MW', 'intensity_max', 'errt', 'errh', 'errz', 'errM']
#cat.export_csv(csv_file, columns=columns)

cat.plot_map()
