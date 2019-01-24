"""
Export catalog for Gruenthal's EMEC catalog
"""

import eqcatalog

start_date = 1985
end_date = 2018
region = (1, 8, 49, 52)
ML_min = 2.0

cat = eqcatalog.rob.query_local_eq_catalog(start_date=start_date, end_date=end_date,
											event_type='all')
#cat = eqcatalog.read_named_catalog("ROB")
cat = cat.subselect(start_date=start_date, end_date=end_date, region=region,
					Mmin=ML_min, Mtype='ML', attr_val=('event_type', 'ke'))

csv_file = r"C:\Temp\ROB_catalog_%d-%d.csv" % (start_date, end_date)
columns = ['ID', 'date', 'time', 'name', 'lon', 'lat', 'depth',
			'ML', 'MS', 'MW', 'intensity_max', 'errt', 'errh', 'errz', 'errM']
cat.export_csv(csv_file, columns=columns)
