"""
"""

import os
import eqcatalog

region = (1, 8, 49, 52)

columns = ['ID', 'date', 'time', 'lon', 'lat', 'depth', 'MW', 'intensity_max']

out_folder = "C:\\Temp"

for cat_name in ('CENEC', 'SHEEC', 'EMEC'):
	cat = eqcatalog.read_named_catalog(cat_name)
	cat = cat.subselect(region=region)
	html_file = os.path.join(out_folder, "%s.html" % cat_name)
	with open(html_file, 'w') as of:
		of.write(cat.print_list(as_html=True))
	csv_file = os.path.join(out_folder, "%s.csv" % cat_name)
	cat.export_csv(csv_file, columns=columns)
