"""
Fetch Bensberg earthquake catalog from the internet

The catalog is available as annual files from 1975 onward
http://www.seismo.uni-koeln.de/data/%Y.js

In addition, there are also monthly location files for earthquakes (Q)
from May 2005 onward
http://www.seismo.uni-koeln.de/catfiles/%y%MQ.TXT
and for induced earthquakes (M)
http://www.seismo.uni-koeln.de/catfiles/%y%MM.TXT
"""

import os
import sys
import datetime
import numpy as np

if sys.version[0] == '3':
	from urllib.request import urlopen
else:
	from urllib import urlopen

#from ..eqrecord import LocalEarthquake
#from ..eqcatalog import EQCatalog
from eqcatalog import LocalEarthquake, EQCatalog


def fetch_bensberg_catalog():
	"""
	Read Bensberg earthquake catalog from the internet

	:return:
		instance of :class:`eqcatalog.EQCatalog`
	"""
	base_url = 'http://www.seismo.uni-koeln.de'
	start_year = 1975
	end_year = datetime.date.today().year

	agency = 'BENS'
	event_type_dict = {'Q': 'ke',
						'M': 'ki',
						'S': 'cb'}

	eq_list = []
	i = 0
	for year in range(start_year, end_year + 1):
		url = '%s/data/%d.js' % (base_url, year)
		print(url)
		web = urlopen(url)
		for line in web.readlines():
			try:
				line = str(line, encoding='latin-1')
			except:
				print(line)
				raise
			if line[:11] == 'EventArray[':
				i += 1
				print(i)
				line = line.split('"')[1]
				day = int(line[:2])
				month = int(line[3:5])
				year = int(line[6:10])
				date = datetime.date(year, month, day)
				time = line[11:21] + '00000'
				if time[6:] == '60.000000':
					time = time[:6] + '59.990000'
				time = datetime.time.fromisoformat(time)
				lat = float(line[22:28])
				lon = float(line[29:35])
				try:
					depth = float(line[36:40])
				except:
					depth = np.nan
				try:
					mag = float(line[43:46])
				except:
					mag = np.nan
				event_type = line[47]
				event_type = event_type_dict[event_type]
				name = line[48:].strip()
				name = name.replace(',', ' ')
				if 'explosion' in name.lower():
					event_type = 'cb'

				eq = LocalEarthquake(i, date, time, lon, lat, depth, mag={'ML': mag},
									name=name, agency=agency, event_type=event_type)
				eq_list.append(eq)

	start_date = np.datetime64('1975-01-01')
	end_date = np.datetime64('today')
	catalog_name = 'Erbebenkatalog der Erbebenstation Bensberg'
	catalog = EQCatalog(eq_list, start_date=start_date, end_date=end_date,
						name=catalog_name)

	return catalog



if __name__ == "__main__":
	bens = fetch_bensberg_catalog()
	bens = bens.get_sorted()
	bens.print_info()

	#out_folder = r'C:\Temp'
	out_folder = r'D:\seismo-gis\collections\Bensberg_seismology'
	columns=['datetime', 'lon', 'lat', 'depth', 'ML', 'name', 'event_type']
	csv_file = os.path.join(out_folder, 'CSV', 'Bensberg_catalog.csv')
	bens.export_csv(csv_file, columns=columns)
	gis_file = os.path.join(out_folder, 'SHP', 'Bensberg_catalog.shp')
	bens.export_gis('ESRI Shapefile', gis_file,
					encoding='latin-1', columns=columns)
