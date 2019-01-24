"""
Write macroseismic geocoding results to database
"""

import os
import csv

import db.simpledb as simpledb
from seismodb_secrets import (host, database, user, passwd)


## Read geocoding results from CSV file
csv_folder = "D:\\Earthquake Reports\\20180525\\DYFI geocoding"
csv_filename = "macroseismic_inq_for_6625_modified_floorcorrected_no dupl_geocoded-ArcGIS_with comm.csv"
csv_file = os.path.join(csv_folder, csv_filename)

db_recs = []
with open(csv_file) as csvf:
	reader = csv.DictReader(csvf)
	for row in reader:
		rec = {}
		rec['id_web'] = id_web = int(row['id_web'])
		rec['longitude'] = lon = float(row['Geo_Longitude'])
		rec['latitude'] = lat = float(row['Geo_Latitude'])
		#rec['address_type'] = address_type = row['address type']

		## Geocoder confidence:
		## 10 : 250 m
		##  9 : 500 m
		##  8 : 1 km
		##  7 : 5 km
		##  6 : 7.5 km
		##  5 : 10 km
		##  4 : 15 km
		##  3 : 20 km
		##  2 : 25 km
		##  1 : > 25 km
		##  0 : undetermined

		## Google API v2 GGeoAddressAccuracy:
		## Constant | Description
		## 0 Unknown location.
		## 1 Country level accuracy.
		## 2 Region (state, province, prefecture, etc.) level accuracy.
		## 3 Sub-region (county, municipality, etc.) level accuracy.
		## 4 Town (city, village) level accuracy.
		## 5 Post code (zip code) level accuracy.
		## 6 Street level accuracy.
		## 7 Intersection level accuracy.
		## 8 Address level accuracy.
		## 9 Premise (building name, property name, shopping center, etc.) level accuracy.

		rec['quality'] = {'SubAddress': 10,
							'PointAddress': 10,
							'StreetAddress': 10,
							'StreetInt': 9,
							'StreetAddressExt': 9,
							'DistanceMarker': 9,
							'StreetName': 8,
							'Locality': 8,
							'PostalLoc': 7,
							'PostalExt': 7,
							'Postal': 7,
							'POI': 7}.get(row['address type'], 0)
		db_recs.append(rec)

		# ['PointAddress', 'Locality', 'PostalLoc', 'StreetAddress', 'StreetAddressExt', 'POI', 'StreetName']
		print(id_web, lon, lat, rec['quality'])

#print set([rec['address_type'] for rec in db_recs])
exit()

## Write to database
user = 'kris'
passwd = '***REMOVED***'
seismodb = simpledb.MySQLDB(database, host, user, passwd)
table_name = 'web_location'
seismodb.add_records(table_name, db_recs, dry_run=True)
