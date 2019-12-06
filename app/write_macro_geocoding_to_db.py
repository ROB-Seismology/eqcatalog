"""
Write macroseismic geocoding results to database
"""

import os
import csv

import db.simpledb as simpledb
from secrets.seismodb import (host, database, user_rw, passwd_rw)

from eqcatalog.rob.seismodb import query_seismodb_table


## Read all id_web IDs currently in web_location database table
table_name = 'web_location'
column_clause = ['id_web', 'quality']
db_recs = query_seismodb_table(table_name, column_clause=column_clause)
web_ids = [rec['id_web'] for rec in db_recs]
qualities = [rec['quality'] for rec in db_recs]


## Read geocoding results from CSV file
csv_folder = "D:\\Earthquake Reports\\20190623\\DYFI geocoding"
csv_filename = "macroseismic_inq_for_8285_corrected_geocoded-ArcGIS.csv"
csv_file = os.path.join(csv_folder, csv_filename)

recs_to_add, recs_to_modify = [], []
with open(csv_file) as csvf:
	reader = csv.DictReader(csvf)
	for row in reader:
		rec = {}
		rec['id_web'] = id_web = int(row['id_web'])
		rec['longitude'] = lon = float(row['Geo_Longitude'])
		rec['latitude'] = lat = float(row['Geo_Latitude'])
		address_type = row['address type']

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

		"""
		rec['confidence'] = {'SubAddress': 10,
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
							'POI': 7}.get(address_type, 0)
		"""

		rec['quality'] = {'SubAddress': 9,
							'PointAddress': 9,
							'StreetAddress': 8,
							'StreetInt': 7,
							'StreetAddressExt': 7,
							'DistanceMarker': 7,
							'StreetName': 6,
							'Locality': 5,
							'PostalLoc': 5,
							'PostalExt': 5,
							'Postal': 4,
							'POI': 4}.get(address_type, 0)

		try:
			r = web_ids.index(id_web)
		except:
			recs_to_add.append(rec)
		else:
			## Only overwrite existing locations if quality is better
			if rec['quality'] > qualities[r]:
				recs_to_modify.append(rec)

		#print(id_web, lon, lat, rec['quality'])

#print set([rec['address_type'] for rec in db_recs])


## Write to database
seismodb = simpledb.MySQLDB(database, host, user_rw, passwd_rw)
table_name = 'web_location'
if len(recs_to_add):
	print("Adding %d new records" % len(recs_to_add))
	#seismodb.add_records(table_name, recs_to_add, dry_run=True)
if len(recs_to_modify):
	print("Updating %d existing records" % len(recs_to_modify))
	#seismodb.update_rows(table_name, recs_to_modify, 'id_web', dry_run=True)
