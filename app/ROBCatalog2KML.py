#! /usr/bin/python

import sys, os

#pythondir = "/datas2/apache/htdocs/EQMapper/python"
#sys.path.insert(0, pythondir)
try:
	#import users.kris.Seismo.db.seismodb as seismodb
	import eqcatalog.seismodb as seismodb
except:
	import seismodb


#Response.ContentType = "text/xml"
#Response.CharSet = "iso-8859-1"

#print "Content-Type: text/xml"
#print

#qs = os.environ["QUERY_STRING"]


def rob_catalog_to_kml(
		kml_filespec=None,
		region=None,
		start_date=None,
		end_date=None,
		Mmin=None,
		Mmax=None,
		min_depth=None,
		max_depth=None,
		event_type="ke",
		time_folders=True,
		instrumental_start_year=1910,
		color_by_depth=False):
	"""
	Export earthquake catalog to KML.

	:param kml_filespec:
		String, full path to output KML file.
		If None, kml is printed on screen
		(default: None)
	:param region:
		(w, e, s, n) tuple specifying rectangular region of interest in
		geographic coordinates (default: None)
	:param start_date:
		Int or date or datetime object specifying start of time window of interest
		If integer, start_date is interpreted as start year
		(default: None)
	:param end_date:
		Int or date or datetime object specifying end of time window of interest
		If integer, end_date is interpreted as end year
		(default: None)
	:param Mmin:
		Float, minimum magnitude to extract (default: None)
	:param Mmax:
		Float, maximum magnitude to extract (default: None)
	:param min_depth:
		Float, minimum depth in km to extract (default: None)
	:param max_depth:
		Float, maximum depth in km to extract (default: None)
	:param event_type:
		str, event type (one of "all", "cb", "ex", "ke", "ki", "km",
		"kr", "kx", "qb", "sb", "se", "si", "sm", "sr", "sx" or "uk")
		(default: "ke" = known earthquakes)
	:param time_folders:
		Bool, whether or not to organize earthquakes in folders by time
		(default: True)
	:param instrumental_start_year:
		Int, start year of instrumental period (only applies when time_folders
		is True) (default: 1910)
	:param color_by_depth:
		Bool, whether or not to color earthquakes by depth (default: False)

	:return:
		str, KML code (if :param:`kml_filespec` is not set)
	"""
	catalog = seismodb.query_ROB_LocalEQCatalog(region=region,
							start_date=start_date, end_date=end_date,
							Mmin=Mmin, Mmax=Mmax,
							min_depth=min_depth, max_depth=max_depth,
							event_type=event_type, verbose=False)

	return catalog.export_KML(kml_filespec, time_folders=time_folders,
							instrumental_start_year=instrumental_start_year,
							color_by_depth=color_by_depth)


if __name__ == "__main__":
	import datetime
	import argparse

	def get_date(date_string):
		try:
			year = int(date_string)
		except:
			day, month, year = date_string.split(',')
			return datetime.date(year, month, day)
		else:
			return year

	def get_region(region_string):
		return map(float, region_string.split(','))

	parser = argparse.ArgumentParser(prog="ROBCatalog2KML")

	parser.add_argument("--kml_file", dest="kml_filespec", default=None,
					help="Full path to output KML file")
	parser.add_argument("--region", dest="region", type=get_region, default=[-1.25, 8.75, 49.15, 53.30],
					help="Rectangular region: west/east/south/north")
	parser.add_argument("--start_date", dest="start_date", type=get_date, default=1,
					help="Start date DD-MM-YYYY or year")
	parser.add_argument("--end_date", dest="end_date", type=get_date, default=datetime.date.today(),
					help="End date DD-MM-YYYY or year")
	parser.add_argument("--Mmin", dest="Mmin", default=0, type=float,
					help="Minimum magnitude")
	parser.add_argument("--Mmax", dest="Mmax", default=None, type=float,
					help="Maximum magnitude")
	parser.add_argument("--min_depth", dest="min_depth", default=0, type=float,
					help="Minimum depth")
	parser.add_argument("--max_depth", dest="max_depth", default=None, type=float,
					help="Maximum depth")
	parser.add_argument("--event_type", dest="event_type", default="ke",
					help="Event type")
	parser.add_argument("--time_folders", dest="time_folders", default=True, type=bool,
					help="Whether or not to classify earthquakes in time folders")
	parser.add_argument("--instrumental_start_year", dest="instrumental_start_year", default=1910, type=int,
					help="Year to consider as start of instrumental period")
	parser.add_argument("--color_by_depth", dest="color_by_depth", default=False, type=bool,
					help="Whether or not to color earthquakes by depth")

	args = parser.parse_args()
	exit(rob_catalog_to_kml(**vars(args)))

	"""
	start_date = datetime.date(1350,1,1)
	#start_date = datetime.date(1985, 7,1)
	end_date = datetime.datetime.now().date()
	region = (-1.25, 8.75, 49.15, 53.30)
	#region = (4.5, 4.65, 50.60, 50.70)
	Mmin = 0
	Mmax = 7


	#write_KML(catalog, r"C:\Temp\Brabant_swarm.kml", start_time=datetime.datetime(2008,7,1))
	#write_KML(catalog, r"D:\GIS-data\KSB-ORB\KSB-ORB_catalog.kml", time_folders=True)
	"""
