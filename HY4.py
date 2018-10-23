"""
SeismicEruption HY4 earthquake catalog format
"""

import ctypes
import struct


class HYPDAT(ctypes.Structure):
	_fields_ = [("latitude", ctypes.c_int),
				("longitude", ctypes.c_int),
				("year", ctypes.c_short),
				("month", ctypes.c_byte),
				("day", ctypes.c_byte),
				("minutes", ctypes.c_short),
				("tseconds", ctypes.c_short),
				("depth", ctypes.c_long),
				("magnitude", ctypes.c_byte),
				("reserved1", ctypes.c_byte),
				("reserved2", ctypes.c_byte),
				("reserved3", ctypes.c_byte)]

	def pack(self):
		return struct.pack("2ih2B2hi4B", self.latitude, self.longitude, self.year, self.month, self.day, self.minutes, self.tseconds, self.depth, self.magnitude, 0, 0, 0)


def readHY4(filespec, Mtype='ML'):
	from eqcatalog import EQCatalog
	catalog = EQCatalog.from_HY4(filespec, Mtype)
	return catalog


def writeHY4(filespec, catalog):
	ofd = open(filespec, "wb")
	for eq in catalog:
		hyp = eq.to_HY4()
		ofd.write("%s" % hyp.pack())
	ofd.close()



if __name__ == "__main__":
	import os
	import datetime
	import seismodb

	out_folder = r"D:\GIS-data\KSB-ORB"
	#out_folder = r"E:\Home\_kris\Meetings\2018 - Opendeurdagen"
	#out_folder = r"C:\Program Files (x86)\SeismicEruption\OpenDoorDays"

	filespec = os.path.join(out_folder, "ROB.HY4")

	region = (0,8,49,52)
	start_date = datetime.date(1985, 1, 1)
	#end_date = datetime.date(2007, 10, 1)
	end_date = datetime.datetime.now()
	min_mag = 0.0
	max_mag = 7.0
	catalog = seismodb.query_ROB_LocalEQCatalog(region, start_date=start_date, end_date=end_date, Mmin=min_mag, Mmax=max_mag)
	catalog.export_HY4(filespec)
