"""
SeismicEruption HY4 earthquake catalog format
"""

from __future__ import absolute_import, division, print_function, unicode_literals


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
		return struct.pack("2ih2B2hi4B", self.latitude, self.longitude, self.year,
							self.month, self.day, self.minutes, self.tseconds,
							self.depth, self.magnitude, 0, 0, 0)


def readHY4(filespec, Mtype='ML'):
	from .eqcatalog import EQCatalog
	catalog = EQCatalog.from_HY4(filespec, Mtype)
	return catalog


def writeHY4(filespec, catalog):
	ofd = open(filespec, "wb")
	for eq in catalog:
		hyp = eq.to_HY4()
		ofd.write("%s" % hyp.pack())
	ofd.close()
