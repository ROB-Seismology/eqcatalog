
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int

try:
	## Python 2
	basestring
	PY2 = True
except:
	## Python 3
	PY2 = False
	basestring = str


## Third-party modules
import numpy as np


__all__ = ["MacroseismicInfo"]


class MacroseismicInfo():
	"""
	Container class to hold information of (aggregated) records retrieved
	from the official or internet macroseismic enquiry database, and
	used for plotting maps.

	:param id_earth:
		int, ID of earthquake in ROB catalog
		or 'all'
	:param id_com:
		int, ID of commune in ROB database
	:param intensity:
		int or float, macroseismic intensity
	:param agg_type:
		str, type of aggregation, one of:
		- 'id_com' or 'commune'
		- 'id_main' or 'main commune'
		- 'grid_X' (where X is grid spacing in km)
		- None or ''
	:param enq_type:
		str, type of enquirey, one of:
		- 'internet' or 'online'
		- 'official'
	:param num_replies:
		int, number of replies in aggregate
		(default: 1)
	:param lon:
		float, longitude or (if :param:`agg_type` = 'grid_X') easting
		(default: 0)
	:param lat:
		float, latitude or (if :param:`agg_type` = 'grid_X') northing
		(default: 0)
	:param db_ids:
		list of ints, IDs of database records represented in aggregate
	"""
	def __init__(self, id_earth, id_com, intensity, agg_type, enq_type, num_replies=1,
				lon=0, lat=0, db_ids=[]):
		self.id_earth = id_earth
		self.id_com = id_com
		self.intensity = intensity
		self.agg_type = agg_type
		self.enq_type = enq_type
		self.num_replies = num_replies
		self.lon = lon
		self.lat = lat
		self.db_ids = db_ids

	@property
	def I(self):
		return self.intensity

	def get_eq(self):
		"""
		Fetch earthquake from ROB database

		:return:
			instance of :class:`eqcatalog.LocalEarthquake`
		"""
		from ..rob.seismodb import query_local_eq_catalog_by_id

		if isinstance(self.id_earth, (int, str)):
			[eq] = query_local_eq_catalog_by_id(self.id_earth)
			return eq

	def get_enquiries(self, min_fiability=20, verbose=False):
		"""
		Fetch macroseismic enquiry records from the database, based on
		either db_ids or, if this is empty, id_earth

		:param min_fiability:
			int, minimum fiability (ignored if db_ids is not empty)
		:param verbose:
			bool, whether or not to print useful information
		"""
		from ..rob.seismodb import query_web_macro_enquiries

		if self.db_ids:
			ensemble = query_web_macro_enquiries(web_ids=self.db_ids, verbose=verbose)
		else:
			ensemble = query_web_macro_enquiries(self.id_earth, id_com=self.id_com,
								min_fiability=min_fiability, verbose=verbose)

		return ensemble
