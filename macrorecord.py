# -*- coding: iso-Latin-1 -*-

"""
Processing of macroseismic and DYFI data
"""

from __future__ import absolute_import, division, print_function, unicode_literals

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


__all__ = ["MacroseismicInfo", "MacroseismicEnquiryEnsemble",
			"MacroseismicDataPoint", "get_roman_intensity"]


ROMAN_INTENSITY_DICT = {0: '', 1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI',
						7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X', 11: 'XI', 12: 'XII'}

def get_roman_intensity(intensities, include_fraction=True):
	"""
	Convert intensity values to Roman numerals

	:param intensities:
		float or float array, intensities
	:param include_fraction:
		bool, whether or not to represent fractions as multiples of 1/4
		(default: True)

	:return:
		list of strings, Roman numerals
	"""
	scalar = False
	if np.isscalar(intensities):
		intensities = [intensities]
		scalar = True
	decimals = np.remainder(intensities, 1)
	intensities = np.floor_divide(intensities, 1)
	roman_intensities = []
	for i in range(len(intensities)):
		intensity, dec = intensities[i], decimals[i]
		roman_intensity = ROMAN_INTENSITY_DICT[intensity]
		if include_fraction and intensity:
			if 0.125 <= dec < 0.375:
				roman_intensity += ' 1/4'
			elif 0.375 <= dec < 0.625:
				roman_intensity += ' 1/2'
			elif 0.625 <= dec:
				roman_intensity += ' 3/4'
		if PY2:
			roman_intensity = roman_intensity.decode('ascii')
		roman_intensities.append(roman_intensity)
	if scalar:
		return roman_intensities[0]
	else:
		return roman_intensities


def strip_accents(txt):
	"""
	Remove accents (diacritics) from (unicode) string

	:param txt:
		unicode or str, input string

	:return:
		unicode, output string
	"""
	import unicodedata
	if isinstance(txt, bytes):
		txt = txt.decode("latin1")
	nkfd_form = unicodedata.normalize('NFKD', txt)
	return "".join([c for c in nkfd_form if not unicodedata.combining(c)])


class MacroseismicDataPoint:
	pass


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
	:param I:
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
	def __init__(self, id_earth, id_com, I, agg_type, enq_type, num_replies=1,
				lon=0, lat=0, db_ids=[]):
		self.id_earth = id_earth
		self.id_com = id_com
		self.I = I
		self.agg_type = agg_type
		self.enq_type = enq_type
		self.num_replies = num_replies
		self.lon = lon
		self.lat = lat
		self.db_ids = db_ids

	def get_eq(self):
		"""
		Fetch earthquake from ROB database

		:return:
			instance of :class:`eqcatalog.LocalEarthquake`
		"""
		from .rob.seismodb import query_local_eq_catalog_by_id

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
		from .rob.seismodb import query_web_macro_enquiries

		if self.db_ids:
			ensemble = query_web_macro_enquiries(web_ids=self.db_ids, verbose=verbose)
		else:
			ensemble = query_web_macro_enquiries(self.id_earth, id_com=self.id_com,
								min_fiability=min_fiability, verbose=verbose)

		return ensemble


## Disable no-member errors for MacroseismicEnquiryEnsemble
# pylint: disable=no-member

class MacroseismicEnquiryEnsemble():
	"""
	Class representing ensemble (or aggregate) of macroseismic enquiries

	:param id_earth:
		int, ID of earthquake in ROB database
	:param recs:
		list of dicts representing enquiries from the database
	"""
	def __init__(self, id_earth, recs):
		self.id_earth = id_earth
		self.recs = recs
		self._gen_arrays()
		self._define_bins()

	def __len__(self):
		return len(self.recs)

	@property
	def num_replies(self):
		return len(self.recs)

	def __iter__(self):
		#return self.recs.__iter__()
		for i in range(len(self)):
			yield self.__getitem__(i)

	def __getitem__(self, spec):
		## Note: recs are not copied!
		if isinstance(spec, int):
			return MacroseismicEnquiryEnsemble(self.id_earth, [self.recs[spec]])
		elif isinstance(spec, slice):
			return MacroseismicEnquiryEnsemble(self.id_earth, self.recs[spec])
		elif isinstance(spec, (list, np.ndarray)):
			## spec can be list of indexes or list of bools
			recs = []
			if len(spec):
				idxs = np.arange(len(self))
				idxs = idxs[np.asarray(spec)]
				for idx in idxs:
					recs.append(self.recs[idx])
			return MacroseismicEnquiryEnsemble(self.id_earth, recs)

	def _gen_arrays(self):
		self.CII = np.array([rec['CII'] for rec in self.recs])
		self.CDI = np.array([rec['CDI'] for rec in self.recs])
		self.MI = np.array([rec['MI'] for rec in self.recs])
		self.CWS = np.array([rec['CWS'] for rec in self.recs])
		self.fiability = np.array([rec['fiability'] for rec in self.recs])

		for prop in ["situation", "building", "floor", "asleep", "noise",
					"felt", "other_felt", "motion", "duration",
					"reaction", "response", "stand", "furniture",
					"heavy_appliance", "walls"]:
			prop_list = [rec[prop] or None for rec in self.recs]
			prop_list = [val if (val and (isinstance(val, int) or val.isdigit()))
						else None for val in prop_list]
			try:
				ar = np.array(prop_list, dtype='float')
			except:
				print("Warning: Array generation failed for prop %s" % prop)
			else:
				mask = np.isnan(ar)
				ar = np.ma.array(ar.astype(np.int), mask=mask)
				setattr(self, prop, ar)

		for prop in ["sway", "creak", "shelf", "picture"]:
			char_map = {c:num for (num, c) in enumerate('ABCDEFGHIJK')}
			char_map['_'] = -1
			char_map[''] = -2
			prop_list = [rec[prop] or '' for rec in self.recs]
			mask = np.array([True if val in ('', '_') else False for val in prop_list])
			#ar = np.array(prop_list, dtype='c')
			prop_list = [char_map[val] for val in prop_list]
			ar = np.array(prop_list, dtype='int')
			ar = np.ma.array(ar, mask=mask)
			setattr(self, prop, ar)

		self.damage = np.zeros((len(self), 14), dtype='bool')
		for r, rec in enumerate(self.recs):
			damage = rec['d_text']
			damage = [1 if d == '*' else 0 for d in damage]
			self.damage[r] = damage

	def _define_bins(self):
		self.bins = {}
		self.bins['asleep'] = np.array([0, 1, 2])
		self.bins['felt'] = np.array([0, 1, np.nan])
		self.bins['other_felt'] = np.array([0, 1, 2, 3, 4])
		self.bins['motion'] = np.array([0, 1, 2, 3, 4, 5, np.nan])
		self.bins['reaction'] = np.array([0, 1, 2, 3, 4, 5, np.nan])
		self.bins['stand'] = np.array([0, 1, 2, 3, np.nan])
		self.bins['furniture'] = np.array([0, 1, np.nan])
		#self.bins['shelf'] = np.array(['A', 'B', 'C', 'D', 'E', 'F', '_'])
		self.bins['shelf'] = np.array([0, 1, 2, 3, 4, 5, np.nan])
		#self.bins['picture'] = np.array(['A', 'B', 'C', '_'])
		self.bins['picture'] = np.array([0, 1, 2, np.nan])
		self.bins['heavy_appliance'] = np.array([0, 1, 2, 3, 4, np.nan])
		self.bins['situation'] = np.array([0, 1, 2, 3, 4, 5])
		self.bins['building'] = np.array([0, 1, 2, 3, 4, 5, 6, np.nan])
		self.bins['noise'] = np.array([0, 1, 2, 3, 4, np.nan])
		self.bins['response'] = np.array([0, 1, 2, 3, 4, np.nan])
		self.bins['walls'] = np.array([0, 1, 2, 3, np.nan])
		self.bins['floor'] = np.array([-10.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 10.5, 100])
		self.bins['sway'] = np.array([0, 1, 2, np.nan])
		self.bins['creak'] = np.array([0, 1, 2, np.nan])

		self.bins['fiability'] = np.arange(0, 101, 10)
		self.bins['CII'] = np.arange(1, 13) - 0.5
		self.bins['CDI'] = self.bins['MI'] = self.bins['CII']
		self.bins['duration'] = np.arange(0, 61, 5)

		self.bins['num_replies'] = np.array([1, 3, 5, 10, 20, 50, 100, 200, 500, 1000])

	def get_copy(self, deepcopy=False):
		"""
		Return a copy of the ensemble. Note that :prop:`recs` will be
		shared with the original ensemble.

		:param deepcopy:
			bool, if True :prop:`recs` will be copied as well, if False
			they will be shared with the original ensemble
			(default: False)
		:return:
			instance of :class:`MacroseismicEnquiryEnsemble`
		"""
		if not deepcopy:
			return self.__getitem__(slice(None, None, None))
		else:
			recs = [rec.copy() for rec in self.recs]
			return self.__class__(self.id_earth, recs)

	def get_eq(self):
		"""
		Fetch earthquake from ROB database

		:return:
			instance of :class:`eqcatalog.LocalEarthquake`
		"""
		from .rob.seismodb import query_local_eq_catalog_by_id

		if isinstance(self.id_earth, (int, str)):
			[eq] = query_local_eq_catalog_by_id(self.id_earth)
			return eq

	def get_prop_values(self, prop):
		"""
		Get list of values for given property

		:param prop:
			string, name of property (that can only have certain values)

		:return:
			list
		"""
		if prop == "damage":
			prop = "d_text"
		if not len(self.recs) or not prop in self.recs[0]:
			return []
		else:
			first_non_None_value = next((rec[prop] for rec in self.recs
										if rec[prop] is not None), None)
			if isinstance(first_non_None_value, basestring):
				none_val = u""
			else:
				none_val = np.nan
			return [rec[prop] if rec[prop] is not None else none_val
					for rec in self.recs]

	def get_unique_prop_values(self, prop):
		"""
		Get list of unique values for given property

		:param prop:
			string, name of property (that can only have certain values)

		:return:
			list
		"""
		prop_values = self.get_prop_values(prop)
		return sorted(set(prop_values))

	def get_datetimes(self):
		import datetime

		format = "%Y-%m-%d %H:%M:%S"
		date_times = self.get_prop_values('submit_time')
		date_times = [datetime.datetime.strptime(dt, format) for dt in date_times]
		date_times = np.array(date_times, dtype='datetime64[s]')
		return date_times

	def get_elapsed_times(self):
		eq = self.get_eq()
		return self.get_datetimes() - np.datetime64(eq.datetime)

	def subselect_by_property(self, prop, prop_values, negate=False):
		"""
		Select part of ensemble matching given property values

		:param prop:
			str, name of property
		:param prop_values:
			list of values of :param:`prop` that should be matched
		:param negate:
			bool, whether or not to reverse the matching

		:return:
			instance of :class:`MacroseismicEnquiryEnsemble`
		"""
		values = self.get_prop_values(prop)
		if isinstance(values[0], basestring):
			prop_values = list(map(str, prop_values))
		if not negate:
			idxs = [i for i in range(len(values)) if values[i] in prop_values]
		else:
			idxs = [i for i in range(len(values)) if not values[i] in prop_values]
		return self.__getitem__(idxs)

	def set_prop_values(self, prop, values):
		"""
		Set values of individual enquiries for given property

		:param prop:
			str, name of property
		:values:
			list or array, values of individual enquiries for given property
		"""
		if not isinstance(values, (list, tuple, np.ndarray)):
			values = [values] * len(self)
		assert len(values) == len(self)
		for r, rec in enumerate(self.recs):
			rec[prop] = values[r]
		self._gen_arrays()

	def subselect_by_distance(self, lon, lat, radius):
		"""
		Select part of ensemble situated inside given radius around
		given point

		:param lon:
		:param lat:
			see :meth:`calc_distances`
		:param radius:
			float, radius (in km)
		"""
		all_distances = self.calc_distances(lon, lat)
		idxs = np.where(all_distances <= radius)[0]
		return self.__getitem__(idxs)

	def calc_distances(self, lon, lat, z=0):
		"""
		Compute distances with respect to a particular point

		:param lon:
			float, longitude of point (in degrees)
		:param lat:
			float, latitude of point (in degrees)
		:param z:
			float, depth of point (in km)
			(default: 0)

		:return:
			array, distances (in km)
		"""
		import mapping.geotools.geodetic as geodetic
		rec_lons = np.array(self.get_prop_values('longitude'))
		rec_lats = np.array(self.get_prop_values('latitude'))
		dist = geodetic.spherical_distance(lon, lat, rec_lons, rec_lats) / 1000.
		dist = np.sqrt(dist**2 + z**2)
		return dist

	def subselect_by_zip_country_tuples(self, zip_country_tuples):
		"""
		Select part of ensemble matching given (ZIP, country) tuples

		:param zip_country_tuples:
			list of (ZIP, country) tuples

		:return:
			instance of :class:`MacroseismicEnquiryEnsemble`
		"""
		all_zip_country_tuples = self.get_zip_country_tuples()
		zip_ensemble_recs = []
		for zip_country in zip_country_tuples:
			idxs = [i for i in range(len(self)) if all_zip_country_tuples[i] == zip_country]
			ensemble = self.__getitem__(idxs)
			zip_ensemble_recs.extend(ensemble.recs)
		return self.__class__(self.id_earth, zip_ensemble_recs)

	def get_addresses(self, exclude_empty_streets=True):
		"""
		Return list of addresses for each record

		:return:
			list of strings
		"""
		streets = self.get_prop_values('street')
		zips = self.get_prop_values('zip')
		communes = self.get_prop_values('city')
		countries = self.get_prop_values('country')

		addresses = []
		for i in range(len(self)):
			street = streets[i].upper()
			zip = zips[i]
			commune = communes[i].upper()
			country = countries[i].upper()
			zip_commune = "%s %s" % (zip, commune)
			## Sometimes people fill out full address instead of just street
			street = street.replace(zip_commune, '').strip().rstrip(',')

			if exclude_empty_streets and not street:
				## Filter out empty streets or zips
				address = ""
			else:
				address = "%s, %s, %s"
				address %= (street, zip_commune, country)
			addresses.append(address)
		return addresses

	def geocode(self, provider, bbox=None, start_idx=0, max_requests=100,
				sleep_every=10, **kwargs):
		"""
		:param provider:
			str, name of geocoding provider understood by geocoder
		:param bbox:
			[west, south, east, north] tuple of coordinates
			Not supported by many providers.

		:return:
			list with (lon, lat, confidence) tuples
		"""
		import time
		import geocoder

		results = []
		num_requests = 0
		for address in self.get_addresses():
			success = False
			if address:
				if num_requests < max_requests:
					if num_requests % sleep_every == 0:
						time.sleep(1)
					try:
						g = geocoder.get(address, provider=provider, proximity=bbox, **kwargs)
					except:
						pass
					else:
						num_requests += 1
						if g.ok:
							success = True

			if success:
				if (bbox and (bbox[0] <= g.lng <= bbox[2])
						and (bbox[1] <= g.lat <= bbox[3])):
					results.append((g.lng, g.lat, g.confidence))
				else:
					success = False
			if not success:
				results.append(tuple())

		return results

	def get_zip_country_tuples(self):
		"""
		Return list of (ZIP, country) tuples for each record

		:return:
			list of (ZIP, country) tuples
		"""
		zips = self.get_prop_values('zip')
		countries = self.get_prop_values('country')
		zip_country_tuples = list(zip(zips, countries))
		return zip_country_tuples

	def get_unique_zip_country_tuples(self):
		"""
		Return list of unique (ZIP, country) tuples in ensemble

		:return:
			list of (ZIP, country) tuples
		"""
		zip_country_tuples = set(self.get_zip_country_tuples())
		return list(zip_country_tuples)

	def get_communes_from_db(self, comm_key='id_com', verbose=False):
		"""
		Extract communes from database

		:param comm_key:
			string, commune key, one of "id_com', 'id_main', 'zip'
			(default: 'id_com')
		:param verbose:
			bool, whether or not to print SQL queries
			(default: False)

		:return:
			dict, mapping comm_key values to database records (dicts)
		"""
		from .rob.seismodb import query_seismodb_table
		from difflib import SequenceMatcher as SM

		if comm_key in ("id_com", "id_main"):
			if comm_key == "id_main" and not hasattr(self, "id_main"):
				self.set_main_commune_ids()
			unique_ids = sorted(set(self.get_prop_values(comm_key)))
			table_clause = ['communes']
			column_clause = ['*']
			query_values = ','.join(map(str, unique_ids))
			where_clause = 'id in (%s)' % query_values
			comm_recs = query_seismodb_table(table_clause,
						column_clause=column_clause, where_clause=where_clause,
						verbose=verbose)
			comm_rec_dict = {rec['id']: rec for rec in comm_recs}
		elif comm_key == "zip":
			comm_rec_dict = {}
			column_clause = ['*']
			for country in ("BE", "NL", "DE", "FR", "LU", "GB"):
				## Some tables have commune, some city, some both...
				## BE, LU, NL: city < commune
				## DE, FR: commune
				## GB: city
				if country in ("BE", "LU", "NL", "GB"):
					com_col = 'city'
				else:
					com_col = 'commune'

				if country == "BE":
					com_tables = ['com_zip_BE_fr', 'com_zip_BE_nl']
				elif country == "LU":
					com_tables = ['com_zip_LU_de', 'com_zip_LU_fr']
				else:
					com_tables = ['com_zip_%s' % country]
				#table_clause = 'com_zip_%s' % country
				#if country in ("BE", "LU"):
				#	table_clause += '_fr'

				ensemble = self.subselect_by_property('country', [country])
				zips = ensemble.get_prop_values('zip')
				unique_zips = sorted(set(zips))
				cities = [strip_accents(city).title()
						for city in ensemble.get_prop_values('city')]
				unique_zip_cities = set(zip(zips, cities))
				#join_clause = [('RIGHT JOIN', 'communes', '%s.id = communes.id_main' % table_clause)]

				if len(unique_zips):
					country_comm_rec_dict = {}
					if country == "NL":
						query_values = '|'.join(['%s' % ZIP for ZIP in unique_zips])
						where_clause = 'zip REGEXP "%s"' % query_values
					else:
						query_values = ','.join(['"%s"' % ZIP for ZIP in unique_zips])
						where_clause = 'zip IN (%s)' % query_values
					for table in com_tables:
						table_clause = table
						comm_recs = query_seismodb_table(table_clause,
							column_clause=column_clause, where_clause=where_clause,
							verbose=verbose)
						if country == "NL":
							comm_recs = {(country, rec['zip'][:-3],
									strip_accents(rec[com_col]).title()): rec
									for rec in comm_recs}
						else:
							comm_recs = {(country, rec['zip'],
									strip_accents(rec[com_col]).title()): rec
									for rec in comm_recs}
						country_comm_rec_dict.update(comm_recs)

					for (ZIP, city) in sorted(unique_zip_cities):
						#try:
						key = (country, ZIP, strip_accents(city).title())
						#except:
						#	key = (country, ZIP, city)
						rec = country_comm_rec_dict.get(key)
						if rec:
							## Zip and commune name matched
							comm_rec_dict[key] = rec
						else:
							## Only zip matched, try fuzzy text matching
							country, ZIP, city = key
							matching_zips, match_ratios, id_coms = [], [], []
							for k in country_comm_rec_dict.keys():
								if k[1] == ZIP:
									#matching_zips.append(country_comm_rec_dict[k])
									matching_zips.append(k)
									match_ratios.append(SM(None, k[2], city).ratio())
									id_coms.append(country_comm_rec_dict[k]['id_com'])
							#id_coms = [r['id_com'] for r in matching_zips]
							if len(matching_zips):
								idx = np.argmax(match_ratios)
								if match_ratios[idx] >= 0.4:
									comm_rec_dict[key] = country_comm_rec_dict[matching_zips[idx]]
									if verbose:
										msg = "Commune %s-%s: %s was fuzzy-matched with %s"
										msg %= (country, ZIP, city, matching_zips[idx][2])
										print(msg)
								else:
									## Take smallest id_com
									idx = np.argmin(id_coms)
									#comm_rec_dict[key] = matching_zips[idx]
									comm_rec_dict[key] = country_comm_rec_dict[matching_zips[idx]]
									if verbose:
										msg = "Commune %s-%s: %s was matched with main commune %s"
										msg %= (country, ZIP, city, matching_zips[idx][2])
										print(msg)
							elif verbose:
								## Unmatched ZIP, probably wrong
								# TODO: we could still try to match commune name?
								msg = "Commune %s-%s: %s could not be matched"
								msg %= (country, ZIP, city)
								print(msg)
					"""
					for rec in comm_recs:
						if country == "NL":
							ZIP = rec['zip'][:-3]
						else:
							ZIP = rec['zip']
						key = (ZIP, country)
						if not key in comm_rec_dict:
							comm_rec_dict[key] = rec
						else:
							## There may be more than one record with the same ZIP...
							## The lowest id_com should correspond to id_main
							# TODO: try matching name first (city/commune, city,
							# we will also need lang...)
							if rec['id_com'] < comm_rec_dict[key]['id_com']:
								comm_rec_dict[key] = rec
					"""
		return comm_rec_dict

	def fix_commune_ids(self, keep_existing=True, keep_unmatched=False):
		"""
		Reset commune ID of all records based on ZIP and country

		:param keep_existing:
			bool, whether or not to keep existing (= non-zero) commune ids
			(default: True)
		:param keep_unmatched:
			bool, whether or not to keep current commune id of unmatched
			records
			(default: False)

		:return:
			None, 'id_com' values of :prop:`recs` are modified in place
		"""
		comm_rec_dict = self.get_communes_from_db(comm_key='zip')
		for rec in self.recs:
			if not (rec['id_com'] and keep_existing):
				#comm_rec = comm_rec_dict.get((rec['zip'], rec['country']))
				comm_name = rec['city']
				if comm_name:
					comm_name = comm_name.title()
				comm_rec = comm_rec_dict.get((rec['country'], rec['zip'], comm_name))
				if comm_rec:
					## Zip and name matched
					rec['id_com'] = comm_rec['id_com']
				else:
					comm_rec = comm_rec_dict.get((rec['country'], rec['zip'], u''))
					if comm_rec:
						## Only zip matched
						rec['id_com'] = comm_rec['id_com']
					elif not keep_unmatched:
						## Nothing matched
						rec['id_com'] = 0

	def get_main_commune_ids(self):
		"""
		Get IDs of main communes for each record

		:return:
			list of ints, main commune IDs
		"""
		comm_rec_dict = self.get_communes_from_db(comm_key='id_com')
		main_commune_ids = []
		for rec in self.recs:
			comm_rec = comm_rec_dict.get(rec['id_com'])
			if comm_rec:
				main_commune_ids.append(comm_rec['id_main'])
			else:
				main_commune_ids.append(None)
		return main_commune_ids

	def set_main_commune_ids(self, keep_existing=True):
		"""
		Set main commune ID of all records based on id_com

		:return:
			None, 'id_main' values of :prop:`recs` are created or
			modified in place
		"""
		main_commune_ids = self.get_main_commune_ids()
		for r, rec in enumerate(self.recs):
			## Note: records do not initially have 'id_main' key!
			if not (rec.get('id_main') and keep_existing):
				rec['id_main'] = main_commune_ids[r]

	def set_locations_from_communes(self, comm_key="id_com", keep_unmatched=True,
									max_quality=5):
		"""
		Set location of all records from corresponding communes

		:param comm_key:
			see :meth:`get_communes_from_db`
		:param keep_unmatched:
			bool, whether or not to keep unmatched records untouched
			(default: True)
		:param max_quality:
			int, maximum location quality, only overwrite location
			if location quality is <= max_quality
			(default: 5)

		:return:
			None, 'longitude' and 'latitude' values of :prop:`recs`
			are created or modified in place
		"""
		comm_rec_dict = self.get_communes_from_db(comm_key=comm_key)
		for rec in self.recs:
			if rec['quality'] <= max_quality:
				if comm_key in ("id_com", "id_main"):
					key = rec[comm_key]
				elif comm_key == "zip":
					key = (rec[comm_key], rec['country'])
				comm_rec = comm_rec_dict.get(key)
				if comm_rec:
					rec['longitude'] = comm_rec['longitude']
					rec['latitude'] = comm_rec['latitude']
					# TODO: decide on location quality
					rec['quality'] = {'id_com': 7,
										'zip': 7,
										'id_main': 5}[comm_key]
				elif not keep_unmatched:
					rec['longitude'] = rec['latitude'] = rec['quality'] = np.nan

	def set_locations_from_geolocation(self, keep_unmatched=True):
		"""
		Set location of all records from geolocation in database

		:param keep_unmatched:
			bool, whether or not to keep unmatched records untouched
			(default: True)

		:return:
			None, 'longitude' and 'latitude' values of :prop:`recs`
			are created or modified in place
		"""
		from .rob.seismodb import query_seismodb_table

		table_clause = ['web_location']
		column_clause = ['*']
		web_ids = self.get_prop_values('id_web')
		query_values = ','.join(map(str, web_ids))
		where_clause = 'id_web in (%s)' % query_values
		db_recs = query_seismodb_table(table_clause,
					column_clause=column_clause, where_clause=where_clause)
		db_rec_dict = {rec['id_web']: rec for rec in db_recs}
		for rec in self.recs:
			db_rec = db_rec_dict.get('id_web')
			if db_rec:
				rec['longitude'] = db_rec['longitude']
				rec['latitude'] = db_rec['latitude']
				rec['quality'] = db_rec['quality']
			elif not keep_unmatched:
				rec['longitude'] = rec['latitude'] = rec['quality'] = np.nan

	def get_bad_zip_country_tuples(self):
		zip_country_tuples = set(self.get_unique_zip_country_tuples())
		comm_recs = self.get_communes_from_db().values()
		db_zip_country_tuples = set([(rec['code_p'], rec['country']) for rec in comm_recs])
		bad_zip_country_tuples = zip_country_tuples.difference(db_zip_country_tuples)
		return list(bad_zip_country_tuples)

	def get_bad_zip_ensemble(self):
		bad_zip_country_tuples = self.get_bad_zip_country_tuples()
		return self.subselect_by_zip_country_tuples(bad_zip_country_tuples)

	def aggregate_by_commune(self, comm_key='id_com'):
		"""
		Aggregate enquiries by commune

		:param comm_key:
			see :meth:`get_communes_from_db`

		:return:
			dict, mapping comm_key values to instances of
			:class:`MacroseismicEnquiryEnsemble`
		"""
		comm_rec_dict = self.get_communes_from_db(comm_key=comm_key)
		if comm_key in ('id_com', 'id_main'):
			all_comm_key_values = self.get_prop_values(comm_key)
		elif comm_key == "zip":
			all_comm_key_values = self.get_zip_country_tuples()
		comm_ensemble_dict = {}
		for comm_key_val in comm_rec_dict.keys():
			idxs = [i for i in range(len(self)) if all_comm_key_values[i] == comm_key_val]
			ensemble = self.__getitem__(idxs)
			comm_ensemble_dict[comm_key_val] = ensemble
		return comm_ensemble_dict

	def aggregate_by_grid(self, grid_spacing=5):
		"""
		Aggregate enquiries into rectangular grid cells

		:param grid_spacing:
			grid spacing (in km)
			(default: 5)

		:return:
			dict, mapping (x_left, y_bottom) tuples to instances of
			:class:`MacroseismicEnquiryEnsemble`
		"""
		grid_spacing *= 1000
		import mapping.geotools.coordtrans as ct
		lons = self.get_prop_values('longitude')
		lats = self.get_prop_values('latitude')
		mask = np.isnan(lons)
		lons = np.ma.array(lons, mask=mask)
		lats = np.ma.array(lats, mask=mask)
		X, Y = ct.transform_array_coordinates(ct.WGS84, ct.LAMBERT1972, lons, lats)

		bin_rec_dict = {}
		for r, rec in enumerate(self.recs):
			x, y = X[r], Y[r]
			x_bin = np.floor(x / grid_spacing) * grid_spacing
			y_bin = np.floor(y / grid_spacing) * grid_spacing
			key = (x_bin, y_bin)
			if not key in bin_rec_dict:
				bin_rec_dict[key] = [rec]
			else:
				bin_rec_dict[key].append(rec)

		for key in bin_rec_dict.keys():
			bin_rec_dict[key] = self.__class__(self.id_earth, bin_rec_dict[key])

		return bin_rec_dict

	def aggregate_by_distance(self, lon, lat, distance_interval):
		"""
		Aggregate enquiriess in different distance bins

		:param lon:
			float, longitude of point to compute distance to
		:param lat:
			float, latitude of point to compute distance to
		:param distance_interval:
			float, distance interval for binning (in km)

		:return:
			dict, mapping distance bins to instances of
			:class:`MacroseismicEnquiryEnsemble`
		"""
		distances = self.calc_distances(lon, lat)
		binned_distances = np.floor(distances / distance_interval) * distance_interval
		bin_rec_dict = {}
		for r, rec in enumerate(self.recs):
			bin = binned_distances[r]
			if np.isnan(bin):
				bin = None
			if not bin in bin_rec_dict:
				bin_rec_dict[bin] = [rec]
			else:
				bin_rec_dict[bin].append(rec)

		for key in bin_rec_dict.keys():
			bin_rec_dict[key] = self.__class__(self.id_earth, bin_rec_dict[key])

		return bin_rec_dict

	def aggregate_by_zip(self):
		# TODO: can be removed
		all_zip_country_tuples = self.get_zip_country_tuples()
		unique_zip_country_tuples = set(all_zip_country_tuples)
		zip_ensemble_dict = {}
		for zip_country in unique_zip_country_tuples:
			idxs = [i for i in range(len(self)) if all_zip_country_tuples[i] == zip_country]
			ensemble = self.__getitem__(idxs)
			zip_ensemble_dict[zip_country] = ensemble
		return zip_ensemble_dict

	def aggregate_by_main_zip(self):
		# TODO: can be removed
		comm_recs = self.get_communes_from_db()
		zip_main_id_dict = {}
		## Note: Store id_main as string property (required for subselect_by_property)
		for rec in comm_recs:
			zip_main_id_dict[(rec['zip'], rec['country'])] = str(rec['id_main'])
		for rec in self.recs:
			try:
				rec['id_main'] = zip_main_id_dict[(rec['zip'], rec['country'])]
			except:
				rec['id_main'] = '0'
				print("Warning: id_main not found for ZIP %s-%s" %
						(rec['country'], rec['zip']))
		unique_main_ids = np.unique(self.get_prop_values('id_main'))
		main_id_ensemble_dict = {}
		for id_main in unique_main_ids:
			ensemble = self.subselect_by_property('id_main', [id_main])
			main_id_ensemble_dict[int(id_main)] = ensemble
		return main_id_ensemble_dict

	def fix_all(self):
		"""
		Fix various issues:
		- repair records that have 'felt' unspecified
		- set 'motion', 'reaction' and 'stand' to 0 for not-felt records
		- match unmatched commune IDs
		- set main commune IDs
		- remove duplicate records

		:return:
			instance of :class:`MacroseismicEnquiryEnsemble`
		"""
		ensemble = self.get_copy()
		ensemble.fix_felt_is_none()
		ensemble.fix_not_felt()
		ensemble.fix_commune_ids()
		ensemble.set_main_commune_ids()
		ensemble = ensemble.remove_duplicate_records()
		return ensemble

	def fix_felt_is_none(self):
		"""
		Fix enquiries where 'felt' has not been filled out, based on
		reply to 'asleep', 'motion' and 'stand' questions.

		:return:
			None, 'felt' values of :prop:`recs` are modified in place
		"""
		ensemble = self.subselect_by_property('felt', ['', np.nan])
		## Slept through it --> not felt
		ensemble[ensemble.asleep == 1].set_prop_values('felt', '0')
		## Awoken --> felt
		ensemble[ensemble.asleep == 2].set_prop_values('felt', '1')
		## Awake and (difficult to stand or motion) --> felt
		ensemble[(ensemble.asleep == 0) &
				(ensemble.motion.filled(-1) > 0) &
				(ensemble.stand.filled(-1) > 1)].set_prop_values('felt', '1')
		self._gen_arrays()

	def fix_not_felt(self):
		"""
		For 'not felt' enquiries, set motion, reaction and stand to 0
		to avoid bias in the aggregated computation

		:return:
			None, 'motion', 'reaction' and 'stand' arrays of :prop:`recs`
			are modified in place
		"""
		not_felt_idxs = np.where(self.felt == 0)
		self.motion[not_felt_idxs] = 0
		self.reaction[not_felt_idxs] = 0
		self.stand[not_felt_idxs] = 0

	def calc_felt_index(self, include_other_felt=True):
		"""
		Compute felt indexes for individual questionnaires
		following Wald et al. (1999)

		:param include_other_felt:
			bool, whether or not to include the replies to the question
			"Did others nearby feel the earthquake ?"
			The 'other_felt' classes correspond to fractional values of
			[0.72, 0.36, 0.72, 1, 1]

			Note: A small modification is applied for cases where 'felt'
			is zero or undefined. In that case, the fractional values are:
			[0., 0., 0.36, 0.72, 1]

			(default: True)

		:return:
			float array, felt indexes [range 0 - 1]
		"""
		# TODO: there are only 5 classes for other_felt,
		# but values in database range from 0 to 5 !
		other_felt_classes = np.array([0.72, 0.36, 0.72, 1, 1])
		other_felt_classes_if_felt_is_zero = np.ma.array([0., 0., 0.36, 0.72, 1])
		other_felt_classes_if_felt_is_zero.mask = np.isnan(other_felt_classes_if_felt_is_zero)

		felt_index = self.felt
		if include_other_felt:
			## Note: do not use *= here, it doesn't work
			#felt_index = felt_index * other_felt_classes[self.other_felt]

			## More complex, taking into account other_felt if felt is zero or undefined
			felt_index = np.ma.zeros(len(self.felt))
			felt_index[self.felt == 1] = other_felt_classes[self.other_felt][self.felt == 1]
			felt_index[self.felt == 0] = (
				other_felt_classes_if_felt_is_zero[self.other_felt][self.felt == 0])
			other_felt_classes_if_felt_is_zero.mask = [1, 0, 0, 0, 0]
			felt_index[self.felt.mask] = (
				other_felt_classes_if_felt_is_zero[self.other_felt][self.felt.mask])

		return felt_index

	def calc_motion_index(self):
		"""
		Compute motion indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, motion indexes [range 0 - 1]
		"""
		return self.motion.astype('float')

	def calc_reaction_index(self):
		"""
		Compute reaction indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, reaction indexes [range 0 - 1]
		"""
		return self.reaction.astype('float')

	def calc_shelf_index(self):
		"""
		Compute shelf indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, shelf indexes [range 0 - 3]
		"""
		return np.maximum(0., (self.shelf - 2))

	def calc_picture_index(self):
		"""
		Compute picture indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, picture indexes [range 0 - 1]
		"""
		return np.minimum(1., self.picture)

	def calc_stand_index(self):
		"""
		Compute stand indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, stand indexes [range 0 - 1]
		"""
		return np.minimum(1., self.stand)

	def calc_furniture_index(self, include_heavy_appliance=False):
		"""
		Compute furniture indexes for individual questionnaires
		following Wald et al. (1999)

		:param include_heavy_appliance:
			bool, whether or not to take heavy_appliance into account
			as well (not standard, but occurs with ROB forms)
			(default: False)

		:return:
			float array, furniture indexes [range 0 - 1]
		"""
		if include_heavy_appliance:
			return ((self.furniture) | (self.heavy_appliance > 1)).astype('float')
		else:
			return self.furniture

	def calc_damage_index(self):
		"""
		Compute damage indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, damage indexes [range 0 - 3]
		"""
		## Note that "one or several cracked windows" [0.5] has position 2 in
		## Wald et al. (1999), but position 6 in our questionnaire!
		damage_classes = np.array([0, 0.5, 0.75, 1, 1, 1, 0.5, 2, 2, 2, 3, 3, 3, 3])
		damage_index = np.zeros(len(self), dtype='float')
		for i in range(len(self)):
			if self.damage[i].any():
				## Avoid crash all damage classes are set to False
				damage_index[i] = np.max(damage_classes[self.damage[i]])
			else:
				damage_index[i] = 0.
		return damage_index

	def filter_floors(self, min_level=0, max_level=4, keep_nan_values=True):
		"""
		Filter out responses from caves and floor levels 5 and above

		:param min_level:
			int, lower floor level to keep
			(default: 0)
		:param max_level:
			int, upper floor level to keep
			(default: 4)
		:param keep_nan_values:
			bool, whether or not to keep enquiries where floor is not
			specified

		:return:
			instance of :class:`MacroseismicEnquiryEnsemble`
		"""
		condition = (self.floor >= min_level) & (self.floor <= max_level)
		if keep_nan_values:
			condition.mask = False
			condition = (condition | self.floor.mask)
		idxs = np.where(condition)[0]
		return self.__getitem__(idxs)

	def calc_cws(self, aggregate=True, filter_floors=(0, 4), include_other_felt=True,
				include_heavy_appliance=False):
		"""
		Compute Community Weighted Sum (CWS) following Wald et al. (1999)

		:param aggregate:
			bool, whether to compute overall CWS (True) or CWS for
			individual questionnaires (False)
			(default: True)
		:param filter_floors:
			(min_floor, max_floor) tuple, floors outside this range
			(basement floors and upper floors) are filtered out
			(default: (0, 4))
		:param include_other_felt:
			see :meth:`calc_felt_index`
		:param include_heavy_appliance:
			see :meth:`calc_furniture_index`

		:return:
			float or float array, CWS
		"""
		if filter_floors:
			min_floor, max_floor = filter_floors
			ensemble = self.filter_floors(min_floor, max_floor)
		else:
			ensemble = self

		if aggregate:
			# TODO: remove outliers ? Not possible for these separate indexes!
			## Masked (NaN) values are not taken into account to compute the mean
			## If all values are masked, index is set to zero

			# TODO: do not take into account records where felt_index is masked!
			felt_index = ensemble.calc_felt_index(include_other_felt).mean()
			if np.ma.is_masked(felt_index) and felt_index.mask:
				felt_index = 0.
			motion_index = ensemble.calc_motion_index().mean()
			if np.ma.is_masked(motion_index) and motion_index.mask:
				motion_index = 0.
			reaction_index = ensemble.calc_reaction_index().mean()
			if np.ma.is_masked(reaction_index) and reaction_index.mask:
				reaction_index = 0.
			stand_index = ensemble.calc_stand_index().mean()
			if np.ma.is_masked(stand_index) and stand_index.mask:
				stand_index = 0.
			shelf_index = ensemble.calc_shelf_index().mean()
			if np.ma.is_masked(shelf_index) and shelf_index.mask:
				shelf_index = 0.
			picture_index = ensemble.calc_picture_index().mean()
			if np.ma.is_masked(picture_index) and picture_index.mask:
				picture_index = 0.
			furniture_index = ensemble.calc_furniture_index(include_heavy_appliance).mean()
			if np.ma.is_masked(furniture_index) and furniture_index.mask:
				furniture_index = 0.
			damage_index = ensemble.calc_damage_index().mean()

			cws = (5 * felt_index
					+ motion_index
					+ reaction_index
					+ 2 * stand_index
					+ 5 * shelf_index
					+ 2 * picture_index
					+ 3 * furniture_index
					+ 5 * damage_index)
		else:
			## Masked (NaN) values are replaced with zeros (including felt)
			felt_indexes = ensemble.calc_felt_index(include_other_felt).filled(0)
			motion_indexes = ensemble.calc_motion_index().filled(0)
			reaction_indexes = ensemble.calc_reaction_index().filled(0)
			stand_indexes = ensemble.calc_stand_index().filled(0)
			shelf_indexes = ensemble.calc_shelf_index().filled(0)
			picture_indexes = ensemble.calc_picture_index().filled(0)
			furniture_indexes = ensemble.calc_furniture_index(include_heavy_appliance).filled(0)
			damage_indexes = ensemble.calc_damage_index()
			cws = (5 * felt_indexes
					+ motion_indexes
					+ reaction_indexes
					+ 2 * stand_indexes
					+ 5 * shelf_indexes
					+ 2 * picture_indexes
					+ 3 * furniture_indexes
					+ 5 * damage_indexes)
			## NaN felt values have CWS zero
			#cws = cws.filled(0)
		return cws

	def calc_cdi(self, aggregate=True, filter_floors=(0, 4), include_other_felt=True,
				include_heavy_appliance=False):
		"""
		Compute original Community Decimal Intensity sensu Dengler &
		Dewey (1998)

		:param aggregate:
		:param filter_floors:
		:param include_other_felt:
		:param include_heavy_appliance:
			see :meth:`calc_cws`

		:return:
			float or float array, CDI
		"""
		cws = self.calc_cws(aggregate=aggregate, filter_floors=filter_floors,
							include_other_felt=include_other_felt,
							include_heavy_appliance=include_heavy_appliance)
		return 3.3 + 0.13 * cws

	def calc_cii(self, aggregate=True, filter_floors=(0, 4), include_other_felt=True,
				include_heavy_appliance=False):
		"""
		Compute Community Internet Intensity following Wald et al. (1999),
		later renamed into Community Decimal Intensity (CDI)

		:param aggregate:
		:param filter_floors:
		:param include_other_felt:
		:param include_heavy_appliance:
			see :meth:`calc_cws`

		:return:
			float or float array, CII
		"""
		cws = self.calc_cws(aggregate=aggregate, filter_floors=filter_floors,
							include_other_felt=include_other_felt,
							include_heavy_appliance=include_heavy_appliance)
		cii = 3.40 * np.log(cws) - 4.38

		## Needed to recompute felt index
		if filter_floors:
			min_floor, max_floor = filter_floors
			ensemble = self.filter_floors(min_floor, max_floor)
		else:
			ensemble = self

		## We set a minimum CDI of 2 if the CWS is nonzero (so the result is at
		## least 2 �Felt�, or 1 �Not felt�), and cap the result at 9.0
		if aggregate:
			felt_index = ensemble.calc_felt_index(include_other_felt).mean()
			if np.ma.is_masked(felt_index) and felt_index.mask:
				felt_index = 0.

			## For 'not felt' responses, CII = 1
			if cws == 0 or felt_index == 0:
				cii = 1
			## For any 'felt' response, CII is at least 2
			elif cws > 0 and felt_index > 0:
				cii = np.maximum(2., cii)
		else:
			felt_index = ensemble.calc_felt_index(include_other_felt=include_other_felt)
			felt_index = felt_index.filled(0)
			## For 'not felt' responses, CII = 1
			cii[(cws == 0) | (felt_index == 0)] = 1.
			## Note: the following is identical to setting CII = 2 for CWS < 6.53
			## for any 'felt' response (Wald et al., 1999)
			idxs = (cws > 0) & (felt_index > 0)
			cii[idxs] = np.maximum(2., cii[idxs])
		cii = np.minimum(9., cii)
		return cii

	def calc_mean_cii(self, filter_floors=(0, 4), include_other_felt=True,
					include_heavy_appliance=False, remove_outliers=(2.5, 97.5)):
		"""
		Compute mean CII value from CII values of individual enquiries,
		ignoring outliers. This is an alternative to the aggregated
		CII computation in :meth:`calc_cii`

		:param filter_floors:
		:param include_other_felt:
		:param include_heavy_appliance:
			see :meth:`calc_cii`
		:param remove_outliers:
			(min_pct, max_pct) tuple, percentile range to use
			(default: 2.5, 97.5)

		:return:
			float, mean CII
		"""
		cii = self.calc_cii(aggregate=False, filter_floors=filter_floors,
					include_other_felt=include_other_felt,
					include_heavy_appliance=include_heavy_appliance)
		min_pct, max_pct = remove_outliers
		pct0 = np.percentile(cii, min_pct)
		pct1 = np.percentile(cii, max_pct)
		cii = cii[(cii >= pct0) & (cii <= pct1)]
		if len(cii):
			return cii.mean()
		else:
			return 1.

	def calc_fiability(self, include_other_felt=True, include_heavy_appliance=False,
						aggregate=False, filter_floors=False):
		"""
		Compute reliability of individual enquiries following ROB web
		procedure

		:param include_other_felt:
		:param include_heavy_appliance:
			see :meth:`calc_cws`
		:param aggregate:
		:param filter_floors:
			dummy arguments to have same call signature as :meth:`calc_cws`
			will be ignored

		:return:
			float array, fiabilities
		"""
		emails = self.get_prop_values('email')
		felt_indexes = self.calc_felt_index(include_other_felt)
		motion_indexes = self.calc_motion_index().filled(0)
		reaction_indexes = self.calc_reaction_index().filled(0)
		stand_indexes = self.calc_stand_index().filled(0)
		shelf_indexes = self.calc_shelf_index().filled(0)
		picture_indexes = self.calc_picture_index().filled(0)
		furniture_indexes = self.calc_furniture_index(include_heavy_appliance).filled(0)
		damage_indexes = self.calc_damage_index()

		fiability = np.zeros(len(self))
		for i in range(len(self)):
			fiability[i] = 80

			if emails[i]:
				fiability[i] += 10

			if felt_indexes[i] == 0:
				if motion_indexes[i] > 0:
					fiability[i] -= (10 * motion_indexes[i])
				if reaction_indexes[i] > 0:
					fiability[i] -= (10 * reaction_indexes[i])
				if stand_indexes[i] > 0:
					fiability[i] -= 50
				if (shelf_indexes[i] > 1 or furniture_indexes[i] > 0
					or damage_indexes[i] > 1.5):
					fiability[i] -= (20 * damage_indexes[i])
			else:
				if (stand_indexes[i] == 1 and
					(motion_indexes[i] < 3 or reaction_indexes[i] < 2)):
					fiability[i] -= 30
				elif (motion_indexes[i] < 3 and reaction_indexes[i] > 3):
					fiability[i] -= 30

			if (damage_indexes[i] > 2 and shelf_indexes[i] < 2
				and picture_indexes[i] == 0):
				fiability[i] -= ((damage_indexes[i] - shelf_indexes[i]) * 20)

		fiability = np.maximum(0, np.minimum(100, fiability))
		return fiability

	def plot_analysis_comparison(self, prop='CWS', include_other_felt=True,
								include_heavy_appliance=False):
		"""
		Plot comparison between values in database and computation
		in this module for analysis of individual enquiries.

		:param prop:
			str, property name, either 'CWS', 'CDI', 'CII' or 'fiability'
			(default: 'CWS')
		:param include_other_felt:
		:param include_heavy_appliance:
			see :meth:`calc_cii`

		:return:
			instance of :class:`MacroseismicEnquiryEnsemble`, containing
			all enquiries where calculated property does not match the
			value in the database
		"""
		import pylab

		db_result = getattr(self, prop.upper() if prop != 'fiability' else prop)
		func_name = 'calc_%s' % prop.lower()
		func = getattr(self, func_name)
		recalc_result = func(aggregate=False, filter_floors=False,
								include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance)
		non_matching_ids = []
		for i in range(len(self)):
			db_val = db_result[i]
			recalc_val = recalc_result[i]
			id_web = self.recs[i]['id_web']
			try:
				if not np.allclose(db_val, recalc_val):
					print("#%d: %s != %s" % (id_web, db_val, recalc_val))
					non_matching_ids.append(id_web)
			except:
				print("#%d: %s != %s" % (id_web, db_val, recalc_val))
				non_matching_ids.append(id_web)

		pylab.plot(db_result, recalc_result, 'r+', ms=10)
		xmin, xmax, ymin, ymax = pylab.axis()
		max_val = max(xmax, ymax)
		pylab.plot([0, max_val], [0, max_val], 'k--')
		pylab.xlabel('%s (database)' % prop.upper())
		pylab.ylabel('%s (recomputed)' % prop.upper())
		pylab.grid()
		pylab.show()

		return self.subselect_by_property('id_web', non_matching_ids)

	def bincount(self, prop, bins=None, include_nan=True):
		"""
		Count number of occurrences of possible values for given property

		:param prop:
			string, name of property (that can only have certain values)
			or list or array
		:param bins:
			list or array of bins (values, not edges)
			Should not be None if :param:`prop` corresponds to list or array
			(default: None, will auto-determine)
		:param include_nan:
			bool, whether or not to count NaN values
			(default: True)

		:return:
			(bins, counts) tuple
		"""
		if isinstance(prop, (list, np.ndarray, np.ma.MaskedArray)):
			ar = prop
		elif prop == "damage":
			bins = np.arange(self.damage.shape[1])
			counts = np.sum(self.damage, axis=0)
			return (bins, counts)
		else:
			try:
				ar = getattr(self, prop)
			except AttributeError:
				ar = np.ma.array(self.get_prop_values(prop))
		if bins is None and isinstance(prop, basestring):
			bins = self.bins.get(prop, None)
		if bins is None:
			if include_nan:
				if ar.dtype == 'bool':
					ar = ar.astype('int')
					ar.fill_value = 2
				bins, counts = np.unique(np.ma.filled(ar), return_counts=True)
				bins = bins.astype('float')
				if len(np.ma.compressed(ar)) < len(ar):
					bins[-1] = np.nan
			else:
				bins, counts = np.unique(np.ma.compressed(ar), return_counts=True)
		else:
			if not include_nan:
				#bins.pop(bins.index(np.nan))
				bins = np.delete(bins, np.where(np.isnan(bins)))
			counts = np.zeros(len(bins))
			partial_counts = np.bincount(np.digitize(np.ma.compressed(ar), bins,
										right=True))
			counts[:len(partial_counts)] = partial_counts
			if include_nan and np.ma.is_masked(ar):
				counts[-1] = np.sum(ar.mask)
		return bins, counts

	def get_histogram(self, prop, bin_edges=None):
		"""
		Compute histogram for given property

		:param prop:
			string, name of property (that can have a range of values)
			or list or array
		:param bin_edges:
			list or array of bin edges
			Should not be None if :param:`prop` corresponds to list or array
			(default: None, will auto-determine)

		:return:
			(bin_edges, counts) tuple
		"""
		if isinstance(prop, (list, np.ndarray)):
			ar = prop
		else:
			try:
				ar = getattr(self, prop)
			except AttributeError:
				ar = np.ma.array(self.get_prop_values(prop))
		if bin_edges is None:
			bin_edges = self.bins.get(prop, None)
		if bin_edges is None:
			bin_edges = 10
		counts, bin_edges = np.histogram(ar, bins=bin_edges)
		return bin_edges, counts

	def get_prop_title_and_labels(self, prop, lang='EN'):
		"""
		Extract title and labels for given property from PHP enquiry
		templates for different languages

		:param prop:
			str, name of property
		:param lang:
			str, language, one of 'EN', 'NL', 'FR', 'DE'
			(default: 'EN')

		:return:
			(title, labels) tuple
		"""
		import os
		from .io.parse_php_vars import parse_php_vars

		base_path = os.path.split(__file__)[0]
		php_file = os.path.join(base_path, 'rob', 'webenq', 'const_inq%s.php' % lang.upper())
		php_var_dict = parse_php_vars(php_file)
		if prop == 'felt':
			title = php_var_dict['$form23']
			labels = [php_var_dict['$no'], php_var_dict['$yes']]
		elif prop == 'floor':
			title = php_var_dict['$form21']
			labels = []
		elif prop == 'duration':
			title = php_var_dict['$form25']
			labels = []
		else:
			php_var_name = {'situation': 'sit',
							'building': 'build',
							'asleep': 'sleep',
							'other_felt': 'ofelt',
							'damage': 'd_text'}.get(prop, prop)
			labels = php_var_dict.get('$o_' + php_var_name, [])
			title = php_var_dict.get('$t_' + php_var_name, "")

		## Move 'no answer' labels to end, to match with bincount
		if prop in ["motion", "reaction", "response", "stand", "furniture",
					"heavy_appliance", "walls", "sway", "creak", "shelf",
					"picture"]:
			labels.append(labels.pop(0))

		return (title, labels)

	def plot_pie(self, prop, bins=None, include_nan=True, start_angle=0,
				colors=None, label_lang='EN', fig_filespec=None):
		"""
		Plot pie chart for particular property

		:param prop:
			string, name of property (that can only have certain values)
		:param bins:
			list or array of bins (values, not edges)
			(default: None, will auto-determine)
		:param include_nan:
			bool, whether or not to count NaN values
			(default: True)
		:param start_angle:
			int, angle with respect to X axis (in degrees) where pie
			chart should start
			(default: 0)
		:param colors:
			list of matplotlib color specs for pies
			(default: None, will use matplotlib default colors)
		:param label_lang":
			string, label language ('EN', 'NL', 'FR' or 'DE')
			(default: 'EN')
		:param fig_filespec:
			string, full path to output file
			(default: None, will plot on screen)
		"""
		import pylab

		pylab.clf()
		bins, counts = self.bincount(prop, bins=bins, include_nan=include_nan)
		#colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

		## Extract title and labels from PHP files for different languages
		title, labels = "", []
		if isinstance(prop, basestring):
			title, labels = self.get_prop_title_and_labels(prop, label_lang)
		if labels and len(labels) < len(bins):
			labels.append('No answer')
		if not labels:
			try:
				labels = ["%.0f" % b for b in bins]
			except TypeError:
				labels = ["%s" % b for b in bins]
		if not title and isinstance(prop, basestring):
			title = prop.title()
		pylab.pie(counts, colors=colors, labels=labels, autopct='%1.1f%%',
					startangle=start_angle)
		#draw circle
		#centre_circle = pylab.Circle((0,0), 0.70, fc='white')
		#fig = pylab.gcf()
		#fig.gca().add_artist(centre_circle)

		# Equal aspect ratio ensures that pie is drawn as a circle
		pylab.axis('equal')
		#pylab.tight_layout()
		pylab.title(title)

		if fig_filespec:
			pylab.savefig(fig_filespec)
		else:
			pylab.show()

	def plot_histogram(self, prop, bin_edges=None, fig_filespec=None):
		"""
		Plot histogram for given property

		:param prop:
			string, name of property
		:param bin_edges:
			list or array of bin edges
			(default: None, will auto-determine)

		:param fig_filespec:
			string, full path to output file
			(default: None, will plot on screen)
		"""
		import pylab

		pylab.clf()

		if prop == "num_replies":
			## Histogram of number of communes / number of replies
			comm_rec_dict = self.aggregate_by_commune(comm_key='id_com')
			ar = np.array([ensemble.num_replies for ensemble in comm_rec_dict.values()])
			bin_edges = self.bins['num_replies']
		else:
			#ar = getattr(self, prop)
			ar = prop
		bin_edges, counts = self.get_histogram(ar, bin_edges=bin_edges)

		pylab.bar(bin_edges[:-1], counts, width=np.diff(bin_edges))
		#pylab.hist(ar[ar>0], bins=bin_edges)

		title = prop.title()
		pylab.title(title)

		if fig_filespec:
			pylab.savefig(fig_filespec)
		else:
			pylab.show()

	def report_by_commune(self, comm_key='id_com', sort_column=0, sort_order="asc"):
		"""
		Print a sorted table of commune names, number of replies,
		mean CCI in database and aggregated CII

		:param comm_key:
			str, commune key, either 'id_com' or 'id_main'
			(default: 'id_com')
		:param sort_column:
			int, column number to sort table with
			(default: 0)
		:param sort_order:
			str, either "asc" (ascending) or "desc" (descending)
			(default: "asc")
		"""
		from operator import itemgetter
		from prettytable import PrettyTable

		table = PrettyTable(["Commune", "ID", "Num replies", "Mean CII", "Aggregated CII"])
		comm_ensemble_dict = self.aggregate_by_commune(comm_key)
		comm_rec_dict = self.get_communes_from_db(comm_key)
		for comm_id, ensemble in comm_ensemble_dict.items():
			comm_name = comm_rec_dict[comm_id]['name']
			mean_cii = np.mean(ensemble.CII)
			agg_cii = ensemble.calc_cii(filter_floors=(0,4), include_other_felt=True)
			table.add_row([comm_name, comm_id, len(ensemble), "%.1f" % mean_cii,
							"%.1f" % agg_cii])

		reverse_order = {"asc": False, "desc": True}[sort_order]
		table._rows = sorted(table._rows, key=itemgetter(sort_column), reverse=reverse_order)
		print(table)

	def report_bincount(self, prop, bins=None, include_nan=True, include_labels=False,
						include_empty=False):
		"""
		Print table with bincounts for given property

		:param prop:
		:param bins:
		:param include_nan:
			see :meth:`bincount`
		:param include_labels:
			bool, whether or not to add a column with corresponding labels
			(default: False)
		:param include_empty:
			bool, whether or not to print empty bins
			(default: False)
		"""
		from prettytable import PrettyTable
		bins, numbers = self.bincount(prop, bins=bins, include_nan=include_nan)
		column_names = ['Value', 'Num records']
		if include_labels:
			title, labels = self.get_prop_title_and_labels(prop, lang="EN")
			if labels and len(labels) < len(bins):
				labels.append('No answer')
			column_names.append('Label')

		table = PrettyTable(column_names)
		for i in range(len(bins)):
			bin, num = bins[i], numbers[i]
			if num > 0 or include_empty:
				row = [bin, int(num)]
				if include_labels:
					row.append(labels[i])
				table.add_row(row)
		print(table)

	def evaluate_cws_calculation(self, aggregate=False, include_other_felt=True,
							include_heavy_appliance=False, filter_floors=(0, 4)):
		"""
		Print values of properties used for CWS calculation, and the
		derived indexes.

		:param aggregate:
		:param include_other_felt:
		:param include_heavy_appliance:
		:param filter_floors:
			see :meth:`calc_cws`
		"""
		print("felt:")
		print("  Values: %s" % self.felt)
		felt_index = self.calc_felt_index(include_other_felt=False)
		if aggregate:
			felt_index = felt_index.mean()
		print("  Felt index (without other_felt) [x5]: %s" % (5 * felt_index))

		print("other_felt:")
		print("  Values: %s" % self.other_felt)
		felt_index = self.calc_felt_index(include_other_felt=True)
		if aggregate:
			felt_index = felt_index.mean()
		print("  Felt index (incl. other_felt) [x5]: %s" % (5 * felt_index))

		print("motion:")
		print("  Values: %s" % self.motion)
		motion_index = self.calc_motion_index()
		if aggregate:
			motion_index = motion_index.mean()
		else:
			motion_index = motion_index.filled(0)
		print("  Motion index [x1]: %s" % motion_index)

		print("reaction:")
		print("  Values: %s" % self.reaction)
		reaction_index = self.calc_reaction_index()
		if aggregate:
			reaction_index = reaction_index.mean()
		else:
			reaction_index = reaction_index.filled(0)
		print("  Reaction index [x1]: %s" % reaction_index)

		print("stand:")
		print("  Values: %s" % self.stand)
		stand_index = self.calc_stand_index()
		if aggregate:
			stand_index = stand_index.mean()
		else:
			stand_index = stand_index.filled(0)
		print("  Stand index [x2]: %s" % (2 * stand_index))

		print("shelf:")
		print("  Values: %s" % self.shelf)
		shelf_index = self.calc_shelf_index()
		if aggregate:
			shelf_index = shelf_index.mean()
		else:
			shelf_index = shelf_index.filled(0)
		print("  Shelf index [x5]: %s" % (5 * shelf_index))

		print("picture:")
		print("  Values: %s" % self.picture)
		picture_index = self.calc_picture_index()
		if aggregate:
			picture_index = picture_index.mean()
		else:
			picture_index = picture_index.filled(0)
		print("  Picture index [x2]: %s" % (2 * picture_index))

		print("furniture:")
		print("  Values: %s" % self.furniture)
		furniture_index = self.calc_furniture_index()
		if aggregate:
			furniture_index = furniture_index.mean()
		else:
			furniture_index = furniture_index.filled(0)
		print("  Furniture index [x3]: %s" % (3 * furniture_index))
		furniture_index = self.calc_furniture_index(include_heavy_appliance=True)
		if aggregate:
			furniture_index = furniture_index.mean()
		else:
			furniture_index = furniture_index.filled(0)
		print("  Furniture index (incl. heavy_appliance) [x3]: %s" %
			(3 * furniture_index))

		print("damage:")
		print("  Values: %s" % self.get_prop_values('d_text'))
		damage_index = self.calc_damage_index()
		if aggregate:
			damage_index = damage_index.mean()
		print("  Damage index [x5]: %s" % (5 * damage_index))

		print("CWS:")
		cws = self.CWS
		if aggregate:
			cws = np.mean(cws)
		print("  Database: %s" % cws)
		print("  Recomputed: %s" % self.calc_cws(aggregate=aggregate,
			filter_floors=filter_floors, include_other_felt=include_other_felt,
			include_heavy_appliance=include_heavy_appliance))
		if not aggregate:
			print("  Aggregated: %s" % self.calc_cws(filter_floors=filter_floors,
								include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance))

	def find_duplicate_addresses(self, verbose=True):
		"""
		Find duplicate records based on their address (street name and ZIP).

		:param verbose:
			bool, whether or not to print some useful information

		:return:
			list of lists containing indexes of duplicate records
		"""
		all_streets = self.get_prop_values('street')
		all_zips = self.get_prop_values('zip')
		#all_communes = self.get_prop_values('city')
		unique_streets = []
		unique_idxs = []
		duplicate_idxs = {}
		for s, street in enumerate(all_streets):
			## Only consider non-empty strings containing numbers
			#if street and not street.replace(' ', '').isalpha():
			if street and any(char.isdigit() for char in street.replace(' ', '')):
				## Combine with ZIP
				zip_street = (all_zips[s], street)
				if not zip_street in unique_streets:
					unique_streets.append(zip_street)
					unique_idxs.append(s)
				else:
					## Duplicate
					unique_idx = unique_streets.index(zip_street)
					unique_idx = unique_idxs[unique_idx]
					if unique_idx in duplicate_idxs:
						duplicate_idxs[unique_idx].append(s)
					else:
						duplicate_idxs[unique_idx] = [s]
		duplicate_idxs = [[key] + values for key, values in duplicate_idxs.items()]
		duplicate_idxs = sorted(duplicate_idxs)

		if verbose:
			print("Duplicate streets:")
			for idxs in duplicate_idxs:
				ensemble = self.__getitem__(idxs)
				for rec in ensemble.recs:
					street = rec['street']
					if street:
						street = street.encode('ascii', 'replace')
					zip = rec['zip']
					cii = rec['CII']
					fiability = rec['fiability']
					name = rec['name']
					if name:
						name = name.encode('ascii', errors='replace')
					print("  %s [CII=%s, fiab=%d] %s - %s" % (zip, cii, fiability,
						name, street))
				print("")

		return duplicate_idxs

	def get_duplicate_records(self, verbose=True):
		"""
		Get duplicate records based on their address (street name and ZIP).

		:param verbose:
			bool, whether or not to print some useful information

		:return:
			list of instances of :class:`MacroseismicEnquiryEnsemble`
			for each set of duplicate records
		"""
		duplicate_idxs = self.find_duplicate_addresses(verbose=verbose)
		ensemble_list = []
		for idxs in duplicate_idxs:
			ensemble = self.__getitem__(idxs)
			ensemble_list.append(ensemble)
		return ensemble_list

	def remove_duplicate_records(self, verbose=True):
		"""
		Remove duplicate records based on street name and ZIP code.
		For duplicate records, the one with the highest fiability and
		most recent submit time is kept.

		Note that records are not removed from current instance,
		but a new instance without the duplicate records is returned!

		:param verbose:
			bool, whether or not to print some useful information

		:return:
			instance of :class:`MacroseismicEnquiryEnsemble`
		"""
		duplicate_idxs = self.find_duplicate_addresses(verbose=verbose)
		web_ids_to_remove = []
		for idxs in duplicate_idxs:
			ensemble = self.__getitem__(idxs)
			## Keep the one with highest fiability and most recent submit time
			# TODO: should we also consider CII (lowest = most reliable?)
			submit_times = ensemble.get_prop_values('submit_time')
			ft = list(zip(ensemble.fiability, submit_times))
			## numpy argsort doesn't work for tuples, this works similar
			#order = np.argsort(fiability_times)
			order = sorted(range(len(ft)), key=ft.__getitem__)
			subensemble = ensemble[order[:-1]]
			web_ids_to_remove.extend(subensemble.get_prop_values('id_web'))

		if verbose:
			print("Removing %d duplicates" % len(web_ids_to_remove))
		return self.subselect_by_property('id_web', web_ids_to_remove, negate=True)

	def plot_cumulative_responses_vs_time(self):
		import pylab
		date_times = np.sort(self.get_datetimes()).astype(object)
		pylab.plot(date_times, np.arange(self.num_replies)+1)
		pylab.gcf().autofmt_xdate()
		pylab.xlabel("Time")
		pylab.ylabel("Number of replies")
		pylab.grid(True)
		pylab.show()

	def get_inconsistent_damage_records(self):
		idxs = np.where((self.damage[:, 0] == True)
						& (np.sum(self.damage[:,1:], axis=1) > 0))[0]
		return self.__getitem__(idxs)
