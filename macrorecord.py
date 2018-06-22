# -*- coding: iso-Latin-1 -*-
"""
"""

## Import standard python modules
import datetime

## Third-party modules
import numpy as np


__all__ = ["MacroseismicRecord", "MacroseismicEnquiryEnsemble",
			"MacroseismicDataPoint"]




class MacroseismicDataPoint:
	pass


class MacroseismicRecord():
	"""
	Container class to hold information of records retrieved from the macrocatalog database table.
	Currently has the following properties:
		id_earth
		id_com
		I
		lon
		lat
		num_replies
	"""
	def __init__(self, id_earth, id_com, I, num_replies=1, lon=0, lat=0, web_ids=[]):
		self.id_earth = id_earth
		self.id_com = id_com
		self.I = I
		self.num_replies = num_replies
		self.lon = lon
		self.lat = lat
		self.web_ids = web_ids

	def get_enquiries(self, min_fiability=20, verbose=False):
		from seismodb import query_ROB_Web_enquiries

		if self.web_ids:
			ensemble = query_ROB_Web_enquiries(web_ids=self.web_ids, verbose=verbose)
		else:
			ensemble = query_ROB_Web_enquiries(self.id_earth, id_com=self.id_com,
								min_fiability=min_fiability, verbose=verbose)

		return ensemble


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
			recs = []
			for idx in spec:
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
			prop_list = [val if (val and (isinstance(val, int) or val.isdigit())) else None for val in prop_list]
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

		self.bins['fiability'] = np.arange(0, 101, 10)
		self.bins['CII'] = np.arange(1, 13) - 0.5
		self.bins['CDI'] = self.bins['MI'] = self.bins['CII']
		self.bins['duration'] = np.arange(0, 61, 5)

	def get_list(self, prop):
		"""
		Get list of values for given property

		:param prop:
			string, name of property (that can only have certain values)

		:return:
			list
		"""
		if len(self.recs) and isinstance(self.recs[0][prop], (str, unicode)):
			none_val = u""
		else:
			none_val = np.nan
		return [rec[prop] if rec[prop] is not None else none_val for rec in self.recs]

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
		values = self.get_list(prop)
		if isinstance(values[0], (str, unicode)):
			prop_values = map(str, prop_values)
		if not negate:
			idxs = [i for i in range(len(values)) if values[i] in prop_values]
		else:
			idxs = [i for i in range(len(values)) if not values[i] in prop_values]
		return self.__getitem__(idxs)

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
		idxs = np.argwhere(all_distances <= radius)
		return self.__getitem__(idxs)

	def calc_distances(self, lon, lat):
		"""
		Compute distances with respect to a particular point

		:param lon:
			float, longitude of point (in degrees)
		:param lat:
			float, latitude of point (in degrees)

		:return:
			array, distances (in km)
		"""
		import mapping.geotools.geodetic as geodetic
		rec_lons = np.array(self.get_list('longitude'))
		rec_lats = np.array(self.get_list('latitude'))
		return geodetic.spherical_distance(lon, lat, rec_lons, rec_lats) / 1000.

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

	def get_zip_country_tuples(self):
		"""
		Return list of (ZIP, country) tuples for each record

		:return:
			list of (ZIP, country) tuples
		"""
		zips = self.get_list('zip')
		countries = self.get_list('country')
		zip_country_tuples = zip(zips, countries)
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
		from seismodb import query_seismodb_table

		if comm_key in ("id_com", "id_main"):
			unique_ids = sorted(set(self.get_list(comm_key)))
			table_clause = ['communes']
			column_clause = ['*']
			query_values = ','.join(map(str, unique_ids))
			db_key = {'id_com': 'id', 'id_main': 'id_main'}[comm_key]
			where_clause = '%s in (%s)' % (db_key, query_values)
			comm_recs = query_seismodb_table(table_clause,
						column_clause=column_clause, where_clause=where_clause)
			comm_rec_dict = {rec[db_key]: rec for rec in comm_recs}
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
				unique_zips = sorted(set(ensemble.get_list('zip')))
				unique_zip_cities = set(zip(ensemble.get_list('zip'), ensemble.get_list('city')))
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
							comm_recs = {(country, rec['zip'][:-3], rec[com_col]): rec for rec in comm_recs}
						else:
							comm_recs = {(country, rec['zip'], rec[com_col]): rec for rec in comm_recs}
						country_comm_rec_dict.update(comm_recs)

					for (ZIP, city) in unique_zip_cities:
						try:
							key = (country, ZIP, city.title())
						except:
							key = (country, ZIP, city)
						rec = country_comm_rec_dict.get(key)
						if rec:
							## Zip and commune name matched
							comm_rec_dict[key] = rec
						else:
							## Only zip matched, keep only smallest id_com
							country, ZIP, city = key
							matching_zips = []
							for k in country_comm_rec_dict.keys():
								if k[1] == ZIP:
									matching_zips.append(country_comm_rec_dict[k])
							id_coms = [r['id_com'] for r in matching_zips]
							if len(matching_zips):
								idx = np.argmin(id_coms)
								comm_rec_dict[key] = matching_zips[idx]
							else:
								## Unmatched ZIP, probably wrong
								pass
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
							# TODO: try matching name first (city/commune, city, we will also need lang...)
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
			if not (rec['id_main'] and keep_existing):
				rec['id_main'] = main_commune_ids[r]

	def set_locations_from_communes(self, comm_key="id_com", keep_unmatched=True):
		"""
		Set location of all records from corresponding communes

		:param comm_key:
			see :meth:`get_communes_from_db`
		:param keep_unmatched:
			bool, whether or not to keep unmatched records untouched

		:return:
			None, 'longitude' and 'latitude' values of :prop:`recs`
			are created or modified in place
		"""
		comm_rec_dict = self.get_communes_from_db(comm_key=comm_key)
		for rec in self.recs:
			if comm_key in ("id_com", "id_main"):
				key = rec[comm_key]
			elif comm_key == "zip":
				key = (rec[comm_key], rec['country'])
			comm_rec = comm_rec_dict.get(key)
			if comm_rec:
				rec['longitude'] = comm_rec['longitude']
				rec['latitude'] = comm_rec['latitude']
			elif not keep_unmatched:
				rec['longitude'] = rec['latitude'] = np.nan

	def set_locations_from_geolocation(self, keep_unmatched=True):
		"""
		Set location of all records from geolocation in database

		:param keep_unmatched:
			bool, whether or not to keep unmatched records untouched

		:return:
			None, 'longitude' and 'latitude' values of :prop:`recs`
			are created or modified in place
		"""
		from seismodb import query_seismodb_table

		table_clause = ['web_location']
		column_clause = ['*']
		web_ids = self.get_list('id_web')
		query_values = ','.join(map(str, web_ids))
		where_clause = 'id_web in (%s)' % query_values
		db_recs = query_seismodb_table(table_clause,
					column_clause=column_clause, where_clause=where_clause)
		db_rec_dict = {rec['id_web']: rec for rec in db_recs}
		for rec in self.recs:
			db_rec = comm_recs.get(key)
			if db_rec:
				rec['longitude'] = db_rec['longitude']
				rec['latitude'] = db_rec['latitude']
			elif not keep_unmatched:
				rec['longitude'] = rec['latitude'] = np.nan

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
			all_comm_key_values = self.get_list(comm_key)
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
		import mapping.geotools.coordtrans as ct
		lons = self.get_list('longitude')
		lats = self.get_list('latitude')
		mask = np.isnan(lons)
		lons = np.ma.array(lons, mask=mask)
		lats = np.ma.array(lats, mask=mask)
		X, Y = ct.transform_array_coordinates(ct.wgs84, ct.lambert1972, lons, lats)

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

	def aggregate_by_zip(self):
		all_zip_country_tuples = self.get_zip_country_tuples()
		unique_zip_country_tuples = set(all_zip_country_tuples)
		zip_ensemble_dict = {}
		for zip_country in unique_zip_country_tuples:
			idxs = [i for i in range(len(self)) if all_zip_country_tuples[i] == zip_country]
			ensemble = self.__getitem__(idxs)
			zip_ensemble_dict[zip_country] = ensemble
		return zip_ensemble_dict

	def aggregate_by_main_zip(self):
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
		unique_main_ids = np.unique(self.get_list('id_main'))
		main_id_ensemble_dict = {}
		for id_main in unique_main_ids:
			ensemble = self.subselect_by_property('id_main', [id_main])
			main_id_ensemble_dict[int(id_main)] = ensemble
		return main_id_ensemble_dict

	def calc_felt_index(self, include_other_felt=True):
		"""
		Compute felt indexes for individual questionnaires
		following Wald et al. (1999)

		:param include_other_felt:
			bool, whether or not to include the replies to the question
			Did others nearby feel the earthquake ?
			(default: True)

		:return:
			float array, felt indexes [range 0 - 1]
		"""
		other_felt_classes = np.array([0.72, 0.36, 0.72, 1, 1])
		felt_index = self.felt
		if include_other_felt:
			## Note: do not use *= here, it doesn't work
			felt_index = felt_index * other_felt_classes[self.other_felt]
		return felt_index

	def calc_shelf_index(self):
		"""
		Compute shelf indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, shelf indexes [range 0 - 1]
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

	def calc_damage_index(self):
		"""
		Compute damage indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, damage indexes [range 0 - 3]
		"""
		damage_classes = np.array([0, 0.5, 0.5, 0.75, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
		damage_index = np.zeros(len(self), dtype='float')
		for i in range(len(self)):
			damage_index[i] = np.max(damage_classes[self.damage[i]])
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
		idxs = np.argwhere(condition)
		return self.__getitem__(idxs)

	def calc_cws(self, aggregate=True, filter_floors=(0, 4), include_other_felt=True):
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

		:return:
			float or float array, CWS
		"""
		if filter_floors:
			min_floor, max_floor = filter_floors
			ensemble = self.filter_floors(min_floor, max_floor)
		else:
			ensemble = self
		# TODO: if felt_index is zero, shouldn't CWS be zero as well?
		if aggregate:
			# TODO: remove outliers ? Not possible for these separate indexes!
			felt_index = ensemble.calc_felt_index(include_other_felt).mean()
			if np.ma.is_masked(felt_index) and felt_index.mask:
				felt_index = 0.
			motion_index = ensemble.motion.mean(dtype='float')
			if np.ma.is_masked(motion_index) and motion_index.mask:
				motion_index = 0.
			reaction_index = ensemble.reaction.mean(dtype='float')
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
			furniture_index = ensemble.furniture.mean(dtype='float')
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
			cws = (5 * ensemble.calc_felt_index(include_other_felt)
					+ ensemble.motion
					+ ensemble.reaction
					+ 2 * ensemble.calc_stand_index()
					+ 5 * ensemble.calc_shelf_index()
					+ 2 * ensemble.calc_picture_index()
					+ 3 * ensemble.furniture
					+ 5 * ensemble.calc_damage_index())
		return cws

	def calc_cdi(self, aggregate=True, filter_floors=(0, 4), include_other_felt=True):
		"""
		Compute original Community Decimal Intensity sensu Dengler &
		Dewey (1998)

		:param aggregate:
		:param filter_floors:
		:param include_other_felt:
			see :meth:`calc_cws`

		:return:
			float or float array, CDI
		"""
		cws = self.calc_cws(aggregate=aggregate, filter_floors=filter_floors,
							include_other_felt=include_other_felt)
		return 3.3 + 0.13 * cws

	def calc_cii(self, aggregate=True, filter_floors=(0, 4), include_other_felt=True):
		"""
		Compute Community Internet Intensity following Wald et al. (1999),
		later renamed into Community Decimal Intensity (CDI)

		:param aggregate:
		:param filter_floors:
		:param include_other_felt:
			see :meth:`calc_cws`

		:return:
			float or float array, CII
		"""
		cws = self.calc_cws(aggregate=aggregate, filter_floors=filter_floors,
							include_other_felt=include_other_felt)
		cii = 3.40 * np.log(cws) - 4.38
		if aggregate:
			if cws == 0:
				cii = 1
			elif cws > 0:
				cii = np.maximum(2., cii)
		else:
			cii[cws == 0] = 1
			cii[cws > 0] = np.maximum(2., cii[cws > 0])
		cii = np.minimum(9., cii)
		return cii

	def calc_mean_cii(self, filter_floors=(0, 4), include_other_felt=True,
						remove_outliers=(5, 95)):
		cii = self.calc_cii(aggregate=False, filter_floors=filter_floors,
					include_other_felt=include_other_felt)
		min_pct, max_pct = remove_outliers
		pct0 = np.percentile(cii, min_pct)
		pct1 = np.percentile(cii, max_pct)
		cii = cii[(cii >= pct0) & (cii <= pct1)]
		return cii.mean()

	def plot_cii_comparison(self, include_other_felt=True):
		import pylab

		db_cii = self.CII
		recalc_cii = self.calc_cii(aggregate=False, filter_floors=False,
								include_other_felt=include_other_felt)
		idxs = np.argsort(db_cii)
		pylab.plot([0,9], [0,9])
		pylab.plot(db_cii[idxs], recalc_cii[idxs], '+')
		pylab.xlabel('CII (database)')
		pylab.ylabel('CII (recomputed)')
		pylab.grid()
		pylab.show()

	def bincount(self, prop, bins=None, include_nan=True):
		"""
		Count number of occurrences of possible values for given property

		:param prop:
			string, name of property (that can only have certain values)
		:param bins:
			list or array of bins (values, not edges)
			(default: None, will auto-determine)
		:param include_nan:
			bool, whether or not to count NaN values
			(default: True)

		:return:
			(bins, counts) tuple
		"""
		try:
			ar = getattr(self, prop)
		except AttributeError:
			ar = np.ma.array(self.get_list(prop))
		if bins is None:
			bins = self.bins.get(prop, None)
		if bins is None:
			if include_nan:
				bins, counts = np.unique(ar.filled(), return_counts=True)
				bins = bins.astype('float')
				bins[-1] = np.nan
			else:
				bins, counts = np.unique(ar.compressed(), return_counts=True)
		else:
			if not include_nan:
				#bins.pop(bins.index(np.nan))
				bins = np.delete(bins, np.argwhere(bins == np.nan))
			counts = np.zeros(len(bins))
			partial_counts = np.bincount(np.digitize(ar, bins, right=True))
			counts[:len(partial_counts)] = partial_counts
			if include_nan:
				counts[-1] = np.sum(ar.mask)
		return bins, counts

	def get_histogram(self, prop, bin_edges=None):
		"""
		Compute histogram for given property

		:param prop:
			string, name of property (that can have a range of values)
		:param bin_edges:
			list or array of bin edges
			(default: None, will auto-determine)

		:return:
			(bin_edges, counts) tuple
		"""
		ar = getattr(self, prop)
		if bin_edges is None:
			bin_edges = self.bins.get(prop, None)
		if bin_edges is None:
			bin_edges = 10
		counts, bin_edges = np.histogram(ar, bins=bin_edges)
		return bin_edges, counts

	def plot_pie(self, prop, bins=None, include_nan=True, start_angle=0,
				colors=None, fig_filespec=None):
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
		:param fig_filespec:
			string, full path to output file
			(default: None, will plot on screen)
		"""
		# TODO: extract title and labels from PHP files for different languages
		import pylab

		pylab.clf()
		bins, counts = self.bincount(prop, bins=bins, include_nan=include_nan)
		#colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
		try:
			labels = ["%.0f" % b for b in bins]
		except TypeError:
			labels = ["%s" % b for b in bins]
		pylab.pie(counts, colors=colors, labels=labels, autopct='%1.1f%%',
					startangle=start_angle)
		#draw circle
		#centre_circle = pylab.Circle((0,0), 0.70, fc='white')
		#fig = pylab.gcf()
		#fig.gca().add_artist(centre_circle)

		# Equal aspect ratio ensures that pie is drawn as a circle
		pylab.axis('equal')
		#pylab.tight_layout()
		pylab.title(prop.title())

		if fig_filespec:
			pylab.savefig(fig_filespec)
		else:
			pylab.show()

	def plot_histogram(self, prop, bin_edges=None, fig_filespec=None):
		# TODO: plot number of communes / number of replies histogram
		import pylab

		pylab.clf()

		ar = getattr(self, prop)
		bin_edges, counts = self.get_histogram(prop, bin_edges=bin_edges)
		pylab.bar(bin_edges[:-1], counts, width=np.diff(bin_edges))
		#pylab.hist(ar[ar>0], bins=bin_edges)

		pylab.title(prop.title())

		if fig_filespec:
			pylab.savefig(fig_filespec)
		else:
			pylab.show()

		if prop in ("asleep", "noise"):
			pass

		elif prop == "duration":
			print(np.nanmin(ar[ar>0]), np.nanmean(ar[ar>0]), np.nanmax(ar[ar>0]))
			bins = np.arange(21)
			ticks = None
			#pylab.hist(ar[ar>0], bins=bins)

		else:
			print("Don't know how to plot %s" % prop)
			return
