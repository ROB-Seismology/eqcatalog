# -*- coding: iso-Latin-1 -*-

"""
Processing of macroseismic and DYFI data
"""

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

from ..time import as_np_datetime

__all__ = ["DYFIEnsemble", "MacroseismicEnquiryEnsemble",
			"ROBDYFIEnsemble", "ROBMacroseismicEnquiryEnsemble"]


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
	if isinstance(txt, basestring):
		nkfd_form = unicodedata.normalize('NFKD', txt)
		return "".join([c for c in nkfd_form if not unicodedata.combining(c)])
	else:
		return str(txt)


## Disable no-member errors for MacroseismicEnquiryEnsemble
# pylint: disable=no-member


class DYFIEnsemble(object):
	"""
	Ensemble of online 'Did You Feel It?' macroseismic enquiry results

	:param ids:
		array-like, enquiry  IDs
	:param event_ids:
		array-like, earthquake IDs (one for each enquiry!)
	:param event_times:
		array-like, earthquake origin datetimes
	:param event_longitudes:
		array-like, earthquake longitudes (in degrees)
	:param event_latitudes:
		array-like, earthquake latitudes (in degrees)
	:param submit_times:
		array-like, datetimes when enquiries were submitted
	:param longitudes:
		array-like, enquiry longitudes (in degrees)
	:param latitudes:
		array-like, enquiry latitudes (in degrees)
	:param asleep:
		array-like, answer to the question
		'Were you asleep during the earthquake?'
		- 0: no
		- 1: slept through it
		- 2: woke me up
	:param felt:
		array-like, answer to the question
		'Did you feel the earthquake?'
		- 0: no
		- 1: yes
		- nan: no answer
	:param other_felt:
		array-like, answer to the question
		'Did others nearby you feel the earthquake?'
		- 0: no answer (don't know / nobody else nearby)
		- 1: no others felt it
		- 2: some felt it, but most did not
		- 3: most others felt it, but some did not
		- 4: (almost) everyone felt it
	:param motion:
		array-like, answer to the question
		'How would you best describe the ground shaking'
		- 0: not felt
		- 1: weak
		- 2: mild
		- 3: moderate
		- 4: strong
		- 5: violent
		- nan: no answer
	:param duration:
		array-like, answer to the question
		'About how many seconds did the shaking last?'
		int or nan
	:param reaction:
		array-like, answer to the question
		'How would you best describe your reaction?'
		- 0: no reaction
		- 1: very little reaction
		- 2: excitement
		- 3: somewhat frightened
		- 4: very frightened
		- 5: extremely frightened
		- nan: no answer
	:param response:
		array-like, answer to the question
		'How did you respond?'
		- 0: took no action
		- 1: moved to doorway
		- 2: ducked and covered
		- 3: ran outside
		- 4: other
		- nan: no answer
	:param stand:
		array-like, answer to the question
		'Was it difficult to stand or walk?'
		- 0: no
		- 1: yes
		- 2: yes, I fell
		- 3: yes, I was thrown to the ground
		- nan: no answer
		Note: categories 2 and 3 have been added with respect to USGS!
	:param sway:
		array-like, answer to the question
		'Did you notice the swinging/swaying of doors or hanging objects?'
		- 0: no
		- 1: yes, slight swinging
		- 2: yes, violent swinging
		- nan: no answer
	:param creak:
		array-like, answer to the question
		'Did you notice creaking or other noise?'
		- 0: no
		- 1: yes, slight noise
		- 2: yes, loud noise
		- nan: no answer
	:param shelf:
		array-like, answer to the question
		'Did objects topple over or fell off shelves?'
		- 0: no
		- 1: rattled slightly
		- 2: rattled loudly
		- 3: a few toppled or fell off
		- 4: many fell off
		- 5: nearly everything fell off
		- nan: no answer
	:param picture:
		array-like, answer to the question
		'Did pictures on walls move or get knocked askew?'
		- 0: no
		- 1: yes, but did not fall
		- 2: yes, and some fell
		- nan: no answer
	:param furniture:
		array-like, answer to the question
		'Did any furniture or appliances slide, tip over or become displaced?'
		- 0: no
		- 1: yes
		- nan: no answer
	:param heavy_appliance:
		array-like, answer to the question
		'Was a heavy appliance (refrigerator or range) affected?'
		- 0: no
		- 1: yes, some contents fell out
		- 2: yes, shifted by inches
		- 3: yes, shifted by a foot or more
		- 4: yes, overturned
		- nan: no answer
	:param walls:
		array-like, answer to the question
		'Were free-standing walls or fences damaged?'
		- 0: no
		- 1: yes, some were cracked
		- 2: yes, some partially fell
		- 3: yes, some fell completely
		- nan: no answer
	:param damage:
		2D bool array or masked array, answer to the question
		'Was there any damage to the building?'
		Note: several answers are possible
		Note: order is different from questionnaire in Wald et al (1999)!
		- 0: no damage
		- 1: hairline cracks in walls
		- 2: a few large cracks in walls [Wald: 3]
		- 3: many large cracks in walls [Wald: 4]
		- 4: ceiling tiles or lighting fixtures fell [Wald: 5]
		- 5: cracks in chimney [Wald: 6]
		- 6: one or several cracked windows [Wald: 2]
		- 7: many windows cracked or some broken out
		- 8: masonry fell from block or brick wall
		- 9: old chimney, major damage or fell down
		- 10: modern chimney, major damage or fell down
		- 11: outside wall(s) tilted over or collapsed completely
		- 12: separation of porch, balcony, or other addition from building
		- 13: building shifted over foundation
	:param situation:
		array-like, answer to the question
		'What was your situation during the earthquake?'
		- 0: no answer
		- 1: inside
		- 2: outside
		- 3: in stopped vehicle
		- 4: in moving car
		- 5: other
	:param building:
		array-like, answer to the question
		'If you were inside, please select the type of building or structure'
		- 0: no building
		- 1: family home
		- 2: apartment building
		- 3: office building / school
		- 4: mobile home with permanent foundation
		- 5: trailer or recr. vehicle with no foundation
		- 6: other
		- nan: no answer
	:param floor:
		array-like, answer to the question
		'If you know the floor, please specify it'
		int or nan
	:param noise:
		array-like, answer to the question
		'Did you hear a noise?'
		- 0: no
		- 1: yes, light and brief noise
		- 2: yes, light and prolonged noise
		- 3: yes, strong and brief noise
		- 4: yes, strong and prolonged noise
		- nan: no answer
	:param CWS:
		array-like, Community Weighted Sum
	:param CII:
		array-like, Community Internet Intensity (sensu Wald et al., 1999)
	:param CDI:
		array-like, Community Decimal Intensity (sensu Wald et al., 1999)
	:param MI:
		array-like, corresponds to rounded value of CII
	:param fiability:
		array-like, reliability (in percent)
	"""
	## Define bins for most record properties
	bins = {}
	bins['asleep'] = np.array([0, 1, 2])
	bins['felt'] = np.array([0, 1, np.nan])
	bins['other_felt'] = np.array([0, 1, 2, 3, 4])

	bins['motion'] = np.array([0, 1, 2, 3, 4, 5, np.nan])
	bins['duration'] = np.arange(0, 61, 5)
	bins['reaction'] = np.array([0, 1, 2, 3, 4, 5, np.nan])
	bins['response'] = np.array([0, 1, 2, 3, 4, np.nan])
	bins['stand'] = np.array([0, 1, 2, 3, np.nan])

	bins['sway'] = np.array([0, 1, 2, np.nan])
	bins['creak'] = np.array([0, 1, 2, np.nan])
	#bins['shelf'] = np.array(['A', 'B', 'C', 'D', 'E', 'F', '_'])
	bins['shelf'] = np.array([0, 1, 2, 3, 4, 5, np.nan])
	#bins['picture'] = np.array(['A', 'B', 'C', '_'])
	bins['picture'] = np.array([0, 1, 2, np.nan])
	bins['furniture'] = np.array([0, 1, np.nan])
	bins['heavy_appliance'] = np.array([0, 1, 2, 3, 4, np.nan])
	bins['walls'] = np.array([0, 1, 2, 3, np.nan])

	bins['situation'] = np.array([0, 1, 2, 3, 4, 5])
	bins['building'] = np.array([0, 1, 2, 3, 4, 5, 6, np.nan])
	bins['floor'] = np.array([-10.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 10.5, 100])
	bins['noise'] = np.array([0, 1, 2, 3, 4, np.nan])
	bins['fiability'] = np.arange(0, 101, 10)

	bins['CII'] = np.arange(1, 13) - 0.5
	bins['CDI'] = bins['MI'] = bins['CII']

	bins['num_replies'] = np.array([1, 3, 5, 10, 20, 50, 100, 200, 500, 1000])

	def __init__(self, ids, event_ids, event_times, event_longitudes,
				event_latitudes, submit_times, longitudes, latitudes, commune_ids,
				asleep, felt, other_felt, motion, duration, reaction, response,
				stand, sway, creak, shelf, picture, furniture, heavy_appliance,
				walls, damage,
				situation, building, floor, noise,
				CWS, CII, CDI, MI, fiability):
		## Earthquake(s) parameters
		assert(len(ids) == len(event_ids) == len(event_times) == len(event_longitudes)
				== len(event_latitudes))
		self.ids = np.asarray(ids)
		self.event_ids = np.asarray(event_ids)
		self.event_times = as_np_datetime(event_times)
		self.event_longitudes = np.asarray(event_longitudes)
		self.event_latitudes = np.asarray(event_latitudes)

		## Locations and times
		assert len(submit_times) == len(longitudes) == len(latitudes)
		self.submit_times = as_np_datetime(submit_times)
		self.longitudes = np.asarray(longitudes)
		self.latitudes = np.asarray(latitudes)
		self.commune_ids = np.asarray(commune_ids)

		# TODO: set masks from nan values??
		assert len(asleep) == len(felt) == len(other_felt) == len(self)
		self.asleep = np.asarray(asleep)
		self.felt = np.asarray(felt)
		self.other_felt = np.asarray(other_felt)

		## Your experience of the earthquake
		assert(len(motion) == len(duration) == len(reaction) == len(response)
				== len(stand) == len(self))
		self.motion = np.asarray(motion)
		self.duration = np.asarray(duration)
		self.reaction = np.asarray(reaction)
		self.response = np.asarray(response)
		self.stand = np.asarray(stand)

		## Earthquake effects
		assert(len(sway) == len(creak) == len(shelf) == len(picture)
				== len(furniture) == len(heavy_appliance) == len(walls)
				== len(damage) == len(self))
		self.sway = np.asarray(sway)
		self.creak = np.asarray(creak)
		self.shelf = np.asarray(shelf)
		self.picture = np.asarray(picture)
		self.furniture = np.asarray(furniture)
		self.heavy_appliance = np.asarray(heavy_appliance)
		self.walls = np.asarray(walls)
		self.damage = np.asarray(damage)

		## Possibly ROB-specific, but useful
		assert(len(situation) == len(building) == len(floor) == len(noise)
				== len(fiability) == len(self))
		self.situation = np.asarray(situation)
		self.building = np.asarray(building)
		self.floor = np.asarray(floor)
		self.noise = np.asarray(noise)
		self.fiability = np.asarray(fiability)

		## Evaluation
		# TODO: masked arrays as well??
		self.CWS = CWS
		if CWS is not None:
			assert len(CWS) == len(self)
			self.CWS = np.asarray(CWS)
		else:
			self.CWS = CWS

		self.CII = CII
		if CII is not None:
			assert len(CII) == len(self)
			self.CII = np.asarray(CII)

		self.CDI = CDI
		if CDI is not None:
			assert len(CDI) == len(self)
			self.CDI = np.asarray(CDI)

		self.MI = MI
		if MI is not None:
			assert len(MI) == len(self)
			self.MI = np.asarray(MI)

	def __len__(self):
		return len(self.ids)

	def __iter__(self):
		for i in range(len(self)):
			yield self.__getitem__(i)

	def __getitem__(self, spec):
		"""
		Note: slicing results in a view, fancy indexing in a copy !
		"""
		ids = self.ids.__getitem__(spec)
		event_ids = self.event_ids.__getitem__(spec)
		event_times = self.event_times.__getitem__(spec)
		event_longitudes = self.event_longitudes.__getitem__(spec)
		event_latitudes = self.event_latitudes.__getitem__(spec)

		submit_times = self.submit_times.__getitem__(spec)
		longitudes = self.longitudes.__getitem__(spec)
		latitudes = self.latitudes.__getitem__(spec)
		commune_ids = self.commune_ids.__getitem__(spec)

		asleep = self.asleep.__getitem__(spec)
		felt = self.felt.__getitem__(spec)
		other_felt = self.other_felt.__getitem__(spec)

		motion = self.motion.__getitem__(spec)
		duration = self.duration.__getitem__(spec)
		reaction = self.reaction.__getitem__(spec)
		response = self.response.__getitem__(spec)
		stand = self.stand.__getitem__(spec)

		sway = self.sway.__getitem__(spec)
		creak = self.creak.__getitem__(spec)
		shelf = self.shelf.__getitem__(spec)
		picture = self.picture.__getitem__(spec)
		furniture = self.furniture.__getitem__(spec)
		heavy_appliance = self.heavy_appliance.__getitem__(spec)
		walls = self.walls.__getitem__(spec)
		damage = self.damage.__getitem__(spec)

		situation = self.situation.__getitem__(spec)
		building = self.building.__getitem__(spec)
		floor = self.floor.__getitem__(spec)
		noise = self.noise.__getitem__(spec)
		fiability = self.fiability.__getitem__(spec)

		CWS = self.CWS.__getitem__(spec) if self.CWS is not None else None
		CII = self.CII.__getitem__(spec) if self.CII is not None else None
		CDI = self.CDI.__getitem__(spec) if self.CDI is not None else None
		MI = self.MI.__getitem__(spec) if self.MI is not None else None

		return self.__class__(ids, event_ids, event_times, event_longitudes,
				event_latitudes, submit_times, longitudes, latitudes, commune_ids,
				asleep, felt, other_felt, motion, duration, reaction, response,
				stand, sway, creak, shelf, picture, furniture, heavy_appliance,
				walls, damage,
				situation, building, floor, noise,
				CWS, CII, CDI, MI, fiability)

	@property
	def num_replies(self):
		return self.__len__()

	def copy(self):
		#return self.__getitem__(slice(None, None, None))

		ids = self.ids.copy()
		event_ids = self.event_ids.copy()
		event_times = self.event_times.copy()
		event_longitudes = self.event_longitudes.copy()
		event_latitudes = self.event_latitudes.copy()

		submit_times = self.submit_times.copy()
		longitudes = self.longitudes.copy()
		latitudes = self.latitudes.copy()
		commune_ids = self.commune_ids.copy()

		asleep = self.asleep.copy()
		felt = self.felt.copy()
		other_felt = self.other_felt.copy()

		motion = self.motion.copy()
		duration = self.duration.copy()
		reaction = self.reaction.copy()
		response = self.response.copy()
		stand = self.stand.copy()

		sway = self.sway.copy()
		creak = self.creak.copy()
		shelf = self.shelf.copy()
		picture = self.picture.copy()
		furniture = self.furniture.copy()
		heavy_appliance = self.heavy_appliance.copy()
		walls = self.walls.copy()
		damage = self.damage.copy()

		situation = self.situation.copy()
		building = self.building.copy()
		floor = self.floor.copy()
		noise = self.noise.copy()
		fiability = self.fiability.copy()

		CWS = self.CWS.copy() if self.CWS is not None else None
		CII = self.CII.copy() if self.CII is not None else None
		CDI = self.CDI.copy() if self.CDI is not None else None
		MI = self.MI.copy() if self.MI is not None else None

		return DYFIEnsemble(ids, event_ids, event_times, event_longitudes,
				event_latitudes, submit_times, longitudes, latitudes, commune_ids,
				asleep, felt, other_felt, motion, duration, reaction, response,
				stand, sway, creak, shelf, picture, furniture, heavy_appliance,
				walls, damage,
				situation, building, floor, noise,
				CWS, CII, CDI, MI, fiability)

	def get_prop_values(self, prop):
		"""
		Get values corresponding to given property

		:param prop:
			str, name of property
		"""
		if prop == 'd_text':
			map = {True: '*', False: '-'}
			d_text = []
			for i in range(len(self)):
				d_text.append(''.join([map[self.damage[i,j]] for j in range(14)]))
			return d_text
		else:
			return getattr(self, prop)

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

	def set_prop_values(self, prop, values, idxs=None, regenerate_arrays=True):
		"""
		Set values of individual enquiries for given property

		:param prop:
			str, name of property
		:param values:
			list or array, values of individual enquiries for given property
		:param idxs:
			slice, array of indexes or array of bools,
			indexes for which to overwrite values
			(default: None)
		:param regenerate_arrays:
			bool, this argument is only present to be compatible with
			the same method in :class:`ROBMacroseismicEnquiryEnsemble`.
			It is ignored.
		"""
		#if not isinstance(values, (list, tuple, np.ndarray)):
		#	values = [values] * len(self)
		#assert len(values) == len(self)
		## Change values in place
		ar = getattr(self, prop)
		if idxs is None:
			ar[:] = values
		else:
			ar[idxs] = values

	def subselect_by_property(self, prop, prop_values, negate=False):
		"""
		Select part of ensemble matching given property values
		Note that this results in a copy of the ensemble, changes made
		to this copy are NOT reflected in the original ensemble!
		So, this cannot be used to change properties of a subselection
		in place, as it is the case for :class:`ROBMacroseismicEnquiryEnsemble`

		:param prop:
			str, name of property
		:param prop_values:
			list of values of :param:`prop` that should be matched
		:param negate:
			bool, whether or not to reverse the matching

		:return:
			instance of :class:`MacroseismicEnquiryEnsemble`
		"""
		if np.isscalar(prop_values):
			prop_values = [prop_values]
		values = self.get_prop_values(prop)

		idxs = np.zeros_like(values, dtype='bool')
		for pv in prop_values:
			idxs |= np.isclose(values, pv, equal_nan=True)
		if negate:
			idxs = ~idxs

		return self.__getitem__(idxs)

	def get_elapsed_times(self):
		"""
		Get time interval between earthquake origin time and submit time
		of each enquiry

		:return:
			np.timedelta64 array
		"""
		return self.submit_times - self.event_times

	def plot_cumulative_responses_vs_time(self):
		import pylab
		date_times = np.sort(self.submit_times).astype(object)
		pylab.plot(date_times, np.arange(self.num_replies)+1)
		pylab.gcf().autofmt_xdate()
		pylab.xlabel("Time")
		pylab.ylabel("Number of replies")
		pylab.grid(True)
		pylab.show()

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

		rec_lons = self.longitudes
		rec_lats = self.latitudes
		dist = geodetic.spherical_distance(lon, lat, rec_lons, rec_lats) / 1000.
		dist = np.sqrt(dist**2 + z**2)
		return dist

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

	def get_region(self, percentile_width=100):
		"""
		Return geographic extent of ensemble

		:param percentile_width:
			float, difference between upper and lower percentile
			(range 0 - 100) to include in calculation
			(default: 100 = include all points)

		:return:
			(W, E, S, N) tuple
		"""
		dp = 100. - percentile_width
		percentiles = [0 + dp/2., 100 - dp/2.]
		longitudes = self.longitudes
		longitudes = longitudes[np.isfinite(longitudes)]
		latitudes = self.latitudes
		latitudes = latitudes[np.isfinite(latitudes)]
		lonmin, lonmax = np.percentile(longitudes, percentiles)
		latmin, latmax = np.percentile(latitudes, percentiles)
		return (lonmin, lonmax, latmin, latmax)

	def get_centroid(self):
		"""
		Determine geographic centre (mean longitude and latitude)

		:return:
			(lon, lat) tuple
		"""
		lon = np.nanmean(self.longitudes)
		lat = np.nanmean(self.latitudes)
		return (lon, lat)

	def subselect_by_region(self, region):
		"""
		Select part of ensemble situated inside given geographic extent

		:param region:
			(W, E, S, N) tuple
		"""
		lonmin, lonmax, latmin, latmax = region
		longitudes = self.longitudes
		latitudes = self.latitudes
		lon_idxs = (lonmin <= longitudes) & (longitudes <= lonmax)
		lat_idxs = (latmin <= latitudes) & (latitudes <= latmax)
		return self.__getitem__(lon_idxs & lat_idxs)

	def is_geo_located(self):
		"""
		:return:
			bool array, indicating whether or not location is available
		"""
		return ~(np.isnan(self.longitudes) | np.isnan(self.latitudes))

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
		elif prop == "id_earth":
			ar = np.ma.array(self.get_prop_values(prop))
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
			counts = np.zeros(len(bins), dtype='int')
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

	@classmethod
	def get_prop_title_and_labels(cls, prop, lang='EN'):
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
		from ..io.parse_php_vars import parse_php_vars

		base_path = os.path.split(__file__)[0]
		php_file = os.path.join(base_path, '..', 'rob', 'webenq', 'const_inq%s.php' % lang.upper())
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

	def to_mdp_collection(self):
		"""
		Convert to MDP collection

		:return:
			instance of :class:`MDPCollection`
		"""
		from .mdp import MacroseismicDataPoint, MDPCollection

		mdp_list = []
		for dyfi_rec in self:
			id = dyfi_rec.ids[0]
			id_earth = dyfi_rec.event_ids[0]
			Imin = dyfi_rec.CII[0]
			Imax = dyfi_rec.CII[0]
			imt = 'CII'
			lon = dyfi_rec.longitudes[0]
			lat = dyfi_rec.latitudes[0]
			data_type = 'dyfi'
			id_com = 0
			id_main = 0
			fiability = dyfi_rec.fiability[0]
			mdp = MacroseismicDataPoint(id, id_earth, Imin, Imax, imt, lon, lat,
										data_type, id_com, id_main, fiability)
			mdp_list.append(mdp)

		return MDPCollection(mdp_list)

	def aggregate_by_grid(self, grid_spacing=5, srs='LAMBERT1972'):
		"""
		Aggregate enquiries into rectangular grid cells

		:param grid_spacing:
			grid spacing (in km)
			(default: 5)
		:param srs:
			osr spatial reference system or str, name of known srs
			(default: 'LAMBERT1972')

		:return:
			dict, mapping (center_lon, center_lat) tuples to instances of
			:class:`MacroseismicEnquiryEnsemble`
		"""
		import mapping.geotools.coordtrans as ct

		if isinstance(srs, basestring):
			srs = getattr(ct, srs)
		grid_spacing *= 1000
		lons = self.longitudes
		lats = self.latitudes
		mask = np.isnan(lons)
		lons = np.ma.array(lons, mask=mask)
		lats = np.ma.array(lats, mask=mask)
		X, Y = ct.transform_array_coordinates(ct.WGS84, srs, lons, lats)

		bin_rec_dict = {}
		for r, rec in enumerate(self.recs):
			x, y = X[r], Y[r]
			## Center X, Y
			x_bin = np.floor(x / grid_spacing) * grid_spacing + grid_spacing/2.
			y_bin = np.floor(y / grid_spacing) * grid_spacing + grid_spacing/2.
			## Center longitude and latitude
			[(lon_bin, lat_bin)] = ct.transform_coordinates(srs, ct.WGS84,
															[(x_bin, y_bin)])
			key = (lon_bin, lat_bin)
			if not key in bin_rec_dict:
				bin_rec_dict[key] = [rec]
			else:
				bin_rec_dict[key].append(rec)

		for key in bin_rec_dict.keys():
			bin_rec_dict[key] = self.__class__(self.id_earth, bin_rec_dict[key])

		return bin_rec_dict

	def aggregate_by_distance(self, lon, lat, distance_interval):
		"""
		Aggregate enquiries in different distance bins

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

	def aggregate_by_property(self, prop):
		# event_ids, commune_ids
		pass

	def subselect_polygon(self, poly_obj):
		pass

	def aggregate_by_polygon(self, poly_obj):
		pass

	def remove_outliers(self, min_pct=2.5, max_pct=97.5):
		"""
		Remove outliers (with CII outside of confidence range)
		from ensemble

		:param min_pct:
			float, lower percentile
			(default: 2.5)
		:param max_pct:
			float, upper percentile
			(default: 97.5)

		:return:
			instance of :class:`MDPCollection`
		"""
		# TODO: add recalc_cii option, but requires additional parameters...?
		pct0 = np.percentile(self.CII, min_pct)
		pct1 = np.percentile(self.CII, max_pct)
		within_confidence = (self.CII >= pct0) & (self.CII <= pct1)
		idxs = np.where(within_confidence)[0]
		return self.__getitem__(idxs)

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
		with np.errstate(invalid='ignore'):
			condition = (self.floor >= min_level) & (self.floor <= max_level)
		if keep_nan_values:
			condition |= np.isnan(self.floor)
		return self.__getitem__(condition)

	def fix_felt_is_none(self):
		"""
		Fix enquiries where 'felt' has not been filled out, based on
		reply to 'asleep', 'motion' and 'stand' questions.

		:return:
			None, 'felt' values of :prop:`recs` are modified in place
		"""
		felt_is_none = np.isnan(self.felt)
		## Slept through it --> not felt
		idxs = felt_is_none & (self.asleep == 1)
		self.set_prop_values('felt', 0, idxs=idxs, regenerate_arrays=False)
		## Awoken --> felt
		idxs = felt_is_none & (self.asleep == 2)
		self.set_prop_values('felt', 1, idxs=idxs, regenerate_arrays=False)
		## Awake and (difficult to stand or motion) --> felt
		idxs = (felt_is_none & (self.asleep == 0) & (self.motion > 0)
				& (self.stand > 1))
		self.set_prop_values('felt', 1, idxs=idxs, regenerate_arrays=True)
		print('Fixed %d values' % np.sum(felt_is_none))

	def fix_not_felt(self):
		"""
		For 'not felt' enquiries, set motion, reaction and stand to 0
		to avoid bias in the aggregated computation

		:return:
			None, 'motion', 'reaction' and 'stand' values of :prop:`recs`
			are modified in place
		"""
		not_felt = self.felt == 0
		self.set_prop_values('motion', 0, idxs=not_felt, regenerate_arrays=False)
		self.set_prop_values('reaction', 0, idxs=not_felt, regenerate_arrays=False)
		self.set_prop_values('stand', 0, idxs=not_felt, regenerate_arrays=True)
		print('Fixed %d values' % np.sum(not_felt))

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
		other_felt_classes = np.array([0.72, 0.36, 0.72, 1, 1])
		other_felt_classes_if_felt_is_zero = np.ma.array([0., 0., 0.36, 0.72, 1])

		if include_other_felt:
			other_felt = self.other_felt.astype('int')
			felt_index = np.ma.zeros(len(self.felt))
			is_felt = (self.felt == 1)
			felt_index[is_felt] = other_felt_classes[other_felt[is_felt]]
			## Take into account other_felt if felt is zero or undefined
			is_not_felt = (self.felt == 0)
			felt_index[is_not_felt] = (
				other_felt_classes_if_felt_is_zero[other_felt[is_not_felt]])
			other_felt_classes_if_felt_is_zero.mask = [1, 0, 0, 0, 0]
			is_undefined = np.isnan(self.felt)
			felt_index[is_undefined] = (
				other_felt_classes_if_felt_is_zero[other_felt[is_undefined]])
		else:
			felt_index = np.ma.array(felt_index, mask=np.isnan(self.felt))

		return felt_index

	def calc_motion_index(self):
		"""
		Compute motion indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, motion indexes [range 0 - 1]
		"""
		return np.ma.array(self.motion, mask=np.isnan(self.motion))

	def calc_reaction_index(self):
		"""
		Compute reaction indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, reaction indexes [range 0 - 1]
		"""
		return np.ma.array(self.reaction, mask=np.isnan(self.reaction))

	def calc_shelf_index(self):
		"""
		Compute shelf indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, shelf indexes [range 0 - 3]
		"""
		mask = (np.isnan(self.shelf) | (self.shelf == 1) | (self.shelf == 2))
		shelf = np.ma.array(self.shelf, mask=mask)
		with np.errstate(invalid='ignore'):
			return np.maximum(0., (shelf - 2))

	def calc_picture_index(self):
		"""
		Compute picture indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, picture indexes [range 0 - 1]
		"""
		picture = np.ma.array(self.picture, mask=np.isnan(self.picture))
		with np.errstate(invalid='ignore'):
			return np.minimum(1., picture)

	def calc_stand_index(self):
		"""
		Compute stand indexes for individual questionnaires
		following Wald et al. (1999)

		:return:
			float array, stand indexes [range 0 - 1]
		"""
		stand = np.ma.array(self.stand, mask=np.isnan(self.stand))
		with np.errstate(invalid='ignore'):
			return np.minimum(1., stand)

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
		furniture = np.ma.array(self.furniture, mask=np.isnan(self.furniture))
		with np.errstate(invalid='ignore'):
			furniture = np.minimum(1, furniture)
		if include_heavy_appliance:
			## Note: replace nan values with zeros in heavy_appliance,
			## ensuring that furniture gets priority over heavy_appliance!
			heavy_appliance = np.ma.array(self.heavy_appliance,
											mask=np.isnan(self.heavy_appliance))
			return ((furniture > 0) | (heavy_appliance.filled(0) > 1)).astype('float')
		else:
			return furniture

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

	def calc_cws(self, aggregate=True, filter_floors=(0, 4), include_other_felt=True,
				include_heavy_appliance=False, overwrite=False):
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
		:param overwrite:
			bool, whether or not :prop:`CWS` should be overwritten
			Only applies if :param:`aggregate` is False
			(default: False)

		:return:
			float or float array, CWS
		"""
		if filter_floors:
			min_floor, max_floor = filter_floors
			ensemble = self.filter_floors(min_floor, max_floor)
		else:
			ensemble = self

		felt_indexes = ensemble.calc_felt_index(include_other_felt)
		motion_indexes = ensemble.calc_motion_index()
		reaction_indexes = ensemble.calc_reaction_index()
		stand_indexes = ensemble.calc_stand_index()
		shelf_indexes = ensemble.calc_shelf_index()
		picture_indexes = ensemble.calc_picture_index()
		furniture_indexes = ensemble.calc_furniture_index(include_heavy_appliance)
		damage_indexes = ensemble.calc_damage_index()

		## Masked (NaN) values are replaced with zeros (including felt)
		cws_individual = (5 * felt_indexes.filled(0)
						+ motion_indexes.filled(0)
						+ reaction_indexes.filled(0)
						+ 2 * stand_indexes.filled(0)
						+ 5 * shelf_indexes.filled(0)
						+ 2 * picture_indexes.filled(0)
						+ 3 * furniture_indexes.filled(0)
						+ 5 * damage_indexes)

		if aggregate:
			## Aggregate calculation may be affected by biases
			## It is not possible to remove outliers for individual indexes,
			## but we can compute non-aggregated intensities first,
			## and determine outliers from that distribution
			remove_outliers = (2.5, 97.5)
			if remove_outliers:
				min_pct, max_pct = remove_outliers
				cii = 3.40 * np.log(cws_individual) - 4.38
				pct0 = np.percentile(cii, min_pct)
				pct1 = np.percentile(cii, max_pct)
				idxs = (cii >= pct0) & (cii <= pct1)
				felt_indexes = felt_indexes[idxs]
				motion_indexes = motion_indexes[idxs]
				reaction_indexes = reaction_indexes[idxs]
				stand_indexes = stand_indexes[idxs]
				shelf_indexes = shelf_indexes[idxs]
				picture_indexes = picture_indexes[idxs]
				furniture_indexes = furniture_indexes[idxs]
				damage_indexes = damage_indexes[idxs]

			## Masked (NaN) values are not taken into account to compute the mean
			## If all values are masked, index is set to zero

			## In addition, we can remove outliers by only taking into account
			## an index if not more than a given percentage (e.g., 80%) is masked
			max_nan_pct = 0.8

			felt_index = felt_indexes.mean()
			if ((np.ma.is_masked(felt_index) and felt_index.mask)
				or np.sum(felt_indexes.mask)/float(len(felt_indexes)) > max_nan_pct):
				felt_index = 0.
			motion_index = motion_indexes.mean()
			if ((np.ma.is_masked(motion_index) and motion_index.mask)
				or np.sum(motion_indexes.mask)/float(len(motion_indexes)) > max_nan_pct):
				motion_index = 0.
			reaction_index = reaction_indexes.mean()
			if ((np.ma.is_masked(reaction_index) and reaction_index.mask)
				or np.sum(reaction_indexes.mask)/float(len(reaction_indexes)) > max_nan_pct):
				reaction_index = 0.
			stand_index = stand_indexes.mean()
			if ((np.ma.is_masked(stand_index) and stand_index.mask)
				or np.sum(stand_indexes.mask)/float(len(stand_indexes)) > max_nan_pct):
				stand_index = 0.
			shelf_index = shelf_indexes.mean()
			if ((np.ma.is_masked(shelf_index) and shelf_index.mask)
				or np.sum(shelf_indexes.mask)/float(len(ensemble)) > max_nan_pct):
				shelf_index = 0.
			picture_index = picture_indexes.mean()
			if ((np.ma.is_masked(picture_index) and picture_index.mask)
				or np.sum(picture_indexes.mask)/float(len(picture_indexes)) > max_nan_pct):
				picture_index = 0.
			furniture_index = furniture_indexes.mean()
			if ((np.ma.is_masked(furniture_index) and furniture_index.mask)
				or np.sum(furniture_indexes.mask)/float(len(furniture_indexes)) > max_nan_pct):
				furniture_index = 0.
			damage_index = damage_indexes.mean()
			if np.sum(damage_indexes == 0)/float(len(damage_indexes)) > max_nan_pct:
				damage_index = 0.

			cws = (5 * felt_index
					+ motion_index
					+ reaction_index
					+ 2 * stand_index
					+ 5 * shelf_index
					+ 2 * picture_index
					+ 3 * furniture_index
					+ 5 * damage_index)
		else:
			cws = cws_individual
			## NaN felt values have CWS zero
			#cws = cws.filled(0)

			if overwrite:
				self.set_prop_values('CWS', cws)

		return cws

	def calc_cdi(self, aggregate=True, filter_floors=(0, 4), include_other_felt=True,
				include_heavy_appliance=False, overwrite=False):
		"""
		Compute original Community Decimal Intensity sensu Dengler &
		Dewey (1998)

		:param aggregate:
		:param filter_floors:
		:param include_other_felt:
		:param include_heavy_appliance:
		:param overwrite:
			see :meth:`calc_cws`

		:return:
			float or float array, CDI
		"""
		cws = self.calc_cws(aggregate=aggregate, filter_floors=filter_floors,
							include_other_felt=include_other_felt,
							include_heavy_appliance=include_heavy_appliance,
							overwrite=overwrite)
		cdi = 3.3 + 0.13 * cws
		if overwrite and aggregate is False:
			self.set_prop_values('CDI', cdi)

		return cdi

	def calc_cii(self, aggregate=True, filter_floors=(0, 4), include_other_felt=True,
				include_heavy_appliance=False, overwrite=False):
		"""
		Compute Community Internet Intensity following Wald et al. (1999),
		later renamed into Community Decimal Intensity (CDI)

		:param aggregate:
		:param filter_floors:
		:param include_other_felt:
		:param include_heavy_appliance:
		:param overwrite:
			see :meth:`calc_cws`

		:return:
			float or float array, CII
		"""
		cws = self.calc_cws(aggregate=aggregate, filter_floors=filter_floors,
							include_other_felt=include_other_felt,
							include_heavy_appliance=include_heavy_appliance,
							overwrite=overwrite)
		cii = 3.40 * np.log(cws) - 4.38

		## Needed to recompute felt index
		if filter_floors:
			min_floor, max_floor = filter_floors
			ensemble = self.filter_floors(min_floor, max_floor)
		else:
			ensemble = self

		## We set a minimum CDI of 2 if the CWS is nonzero (so the result is at
		## least 2 “Felt”, or 1 “Not felt”), and cap the result at 9.0
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

		if overwrite and aggregate is False:
			self.set_prop_values('CII', cii)

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
			return 0.

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
			id_web = self.ids[i]
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

		return self.subselect_by_property('ids', non_matching_ids)

	def get_inconsistent_damage_records(self):
		"""
		Extract ensemble containing inconsistent damage field

		:return:
			instance of :class:`DYFIEnsemble` or subclass
		"""
		idxs = np.where((self.damage[:, 0] == True)
						& (np.sum(self.damage[:,1:], axis=1) > 0))[0]
		return self.__getitem__(idxs)

	def plot_cii_histogram(self, include_other_felt=True, include_heavy_appliance=False,
						color=None, label='', **kwargs):
		"""
		Plot histogram of individual questionnaire intensities
		"""
		from plotting.generic_mpl import plot_histogram

		cii = self.calc_cii(aggregate=False, include_other_felt=include_other_felt,
							include_heavy_appliance=include_heavy_appliance)
		bins = np.arange(int(np.ceil(cii).max()))
		if label:
			kwargs['labels'] = [label]
		if color:
			kwargs['colors'] = [color]
		kwargs['xticks'] = bins

		return plot_histogram([cii], bins + 0.5, **kwargs)



MacroseismicEnquiryEnsemble = DYFIEnsemble


class ROBDYFIEnsemble(DYFIEnsemble):
	"""
	DYFI ensemble linked to ROB database

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

	def __getitem__(self, spec):
		## Note: recs are shared!
		if isinstance(spec, int):
			return self.__class__(self.id_earth, [self.recs[spec]])
		elif isinstance(spec, slice):
			return self.__class__(self.id_earth, self.recs[spec])
		elif isinstance(spec, (list, np.ndarray)):
			## spec can be list of indexes or list of bools
			recs = []
			if len(spec):
				idxs = np.arange(len(self))
				idxs = idxs[np.asarray(spec)]
				for idx in idxs:
					recs.append(self.recs[idx])
			return self.__class__(self.id_earth, recs)

	def _gen_arrays(self):
		"""
		Generate masked arrays from the following record properties:
		- CII
		- CDI
		- MI
		- CWS
		- fiability

		- situation
		- building
		- floor
		- asleep
		- noise
		- felt
		- other_felt
		- motion
		- duration
		- reaction
		- response
		- stand
		- furniture
		- heavy-appliance
		- walls
		- sway
		- creak
		- shelf
		- picture
		- damage

		- id_web

		Note that in many cases, string values are automatically
		converted to floats, and None values to np.nan

		Note: these arrays should not be modified, as they will be
		regenerated by :meth:`set_prop_values` !

		:return:
			None, arrays are stored as class properties
		"""
		for prop in ["CII", "CDI", "MI", "CWS", "fiability"]:
			ar = np.array([rec[prop] for rec in self.recs])
			ar.setflags(write=False)
			setattr(self, prop, ar)

		for prop in ["situation", "building", "floor", "asleep", "noise",
					"felt", "other_felt", "motion", "duration",
					"reaction", "response", "stand", "furniture",
					"heavy_appliance", "walls", "id_web"]:
			prop_list = []
			for rec in self.recs:
				val = rec[prop]
				if val in ('', None) or not (isinstance(val, int) or val.isdigit()):
					val = None
				prop_list.append(val)
			#prop_list = [rec[prop] if rec[prop] != '' else None for rec in self.recs]
			#prop_list = [val if val is not None and (isinstance(val, int) or val.isdigit())
			#			else None for val in prop_list]
			try:
				## Note: dtype='float' ensures that None values are converted to nan
				ar = np.array(prop_list, dtype='float')
			except:
				print("Warning: Array generation failed for prop %s" % prop)
			else:
				#mask = np.isnan(ar)
				#ar = np.ma.array(ar.astype(np.int), mask=mask)
				ar.setflags(write=False)
				setattr(self, prop, ar)

		## Fix other_felt array
		## Only 5 classes are defined, but values in database range from 0 to 5 !
		## other_felt = 1 means no answer, but there are also zero values...
		## The solution is to subtract 1 from values > 0,
		## so that 0 and 1 collapse to the same class
		self.other_felt.setflags(write=True)
		self.other_felt[self.other_felt > 0] -= 1
		## There is even one nan value...
		#self.other_felt[self.other_felt.mask] = 0
		self.other_felt[np.isnan(self.other_felt)] = 0
		self.other_felt.setflags(write=False)

		for prop in ["sway", "creak", "shelf", "picture"]:
			char_map = {c:num for (num, c) in enumerate('ABCDEFGHIJK')}
			char_map['_'] = None
			char_map[''] = None
			prop_list = [rec[prop] or '' for rec in self.recs]
			#mask = np.array([True if val in ('', '_') else False for val in prop_list])
			#ar = np.array(prop_list, dtype='c')
			prop_list = [char_map[val] for val in prop_list]
			ar = np.array(prop_list, dtype='float')
			#ar = np.ma.array(ar, mask=mask)
			ar.setflags(write=False)
			setattr(self, prop, ar)

		self.damage = np.zeros((len(self), 14), dtype='bool')
		for r, rec in enumerate(self.recs):
			damage = rec['d_text']
			damage = [1 if d == '*' else 0 for d in damage]
			self.damage[r] = damage
		self.damage.setflags(write=False)

	def copy(self, deepcopy=False):
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

	def to_simple_dyfi_ensemble(self):
		"""
		Convert to simple DYFI ensemble

		:return:
			instance of :class:`DYFIEnsemble`
		"""
		return super(ROBMacroseismicEnquiryEnsemble, self).copy()

	def get_prop_values(self, prop):
		"""
		Get list of values for given property, as read from :prop:`recs`
		Note that type is not converted, and that empty values are
		represented as None values!

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
			if prop in ["longitude", "latitude", "CII", "CDI", "MI", "CWS",
					"fiability", "situation", "building", "floor", "asleep",
					"noise", "felt", "other_felt", "motion", "duration",
					"reaction", "response", "stand", "furniture",
					"heavy_appliance", "walls"]:
				#none_val = np.nan
				none_val = None
			elif prop in ["d_text", "sway", "creak", "shelf", "picture"]:
				none_val = u""
			else:
				## Note that this may fail if there is only 1 enquiry
				first_non_None_value = next((rec[prop] for rec in self.recs
											if rec[prop] is not None), None)
				if isinstance(first_non_None_value, basestring):
					none_val = u""
				elif first_non_None_value is None:
					none_val = None
				else:
					none_val = np.nan
			return [rec[prop] if rec[prop] is not None else none_val
					for rec in self.recs]

	def set_prop_values(self, prop, values, idxs=None, regenerate_arrays=True):
		"""
		Set values of individual enquiries for given property
		This is done by overwriting the corresponding values in :prop:`recs`,
		optionally followed by regenerating the arrays

		:param prop:
			str, name of property
		:param values:
			list or array, values of individual enquiries for given property
		:param idxs:
			slice, array of indexes or array of bools,
			indexes for which to overwrite values
			(default: None)
		:param regenerate_arrays:
			bool, whether or not to regenerate porperty arrays
			(default: True)
		"""
		if not isinstance(values, (list, tuple, np.ndarray)):
			values = [values] * len(self)
		if idxs is not None:
			ensemble = self.__getitem__(idxs)
		else:
			ensemble = self
		#assert len(values) == len(ensemble)
		for r, rec in enumerate(ensemble.recs):
			rec[prop] = values[r]
		if regenerate_arrays:
			self._gen_arrays()

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
		if np.isscalar(prop_values):
			prop_values = [prop_values]
		if hasattr(self, prop):
			## Regular array
			values = getattr(self, prop)
			idxs = np.ones_like(values, dtype='bool')
			for pv in prop_values:
				idxs |= np.isclose(values, pv, equal_nan=True)
			if negate:
				idxs = ~idxs
		else:
			## List
			values = self.get_prop_values(prop)
			if len(values) and isinstance(values[0], basestring):
				prop_values = [str(pv) if pv is not None else pv for pv in prop_values]
			if not negate:
				idxs = [i for i in range(len(values)) if values[i] in prop_values]
			else:
				idxs = [i for i in range(len(values)) if not values[i] in prop_values]

		return self.__getitem__(idxs)

	@property
	def longitudes(self):
		return np.array(self.get_prop_values('longitude'), dtype='float')

	@property
	def latitudes(self):
		return np.array(self.get_prop_values('latitude'), dtype='float')

	@property
	def ids(self):
		return np.array(self.get_prop_values('id_web'))

	@property
	def event_ids(self):
		return self.get_prop_values('id_earth')

	@property
	def commune_ids(self):
		return self.get_prop_values('id_com')

	def get_catalog(self):
		"""
		Fetch earthquake catalog from ROB database

		:return:
			instance of :class:`eqcatalog.EQCatalog`
		"""
		from ..rob.seismodb import query_local_eq_catalog_by_id

		return query_local_eq_catalog_by_id(list(np.unique(self.event_ids)))

	def get_catalog_indexes(self):
		"""
		Determine index for each enquiry in earthquake catalog

		:return:
			int array
		"""
		_, catalog_indexes = np.unique(self.event_ids, return_inverse=True)
		return catalog_indexes

	@property
	def event_longitudes(self):
		idxs = self.get_catalog_indexes()
		catalog = self.get_catalog()
		return catalog.lons[idxs]

	@property
	def event_latitudes(self):
		idxs = self.get_catalog_indexes()
		catalog = self.get_catalog()
		return catalog.lats[idxs]

	@property
	def event_times(self):
		"""
		Event times for each enquiry as reported in database

		:return:
			datetime64 array
		"""
		import datetime
		years = self.get_prop_values('time_year')
		months = self.get_prop_values('time_month')
		days = self.get_prop_values('time_day')
		hrmin = self.get_prop_values('time_hrmin')

		event_times = []
		for i in range(len(self)):
			dt = datetime.datetime(years[i], months[i], days[i]) + hrmin[i]
			event_times.append(dt)
		return np.array(event_times, dtype='datetime64[s]')

	@property
	def submit_times(self):
		"""
		Get submit time of each enquiry
		Note: NULL date values will be replace by the event time
		(see :meth:`get_event_times`)

		:return:
			datetime64 array
		"""
		submit_times = self.get_prop_values('submit_time')
		NULL_DATE = '0000-00-00 00:00:00'
		if NULL_DATE in submit_times:
			event_times = self.event_times
			submit_times = [submit_times[i] if not submit_times[i] == NULL_DATE
							else event_times[i] for i in range(len(self))]

		submit_times = np.array(submit_times, dtype='datetime64[s]')
		return submit_times

	def get_elapsed_times(self, use_enq_event_time=False):
		"""
		Get time interval between earthquake origin time and submit time
		of each enquiry

		:param use_enq_event_time:
			bool, whether or not to use event time reported in enquiry
			If False, earthquake origin time will be fetched from the
			earthquakes database, but this only works if all enquiries
			belong to the same earthquake
			(default: False)

		:return:
			np.timedelta64 array
		"""
		if use_enq_event_time:
			event_times = self.event_times
		else:
			idxs = self.get_catalog_indexes()
			catalog = self.get_catalog()
			event_times = catalog.get_datetimes()[idxs]

		return self.submit_times - event_times

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
			#street = street.replace(zip_commune, '').strip().rstrip(',')
			street = street.replace('%s' % zip, '')
			street = street.replace(commune, '')
			street = street.strip().rstrip(',').strip()

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
		Determine locations based on addresses

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
						g = geocoder.get(address, provider=provider, proximity=bbox,
										**kwargs)
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
		return sorted(zip_country_tuples)

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
		from ..rob.seismodb import query_seismodb_table
		from difflib import SequenceMatcher as SM

		if comm_key in ("id_com", "id_main"):
			if comm_key == "id_main" and not hasattr(self, "id_main"):
				self.set_main_commune_ids()
			unique_ids = sorted(set(self.get_prop_values(comm_key)))
			table_clause = ['communes']
			column_clause = ['*']
			query_values = ','.join(map(str, unique_ids))
			where_clause = 'id in (%s)' % query_values
			comm_recs = query_seismodb_table(table_clause, column_clause,
								where_clause=where_clause, verbose=verbose)
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
				if len(unique_zips):
					cities = [strip_accents(city).title()
							for city in ensemble.get_prop_values('city')]
					unique_zip_cities = set(zip(zips, cities))
					#join_clause = [('RIGHT JOIN', 'communes', '%s.id = communes.id_main' % table_clause)]

					country_comm_rec_dict = {}
					if country == "NL":
						query_values = '|'.join(['%s' % ZIP for ZIP in unique_zips if ZIP])
						where_clause = 'zip REGEXP "%s"' % query_values
					else:
						query_values = ','.join(['"%s"' % ZIP for ZIP in unique_zips if ZIP])
						where_clause = 'zip IN (%s)' % query_values
					for table in com_tables:
						table_clause = table
						comm_recs = query_seismodb_table(table_clause, column_clause,
										where_clause=where_clause, verbose=verbose)
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
			bool, keep unmatched records untouched (True) or set to nan
			(False)
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
			if not rec['location_quality'] or rec['location_quality'] <= max_quality:
				if comm_key in ("id_com", "id_main"):
					key = rec[comm_key]
				elif comm_key == "zip":
					key = (rec[comm_key], rec['country'])
				comm_rec = comm_rec_dict.get(key)
				if comm_rec:
					rec['longitude'] = comm_rec['longitude']
					rec['latitude'] = comm_rec['latitude']
					# TODO: decide on location quality
					rec['location_quality'] = {'id_com': 5,
										'zip': 5,
										'id_main': 4}[comm_key]
				elif not keep_unmatched:
					rec['longitude'] = rec['latitude'] = rec['location_quality'] = np.nan

	def read_locations_from_db(self):
		"""
		Read locations for each record from the database

		:return:
			dict, mapping id_web to database records (with 'longitude',
			'latitude' and 'quality' keys)
		"""
		from ..rob.seismodb import query_seismodb_table

		table_clause = ['web_location']
		column_clause = ['*']
		web_ids = self.get_prop_values('id_web')
		query_values = ','.join(map(str, web_ids))
		where_clause = 'id_web in (%s)' % query_values
		db_recs = query_seismodb_table(table_clause, column_clause,
										where_clause=where_clause)
		db_rec_dict = {rec['id_web']: rec for rec in db_recs}
		return db_rec_dict

	def set_locations_from_geolocation(self, keep_unmatched=True):
		"""
		Set location of all records from geolocation in database

		:param keep_unmatched:
			bool, keep unmatched records untouched (True) or set to nan
			(False)
			(default: True)

		:return:
			None, 'longitude' and 'latitude' values of :prop:`recs`
			are created or modified in place
		"""
		db_rec_dict = self.read_locations_from_db()
		for rec in self.recs:
			db_rec = db_rec_dict.get('id_web')
			if db_rec:
				rec['longitude'] = db_rec['longitude']
				rec['latitude'] = db_rec['latitude']
				rec['location_quality'] = db_rec['quality']
			elif not keep_unmatched:
				rec['longitude'] = rec['latitude'] = rec['location_quality'] = np.nan

	def write_locations_to_db(self, user, passwd, min_quality=6, overwrite=False,
							dry_run=False):
		"""
		Write locations to database

		:param user:
			str, name of user with write permission
		:param passwd:
			str, password for given user
		:param min_quality:
			int, minimum location quality to write to database
			(default: 6)
		:param overwrite:
			bool, whether or not existing locations should be overwritten
			(default: False)
		:param dry_run:
			bool, whether to actually write locations to database (False)
			or just report how many records would be added/modified (True)
			(default: False)
		"""
		import db.simpledb as simpledb
		from secrets.seismodb import host, database
		#from ..rob.seismodb import query_seismodb_table

		db_rec_dict = self.read_locations_from_db()
		recs_to_add, recs_to_modify = [], []
		for rec in self.recs:
			if rec['location_quality'] >= min_quality:
				if rec['id_web'] in db_rec_dict:
					if overwrite:
						recs_to_modify.append(rec)
				else:
					recs_to_add.append(rec)

		## Write to database
		seismodb = simpledb.MySQLDB(database, host, user, passwd)
		table_name = 'web_location'
		if len(recs_to_add):
			print("Adding %d new records" % len(recs_to_add))
			if not dry_run:
				seismodb.add_records(table_name, recs_to_add)
		if len(recs_to_modify):
			print("Updating %d existing records" % len(recs_to_modify))
			if not dry_run:
				seismodb.update_rows(table_name, recs_to_modify, 'id_web')

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

	def get_aggregated_info(self, aggregate_by='id_com', min_replies=3,
							min_fiability=80, filter_floors=(0, 4),
							agg_info='cii', agg_method='mean', fix_records=True,
							include_other_felt=True, include_heavy_appliance=False,
							remove_outliers=(2.5, 97.5)):
		"""
		Get aggregated macroseismic information.

		:param aggregate_by:
			str, type of aggregation, specifying how macroseismic data
			should be aggregated, one of:
			- 'id_com' or 'commune'
			- 'id_main' or 'main commune'
			- 'grid_X' (where X is grid spacing in km)
			- None or '' (= no aggregation, i.e. info is returned for
			  all replies individually)
			(default: 'id_com')
		:param min_replies:
			int, minimum number of replies to use for aggregating
			(default: 3)
		:param min_fiability:
			int, minimum fiability of enquiries to include in aggregate
			(default: 80)
		:param filter_floors:
			(lower_floor, upper_floor) tuple, floors outside this range
			(basement floors and upper floors) are filtered out
			(default: (0, 4))
		:param agg_info:
			str, info to aggregate, either 'cii', 'cdi' or 'num_replies'
			(default: 'cii')
		:param agg_method:
			str, how to aggregate individual enquiries,
			either 'mean' (= ROB practice), 'aggregated' (= DYFI practice),
			'mean-aggregated' or 'aggregated-mean'
			(default: 'mean')
		:param fix_records:
			bool, whether or not to fix various issues (see :meth:`fix_all`)
			(default: True)
		:param include_other_felt:
			bool, whether or not to take into acoount the replies to the
			question "Did others nearby feel the earthquake ?"
			(default: True)
		:param include_heavy_appliance:
			bool, whether or not to take heavy_appliance into account
			as well (not standard, but occurs with ROB forms)
			(default: False)
		:param remove_outliers:
			(min_pct, max_pct) tuple, percentile range to use.
			Only applies if :param:`agg_method` = 'mean'
			and if :param:`agg_info` = 'cii'
			(default: 2.5, 97.5)

		:return:
			instance of :class:`MacroInfoCollection`
		"""
		from .macro_info import MacroseismicInfo, MacroInfoCollection

		if agg_method[:3] in ('min', 'max'):
			print("Warning: aggregation method %s not supported!" % agg_method)
			return

		ensemble = self
		if fix_records:
			ensemble = ensemble.fix_all()

		ensemble = ensemble[ensemble.fiability >= min_fiability]
		if filter_floors:
			ensemble = ensemble.filter_floors(*filter_floors, keep_nan_values=True)

		if len(ensemble) == 0:
			return MacroInfoCollection([], aggregate_by, 'internet')

		if aggregate_by == 'commune':
			aggregate_by = 'id_com'
		elif aggregate_by == 'main commune':
			aggregate_by = 'id_main'
		if aggregate_by in ('id_com', 'id_main'):
			comm_key = aggregate_by
			ensemble.set_locations_from_communes(comm_key=comm_key, max_quality=10,
												keep_unmatched=False)
			agg_ensemble_dict = ensemble.aggregate_by_commune(comm_key=comm_key)
		elif not aggregate_by:
			min_replies = 1
			## If there are no locations, get them from the communes
			if (~ensemble.is_geo_located()).all():
				ensemble.set_locations_from_communes(comm_key='id_com')
			agg_ensemble_dict = {}
			for subensemble in ensemble:
				if subensemble.is_geo_located()[0]:
					id_web = subensemble.recs[0]['id_web']
					agg_ensemble_dict[id_web] = subensemble
		elif aggregate_by[:4] == 'grid':
			## Include non-geocoded enquiries
			# TODO: should this be an option?
			ensemble.set_locations_from_communes(comm_key='id_com', max_quality=5,
												keep_unmatched=False)
			if '_' in aggregate_by:
				_, grid_spacing = aggregate_by.split('_')
				grid_spacing = float(grid_spacing)
			else:
				grid_spacing = 5
			agg_ensemble_dict = ensemble.aggregate_by_grid(grid_spacing)

		try:
			unassigned = agg_ensemble_dict.pop(0)
		except KeyError:
			unassigned = None
		else:
			print("Note: %d enquiries are not assigned to a commune"
					% unassigned.num_replies)

		macro_infos = []
		for key in list(agg_ensemble_dict.keys()):
			num_replies = agg_ensemble_dict[key].num_replies
			if num_replies < min_replies:
				agg_ensemble_dict.pop(key)
				continue
			if aggregate_by in ('id_com', 'id_main'):
				id_com = agg_ensemble_dict[key].recs[0][comm_key]
			else:
				id_com = key
			if aggregate_by in ('id_com', 'id_main') or not aggregate_by:
				lon = agg_ensemble_dict[key].longitudes[0]
				lat = agg_ensemble_dict[key].latitudes[0]
			elif aggregate_by[:4] == 'grid':
				lon, lat = key
			web_ids = agg_ensemble_dict[key].ids

			if agg_info == 'cii':
				if 'aggregated' in agg_method:
					Iagg = agg_ensemble_dict[key].calc_cii(filter_floors=False,
								include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance)
				if 'mean' in agg_method:
					Imean = agg_ensemble_dict[key].calc_mean_cii(filter_floors=False,
								include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance,
								remove_outliers=remove_outliers)
			elif agg_info == 'cdi':
				if 'aggregated' in agg_method:
					Iagg = agg_ensemble_dict[key].calc_cdi(filter_floors=False,
								include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance)
				if 'mean' in agg_method:
					Imean = np.mean(agg_ensemble_dict[key].calc_cdi(aggregate=False,
								filter_floors=False, include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance))
			residual = 0
			if agg_info in ('cii', 'cdi'):
				if agg_method in ('mean', 'mean-aggregated'):
					I = Imean
				elif agg_method in ('aggregated', 'aggregated-mean'):
					I = Iagg
				if agg_method == 'mean-aggregated':
					residual = Imean - Iagg
				elif agg_method == 'aggregated-mean':
					residual = Iagg - Imean
			elif agg_info == "num_replies":
				I = 1
			else:
				print("Don't know how to compute %s" % agg_info)
				exit()
			macro_info = MacroseismicInfo(ensemble.id_earth, id_com, I, aggregate_by,
									'internet', num_replies, lon=lon, lat=lat,
									residual=residual, db_ids=web_ids)
			macro_infos.append(macro_info)

		proc_info = dict(min_replies=min_replies, min_fiability=min_fiability,
						filter_floors=filter_floors, agg_method=agg_method,
						fix_records=fix_records, include_other_felt=include_other_felt,
						include_heavy_appliance=include_heavy_appliance,
						remove_outliers=remove_outliers)

		macro_info_coll = MacroInfoCollection(macro_infos, aggregate_by, 'internet',
											proc_info=proc_info)

		return macro_info_coll

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

	def fix_all(self, verbose=False):
		"""
		Fix various issues:
		- repair records that have 'felt' unspecified
		- set 'motion', 'reaction' and 'stand' to 0 for not-felt records
		- match unmatched commune IDs
		- set main commune IDs
		- remove duplicate records
		- recompute fiabilities

		:param verbose:
			bool, whether or not to print information about
			duplicate records
			(default: False)

		:return:
			instance of :class:`MacroseismicEnquiryEnsemble`
		"""
		ensemble = self.copy()
		ensemble.fix_felt_is_none()
		ensemble.fix_not_felt()
		ensemble.fix_commune_ids()
		if len(ensemble) > 0:
			ensemble.set_main_commune_ids()
		ensemble = ensemble.remove_duplicate_records(verbose=verbose)
		ensemble.set_prop_values('fiability', ensemble.calc_fiability(include_other_felt=False,
													include_heavy_appliance=True))
		return ensemble

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


ROBMacroseismicEnquiryEnsemble = ROBDYFIEnsemble
