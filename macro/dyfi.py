# -*- coding: iso-Latin-1 -*-

"""
Processing of online 'Did You Feel It?' (DYFI) data
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

__all__ = ["DYFIEnsemble", "MacroseismicEnquiryEnsemble"]



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
		Note: A/B/C in ROB database!
	:param creak:
		array-like, answer to the question
		'Did you notice creaking or other noise?'
		- 0: no
		- 1: yes, slight noise
		- 2: yes, loud noise
		- nan: no answer
		Note: A/B/C in ROB database!
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
		Note: A/B/C/D/E/F in ROB database!
	:param picture:
		array-like, answer to the question
		'Did pictures on walls move or get knocked askew?'
		- 0: no
		- 1: yes, but did not fall
		- 2: yes, and some fell
		- nan: no answer
		Note: A/B in ROB database!
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

	def __init__(self, ids, event_ids, event_times,
				event_longitudes, event_latitudes, event_depths,
				submit_times, longitudes, latitudes, commune_ids,
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
		self.event_depths = np.asarray(event_depths)

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

	def __repr__(self):
		return '<DYFIEnsemble (n=%d)>' % len(self)

	def __len__(self):
		return len(self.ids)

	def __iter__(self):
		for i in range(len(self)):
			yield self.__getitem__(i)

	def __getitem__(self, spec):
		"""
		Note: slicing results in a view, fancy indexing in a copy !
		"""
		if isinstance(spec, (int, np.integer)):
			## Turn index in a slice
			spec = slice(spec, spec + 1)

		ids = self.ids.__getitem__(spec)
		event_ids = self.event_ids.__getitem__(spec)
		event_times = self.event_times.__getitem__(spec)
		event_longitudes = self.event_longitudes.__getitem__(spec)
		event_latitudes = self.event_latitudes.__getitem__(spec)
		event_depths = self.event_depths.__getitem__(spec)

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

		return self.__class__(ids, event_ids, event_times,
				event_longitudes, event_latitudes, event_depths,
				submit_times, longitudes, latitudes, commune_ids,
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
		event_depths = self.event_depths.copy()

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

		return DYFIEnsemble(ids, event_ids, event_times,
				event_longitudes, event_latitudes, event_depths,
				submit_times, longitudes, latitudes, commune_ids,
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

	def calc_distances(self, lon, lat, depth=0):
		"""
		Compute distances with respect to a particular point

		:param lon:
			float, longitude of point (in degrees)
		:param lat:
			float, latitude of point (in degrees)
		:param depth:
			float, depth of point (in km)
			(default: 0)

		:return:
			array, distances (in km)
		"""
		import mapping.geotools.geodetic as geodetic

		rec_lons = self.longitudes
		rec_lats = self.latitudes
		dist = geodetic.spherical_distance(lon, lat, rec_lons, rec_lats) / 1000.
		dist = np.sqrt(dist**2 + depth**2)
		return dist

	def calc_epicentral_distances(self):
		"""
		Compute epicentral distances

		:return:
			array, distances (in km)
		"""
		lon, lat = self.event_longitudes, self.event_latitudes
		return self.calc_distances(lon, lat, depth=0)

	def calc_hypocentral_distances(self):
		"""
		Compute hypocentral distances

		:return:
			array, distances (in km)
		"""
		lon, lat = self.event_longitudes, self.event_latitudes
		depth = self.event_depths
		return self.calc_distances(lon, lat, depth=depth)

	def subselect_by_distance(self, ref_pt, radius):
		"""
		Select part of ensemble situated inside given radius around
		given point

		:param ref_pt:
			reference point, either (lon, lat, [depth]) tuple or
			object having 'lon', 'lat' and optionally 'depth' properties
			If None, :prop:`event_longitudes` and :prop:`event_latitudes`
			will be used
		:param radius:
			float, radius (in km)
		"""
		if ref_pt is None:
			lon, lat = self.event_longitudes, self.event_latitudes
			depth = self.event_depths
		elif hasattr(ref_pt, 'lon'):
			lon, lat = ref_pt.lon, ref_pt.lat
			depth = getattr(ref_pt, 'depth', 0.)
		else:
			lon, lat = ref_pt[:2]
			if len(ref_pt) > 2:
				depth = ref_pt[2]
			else:
				depth = 0

		all_distances = self.calc_distances(lon, lat, depth=depth)
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

		:return:
			instance of :class:`DYFIEnsemble` or subclass
		"""
		lonmin, lonmax, latmin, latmax = region
		longitudes, latitudes = self.longitudes, self.latitudes
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
		counts, bin_edges = np.histogram(ar[np.isfinite(ar)], bins=bin_edges)
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
			comm_rec_dict = self.split_by_commune(comm_key='id_com')
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
						include_empty=False, label_lang='EN'):
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
			title, labels = self.get_prop_title_and_labels(prop, lang=label_lang)
			if labels and len(labels) < len(bins):
				labels.append({'EN': 'No answer',
									'NL': 'Geen antwoord',
									'FR': 'Pas de réponse',
									'DE': 'Kein Antwort'}[label_lang])
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

	def print_form(self, lang='EN', institute='USGS'):
		"""
		Print enquiry form for each individual record

		:param lang:
			str, language
		:param institute:
			str, determines number of fields and their order,
			either 'USGS' or 'ROB'
		"""
		if institute.upper() == 'USGS':
			fields = ['asleep', 'felt', 'other_felt', 'stand', 'motion',
						'duration', 'reaction', 'response', 'stand',
						'sway', 'creak', 'shelf', 'picture', 'furniture',
						'heavy_appliance', 'walls', 'damage']
		elif institute.upper() == 'ROB':
			fields = ['situation', 'building', 'floor', 'asleep', 'felt',
					'noise', 'other_felt', 'motion', 'duration', 'reaction',
					'response', 'stand', 'sway', 'creak', 'shelf', 'picture',
					'furniture', 'heavy_appliance', 'walls', 'damage']

		for prop in fields:
			question, labels = self.get_prop_title_and_labels(prop, lang=lang)
			print(question)
			for d, dyfi in enumerate(self):
				if prop in ('duration', 'floor'):
					print('  [%2d] %s' % (d, getattr(dyfi, prop)[0]))
				else:
					bins, counts = dyfi.bincount(prop)
					if len(labels) < len(bins):
						labels.append('No answer')
					for idx, count in enumerate(counts):
						if count:
							answer = labels[idx]
							print('  [%2d] %s' % (d, answer))

	def to_mdp_collection(self, convert_cii=None):
		"""
		Convert to MDP collection

		:param convert_cii:
			str, how to convert CII to Imin/Imax: 'round', 'floor_ceil'
			or None
			(default: None)

		:return:
			instance of :class:`MDPCollection`
		"""
		from .mdp import MacroseismicDataPoint, MDPCollection

		mdp_list = []
		for dyfi_rec in self:
			id = dyfi_rec.ids[0]
			id_earth = dyfi_rec.event_ids[0]
			if convert_cii == 'round':
				Imin = Imax = np.round(dyfi_rec.CII[0])
			elif convert_cii == 'floor_ceil':
				Imin = np.floor(dyfi_rec.CII[0])
				Imax = np.ceil(dyfi_rec.CII[0])
			else:
				Imin = Imax = dyfi_rec.CII[0]
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

	def split_by_grid(self, grid_spacing=5, srs='LAMBERT1972'):
		"""
		Split enquiries into rectangular grid cells

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

		agg_idx_dict = {}
		for r in range(len(self)):
			x, y = X[r], Y[r]
			## Center X, Y
			x_bin = np.floor(x / grid_spacing) * grid_spacing + grid_spacing/2.
			y_bin = np.floor(y / grid_spacing) * grid_spacing + grid_spacing/2.
			## Center longitude and latitude
			[(lon_bin, lat_bin)] = ct.transform_coordinates(srs, ct.WGS84,
															[(x_bin, y_bin)])
			key = (lon_bin, lat_bin)
			if not key in agg_idx_dict:
				agg_idx_dict[key] = [r]
			else:
				agg_idx_dict[key].append(r)

		for key, idxs in agg_idx_dict.items():
			agg_idx_dict[key] = self.__getitem__(idxs)

		return agg_idx_dict

	def split_by_distance(self, ref_pt, distance_interval):
		"""
		Split enquiries in different distance bins

		:param ref_pt:
			see :meth:`subselect_by_distance`
		:param distance_interval:
			float, distance interval for binning (in km)

		:return:
			dict, mapping distance bins to instances of
			:class:`MacroseismicEnquiryEnsemble`
		"""
		if ref_pt is None:
			lon, lat = self.event_longitudes, self.event_latitudes
			depth = self.event_depths
		elif hasattr(ref_pt, 'lon'):
			lon, lat = ref_pt.lon, ref_pt.lat
			depth = getattr(ref_pt, 'depth', 0.)
		else:
			lon, lat = ref_pt[:2]
			if len(ref_pt) > 2:
				depth = ref_pt[2]
			else:
				depth = 0

		distances = self.calc_distances(lon, lat, depth=depth)
		binned_distances = np.floor(distances / distance_interval) * distance_interval
		binned_distances += distance_interval / 2.

		agg_idx_dict = {}
		for r in range(len(self)):
			bin = binned_distances[r]
			if np.isnan(bin):
				bin = None
			if not bin in agg_idx_dict:
				agg_idx_dict[bin] = [r]
			else:
				agg_idx_dict[bin].append(r)

		for key, idxs in agg_idx_dict.items():
			agg_idx_dict[key] = self.__getitem__(idxs)

		return agg_idx_dict

	def split_by_property(self, prop):
		"""
		Split enquiries in different sets based on property values

		:param prop:
			str, property name (e.g., 'event_ids', 'commune_ids')

		:return:
			dict, mapping property values to instances of
			:class:`DYFIEnsemble` or subclasses thereof
		"""
		agg_idx_dict = {}
		prop_values = self.get_prop_values(prop)
		for r, prop_val in enumerate(prop_values):
			if not prop_val in agg_idx_dict:
				agg_idx_dict[prop_val] = [r]
			else:
				agg_idx_dict[prop_val].append(r)

		for key, idxs in agg_idx_dict.items():
			agg_idx_dict[key] = self.__getitem__(idxs)

		return agg_idx_dict

	def subselect_by_polygon(self, poly_obj):
		"""
		Select DYFI records that are situated inside given polygon

		:param poly_obj:
			polygon or closed linestring object (ogr geometry object
			or oqhazlib.geo.polygon.Polygon object)

		:return:
			(dyfi_inside, dyfi_outside) tuple:
			instances of :class:`DYFIEnsemble` or subclass
		"""
		from mapping.geotools.pt_in_polygon import filter_points_by_polygon

		idxs_inside, idxs_outside = filter_points_by_polygon(self.longitudes,
													self.latitudes, poly_obj)
		return (self.__getitem__(idxs_inside), self.__getitem__(idxs_outside))

	def split_by_polygon_data(self, poly_data, value_key):
		"""
		Split DYFI ensemble according to a set of polygons

		:param poly_data:
			instance of :class:`layeredbasemap.MultiPolygonData`
			or list of instances of :class:`osgeo.ogr.Geometry`
			or str, full path to GIS file containing polygon data
		:param value_key:
			str, key in values dict of :param:`poly_data` that should
			be used to link DYFI records to polygons
			If None, use sequential number

		:return:
			dict, mapping polygon IDs to instances of
			:class:`DYFIEnsemble` or subclass
		"""
		import mapping.layeredbasemap as lbm

		if isinstance(poly_data, basestring):
			gis_data = lbm.GisData(poly_data)
			_, _, poly_data = gis_data.get_data()

		if value_key is not None:
			if len(poly_data) != len(np.unique(poly_data.values[value_key])):
				print("Warning: Polygon data values not unique for key %s!"
						% value_key)

		dyfi_dict = {}
		dyfi_outside = self
		for p, poly_obj in enumerate(poly_data):
			try:
				poly_id = poly_obj.value.get(value_key, p)
			except:
				poly_id = p
			dyfi_inside, dyfi_outside = dyfi_outside.subselect_by_polygon(poly_obj)
			dyfi_dict[poly_id] = dyfi_inside

		return dyfi_dict

	def get_aggregated_intensity(self, agg_info='cii', agg_method='mean',
								include_other_felt=True,
								include_heavy_appliance=False,
								max_deviation=2., max_nan_pct=100):
		"""
		Compute aggregated intensity, and optionally the residual
		(difference) between two methods

		:param agg_info:
			str, info to aggregate, either 'cii' or 'cdi'
			(default: 'cii')
		:param agg_method:
			str, how to aggregate individual enquiries,
			either 'mean' (= ROB practice), 'dyfi' (= DYFI/USGS practice),
			'mean-dyfi' or 'dyfi-mean'
			(default: 'mean')
		:param include_other_felt:
			bool, whether or not to take into acoount the replies to the
			question "Did others nearby feel the earthquake ?"
			(DYFI/USGS practice, but not taken into account in ROB forms)
			(default: True)
		:param include_heavy_appliance:
			bool, whether or not to take heavy_appliance into account
			as well (not standard, but occurs with ROB forms)
			(default: False)
		:param max_deviation:
			float, max. deviation allowed for individual intensities to be
			taken into account to compute the aggregated intensity.
			Applies to both aggregation methods (in a different way)
			(default: 2.)
		:param max_nan_pct:
			int, maximum percentage of nan (i.e. unanswered) values
			to accept for each index in aggregated calculation
			Only applies if :param:`agg_method` == 'dyfi'!
			(default: 100)

		:return:
			(intensity, residual) tuple of floats
		"""
		## Allow 'usgs' as alias for 'dyfi'
		agg_method = agg_method.replace('usgs', 'dyfi')

		if agg_info == 'cii':
			if 'dyfi' in agg_method:
				Iagg = self.calc_cii(aggregate=True,
								include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance,
								max_deviation=max_deviation,
								max_nan_pct=max_nan_pct)
			if 'mean' in agg_method:
				Imean = self.calc_mean_cii_or_cdi('cii',
								include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance,
								max_deviation=max_deviation)
		elif agg_info == 'cdi':
			if 'dyfi' in agg_method:
				Iagg = self.calc_cdi(include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance,
								max_deviation=max_deviation,
								max_nan_pct=max_nan_pct)
			if 'mean' in agg_method:
				Imean = self.calc_mean_cii_or_cdi('cdi',
								include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance,
								max_deviation=max_deviation)

		residual = 0
		if agg_info in ('cii', 'cdi'):
			if agg_method in ('mean', 'mean-dyfi'):
				I = Imean
			elif agg_method in ('dyfi', 'dyfi-mean'):
				I = Iagg
			if agg_method == 'mean-dyfi':
				residual = Imean - Iagg
			elif agg_method == 'dyfi-mean':
				residual = Iagg - Imean
		else:
			raise Exception("Don't know how to compute %s" % agg_info)

		return (I, residual)

	def aggregate(self, aggregate_by='id_com', min_replies=3,
					min_fiability=80, filter_floors=(0, 4),
					agg_info='cii', agg_method='mean',
					include_other_felt=True, include_heavy_appliance=False,
					max_deviation=2, max_nan_pct=100, **kwargs):
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
			- 'polygon': requires 'poly_data' and 'value_key' kwargs,
				and optionally 'include_unmatched_polygons'
			- 'distance': requires 'ref_pt' and 'distance_interval' kwargs
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
		:param agg_method:
		:param include_other_felt:
		:param include_heavy_appliance:
		:param max_deviation:
		:param max_nan_pct:
			see :meth:`get_aggregated_intensity`

		:**kwargs:
			additional keyword arguments required by some aggregation
			methods

		:return:
			instance of :class:`AggregatedMacroInfoCollection`
		"""
		from mapping.geotools.geodetic import spherical_point_at
		import mapping.layeredbasemap as lbm
		from .macro_info import AggregatedMacroInfo, AggregatedMacroInfoCollection

		if agg_method[:3] in ('min', 'max'):
			print("Warning: aggregation method %s not supported!" % agg_method)
			return

		ensemble = self

		ensemble = ensemble[ensemble.fiability >= min_fiability]
		if filter_floors:
			ensemble = ensemble.filter_floors(*filter_floors, keep_nan_values=True)

		if aggregate_by == 'commune':
			aggregate_by = 'id_com'
		elif aggregate_by == 'main commune':
			aggregate_by = 'id_main'

		if len(ensemble) == 0:
			return AggregatedMacroInfoCollection([], aggregate_by, 'internet')

		macro_geoms, geom_key = None, ''

		if aggregate_by in ('id_com', 'id_main', 'zip'):
			## Note: correct commune_ids should have been set when converting
			## ROBDYFIEnsemble to DYFIEnsemble
			agg_ensemble_dict = ensemble.split_by_property('commune_ids')
			try:
				unassigned = agg_ensemble_dict.pop(0)
			except KeyError:
				unassigned = None
			else:
				print("Note: %d enquiries are not assigned to a commune"
						% unassigned.num_replies)

		elif not aggregate_by:
			min_replies = 1
			agg_ensemble_dict = {}
			for subensemble in ensemble:
				if subensemble.is_geo_located()[0]:
					id_web = subensemble.ids[0]
					agg_ensemble_dict[id_web] = subensemble

		elif aggregate_by[:4] == 'grid':
			if '_' in aggregate_by:
				_, grid_spacing = aggregate_by.split('_')
				grid_spacing = float(grid_spacing)
			else:
				grid_spacing = 5
			agg_ensemble_dict = ensemble.split_by_grid(grid_spacing)

		elif aggregate_by == 'polygon':
			poly_data, value_key = kwargs['poly_data'], kwargs['value_key']
			include_unmatched_polygons = kwargs.get('include_unmatched_polygons',
													False)

			if isinstance(poly_data, basestring):
				gis_data = lbm.GisData(poly_data)
				_, _, poly_data = gis_data.get_data()

			agg_ensemble_dict = ensemble.split_by_polygon_data(poly_data, value_key)

			geom_key = value_key
			if include_unmatched_polygons:
				macro_geoms = poly_data
			else:
				polygon_list = []
				for geom_key_val in agg_ensemble_dict.keys():
					poly_idx = poly_data.values[value_key].index(geom_key_val)
					poly_obj = poly_data[poly_idx]
					polygon_list.append(poly_obj)
				macro_geoms = lbm.MultiPolygonData.from_polygons(polygon_list)

		elif aggregate_by == 'distance':
			ref_pt = kwargs['ref_pt']
			distance_interval = kwargs['distance_interval']
			create_polygons = kwargs.get('create_polygons', True)

			agg_ensemble_dict = ensemble.split_by_distance(ref_pt, distance_interval)

			## Create macro_geoms
			if create_polygons:
				geom_key = 'max_radius'
				polygon_list = []
				azimuths = np.linspace(0, 360, 361)
				if hasattr(ref_pt, 'lon'):
					lon, lat = ref_pt.lon, ref_pt.lat
				else:
					lon, lat = ref_pt[:2]
				for max_radius in agg_ensemble_dict.keys():
					lons, lats = spherical_point_at(lon, lat, max_radius, azimuths)
					min_radius = max_radius - distance_interval
					if min_radius:
						interior_lons, interior_lats = spherical_point_at(lon, lat,
																min_radius, azimuths)
						interior_lons, interior_lats = [interior_lons], [interior_lats]
					else:
						interior_lons, interior_lats = None, None

					value = {'min_radius': min_radius, 'max_radius': max_radius}
					poly_obj = lbm.PolygonData(lons, lats, interior_lons=interior_lons,
												interior_lats=interior_lats,
												value=value)
					polygon_list.append(poly_obj)

				macro_geoms = lbm.MultiPolygonData.from_polygons(polygon_list)

		## Compute aggregated intensity and convert to aggregated macro info
		macro_infos = []
		imt = agg_info.upper()
		for key in list(agg_ensemble_dict.keys()):
			num_replies = agg_ensemble_dict[key].num_replies
			if num_replies < min_replies:
				agg_ensemble_dict.pop(key)
				continue

			I, residual = agg_ensemble_dict[key].get_aggregated_intensity(
								agg_info=agg_info, agg_method=agg_method,
								include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance,
								max_deviation=max_deviation,
								max_nan_pct=max_nan_pct)
			if agg_info == "num_replies":
				I = 1

			unique_id_coms = np.unique(agg_ensemble_dict[key].commune_ids)
			id_com = unique_id_coms[0] if len(unique_id_coms) == 1 else None
			unique_id_earths = np.unique(agg_ensemble_dict[key].event_ids)
			id_earth = unique_id_earths[0] if len(unique_id_earths) == 1 else None
			web_ids = agg_ensemble_dict[key].ids

			if aggregate_by in ('distance', 'polygon'):
				geom_key_val = key
			else:
				geom_key_val = None

			if aggregate_by in ('id_com', 'id_main') or not aggregate_by:
				lon = agg_ensemble_dict[key].longitudes[0]
				lat = agg_ensemble_dict[key].latitudes[0]
			elif aggregate_by[:4] == 'grid':
				lon, lat = key
			elif aggregate_by == 'polygon':
				poly_idx = poly_data.values[geom_key].index(geom_key_val)
				poly_obj = poly_data[poly_idx]
				centroid = poly_obj.get_centroid()
				lon, lat = centroid.lon, centroid.lat
			else:
				lon, lat = 0, 0

			macro_info = AggregatedMacroInfo(id_earth, id_com, I, imt, aggregate_by,
									'internet', num_replies, lon=lon, lat=lat,
									residual=residual, db_ids=web_ids,
									geom_key_val=geom_key_val)
			macro_infos.append(macro_info)

		proc_info = dict(min_replies=min_replies, min_fiability=min_fiability,
						filter_floors=filter_floors, agg_method=agg_method,
						include_other_felt=include_other_felt,
						include_heavy_appliance=include_heavy_appliance,
						max_deviation=max_deviation)

		macro_info_coll = AggregatedMacroInfoCollection(macro_infos, aggregate_by,
										'internet', macro_geoms=macro_geoms,
										geom_key=geom_key, proc_info=proc_info)

		return macro_info_coll

	def aggregate_by_nothing(self, min_replies=3, min_fiability=80,
					filter_floors=(0, 4), agg_info='cii', agg_method='mean',
					include_other_felt=True, include_heavy_appliance=False,
					max_deviation=2., max_nan_pct=100, **kwargs):
		"""
		Turn DYFI collection in aggregated macro information, with
		each enquiry corresponding to an aggregate

		See :meth:`aggregate` for parameters

		:**kwargs:
			additional keyword arguments for :class:`ROBDYFIEnsemble`

		:return:
			instance of :class:`AggregatedMacroInfoCollection`
		"""
		aggregate_by = ''
		return self.aggregate(aggregate_by, min_replies=min_replies,
					min_fiability=min_fiability, filter_floors=filter_floors,
					include_other_felt=include_other_felt,
					include_heavy_appliance=include_heavy_appliance,
					max_deviation=max_deviation, max_nan_pct=max_nan_pct,
					**kwargs)

	def aggregate_by_commune(self, min_replies=3, min_fiability=80,
					filter_floors=(0, 4), agg_info='cii', agg_method='mean',
					include_other_felt=True, include_heavy_appliance=False,
					max_deviation=2., max_nan_pct=100):
		"""
		Aggregate DYFI collection by commune

		See :meth:`aggregate` for other parameters

		:**kwargs:
			additional keyword arguments for :class:`ROBDYFIEnsemble`

		:return:
			instance of :class:`AggregatedMacroInfoCollection`
		"""
		# Note: no **kwargs needed because ROBDYFIEnsemble has method with same name
		aggregate_by = 'id_com'
		return self.aggregate(aggregate_by, min_replies=min_replies,
					min_fiability=min_fiability, filter_floors=filter_floors,
					agg_info=agg_info, agg_method=agg_method,
					include_other_felt=include_other_felt,
					include_heavy_appliance=include_heavy_appliance,
					max_deviation=max_deviation, max_nan_pct=max_nan_pct)

	def aggregate_by_distance(self, ref_pt, distance_interval, create_polygons=True,
					min_replies=3, min_fiability=80, filter_floors=(0, 4),
					agg_info='cii', agg_method='mean',
					include_other_felt=True, include_heavy_appliance=False,
					max_deviation=2., max_nan_pct=100, **kwargs):
		"""
		Aggregate DYFI ensemble by distance

		:param ref_pt:
		:param distance_interval:
			see :meth:`split_by_distance`
		:param create_polygons:
			bool, whether or not to create polygon objects necessary
			for plotting on a map
			(default: True)
		:**kwargs:
			additional keyword arguments for :class:`ROBDYFIEnsemble`

		See :meth:`aggregate` for other parameters

		:return:
			instance of :class:`AggregatedMacroInfoCollection`
		"""
		aggregate_by = 'distance'
		return self.aggregate(aggregate_by, min_replies=min_replies,
					min_fiability=min_fiability, filter_floors=filter_floors,
					include_other_felt=include_other_felt,
					include_heavy_appliance=include_heavy_appliance,
					max_deviation=max_deviation, max_nan_pct=max_nan_pct,
					ref_pt=ref_pt,  distance_interval=distance_interval,
					create_polygons=create_polygons, **kwargs)

	def aggregate_by_polygon_data(self, poly_data, value_key,
					include_unmatched_polygons=False,
					min_replies=3, min_fiability=80, filter_floors=(0, 4),
					agg_info='cii', agg_method='mean',
					include_other_felt=True, include_heavy_appliance=False,
					max_deviation=2., max_nan_pct=100, **kwargs):
		"""
		Aggregate DYFI ensemble according to a set of polygons

		:param poly_data:
		:param value_key:
			see :meth:`split_by_polygon_data`
		:param include_unmatched_polygons:
			bool, whether or not unmatched polygons should be included
			in the result (their intensity will be set to nan!)
			(default: True)
		:**kwargs:
			additional keyword arguments for :class:`ROBDYFIEnsemble`

		See :meth:`aggregate` for other parameters

		:return:
			instance of :class:`AggregatedMacroInfoCollection`
		"""
		aggregate_by = 'polygon'
		return self.aggregate(aggregate_by, min_replies=min_replies,
					min_fiability=min_fiability, filter_floors=filter_floors,
					include_other_felt=include_other_felt,
					include_heavy_appliance=include_heavy_appliance,
					max_deviation=max_deviation, max_nan_pct=max_nan_pct,
					poly_data=poly_data, value_key=value_key,
					include_unmatched_polygons=include_unmatched_polygons, **kwargs)

	#def remove_outliers(self, min_pct=2.5, max_pct=97.5):
	def remove_outliers(self, max_deviation=2.):
		"""
		Remove outliers (with CII outside of mean +/- nr. of standard deviations)
		from ensemble

		:param max_deviation:
			float, maximum allowed deviation in terms of number of
			standard deviations
			(default: 2.)

		:return:
			instance of :class:`MDPCollection`
		"""
		# TODO: add recalc_cii option, but requires additional parameters...?
		if max_deviation:
			mean = np.nanmean(self.CII)
			std = np.nanstd(self.CII)
			deviation = np.abs(self.CII - mean)
			is_outlier = deviation > max_deviation * std
		else:
			is_outlier = np.zeros_like(self.CII, dtype=np.bool)

		return self.__getitem__(~is_outlier)

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

	def fix_felt_is_none(self, verbose=True):
		"""
		Fix enquiries where 'felt' has not been filled out, based on
		reply to 'asleep', 'motion' and 'stand' questions.

		:param verbose:
			bool, whether or not to print information about the operation
			(default: True)

		:return:
			None, 'felt' values of :prop:`recs` are modified in place
		"""
		felt_is_none = np.isnan(self.felt)
		## Slept through it --> not felt
		idxs = felt_is_none & (self.asleep == 1)
		num_fixed = np.sum(idxs)
		self.set_prop_values('felt', 0, idxs=idxs, regenerate_arrays=False)
		## Awoken --> felt
		idxs = felt_is_none & (self.asleep == 2)
		num_fixed += np.sum(idxs)
		self.set_prop_values('felt', 1, idxs=idxs, regenerate_arrays=False)
		## Awake and (difficult to stand or motion) --> felt
		with np.errstate(invalid='ignore'):
			idxs = (felt_is_none & (self.asleep == 0) & (self.motion > 0)
					& (self.stand > 1))
		num_fixed += np.sum(idxs)
		self.set_prop_values('felt', 1, idxs=idxs, regenerate_arrays=True)
		if verbose:
			print('Fixed %d/%d felt values' % (num_fixed, np.sum(felt_is_none)))

	def fix_not_felt(self, verbose=True):
		"""
		For 'not felt' enquiries, set motion, reaction and stand to 0
		to avoid bias in the aggregated computation

		:param verbose:
			bool, whether or not to print information about the operation
			(default: True)

		:return:
			None, 'motion', 'reaction' and 'stand' values of :prop:`recs`
			are modified in place
		"""
		not_felt = (self.felt == 0)
		self.set_prop_values('motion', 0, idxs=not_felt, regenerate_arrays=False)
		self.set_prop_values('reaction', 0, idxs=not_felt, regenerate_arrays=False)
		self.set_prop_values('stand', 0, idxs=not_felt, regenerate_arrays=True)
		if verbose:
			print('Set %d motion/reaction/stand values' % np.sum(not_felt))

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
			felt_index = np.ma.array(self.felt, mask=np.isnan(self.felt))

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
			with np.errstate(invalid='ignore'):
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

	def calc_cws(self, aggregate=True, filter_floors=(0, 4),
				include_other_felt=True, include_heavy_appliance=False,
				max_deviation=None, max_nan_pct=100,
				overwrite=False):
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
		:param max_deviation:
			float, maximum deviation of non-aggregated intensities (in terms of
			number of standard deviations) to use for calculating aggregated CWS
			Only applies if :param:`aggregate` is True!
			(default: None)
		:param max_nan_pct:
			int, maximum percentage of nan (i.e. unanswered) values
			to accept for each index to calculate aggregated CWS
			Only applies if :param:`aggregate` is True!
			(default: 100)
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
		with np.errstate(invalid='ignore'):
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
			if max_deviation:
				with np.errstate(invalid='ignore', divide='ignore'):
					cii = 3.40 * np.log(cws_individual) - 4.38
				## Replace -inf CII values (due to zero CWS values) with ones
				cii[np.isinf(cii)] = 1
				_mean = np.nanmean(cii)
				_std = np.nanstd(cii)
				deviation = np.abs(cii - _mean)
				idxs = deviation <= max_deviation * _std
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
			if max_nan_pct > 1:
				max_nan_pct /= 100.
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
				#or np.sum(shelf_indexes.mask)/float(len(shelf_indexes)) > max_nan_pct)):
				or np.sum(np.isnan(ensemble.shelf)/float(len(ensemble)) > max_nan_pct)):
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

	def calc_cdi(self, aggregate=True, filter_floors=(0, 4),
				include_other_felt=True, include_heavy_appliance=False,
				max_deviation=None, max_nan_pct=100,
				overwrite=False):
		"""
		Compute original Community Decimal Intensity sensu Dengler &
		Dewey (1998)

		:param aggregate:
		:param filter_floors:
		:param include_other_felt:
		:param include_heavy_appliance:
		:param max_deviation:
		:param max_nan_pct:
		:param overwrite:
			see :meth:`calc_cws`

		:return:
			float or float array, CDI
		"""
		cws = self.calc_cws(aggregate=aggregate, filter_floors=filter_floors,
							include_other_felt=include_other_felt,
							include_heavy_appliance=include_heavy_appliance,
							max_deviation=max_deviation, max_nan_pct=max_nan_pct,
							overwrite=overwrite)
		cdi = 3.3 + 0.13 * cws
		if overwrite and aggregate is False:
			self.set_prop_values('CDI', cdi)

		return cdi

	def calc_cii(self, aggregate=True, filter_floors=(0, 4),
				include_other_felt=True, include_heavy_appliance=False,
				max_deviation=None, max_nan_pct=100,
				overwrite=False):
		"""
		Compute Community Internet Intensity following Wald et al. (1999),
		later renamed into Community Decimal Intensity (CDI)

		:param aggregate:
		:param filter_floors:
		:param include_other_felt:
		:param include_heavy_appliance:
		:param max_deviation:
		:param max_nan_pct:
		:param overwrite:
			see :meth:`calc_cws`

		:return:
			float or float array, CII
		"""
		cws = self.calc_cws(aggregate=aggregate, filter_floors=filter_floors,
							include_other_felt=include_other_felt,
							include_heavy_appliance=include_heavy_appliance,
							max_deviation=max_deviation, max_nan_pct=max_nan_pct,
							overwrite=overwrite)
		with np.errstate(invalid='ignore', divide='ignore'):
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
			self.set_prop_values('MI', np.round(cii))

		return cii

	def calc_mean_cii_or_cdi(self, which='cii', filter_floors=(0, 4),
						include_other_felt=True, include_heavy_appliance=False,
						max_deviation=2.):
		"""
		Compute mean CII value from CII values of individual enquiries,
		ignoring outliers. This is an alternative to the aggregated
		CII computation in :meth:`calc_cii`

		:param which:
			str, either 'cii' or 'cdi'
		:param filter_floors:
		:param include_other_felt:
		:param include_heavy_appliance:
			see :meth:`calc_cii`
		:param max_deviation:
			float, maximum deviation allowed for individual enquiries to
			be used for calculating the mean
			(default: 2.)

		:return:
			float, mean CII or CDI
		"""
		func = getattr(self, 'calc_%s' % which.lower())
		cii_or_cdi = func(aggregate=False, filter_floors=filter_floors,
					include_other_felt=include_other_felt,
					include_heavy_appliance=include_heavy_appliance)
		if max_deviation:
			_mean = np.nanmean(cii_or_cdi)
			_std = np.std(cii_or_cdi)
			deviation = np.abs(cii_or_cdi - _mean)
			is_outlier = deviation > max_deviation * _std
			cii_or_cdi = cii_or_cdi[~is_outlier]
		if len(cii_or_cdi):
			return cii_or_cdi.mean()
		else:
			return 0.

	def evaluate_cws_calculation(self, aggregate=False, include_other_felt=True,
							include_heavy_appliance=False, filter_floors=(0, 4),
							max_deviation=None, max_nan_pct=100):
		"""
		Print values of properties used for CWS calculation, and the
		derived indexes.

		:param aggregate:
		:param include_other_felt:
		:param include_heavy_appliance:
		:param filter_floors:
		:param max_deviation:
		:param max_nan_pct:
			see :meth:`calc_cws`
		"""
		print("felt:")
		print("  Values: %s" % np.ma.array(self.felt, mask=np.isnan(self.felt)))
		felt_index = self.calc_felt_index(include_other_felt=False)
		if aggregate:
			if np.sum(felt_index.mask)/float(len(felt_index)) > max_nan_pct:
				felt_index = 0.
			else:
				felt_index = felt_index.mean()
		print("  Felt index (without other_felt) [x5]: %s" % (5 * felt_index))

		print("other_felt:")
		print("  Values: %s" % np.ma.array(self.other_felt, mask=np.isnan(self.other_felt)))
		felt_index = self.calc_felt_index(include_other_felt=True)
		if aggregate:
			if np.sum(felt_index.mask)/float(len(felt_index)) > max_nan_pct:
				felt_index = 0.
			else:
				felt_index = felt_index.mean()
		print("  Felt index (incl. other_felt) [x5]: %s" % (5 * felt_index))

		print("motion:")
		print("  Values: %s" % np.ma.array(self.motion, mask=np.isnan(self.motion)))
		motion_index = self.calc_motion_index()
		if aggregate:
			if np.sum(motion_index.mask)/float(len(motion_index)) > max_nan_pct:
				motion_index = 0.
			else:
				motion_index = motion_index.mean()
		else:
			motion_index = motion_index.filled(0)
		print("  Motion index [x1]: %s" % motion_index)

		print("reaction:")
		print("  Values: %s" % np.ma.array(self.reaction, mask=np.isnan(self.reaction)))
		reaction_index = self.calc_reaction_index()
		if aggregate:
			if np.sum(reaction_index.mask)/float(len(reaction_index)) > max_nan_pct:
				reaction_index = 0.
			else:
				reaction_index = reaction_index.mean()
		else:
			reaction_index = reaction_index.filled(0)
		print("  Reaction index [x1]: %s" % reaction_index)

		print("stand:")
		print("  Values: %s" % np.ma.array(self.stand, mask=np.isnan(self.stand)))
		stand_index = self.calc_stand_index()
		if aggregate:
			if np.sum(stand_index.mask)/float(len(stand_index)) > max_nan_pct:
				stand_index = 0.
			else:
				stand_index = stand_index.mean()
		else:
			stand_index = stand_index.filled(0)
		print("  Stand index [x2]: %s" % (2 * stand_index))

		print("shelf:")
		print("  Values: %s" % np.ma.array(self.shelf, mask=np.isnan(self.shelf)))
		shelf_index = self.calc_shelf_index()
		if aggregate:
			if np.sum(np.isnan(self.shelf))/float(len(self.shelf)) > max_nan_pct:
				shelf_index = 0.
			else:
				shelf_index = shelf_index.mean()
		else:
			shelf_index = shelf_index.filled(0)
		print("  Shelf index [x5]: %s" % (5 * shelf_index))

		print("picture:")
		print("  Values: %s" % np.ma.array(self.picture, mask=np.isnan(self.picture)))
		picture_index = self.calc_picture_index()
		if aggregate:
			if np.sum(picture_index.mask)/float(len(picture_index)) > max_nan_pct:
				picture_index = 0.
			else:
				picture_index = picture_index.mean()
		else:
			picture_index = picture_index.filled(0)
		print("  Picture index [x2]: %s" % (2 * picture_index))

		print("furniture:")
		print("  Values: %s" % np.ma.array(self.furniture, mask=np.isnan(self.furniture)))
		furniture_index = self.calc_furniture_index()
		if aggregate:
			if np.sum(furniture_index.mask)/float(len(furniture_index)) > max_nan_pct:
				furniture_index = 0.
			else:
				furniture_index = furniture_index.mean()
		else:
			furniture_index = furniture_index.filled(0)
		print("  Furniture index [x3]: %s" % (3 * furniture_index))
		furniture_index = self.calc_furniture_index(include_heavy_appliance=True)
		if aggregate:
			if np.sum(furniture_index.mask)/float(len(furniture_index)) > max_nan_pct:
				furniture_index = 0.
			else:
				furniture_index = furniture_index.mean()
		else:
			furniture_index = furniture_index.filled(0)
		print("  Furniture index (incl. heavy_appliance) [x3]: %s" %
			(3 * furniture_index))

		print("damage:")
		print("  Values:")
		for d_text in self.get_prop_values('d_text'):
			print("\t%s" % ' '.join(d_text))
		damage_index = self.calc_damage_index()
		if aggregate:
			if np.sum(damage_index == 0)/float(len(damage_index)) > max_nan_pct:
				damage_index = 0.
			else:
				damage_index = damage_index.mean()
		print("  Damage index [x5]: %s" % (5 * damage_index))

		print("CWS:")
		cws = self.CWS
		if aggregate:
			cws = np.mean(cws)
		print("  Database: %s" % cws)
		print("  Recomputed: %s" % self.calc_cws(aggregate=aggregate,
			filter_floors=filter_floors, include_other_felt=include_other_felt,
			include_heavy_appliance=include_heavy_appliance,
			max_deviation=max_deviation, max_nan_pct=max_nan_pct))
		if not aggregate:
			print("  Aggregated: %.2f" % self.calc_cws(filter_floors=filter_floors,
								include_other_felt=include_other_felt,
								include_heavy_appliance=include_heavy_appliance,
								max_deviation=max_deviation, max_nan_pct=max_nan_pct))

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
						color='usgs', label='', **kwargs):
		"""
		Plot histogram of individual questionnaire intensities
		"""
		from plotting.generic_mpl import plot_histogram

		if color in ('usgs', 'rob'):
			import mapping.layeredbasemap as lbm
			colors = lbm.cm.get_cmap('macroseismic', color)
		elif isinstance(color, (basestring, list, np.ndarray)):
			colors = color
		elif color is None:
			colors = None
		else:
			colors = [color]

		cii = self.calc_cii(aggregate=False, include_other_felt=include_other_felt,
							include_heavy_appliance=include_heavy_appliance)

		xmin = kwargs.pop('xmin', 0)

		if color == 'rob':
			Imax = 7
			xmax = kwargs.pop('xmax', Imax + 0.5)
		elif color == 'usgs':
			Imax = 9
			xmax = kwargs.pop('xmax', Imax + 0.5)
		else:
			xmax = kwargs.pop('xmax', 12.5)
			Imax = np.floor(xmax)
		bins = np.arange(1, Imax + 1)

		labels = kwargs.pop('label', [label])
		xticks = kwargs.pop('xticks', bins)

		xlabel = kwargs.pop('xlabel', 'Intensity (CII)')

		return plot_histogram([cii], bins, colors=colors, labels=labels,
							xmin=xmin, xmax=xmax, xlabel=xlabel,
							xticks=xticks, **kwargs)


MacroseismicEnquiryEnsemble = DYFIEnsemble
