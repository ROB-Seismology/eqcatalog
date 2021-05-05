# -*- coding: utf-8 -*-
"""
Create merged catalog based on ROB / KNMI / BENS and optionally SIHEX /BGS
catalogs, keeping solutions by all agencies for each event.

Created on Wed Apr 21 11:25:44 2021

@author: kris
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np
from .eqcatalog import EQCatalog


__all__ = ['MergedEvent', 'MergedCatalog']


class MergedEvent:
	"""
	Event having different solutions from different agencies

	:param ID:
		int or str, event ID
	:param solutions:
		dict, mapping agency names to instances of :class:`LocalEarthquake`
	:param master_agency:
		str, name of master agency
	:param match_criteria:
		dict, mapping tuples of agency names to strings describing matching criteria
		(default: {})
	"""
	def __init__(self, ID, solutions, master_agency, match_criteria={}):
		assert master_agency in solutions
		self.ID = ID
		self.solutions = solutions
		self.master_agency = master_agency
		self.match_criteria = match_criteria or {}

	def __len__(self):
		return len(self.solutions)

	def __iter__(self):
		return self.solutions.values().__iter__()

	## Note: set all defaults to False because otherwise get_event_index
	## method of MergedCatalog takes too long!
	def get_master_solution(self, override_ID=False, update_event_type=False,
								integrate_other_agency_mags=False):
		"""
		Fetch master solution

		:param override_ID:
			bool, whether or not to override earthquake ID (given by master agency)
			with the ID of the merged event
			(default: False)
		:param update_event_type:
			bool, whether or not to update event type of the master solution
			if the other solutions all agree on another event type
			(default: False)
		:param integrate_other_agency_mags:
			bool whether or not to integrate magnitude types from other agencies
			if they are not present in the master solution
			(default: False)

		:return:
			instance of :class:`LocalEarthquake`
		"""
		solution = self.solutions[self.master_agency]
		if override_ID or update_event_type or integrate_other_agency_mags:
			solution = solution.copy()
		if override_ID:
			solution.ID = self.ID
		if update_event_type:
			## If there are 2 or more other solutions, and they all have
			## the same event type differing from that of the master solution,
			## override that event type
			if len(self.solutions) > 2:
				event_types = set()
				for agency in self.get_secondary_agencies():
					sec_solution = self.solutions[agency]
					event_types.add(sec_solution.event_type)
				event_types = list(event_types)
				if len(event_types) == 1 and event_types[0] != solution.event_type:
					print('%s: fixing event type %s -> %s'
							% (solution.ID, solution.event_type, event_types[0]))
					solution.event_type = event_types[0]
		if integrate_other_agency_mags:
			for agency in self.get_secondary_agencies():
				sec_solution = self.solutions[agency]
				for Mtype, mag in sec_solution.mag.items():
					if not Mtype in solution.mag and not np.isnan(mag):
						solution.mag[Mtype] = mag
		return solution

	def get_average_solution(self):
		"""
		Construct average solution for event

		:return:
			instance of :class:`eqcatalog.LocalEarthquake`
		"""
		from . import time as timelib
		from .eqrecord import LocalEarthquake

		master_solution = self.get_master_solution()
		ID = self.ID
		name = master_solution.name
		## np.mean() not supported for np.datetime64 objects
		master_dt = master_solution.datetime
		time_deltas = [eq.datetime - master_dt for eq in self.solutions.values()]
		print(time_deltas, np.mean(time_deltas))
		dt = master_dt + np.mean(time_deltas)
		date = timelib.to_py_date(dt)
		time = timelib.to_py_time(dt)
		lon = np.nanmean([eq.lon for eq in self.solutions.values()])
		lat = np.nanmean([eq.lat for eq in self.solutions.values()])
		depth = np.nanmean([eq.depth for eq in self.solutions.values()])
		mag = {}
		Mtypes = set([])
		for eq in self.solutions.values():
			Mtypes = Mtypes.union(eq.mag.keys())
		for Mtype in Mtypes:
			mag[Mtype] = np.nanmean([eq.mag.get(Mtype, np.nan)
							for eq in self.solutions.values()])

		return LocalEarthquake(ID, date, time, lon, lat, depth, mag, name=name)

	def get_agencies(self):
		"""
		List all agencies with a solution for this event

		:return:
			list of strings
		"""
		return sorted(self.solutions.keys())

	def get_secondary_agencies(self):
		"""
		List secondary (i.e. non-master) agencies for this event

		:return:
			list of strings
		"""
		secondary_agencies = self.get_agencies()
		secondary_agencies.remove(self.master_agency)
		return secondary_agencies

	def has_agency(self, agency):
		"""
		Determine whether event contains solution by given agency

		:return:
			bool
		"""
		return agency in self.solutions

	def has_agencies(self, agencies):
		"""
		Determine whether event contains solutions by all given agencies

		:return:
			bool
		"""
		has_agencies = []
		for agency in agencies:
			has_agencies.append(self.has_agency(agency))

		return np.all(has_agencies)

	def calc_time_deltas(self, wrt='master'):
		"""
		Compute time differences between different solutions

		:param wrt:
			str, with respect to: either 'master' or 'all'
			(default: 'master')

		:return:
			dict, mapping agency names (or 2-tuples of agency names) to floats
			(time differences in seconds)
		"""
		time_deltas = {}

		if wrt == 'master':
			master_solution = self.get_master_solution()
			for agency in self.get_secondary_agencies():
				solution = self.solutions[agency]
				td = np.abs(master_solution.datetime - solution.datetime)
				time_deltas[agency] = td
		elif wrt == 'all':
			from itertools import combinations
			agencies = self.get_agencies()
			for (agency1, agency2) in combinations(agencies, 2):
				solution1 = self.solutions[agency1]
				solution2 = self.solutions[agency2]
				td = solution1.datetime - solution2.datetime
				time_deltas[tuple(sorted([agency1, agency2]))] = td

		return time_deltas

	def calc_distances(self, wrt='master'):
		"""
		Compute distances between different solutions

		:param wrt:
			str, with respect to: either 'master' or 'all'
			(default: 'master')

		:return:
			dict, mapping agency names (or 2-tuples of agency names) to floats
			(distances in km)
		"""
		from mapping.geotools.geodetic import spherical_distance

		distances = {}

		if wrt == 'master':
			master_solution = self.get_master_solution()
			for agency in self.get_secondary_agencies():
				solution = self.solutions[agency]
				d = spherical_distance(master_solution.lon, master_solution.lat,
											solution.lon, solution.lat) / 1000.
				distances[agency] = d
		elif wrt == 'all':
			from itertools import combinations
			agencies = self.get_agencies()
			for (agency1, agency2) in combinations(agencies, 2):
				solution1 = self.solutions[agency1]
				solution2 = self.solutions[agency2]
				d = spherical_distance(solution1.lon, solution1.lat,
											solution2.lon, solution2.lat) / 1000.
				distances[tuple(sorted([agency1, agency2]))] = d

		return distances

	def calc_mag_deltas(self, Mtype='MW', Mrelation={}, wrt='master'):
		"""
		Compute magnitude differences between different solutions

		:param Mtype:
			str, magnitude type
			(default: 'MW')
		:param Mrelation:
			dict, mapping agency names to dicts, mapping Mtypes to instances
			of :class:`eqcatalog.msc.MSCE`
			or dict mapping Mtypes to instances of :class:`eqcatalog.msc.MSCE`
			for converting to given Mtype
			(default: {})
		:param wrt:
			see :meth:`calc_time_deltas`

		:return:
			dict, mapping agency names (or 2-tuples of agency names) to floats
			(magnitude differences)
		"""
		mag_deltas = {}

		if wrt == 'master':
			master_solution = self.get_master_solution()
			Mrel = Mrelation.get(self.master_agency, Mrelation)
			master_mag = master_solution.get_or_convert_mag(Mtype=Mtype, Mrelation=Mrel)
			for agency in self.get_secondary_agencies():
				solution = self.solutions[agency]
				Mrel = Mrelation.get(agency, Mrelation)
				mag = solution.get_or_convert_mag(Mtype=Mtype, Mrelation=Mrel)
				mag_deltas[agency] = master_mag - mag
		elif wrt == 'all':
			from itertools import combinations
			agencies = self.get_agencies()
			for (agency1, agency2) in combinations(agencies, 2):
				solution1 = self.solutions[agency1]
				solution2 = self.solutions[agency2]
				Mrel1 = Mrelation.get(agency1, Mrelation)
				mag1 = solution1.get_or_convert_mag(Mtype=Mtype, Mrelation=Mrel1)
				Mrel2 = Mrelation.get(agency2, Mrelation)
				mag2 = solution2.get_or_convert_mag(Mtype=Mtype, Mrelation=Mrel2)
				mag_deltas[tuple(sorted([agency1, agency2]))] = mag1 - mag2

		return mag_deltas

	def calc_match_scores(self, Mtype='MW', Mrelation={}, wrt='master'):
		"""
		Compute matching scores between different solutions, based on time
		difference, distance and magnitude difference

		:param Mtype:
		:param Mrelation:
		:param wrt:
			see :meth:`calc_mag_deltas`

		:return:
			dict, mapping agency names (or 2-tuples of agency names) to floats
			(scores)
		"""
		from . import time as timelib

		time_deltas = self.calc_time_deltas(wrt=wrt)
		distances = self.calc_distances(wrt=wrt)
		mag_deltas = self.calc_mag_deltas(Mtype=Mtype, Mrelation=Mrelation, wrt=wrt)

		scores = {}
		for agency in time_deltas.keys():
			time_delta = np.abs(time_deltas[agency])
			d = distances[agency]
			mag_delta = np.abs(mag_deltas[agency])
			score = timelib.fractional_time_delta(time_delta, 's')
			if not np.isnan(d):
				score += d / 10.
			else:
				score += 10
			if not np.isnan(mag_delta):
				score += mag_delta / 0.3
			else:
				score += 10
			scores[agency] = score

		return scores

	def calc_max_deltas(self, Mtype='MW', Mrelation={}, wrt='master'):
		"""
		Compute maximum time, distance and magnitude differences between
		different solutions

		:param Mtype:
		:param Mrelation:
		:param wrt:
			see :meth:`calc_mag_deltas`

		:return:
			(max_time_delta, max_dist, max_mag_delta) tuple of floats
		"""
		time_deltas = self.calc_time_deltas(wrt=wrt)
		distances = self.calc_distances(wrt=wrt)
		mag_deltas = self.calc_mag_deltas(Mtype=Mtype, Mrelation=Mrelation, wrt=wrt)

		max_time_delta = np.nanmax(np.abs(list(time_deltas.values())))
		max_dist = np.nanmax(list(distances.values()))
		max_mag_delta = np.nanmax(np.abs(list(mag_deltas.values())))

		return (max_time_delta, max_dist, max_mag_delta)

	def print_info(self, Mtype='MW', Mrelation={}, out_file=None):
		"""
		Print information about merged event and different solutions

		:param Mtype:
			str, magnitude type
			(default: 'MW')
		:param Mrelation:
			dict, mapping agency names to dicts, mapping Mtypes to instances
			of :class:`eqcatalog.msc.MSCE`
			or dict mapping Mtypes to instances of :class:`eqcatalog.msc.MSCE`
			for converting to given Mtype
			(default: {})
		:param out_file:
			str, full path to output file
			(default: None, will print on screen)
		"""
		from . import time as timelib

		info = '<Merged event #%s | n=%d' % (self.ID, len(self))
		if len(self) > 1:
			max_time_delta, max_dist, max_mag_delta = self.calc_max_deltas(Mtype, Mrelation)
			max_time_delta = timelib.fractional_time_delta(max_time_delta, 's')
			info += ' | dt=%.2f s | d=%.2f km | dM=%.2f'
			info %= (max_time_delta, max_dist, max_mag_delta)
		info += ' >'
		print(info, file=out_file)
		master_solution = self.get_master_solution()
		print(master_solution,file=out_file)
		for agency in self.get_secondary_agencies():
			solution = self.solutions[agency]
			print(solution, file=out_file)


class MergedCatalog:
	"""
	Catalog composed of merged events

	:param merged_events:
		list with instances of :class:`MergedEvent`
	:param name:
		str, name of catalog
		(default: '')
	:param default_completeness:
		dict, mapping agency names to instances of :class:`eqcatalog.Completeness`
		(default: {})
	:param default_Mrelations:
		dict, mapping agency names to dicts, mapping magnitude types to dicts,
		mapping magnitude types to instances of :class:`eqcatalog.msc.MSCE`
		(default: {})
	:param agency_areas:
		dict, mapping agency names to polygon or closed linestring objects
		(ogr geometry object or oqhazlib.geo.polygon.Polygon object)
		representing their authoritative areas
		(default: {})
	"""
	def __init__(self, merged_events, name='',
					default_completeness={},
					default_Mrelations={},
					agency_areas={}):
		self.merged_events = merged_events
		self.name = name
		self.default_completeness = default_completeness or {}
		self.default_Mrelations = default_Mrelations or {}
		self.agency_areas = agency_areas or {}

	def __len__(self):
		return len(self.merged_events)

	def __iter__(self):
		return self.merged_events.__iter__()

	def __getitem__(self, item):
		"""
		Indexing --> instance of :class:`MergedEvent`
		Slicing / Index array --> instance of :class:`MergedCatalog`
		"""
		if isinstance(item, (int, np.integer)):
			return self.merged_events.__getitem__(item)
		elif isinstance(item, slice):
			return self.__class__(self.merged_events.__getitem__(item),
										name=self.name + " %s" % item,
										default_completeness=self.default_completeness,
										default_Mrelations=self.default_Mrelations,
										agency_areas=self.agency_areas)
		elif isinstance(item, (list, np.ndarray)):
			## item can contain indexes or bool
			ev_list = []
			if len(item):
				idxs = np.arange(len(self))
				idxs = idxs[np.asarray(item)]
				for idx in idxs:
					ev_list.append(self.merged_events[idx])
			return self.__class__(ev_list, name=self.name + " %s" % item,
										default_completeness=self.default_completeness,
										default_Mrelations=self.default_Mrelations,
										agency_areas=self.agency_areas)

	def get_event_index(self, event_id, agency=None):
		"""
		Determine index in catalog of a particular event

		:param event_id:
			int or str, ID of solution or merged event
		:param agency:
			str, agency name, 'master' or None:
			- None --> use ID of merged event
			- 'master' --> use ID of master solution
			- agency name --> use ID of solution of given agency
			(default: None)

		:return:
			instance of :class:`eqcatalog.LocalEarthquake`
			or instance of :class:`MergedEvent`
		"""
		for i, event in enumerate(self):
			if agency == 'master':
				solution = event.get_master_solution(override_ID=False,
															update_event_type=False,
															integrate_other_agency_mags=False)
			elif agency:
				solution = event.solutions.get(agency)
			else:
				solution = event
			if solution:
				if solution.ID == event_id:
					return i
			else:
				continue

	def get_master_event(self, master_id):
		"""
		Fetch event based on ID of master solution

		:param master_id:
			int or str, ID of master solution

		:return:
			instance of :class:`MergedEvent`
		"""
		return self.merged_events[self.get_event_index(master_id, 'master')]

	def get_event(self, event_id, agency=None):
		"""
		Fetch event based on given ID

		:param event_id:
		:param agency:
			see :meth:`get_event_index`

		:return:
			instance of :class:`MergedEvent`
		"""
		try:
			ev = self.merged_events[self.get_event_index(event_id, agency=agency)]
		except:
			ev = None
		return ev

	def get_master_catalog(self, override_IDs=False, update_event_types=False,
								integrate_other_agency_mags=False):
		"""
		Extract catalog with master solutions for each merged event

		:param override_IDs:
			bool, whether or not to override earthquake IDs (given by master agency)
			with the IDs of the merged events
			(default: False)
		:param update_event_types:
			bool, whether or not to update event types of the master solutions
			if the other solutions all agree on another event type
			(default: False)
		:param integrate_other_agency_mags:
			bool whether or not to integrate magnitude types from other agencies
			if they are not present in the master solutions
			(default: False)

		:return:
			instance of :class:`EQCatalog`
		"""
		eq_list = [ev.get_master_solution(override_ID=override_IDs,
						update_event_type=update_event_types,
						integrate_other_agency_mags=integrate_other_agency_mags)
						for ev in self.merged_events]
		return EQCatalog(eq_list, name=self.name + ' (master)')

	def get_agency_catalog(self, agency):
		"""
		Extract catalog containing only solutions from given agency

		:param agency:
			str, name of agency

		:return:
			instance of :class:`EQCatalog`
		"""
		eq_list = []
		for ev in self.merged_events:
			if ev.has_agency(agency):
				eq_list.append(ev.solutions[agency])
		return EQCatalog(eq_list, name=self.name + ' (%s)' % agency)

	def get_agency_counts(self, unique=True):
		"""
		Count number of earthquakes for each agency or combination of agencies

		:param unique:
			bool, whether or not to count unique occurrences of agency or
			combination of agencies
			(default: True)

		:return:
			dict, mapping agencies or agency combinations to integers
		"""
		from itertools import combinations

		agency_counts = {}
		for ev in self:
			ev_agencies = ev.get_agencies()
			for n in range(len(ev_agencies)):
				for comb_agencies in combinations(ev_agencies, n+1):
					if not(unique and len(ev_agencies) != len(comb_agencies)):
						comb_agency = '+'.join(sorted(comb_agencies))
						if comb_agency in agency_counts:
							agency_counts[comb_agency] += 1
						else:
							agency_counts[comb_agency] = 1
		return agency_counts

	def get_agencies(self):
		"""
		Report all agencies contributing to merged catalog

		:return:
			list of strings, agency names
		"""
		agencies = set([])
		for ev in self.merged_events:
			for agency in ev.get_agencies():
				agencies.add(agency)
		return sorted(agencies)

	def subselect_by_agency(self, agency, master=True, unique=False):
		"""
		Subselect merged catalog by single agency

		:param agency:
			str, name of agency
		:param master:
			bool, whether or not to select only events where given agency is master
			(default: True)
		:param unique:
			bool, whether or not to select only events where given agency is the
			only contributing agency (implies :param:`master` to be True)
			(default: False)

		:return:
			instance of :class:`MergedCatalog`
		"""
		merged_events = []
		for ev in self.merged_events:
			if master and agency != ev.master_agency:
				continue
			ev_agencies = ev.get_agencies()
			if agency in ev_agencies:
				if unique and len(ev_agencies) > 1:
					continue
				merged_events.append(ev)

		return self.__class__(merged_events,
									default_completeness=self.default_completeness,
									default_Mrelations=self.default_Mrelations,
									agency_areas=self.agency_areas)

	def subselect_by_agencies(self, agencies, unique=True):
		"""
		Subselect merged catalog by one or more agencies

		:param agencies:
			list of strings, agency names
		:param unique:
			bool, whether or not to select only events where the given combination
			of agencies are the only contributing agencies

		:return:
			instance of :class:`MergedCatalog`
		"""
		n = len(agencies)
		merged_events = []
		for ev in self.merged_events:
			if ev.has_agencies(agencies):
				if unique and len(ev) > n:
					continue
				else:
					merged_events.append(ev)

		return self.__class__(merged_events,
									default_completeness=self.default_completeness,
									default_Mrelations=self.default_Mrelations,
									agency_areas=self.agency_areas)

	def subselect(self,
		region=None,
		start_date=None, end_date=None,
		Mmin=None, Mmax=None,
		min_depth=None, max_depth=None,
		attr_val=(),
		Mtype="MW", Mrelation={},
		include_right_edges=True,
		catalog_name=""):
		"""
		Subselect merged catalog based on geographic area, time interval,
		magnitude interval, depth interval or other attributes

		:param region:
		:param start_date:
		:param end_date:
		:param Mmin:
		:param Mmax:
		:param min_depth:
		:param max_depth:
		:param attr_val:
		:param Mtype:
		:param Mrelation:
		:param include_right_edges:
		:param catalog_name:
			see :meth:`EQCatalog.subselect`

		:return:
			instance of :class:`MergedCatalog`
		"""
		master_catalog = self.get_master_catalog()
		sel_catalog = master_catalog.subselect(region=region,
												start_date=start_date, end_date=end_date,
												Mmin=Mmin, Mmax=Mmax,
												min_depth=min_depth, max_depth=max_depth,
												attr_val=attr_val,
												Mtype=Mtype, Mrelation=Mrelation,
												include_right_edges=include_right_edges)

		merged_events = []
		for ev in self.merged_events:
			master_solution = ev.get_master_solution()
			if master_solution in sel_catalog:
				merged_events.append(ev)

		return self.__class__(merged_events, name=catalog_name,
									default_completeness=self.default_completeness,
									default_Mrelations=self.default_Mrelations,
									agency_areas=self.agency_areas)

	# TODO: needs testing
	def subselect_completeness(self, completeness=None, Mrelation={}):
		"""
		Subselect merged catalog according to completeness criteria (possibly of
		different agencies)

		:param completeness:
			instance of :class:`eqcatalog.Completeness`, uniform completeness
			for all agencies, overriding :prop:`default_completeness`
		:param Mrelation:
			dict, mapping magnitude types to instances of :class:`eqcatalog.msc.MSCE`
			to convert to Mtype of :param:`completeness`,
			thus overriding :prop:`default_Mrelations`

		:return:
			instance of :class:`MergedCatalog`
		"""
		merged_events = []
		for ev in self.merged_events:
			agency = ev.master_agency
			master_solution = ev.get_master_solution()
			comp = completeness or self.default_completeness.get(agency)
			Mtype = comp.Mtype
			Mrel =  Mrelation or self.default_Mrelations.get(agency, self.default_Mrelations)
			Mrel = Mrel.get(Mtype, {})
			M = master_solution.get_or_convert_mag(Mtype, Mrel)
			if M >= comp.get_completeness_magnitude(master_solution.date):
				merged_events.append(ev)

		return self.__class__(merged_events,
									default_completeness=self.default_completeness,
									default_Mrelations=self.default_Mrelations,
									agency_areas=self.agency_areas)

	def subselect_polygon(self, poly_obj):
		"""
		Subselect merged catalog by polygon

		:param poly_obj:
			polygon or closed linestring object (ogr geometry object
			or oqhazlib.geo.polygon.Polygon object)

		:return:
			instance of :class:`MergedCatalog`
		"""
		master_catalog = self.get_master_catalog()
		master_catalog_inside = master_catalog.subselect_polygon(poly_obj)
		merged_events = []
		for ev in self.merged_events:
			master_solution = ev.get_master_solution()
			if master_solution in master_catalog_inside:
				merged_events.append(ev)

		return self.__class__(merged_events,
									default_completeness=self.default_completeness,
									default_Mrelations=self.default_Mrelations,
									agency_areas=self.agency_areas)

	def subselect_by_agency_area(self, agency, buffer_distance=0):
		"""
		Subselect merged catalog by authoritative area of given agency

		:param agency:
			str, name of agency
		:param buffer_distance:
			float, buffer distance (in km) to apply to polygon
			(default: 0)

		:return:
			instance of :class:`MergedCatalog`
		"""
		from mapping.geotools.buffer import create_buffer_polygon

		area_geom = self.agency_areas.get(agency)
		if area_geom:
			buffer_pg = create_buffer_polygon(area_geom, buffer_distance)
			return self.subselect_polygon(buffer_pg)
		else:
			raise Exception('Authoritative area for agency %s not defined!' % agency)

	def export_csv(self, csv_filespec=None, separator=';'):
		"""
		Export meta-catalog to CSV file

		:param csv_filespec:
			str, full path to CSV file
			(default: None, will print on screen)
		:param separator:
			char, separator character between fields in CSV file
			Note: cannot be ',' because match_criteria column may contain commas
			(default: ';')
		"""
		import sys
		import json

		assert separator != ','

		agencies = self.get_agencies()
		col_names = ['ID']
		col_names += ['ID_%s' % agency for agency in agencies]
		col_names += ['master_agency', 'match_criteria']

		rows = []
		for ev in self.merged_events:
			row = [''] * len(col_names)
			row[0] = ev.ID
			for agency in ev.get_agencies():
				idx = agencies.index(agency) + 1
				solution = ev.solutions[agency]
				row[idx] = str(solution.ID)
			row[-2] = ev.master_agency
			row[-1] = json.dumps(ev.match_criteria)
			rows.append(row)

		if csv_filespec is None:
			of = sys.stdout
		else:
			of = open(csv_filespec, 'w')

		of.write(separator.join(col_names))
		of.write('\n')
		for row in rows:
			of.write(separator.join(row))
			of.write('\n')

		if csv_filespec:
			of.close()

	@classmethod
	def from_csv(cls, csv_filespec, catalogs, separator=';', agency_areas={}):
		"""
		Reconstruct merged catalog from CSV file

		:param csv_filespec:
			str, full path to CSV file containing meta-information
		:param catalogs:
			dict, mapping agency names to instances of :class:`EQCatalog`,
			these are the catalogs that have been merged and must contain
			all events referenced in the CSV file
		:param separator:
			char, separator character between fields in CSV file
			Note: cannot be ',' because match_criteria column may contain commas
			(default: ';')
		:param agency_areas:
			dict, mapping agency names to polygon or closed linestring objects
			(ogr geometry object or oqhazlib.geo.polygon.Polygon object)
			representing their authoritative areas
			(default: {})

		:return:
			instance of :class:`MergedCatalog
		"""
		import csv
		import json

		default_completeness = {agency: cat.default_completeness
									for (agency, cat) in catalogs.items()}
		default_Mrelations = {agency: cat.default_Mrelations
									for (agency, cat) in catalogs.items()}

		with open(csv_filespec) as csvf:
			csv_reader = csv.DictReader(csvf, delimiter=separator)
			col_names = csv_reader.fieldnames
			agency_colnames =  col_names[:-2]
			agencies = [col_name.split('ID_')[1] for col_name in agency_colnames]

			merged_events = []
			for row in csv_reader:
				ID = row['ID']
				master_agency = row['master_agency']
				match_criteria = json.loads(row['match_criteria'])
				solutions = {}
				for agency in agencies:
					agency_ID = row['ID_%s' % agency]
					if agency_ID:
						eq = catalogs[agency].get_event_by_id(agency_ID)
						if eq:
							solutions[agency] = eq
						else:
							msg = 'Warning: ID #%s not found in %s catalog!'
							msg %= (agency_ID, agency)
							print(msg)
				ev = MergedEvent(ID, solutions, master_agency, match_criteria=match_criteria)
				merged_events.append(ev)

		name = os.path.splitext(os.path.split(csv_filespec)[1])[0]
		return cls(merged_events, name=name, default_completeness=default_completeness,
					default_Mrelations=default_Mrelations,
					agency_areas=agency_areas)

	def print_list(self, Mtype='MW', Mrelation={}, out_file=None):
		"""
		Print list of merged events

		:param Mtype:
		:param Mrelation:
		:param out_file:
			see :meth:`MergedEvent.print_info`
		"""
		for ev in self.get_sorted().merged_events:
			agency = ev.master_agency
			Mrel =  Mrelation or self.default_Mrelations.get(agency, self.default_Mrelations)
			Mrel = Mrel.get(Mtype, {})
			ev.print_info(Mtype=Mtype, Mrelation=Mrel, out_file=out_file)
			print('', file=out_file)

	def get_sorted(self, key="datetime", order="asc"):
		"""
		Get copy of `merged catalog sorted by earthquake attribute.

		:param key:
			str, property of :class:`LocalEarthquake` to use as sort key
			(default: "datetime")
		:param order:
			str, sorting order: "asc" or "desc"
			(default: "asc")

		:return:
			instance of :class:`MergedCatalog`
		"""
		master_catalog = self.get_master_catalog()
		idxs = master_catalog.argsort(key=key, order=order)
		return self.__getitem__(idxs)

	def compare_agency_magnitudes(self, agency1, agency2, Mtype):
		"""
		Compare magnitudes reported by 2 different agencies

		:param agency1:
			str, name of 1st agency
		:param agency2:
			str, name of 2nd agency
		:param Mtype:
			str, magnitude type to compare

		:return:
			(mags1, mags2) tuple of 1D arrays, corresponding magnitudes
			given by both agencies
		"""
		sel_catalog = self.subselect_by_agencies([agency1, agency2], unique=False)
		agency1_cat = sel_catalog.get_agency_catalog(agency1)
		agency2_cat = sel_catalog.get_agency_catalog(agency2)
		mags1 = np.array([eq.mag.get(Mtype, np.nan) for eq in agency1_cat])
		mags2 = np.array([eq.mag.get(Mtype, np.nan) for eq in agency2_cat])
		nan_idxs = (np.isnan(mags1) | np.isnan(mags2))
		mags1 = mags1[~nan_idxs]
		mags2 = mags2[~nan_idxs]

		return (mags1, mags2)

	def calc_agency_time_deltas(self):
		"""
		Compute time differences between solutions by different agencies for
		common events

		:return:
			dict, mapping 2-tuples of agencies to lists of floats
		"""
		time_deltas = {}
		for ev in self.merged_events:
			ev_time_deltas = ev.calc_time_deltas(wrt='all')
			for agency_combo, td in ev_time_deltas.items():
				if not np.isnan(td):
					if not agency_combo in time_deltas:
						time_deltas[agency_combo] = [td]
					else:
						time_deltas[agency_combo].append(td)

		return time_deltas

	def calc_agency_mag_deltas(self, Mtype='MW', Mrelation={}):
		"""
		Compute magnitude differences between solutions by different agencies
		for common events

		:param Mtype:
		:param Mrelation:
			see :meth:`MergedEvent.calc_mag_deltas`

		:return:
			dict, mapping 2-tuples of agencies to lists of floats
		"""
		mag_deltas = {}
		for ev in self.merged_events:
			ev_mag_deltas = ev.calc_mag_deltas(Mtype=Mtype, Mrelation=Mrelation, wrt='all')
			for agency_combo, mag_delta in ev_mag_deltas.items():
				if not np.isnan(mag_delta):
					if not agency_combo in mag_deltas:
						mag_deltas[agency_combo] = [mag_delta]
					else:
						mag_deltas[agency_combo].append(mag_delta)

		return mag_deltas

	def calc_agency_distances(self):
		"""
		Compute distances between solutions by different agencies for common
		events

		:return:
			dict, mapping 2-tuples of agencies to lists of floats
		"""
		distances = {}
		for ev in self.merged_events:
			ev_distances = ev.calc_distances(wrt='all')
			for agency_combo, d in ev_distances.items():
				if not np.isnan(d):
					if not agency_combo in distances:
						distances[agency_combo] = [d]
					else:
						distances[agency_combo].append(d)

		return distances

	def plot_mag_delta_histogram(self, Mtype='MW', Mrelation={}, bins=None):
		"""
		Plot histogram of magnitude differences between solutions by different
		agencies for common events

		:param Mtype:
		:param Mrelation:
			see :meth:`calc_mag_deltas`

		:return:
			matplotlib Axes instance
		"""
		from plotting.generic_mpl import plot_histogram

		mag_deltas = self.calc_agency_mag_deltas(Mtype=Mtype, Mrelation=Mrelation)
		datasets = list(mag_deltas.values())
		labels = list(mag_deltas.keys())
		if bins is None:
			bins = [-1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

		return plot_histogram(datasets, bins, labels=labels, xlabel='Mag. difference')

	def plot_distance_histogram(self, bins=None):
		"""
		Plot histogram of distances between solutions by different agencies
		for common events

		:param bins:
			list or array, distance bins
			(default: None)

		:return:
			matplotlib Axes instance
		"""
		from plotting.generic_mpl import plot_histogram

		distances = self.calc_agency_distances()
		datasets = list(distances.values())
		labels = list(distances.keys())
		if bins is None:
			bins = [0, 1, 2.5, 5, 10, 15, 20, 25, 30,40, 50, 60, 70]

		return plot_histogram(datasets, bins, labels=labels, xlabel='Distance (km)')

	def get_folium_map(self, agency_colors={}, Mtype='MW', Mrelation={},
							mag_size_func=lambda M: 2 * np.sqrt(2.25**(M+1) / np.pi),
							edge_width=2, add_popup=True, region=None,
							lat_lon_popup=False, source_model_name =''):
		"""
		Construct folium map of merged catalog

		:param agency_colors:
			dict, mapping agency names to color specs understood by folium
			(default: {})
		:param Mtype:
		:param Mrelation:
		:param mag_size_func:
		:param edge_width:
		:param add_popup:
		:param region:
		:param lat_lon_popup:
		:param source_model_name:
			see :meth:`EQCatalog.get_folium_map`

		:return:
			instance of :class:`folium.Map`
		"""
		layers = []

		if not agency_colors:
			agency_colors = {'ROB': 'blue',
								'KNMI': 'purple',
								'BENS': 'brown',
								'BGS': 'darkgreen',
								'SIHEX': 'peru'}

		## Agency combinations
		agency_counts = self.get_agency_counts(unique=True)
		agency_combos = list(agency_counts.keys())
		agency_combos = [combo.split('+') for combo in agency_combos]
		for combo in sorted(agency_combos, key=lambda x: len(x)):
			combo_catalog = self.subselect_by_agencies(combo, unique=True)
			for agency in combo:
				agency_catalog = combo_catalog.get_agency_catalog(agency)
				agency_catalog.name = '+'.join(sorted(combo))
				if len(combo) > 1:
					agency_catalog.name += ' (%s)' % agency
				color = agency_colors.get(agency, 'r')
				layer = agency_catalog.to_folium_layer(Mtype=Mtype, Mrelation=Mrelation,
																mag_size_func=mag_size_func,
																edge_width=edge_width,
																edge_color=color, add_popup=add_popup)
				layer.show = False
				layers.append(layer)

		## Master catalog
		master_catalog = self.get_master_catalog(override_IDs=True,
														update_event_types=True,
														integrate_other_agency_mags=True)
		color = agency_colors.get('master', 'navy')
		map = master_catalog.get_folium_map(Mtype=Mtype, Mrelation=Mrelation,
														mag_size_func=mag_size_func,
														edge_width=edge_width, edge_color=color,
														add_popup=add_popup, region=region,
														additional_layers=layers,
														source_model_name=source_model_name,
														lat_lon_popup=lat_lon_popup)

		return map

	def find_potential_matches(self, Mtype='MW', Mrelation={}, max_score=1000):
		"""
		Find potential matches for single events in merged catalog

		:param Mtype:
		:param Mrelation:
			see :meth:`MergedEvent.calc_mag_deltas`
		:param max_score:
			float, maximum score to consider as a potential match
			(default: 1000)

		:return:
			ordered dict, mapping merged event IDs to (event ID, score) tuples
			representing the closest match
			dict is ordered according to the matching score
		"""
		from collections import OrderedDict
		from mapping.geotools.geodetic import spherical_distance
		from . import time as timelib

		assert len(set([ev.ID for ev in self.merged_events]))

		match_scores = {}
		for ev in self.merged_events:
			if len(ev.solutions) == 1:
				ev_match_scores = {}
				for other_ev in self.merged_events:
					if not other_ev.has_agency(ev.master_agency):
						eq = ev.get_master_solution()
						Mrel = Mrelation.get(ev.master_agency, Mrelation)
						mag = eq.get_or_convert_mag(Mtype=Mtype, Mrelation=Mrel)

						scores = []
						for other_agency, other_eq in other_ev.solutions.items():
							time_delta = np.abs(eq.datetime - other_eq.datetime)
							d = spherical_distance(eq.lon, eq.lat, other_eq.lon, other_eq.lat)
							Mrel = Mrelation.get(other_agency, Mrelation)
							other_mag = other_eq.get_or_convert_mag(Mtype=Mtype, Mrelation=Mrel)
							mag_delta = np.abs(mag - other_mag)
							score = timelib.fractional_time_delta(time_delta, 's')
							if not np.isnan(d):
								score += d / 10.
							else:
								score += 10
							if not np.isnan(mag_delta):
								score += mag_delta / 0.3
							else:
								score += 10
							scores.append(score)
						min_score = np.min(scores)
						ev_match_scores[other_ev.ID] = min_score
				idx = np.argmin(list(ev_match_scores.values()))
				matching_ev_ID = list(ev_match_scores.keys())[idx]
				score = list(ev_match_scores.values())[idx]
				if score <= max_score:
					match_scores[ev.ID] = (matching_ev_ID, score)

		ordered_match_scores = OrderedDict(sorted(match_scores.items(),
															key=lambda x: x[1][1]))

		return ordered_match_scores

	def find_missing_events(self, agency, buffer_distance=0,
									start_date=None, Mmin=None, Mmax=None,
									Mtype='MW', Mrelation={}):
		"""
		Find events in the authoritative area and period of activity of a given
		agency that have not been detected by this agency, but by one or more
		other agencies

		:param agency:
			str, name of agency for which to find missing events
		:param buffer_distance:
			float, buffer distance (in km) to consider around agency's
			authoritative area
			(default: 0)
		:param start_date:
			date specification understood by eqcatalog, start date
			(default: None, will auto-determine start date of agency catalog)
		:param Mmin:
			float, minimum magnitude to consider
			(default: None)
		:param Mmax:
			float, maximum magnitude to consider
			(default: None)
		:param Mtype:
			str, magnitude type for :param:`Mmin` and :param:`Mmax`
			(default: 'MW')
		:param Mrelation:
			dict, mapping agency names to dicts, mapping Mtypes to instances
			of :class:`eqcatalog.msc.MSCE`
			or dict mapping Mtypes to instances of :class:`eqcatalog.msc.MSCE`
			for converting to given Mtype
			(default: {})

		:return:
			instance of :class:`MergedCatalog`
		"""
		from itertools import combinations

		## Determine start date if not given
		if not start_date:
			agency_merged_cat = self.subselect_by_agency(agency, unique=True)
			start_date = agency_merged_cat.get_master_catalog().Tminmax()[0]

		area_agency_merged_cat = self.subselect_by_agency_area(agency,
															buffer_distance=buffer_distance)
		area_agency_merged_cat = area_agency_merged_cat.subselect(start_date=start_date,
															Mmin=Mmin, Mmax=Mmax,
															Mtype=Mtype, Mrelation=Mrelation)

		other_agencies = self.get_agencies()
		other_agencies.remove(agency)
		merged_events = []
		for n in range(len(other_agencies)):
			for agency_combo in combinations(other_agencies, n+1):
				unique_agency_merged_cat = area_agency_merged_cat.subselect_by_agencies(
																	agency_combo, unique=True)
				merged_events += unique_agency_merged_cat.merged_events

		catalog_name = '%s missing events' % agency
		default_completeness = self.default_completeness.copy().pop(agency, None)
		default_Mrelations = self.default_Mrelations.copy().pop(agency, {})
		agency_areas = self.agency_areas.copy().pop(agency, {})
		merged_catalog = self.__class__(merged_events, name=catalog_name,
											default_completeness=default_completeness,
											default_Mrelations=default_Mrelations,
											agency_areas=agency_areas)

		return merged_catalog.get_sorted()

	def find_ambiguous_event_types(self):
		"""
		Find events with different event types assigned by different agencies

		:return:
			instance of :class:`MergedCatalog`
		"""
		merged_events = []
		for ev in self:
			num_solutions = len(ev)
			if num_solutions > 1:
				event_types = set([solution.event_type for solution in ev])
				if len(event_types) > 1:
					merged_events.append(ev)

		catalog_name = self.name + ' (ambiguous event types)'

		merged_catalog = self.__class__(merged_events, name=catalog_name,
											default_completeness=self.default_completeness,
											default_Mrelations=self.default_Mrelations,
											agency_areas=self.agency_areas)

		return merged_catalog
