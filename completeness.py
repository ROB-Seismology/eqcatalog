"""
Functionality related to catalog completeness (time/magnitude)
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np

from . import time as timelib



__all__ = ['Completeness', 'combine_completenesses']


class Completeness(object):
	"""
	Class defining completeness of earthquake catalog.

	:param min_dates:
		list or array containing initial years or dates of completeness,
		in chronological order (= from old to recent)
	:param min_mags:
		list or array with corresponding lower magnitude for which
		catalog is assumed to be complete (usually from large to small)
	:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
	"""
	def __init__(self, min_dates, min_mags, Mtype):
		if len(min_dates) != len(min_mags):
			raise Exception("Number of magnitudes not equal to number of dates!")

		## Convert years to dates if necessary
		if isinstance(min_dates[0], int):
			min_dates = [timelib.time_tuple_to_np_datetime(yr, 1, 1) for yr in min_dates]
			self.min_dates = np.array(min_dates)
		else:
			self.min_dates = timelib.as_np_datetime(min_dates)

		self.min_mags = np.array(min_mags)
		self.Mtype = Mtype

		## Make sure ordering is chronologcal
		if len(self.min_dates) > 1 and self.min_dates[0] > self.min_dates[1]:
			self.min_dates = self.min_dates[::-1]
			self.min_mags = self.min_mags[::-1]
		if not np.all(np.diff(self.min_dates).astype('float') >= 0):
			raise Exception("Completeness dates not in chronological order")
		if not np.all(np.diff(self.min_mags) < 0):
			print("Warning: Completeness magnitudes not monotonically "
							"decreasing with time!")

	def __len__(self):
		return len(self.min_dates)

	def __str__(self):
		s = "\n".join(["%s, %.2f" % (date, mag) for (date, mag)
						in zip(self.min_dates, self.min_mags)])
		return s

	def print_table(self):
		"""
		Print completeness dates and magnitues in a table
		"""
		try:
			from prettytable import PrettyTable
		except:
			has_prettytable = False
		else:
			has_prettytable = True

		col_names = ["Start date", "Mmin (%s)" % self.Mtype]
		if has_prettytable:
			tab = PrettyTable(col_names)
		else:
			tab = []
		for (date, mag) in zip(self.min_dates, self.min_mags):
			row = [str(date), "%.2f" % mag]
			if has_prettytable:
				tab.add_row(row)
			else:
				tab.append(row)

		if has_prettytable:
			print(tab)
		else:
			print('\t'.join(col_names))
			for row in tab:
				print('\t'.join(row))

	@property
	def start_date(self):
		return self.min_dates.min()

	@property
	def start_year(self):
		return timelib.to_year(self.start_date)

	@property
	def min_mag(self):
		return self.min_mags.min()

	@property
	def min_years(self):
		return timelib.to_fractional_year(self.min_dates)

	def are_mags_monotonously_decreasing(self):
		"""
		Determine whether completeness magnitudes are monotonously decreasing

		:return:
			bool
		"""
		return (np.diff(self.min_mags) <= 0).all()

	def get_completeness_magnitude(self, date):
		"""
		Return completeness magnitude for given date, this corresponds
		to the most recent completeness magnitude.

		:param date:
			datetime.date or Int, date or year

		:return:
			Float, completeness magnitude
		"""
		if isinstance(date, int):
			date = timelib.time_tuple_to_np_datetime(date, 1, 1)
		try:
			index = np.where(self.min_dates <= date)[0][-1]
		except IndexError:
			## Date before completeness start date
			return 10
		else:
			return self.min_mags[index]

	def get_lowest_completeness_magnitude(self, date):
		"""
		Return lowest completeness magnitude before given date,
		this is the lowest magnitude for which the catalog is complete
		(not necessarily corresponding to the most recent completeness
		magnitude!)

		:param date:
			datetime.date or Int, date or year

		:return:
			Float, completeness magnitude
		"""
		if isinstance(date, int):
			date = timelib.time_tuple_to_np_datetime(date, 1, 1)
		return self.min_mags[self.min_dates <= date].min()

	def get_initial_completeness_date(self, M):
		"""
		Return initial date of completeness for given magnitude

		:param M:
			Float, magnitude

		:return:
			np.datetime64, initial date of completeness for given magnitude
		"""
		try:
			index = np.where(M >= self.min_mags)[0][0]
		except:
			## Magnitude below smallest completeness magnitude
			## Return date very far in the future
			return timelib.time_tuple_to_np_datetime(1000000, 1, 1)
		else:
			return self.min_dates[index]

	def get_final_completeness_date(self, M):
		"""
		Return last date of completeness for given magnitude
		(useful for non-monotonically decreasing completeness magnitudes)

		:param M:
			float, magnitude

		:return:
			np.datetime64, final date of completeness for given magnitude
			(or None if magnitude is complete beyond the defined
			completeness interval)
		"""
		try:
			index = np.where(M >= self.min_mags)[0][-1] + 1
		except:
			## Magnitude below smallest completeness magnitude
			## Return oldest date in completeness interval
			return self.min_dates[0]
		else:
			if index < len(self):
				return self.min_dates[index]
			else:
				## Magnitude still complete at end of completeness interval
				return None

	def get_initial_completeness_year(self, M):
		"""
		Return initial year of completeness for given magnitude

		:param M:
			Float, magnitude

		:return:
			Int, initial year of completeness for given magnitude
		"""
		return timelib.to_year(self.get_initial_completeness_date(M))

	def get_completeness_timespans(self, magnitudes, end_date, unit='Y'):
		"""
		Select correct method to compute completeness timespans in
		fractional years for a list of magnitudes, depending on whether
		completeness magnitudes are monotonously decreasing or not

		:param magnitudes:
			list or numpy array, magnitudes
		:param end_date:
			datetime.date object or int, end date or year with respect to which
			timespan has to be computed
		:param unit:
			str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
			(year|week|day|hour|minute|second|millisecond|microsecond)
			(default: 'Y')

		:return:
			numpy float array, completeness timespans in fractions of :param:`unit`
		"""
		if self.are_mags_monotonously_decreasing():
			return self._get_completeness_timespans_monotonous(magnitudes, end_date,
																unit=unit)
		else:
			return self._get_completeness_timespans_non_monotonous(magnitudes, end_date,
																	unit=unit)

	def _get_completeness_timespans_monotonous(self, magnitudes, end_date, unit='Y'):
		"""
		Compute completeness timespans in fractional years for list of magnitudes.
		Works only if completeness magnitudes are monotonously decreasing.

		:param magnitudes:
			list or numpy array, magnitudes
		:param end_date:
			datetime.date object or int, end date or year with respect to which
			timespan has to be computed
		:param unit:
			str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
			(year|week|day|hour|minute|second|millisecond|microsecond)
			(default: 'Y')

		:return:
			numpy float array, completeness timespans in fractions of :param:`unit`
		"""
		if isinstance(end_date, int):
			end_date = timelib.time_tuple_to_np_datetime(end_date, 12, 31)
		completeness_dates = [self.get_initial_completeness_date(M) for M in magnitudes]
		completeness_dates = np.array(completeness_dates)
		completeness_timespans = timelib.timespan(completeness_dates, end_date, unit=unit)
		## Replace negative timespans with zeros
		completeness_timespans[np.where(completeness_timespans < 0)] = 0
		return completeness_timespans

	def _get_completeness_timespans_non_monotonous(self, magnitudes, end_date, unit='Y'):
		"""
		Compute completeness timespans in fractional years for list of
		magnitudes. This method can cope with non-monotonously decreasing
		completeness magnitudes.

		:param magnitudes:
			list or numpy array, magnitudes
		:param end_date:
			datetime.date object or int, end date or year with respect to which
			timespan has to be computed
		:param unit:
			str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
			(year|week|day|hour|minute|second|millisecond|microsecond)
			(default: 'Y')

		:return:
			numpy float array, completeness timespans in fractions of :param:`unit`
		"""
		if isinstance(end_date, int):
			end_date = timelib.time_tuple_to_np_datetime(end_date, 12, 31)
		are_mags_complete = np.zeros((len(self.min_mags), len(magnitudes)), dtype=bool)
		for i, min_mag in enumerate(self.min_mags):
			are_mags_complete[i] = (magnitudes >= min_mag)
		completeness_timespans = np.zeros_like(magnitudes)
		for m in range(len(magnitudes)):
			ts = 0
			is_mag_complete = are_mags_complete[:,m]
			try:
				start_idx = np.where(is_mag_complete == True)[0][0]
			except IndexError:
				start_idx = len(self)
			for i in range(start_idx, len(self)):
				if is_mag_complete[i]:
					interval_start = self.min_dates[i]
					if i < len(self) - 1:
						resolution = {'Y': 'D', 'M': 'D', 'D': 'h', 'h': 'm', 'm': 's'}.get(unit)
						interval_end = self.min_dates[i+1] - np.timedelta64(1, resolution)
					else:
						interval_end = end_date
					ts += timelib.timespan(interval_start, interval_end, unit=unit)
			completeness_timespans[m] = ts
		return completeness_timespans

	def get_total_timespan(self, end_date, unit='Y'):
		"""
		Give total timespan represented by completeness object

		:param end_date:
			datetime.date object or int, end date or year with respect to which
			timespan has to be computed
		:param unit:
			str, one of 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us'
			(year|week|day|hour|minute|second|millisecond|microsecond)
			(default: 'Y')

		:return:
			float, timespan in fractions of :param:`unit`
		"""
		if isinstance(end_date, int):
			end_date = timelib.time_tuple_to_np_datetime(end_date, 12, 31)
		return timelib.timespan(self.start_date, end_date, unit=unit)

	def to_hmtk_table(self, Mmax=None):
		"""
		Convert to a 2-D completeness table used by hazard modeler's toolkit
		"""
		n = len(self)
		if Mmax and Mmax > self.min_mags.max():
			n += 1
		table = np.zeros((n, 2), 'f')
		table[:len(self),0] = self.min_years[::-1]
		table[:len(self),1] = self.min_mags[::-1]
		if Mmax and Mmax > self.min_mags.max():
			table[-1,0] = self.min_years.min()
			table[-1,1] = Mmax
		return table


def combine_completenesses(completenesses, combine_func='min'):
	"""
	Combine different completenesses into a single object

	:param completenesses:
		list with instances of :class:`Completeness`
	:param combine_func:
		str, name of function to apply to minimum magnitudes of different
		completenesses: 'min', 'max' or 'mean'
		(default: 'min')

	:return:
		instance of :class:`Completeness`
	"""
	Mtypes = [completeness.Mtype for completeness in completenesses]
	assert len(set(Mtypes)) == 1
	Mtype = Mtypes[0]

	min_dates = []
	for completeness in completenesses:
		min_dates.extend(completeness.min_dates)
	min_dates = np.sort(np.unique(min_dates))

	min_mags = []
	combine_func = getattr(np, combine_func)
	for min_date in min_dates:
		mag = combine_func([completeness.get_completeness_magnitude(min_date)
								for completeness in completenesses])
		min_mags.append(mag)

	merged_completeness = Completeness(min_dates, min_mags, Mtype)

	return merged_completeness
