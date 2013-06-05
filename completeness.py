import datetime
import numpy as np



class Completeness:
	"""
	Class defining completeness of earthquake catalog.

	:param min_dates:
		list or array containing initial years or dates of completeness,
		in chronological order (= from small to large)
	:param min_mags:
		list or array with corresponding lower magnitude for which
		catalog is assumed to be complete
	:param Mtype:
			String, magnitude type: "ML", "MS" or "MW"
	"""
	def __init__(self, min_dates, min_mags, Mtype):
		## Convert years to dates if necessary
		if isinstance(min_dates[0], int):
			min_dates = [datetime.date(yr, 1, 1) for yr in min_dates]
		self.min_dates = np.array(min_dates)
		self.min_mags = np.array(min_mags)
		self.Mtype = Mtype
		## Make sure ordering is chronologcal
		if len(self.min_dates) > 1 and self.min_dates[0] > self.min_dates[1]:
			self.min_dates = self.min_dates[::-1]
			self.min_mags = self.min_mags[::-1]

	def __len__(self):
		return len(self.min_dates)

	def __str__(self):
		s = "\n".join(["%d, %.2f" % (date, mag) for (date, mag) in zip(self.min_dates, self.min_mags)])
		return s

	@property
	def start_date(self):
		return self.min_dates.min()

	@property
	def start_year(self):
		return self.start_date.year

	@property
	def min_years(self):
		#return np.array([date.year for date in self.min_dates])
		years = np.array([date.year for date in self.min_dates], 'f')
		year_num_days = np.array([(datetime.date(year, 12, 31) - datetime.date(year, 1, 1)).days + 1 for year in years], 'f')
		time_deltas = self.min_dates - np.array([datetime.date(year, 1, 1) for year in years])
		num_days = np.array([td.days for td in time_deltas], 'f')
		return years + num_days / year_num_days

	def get_completeness_magnitude(self, date):
		"""
		Return completeness magnitude for given date, this is the lowest
		magnitude for which the catalog is complete.

		:param date:
			datetime.date or Int, date or year

		:return:
			Float, completeness magnitude
		"""
		if isinstance(date, int):
			date = datetime.date(date, 1, 1)
		try:
			index = np.where(self.min_dates <= date)[0][-1]
		except IndexError:
			return None
		else:
			return self.min_mags[index]

	def get_completeness_date(self, M):
		"""
		Return initial date of completeness for given magnitude

		:param M:
			Float, magnitude

		:return:
			datetime.date, initial date of completeness for given magnitude
		"""
		try:
			index = np.where(M >= self.min_mags)[0][0]
		except:
			return None
		else:
			return self.min_dates[index]

	def get_completeness_year(self, M):
		"""
		Return initial year of completeness for given magnitude

		:param M:
			Float, magnitude

		:return:
			Int, initial year of completeness for given magnitude
		"""
		return self.get_completeness_date(M).year

	def get_completeness_timespans(self, magnitudes, end_date):
		"""
		Compute completeness timespans in fractional years for list of magnitudes

		:param magnitudes:
			list or numpy array, magnitudes
		:param end_date:
			datetime.date object or int, end date or year with respect to which
			timespan has to be computed

		:return:
			numpy float array, completeness timespans (fractional years)
		"""
		if isinstance(end_date, int):
			end_date = datetime.date(end_date, 1, 1)
		completeness_dates = [self.get_completeness_date(M) for M in magnitudes]
		completeness_timespans = [((end_date - start_date).days + 1) / 365.25 for start_date in completeness_dates]
		return np.array(completeness_timespans)

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


## NOTE: I think threshold magnitudes should be a multiple of dM (or dM/2)!
Completeness_Leynaud = Completeness([1350, 1911, 1985], [4.7, 3.3, 1.8], "MS")
#Completeness_Leynaud = Completeness([1350, 1911, 1985], [4.75, 3.25, 1.75], "MS")
Completeness_Rosset = Completeness([1350, 1926, 1960, 1985], [5.0, 4.0, 3.0, 1.8], "MS")
## Following relation is for MW based on conversion from ML using Ahorner (1983)
Completeness_MW_201303a = Completeness([1350, 1750, 1860, 1905, 1960, 1985], [5.2, 4.9, 4.5, 4.0, 3.0, 2.2], "MW")
## Following relation is for MW based on conversion from ML using Reamer and Hinzen (2004)
Completeness_MW_201303b = Completeness([1350, 1750, 1860, 1905, 1960, 1985], [5.2, 4.9, 4.5, 3.9, 2.9, 2.0], "MW")


