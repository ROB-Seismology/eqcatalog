"""
This module extends functionality of nhlib.mfd
"""

import datetime
import numpy as np
import pylab
from matplotlib.font_manager import FontProperties
import nhlib.mfd



class MFD:
	"""
	Generic class containing methods that are common for
	:class:`EvenlyDiscretizedMFD` and :class:`TruncatedGRMFD`
	"""
	def __init__(self):
		pass

	def __len__(self):
		return int(round((self.max_mag - self.get_min_mag_edge()) / self.bin_width))

	def get_magnitude_bin_centers(self):
		"""
		Return center values of magnitude bins

		:return:
			numpy float array
		"""
		return np.array(zip(*self.get_annual_occurrence_rates())[0])

	def get_magnitude_bin_edges(self):
		"""
		Return left edge value of magnitude bins

		:return:
			numpy float array
		"""
		return self.get_magnitude_bin_centers() - self.bin_width / 2

	def get_magnitude_index(self, M):
		"""
		Determine index of given magnitude (edge) value

		:param M:
			Float, magnitude value (left edge of bin)

		:return:
			Int, index
		"""
		return int(round((M - self.get_min_mag()) / self.bin_width))

	def get_cumulative_rates(self):
		"""
		Return cumulative annual occurrence rates

		:return:
			numpy float array
		"""
		return np.add.accumulate(self.occurrence_rates[::-1])[::-1]

	def is_magnitude_compatible(self, M):
		"""
		Determine whether a particular magnitude (edge) value is compatible
		with bin width of MFD

		:param M:
			Float, magnitude value (left edge of bin)

		:return:
			Bool
		"""
		foffset = (M - self.get_min_mag_edge()) / self.bin_width
		offset = int(round(foffset))
		if not np.allclose(foffset, offset):
			return False
		else:
			return True

	def is_compatible(self, other_mfd):
		"""
		Determine if MFD is compatible with another one, in terms of
		bin width, modulus of magnitude, and magnitude type

		:param other_mfd:
			instance of :class:`EvenlyDiscretizedMFD` or :clas:`TruncatedGRMFD`

		:return:
			Bool
		"""
		magnitude_bin_edges = other_mfd.get_magnitude_bin_edges()
		occurrence_rates = other_mfd.occurrence_rates
		if other_mfd.Mtype != self.Mtype:
			return False
		if not np.allclose(other_mfd.bin_width, self.bin_width):
			return False
		elif not self.is_magnitude_compatible(other_mfd.get_min_mag_edge()):
			return False
		else:
			return True


class EvenlyDiscretizedMFD(nhlib.mfd.EvenlyDiscretizedMFD, MFD):
	"""
	Evenly Discretized Magnitude-Frequency Distribution

	:param min_mag:
		Positive float value representing the middle point of the first
		bin in the histogram.
	:param bin_width:
		A positive float value -- the width of a single histogram bin.
	:param occurrence_rates:
		The list of non-negative float values representing the actual
		annual occurrence rates. The resulting histogram has as many bins
		as this list length.
	:param Mtype:
		String, magnitude type, either "MW" or "MS" (default: "MW")
	"""
	def __init__(self, min_mag, bin_width, occurrence_rates, Mtype="MW"):
		nhlib.mfd.EvenlyDiscretizedMFD.__init__(self, min_mag, bin_width, list(occurrence_rates))
		self.occurrence_rates = np.array(self.occurrence_rates)
		self.Mtype = Mtype

	def __div__(self, other):
		if isinstance(other, (int, float)):
			occurrence_rates = np.array(self.occurrence_rates) / other
			return EvenlyDiscretizedMFD(self.min_mag, self.bin_width, list(occurrence_rates), self.Mtype)
		else:
			raise TypeError("Divisor must be integer or float")

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			occurrence_rates = np.array(self.occurrence_rates) * other
			return EvenlyDiscretizedMFD(self.min_mag, self.bin_width, list(occurrence_rates), self.Mtype)
		else:
			raise TypeError("Divisor must be integer or float")

	def __add__(self, other):
		if isinstance(other, (TruncatedGRMFD, EvenlyDiscretizedMFD)):
			return sum_MFDs([self, other])
		else:
			raise TypeError("Operand must be MFD")

	def __sub__(self, other):
		if isinstance(other, (TruncatedGRMFD, EvenlyDiscretizedMFD)):
			if not self.is_compatible(other):
				raise Exception("MFD's not compatible")
			if self.get_min_mag() <= other.get_min_mag() and self.max_mag >= other.max_mag:
				occurrence_rates = self.occurrence_rates.copy()
				start_index = self.get_magnitude_index(other.get_min_mag())
				occurrence_rates[start_index:start_index+len(other)] -= other.occurrence_rates
				# Replace negative values with zeros
				occurrence_rates[np.where(occurrence_rates <0)] = 0
				return EvenlyDiscretizedMFD(self.min_mag, self.bin_width, occurrence_rates)
			else:
				raise Exception("Second MFD must fully overlap with first one!")
		else:
			raise TypeError("Operand must be MFD")

	@property
	def max_mag(self):
		return self.get_min_mag_edge() + (len(self.occurrence_rates) + 1) * self.bin_width

	def get_min_mag_edge(self):
		"""
		Return left edge of minimum magnitude bin

		:return:
			Float
		"""
		return self.min_mag - self.bin_width / 2

	def get_min_mag_center(self):
		"""
		Return center value of minimum magnitude bin

		:return:
			Float
		"""
		return self.min_mag

	#def get_magnitude_bin_edges(self):
	#	return np.array(zip(*self.get_annual_occurrence_rates())[0])

	#def get_magnitude_bin_centers(self):
	#	return self.get_magnitude_bin_edges() + self.bin_width / 2

	#def get_cumulative_rates(self):
	#	return np.add.accumulate(self.occurrence_rates[::-1])[::-1]

	def divide(self, weights):
		"""
		Divide MFD into a number of MFD's that together sum up to the original MFD

		:param weights:
			list or array containing weight of each sub-MFD

		:return:
			List containing instances of :class:`EvenlyDiscretizedMFD`
		"""
		weights = np.array(weights, dtype='d')
		weights /= np.add.reduce(weights)
		mfd_list = []
		for w in weights:
			occurrence_rates = np.array(self.occurrence_rates) * w
			mfd = EvenlyDiscretizedMFD(self.min_mag, self.bin_width, list(occurrence_rates), self.Mtype)
			mfd_list.append(mfd)
		return mfd_list

	def split(self, M):
		"""
		Split MFD at a particular magnitude

		:param M:
			Float, magnitude value where MFD should be split

		:return:
			List containing 2 instances of :class:`EvenlyDiscretizedMFD`
		"""
		if not self.is_magnitude_compatible(M):
			raise Exception("Magnitude value not compatible!")
		elif self.get_min_mag_edge() < M < self.max_mag:
			index = int(round((M - self.get_min_mag_edge()) / self.bin_width))
			occurrence_rates1 = list(self.occurrence_rates[:index])
			occurrence_rates2 = list(self.occurrence_rates[index:])
			mfd1 = EvenlyDiscretizedMFD(self.min_mag, self.bin_width, occurrence_rates1, self.Mtype)
			mfd2 = EvenlyDiscretizedMFD(M+self.bin_width/2, self.bin_width, occurrence_rates2, self.Mtype)
			return [mfd1, mfd2]
		else:
			raise Exception("Split magnitude not in valid range!")

	def extend(self, other_mfd):
		"""
		Extend MFD with another one that covers larger magnitudes.

		:param other_mfd:
			instance of :class:`EvenlyDiscretizedMFD` or :class:`TruncatedGRMFD`
			The minimum magnitude of other_mfd should be equal to or larger than
			the maximum magnitude of this MFD.

		Note:
			Bins between both MFD's will be filled with zero incremental
			occurrence rates!
		"""
		magnitude_bin_edges = other_mfd.get_magnitude_bin_edges()
		occurrence_rates = other_mfd.occurrence_rates
		if not np.allclose(other_mfd.bin_width, self.bin_width):
			raise Exception("Bin width not compatible!")
		fgap = (magnitude_bin_edges[0] - self.max_mag) / self.bin_width
		gap = int(round(fgap))
		if not np.allclose(fgap, gap):
			raise Exception("Bin width not compatible!")

		num_empty_bins = int(round((magnitude_bin_edges[0] - self.max_mag) / self.bin_width)) + 1
		if num_empty_bins >= 0:
			self.occurrence_rates = np.concatenate([self.occurrence_rates, np.zeros(num_empty_bins, dtype='d'), occurrence_rates])
		else:
			raise Exception("Magnitudes must not overlap with MFD magnitude range")

	def append_characteristic_eq(self, Mc, return_period):
		"""
		Append magnitude-frequency of a characteristic earthquake

		:param Mc:
			Float, magnitude of characteristic earthquake
		:param return_period:
			Float, return period in yr of characteristic earthquake
		"""
		characteristic_mfd = EvenlyDiscretizedMFD(Mc, self.bin_width, [1./return_period])
		self.extend(characteristic_mfd)

	def plot(self, color='k', style="o", label="", discrete=True, cumul_or_inc="both", completeness=None, Mrange=(), Freq_range=(), title="", lang="en", fig_filespec=None, fig_width=0, dpi=300):
		"""
		Plot magnitude-frequency distribution

		:param color:
			matplotlib color specification (default: 'k')
		:param style:
			matplotlib symbol style or line style (default: 'o')
		:param label:
			String, plot labels (default: "")
		:param discrete:
			Bool, whether or not to plot discrete MFD (default: True)
		:param cumul_or_inc:
			String, either "cumul", "inc" or "both", indicating
			whether to plot cumulative MFD, incremental MFD or both
			(default: "both")
		:param completeness:
			instance of :class:`Completeness`, used to plot completeness
			limits (default: None)
		:param Mrange:
			(Mmin, Mmax) tuple, minimum and maximum magnitude in X axis
			(default: ())
		:param Freq_range:
			(Freq_min, Freq_max) tuple, minimum and maximum values in frequency
			(Y) axis (default: ())
		:param title:
			String, plot title (default: "")
		:param lang:
			String, language of plot axis labels (default: "en")
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)
		"""
		plot_MFD([self], colors=[color], styles=[style], labels=[label], discrete=[discrete], cumul_or_inc=[cumul_or_inc], completeness=completeness, title=title)


class TruncatedGRMFD(nhlib.mfd.TruncatedGRMFD, MFD):
	"""
	Truncated or modified Gutenberg-Richter MFD

	:param min_mag:
		The lowest possible magnitude for this MFD. The first bin in the
		:meth:`result histogram <get_annual_occurrence_rates>` will be aligned
		to make its left border match this value.
	:param max_mag:
		The highest possible magnitude. The same as for ``min_mag``: the last
		bin in the histogram will correspond to the magnitude value equal to
		``max_mag - bin_width / 2``.
	:param bin_width:
		A positive float value -- the width of a single histogram bin.
	:param a_val:
		Float, the cumulative ``a`` value (``10 ** a`` is the number
		of earthquakes per year with magnitude greater than or equal to 0),
	:param b_val:
		Float, Gutenberg-Richter ``b`` value -- the decay rate
		of exponential distribution. It describes the relative size distribution
		of earthquakes: a higher ``b`` value indicates a relatively larger
		proportion of small events and vice versa.
	:param b_sigma:
		Float, standard deviation on the b value.
	:param Mtype:
		String, magnitude type, either "MW" or "MS" (default: "MW")

	Note:
		Values for ``min_mag`` and ``max_mag`` don't have to be aligned with
		respect to ``bin_width``. They get rounded accordingly anyway so that
		both are divisible by ``bin_width`` just before converting a function
		to a histogram. See :meth:`_get_min_mag_and_num_bins`.
	"""
	def __init__(self, min_mag, max_mag, bin_width, a_val, b_val, b_sigma, Mtype="MW"):
		nhlib.mfd.TruncatedGRMFD.__init__(self, min_mag, max_mag, bin_width, a_val, b_val)
		self.b_sigma = b_sigma
		self.Mtype = Mtype

	def __div__(self, other):
		if isinstance(other, (int, float)):
			N0 = 10**self.a_val
			a_val = np.log10(N0 / float(other))
			return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, a_val, self.b_val, self.b_sigma, self.Mtype)
		else:
			raise TypeError("Divisor must be integer or float")

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			N0 = 10**self.a_val
			a_val = np.log10(N0 * float(other))
			return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, a_val, self.b_val, self.b_sigma, self.Mtype)
		else:
			raise TypeError("Multiplier must be integer or float")

	def __add__(self, other):
		if isinstance(other, (TruncatedGRMFD, EvenlyDiscretizedMFD)):
			return sum_MFDs([self, other])
		else:
			raise TypeError("Operand must be MFD")

	def __sub__(self, other):
		if isinstance(other, TruncatedGRMFD):
			if self.min_mag == other.min_mag and self.max_mag == other.max_mag and self.b_val == other.b_val and self.Mtype == other.Mtype:
				## Note: bin width does not have to be the same here
				N0 = 10 ** self.a_val - 10 ** other.a_val
				a_val = np.log10(N0)
				return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, a_val, self.b_val, self.b_sigma, self.Mtype)
		elif isinstance(other, EvenlyDiscretizedMFD):
			return self.to_evenly_discretized_mfd().__sub__(other)
		else:
			raise TypeError("Operand must be MFD")

	@property
	def occurrence_rates(self):
		return np.array(zip(*self.get_annual_occurrence_rates())[1])

	def get_min_mag_edge(self):
		"""
		Return left edge of minimum magnitude bin

		:return:
			Float
		"""
		return self.min_mag

	def get_min_mag_center(self):
		"""
		Return center value of minimum magnitude bin

		:return:
			Float
		"""
		return self.min_mag + self.bin_width / 2

	def get_cumulative_rates(self):
		"""
		Return cumulative annual occurrence rates

		:return:
			numpy float array
		"""
		a, b = self.a_val, self.b_val
		min_mag, max_mag = self.get_min_mag_edge(), self.max_mag
		mags = self.get_magnitude_bin_edges()
		return (10**(a-b*min_mag))*((10**(-1*b*mags)-10**(-1*b*max_mag))/(10**(-1*b*min_mag)-10**(-1*b*max_mag)))
		## Note: the following is identical
		#return np.add.accumulate(self.occurrence_rates[::-1])[::-1]

	def to_evenly_discretized_mfd(self):
		"""
		Convert to an EvenlyDiscretizedMFD

		:return:
			instance of :class:`EvenlyDiscretizedMFD`
		"""
		return EvenlyDiscretizedMFD(self.get_min_mag_center(), self.bin_width, list(self.occurrence_rates))

	def divide(self, weights):
		"""
		Divide MFD into a number of MFD's that together sum up to the original MFD

		:param weights:
			list or array containing weight of each sub-MFD

		:return:
			List containing instances of :class:`TruncatedGRMFD`
		"""
		weights = np.array(weights, dtype='d')
		weights /= np.add.reduce(weights)
		N0 = 10**self.a_val
		avalues = np.log10(weights * N0)
		return [TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, aw, self.b_val, self.b_sigma, self.Mtype) for aw in avalues]

	def split(self, M):
		"""
		Split MFD at a particular magnitude

		:param M:
			Float, magnitude value where MFD should be split

		:return:
			List containing 2 instances of :class:`TruncatedGRMFD`
		"""
		if not self.is_magnitude_compatible(M):
			raise Exception("Magnitude value not compatible!")
		elif self.get_min_mag_edge() < M < self.max_mag:
			mfd1 = TruncatedGRMFD(self.min_mag, M, self.bin_width, self.a_val, self.b_val, self.b_sigma, self.Mtype)
			mfd2 = TruncatedGRMFD(M, self.max_mag, self.bin_width, self.a_val, self.b_val, self.b_sigma, self.Mtype)
			return [mfd1, mfd2]
		else:
			raise Exception("Split magnitude not in valid range!")

	def extend(self, other_mfd):
		"""
		Extend MFD with another one that covers larger magnitudes.

		:param other_mfd:
			instance of :class:`TruncatedGRMFD`
			The minimum magnitude of other_mfd should be equal to the
			maximum magnitude of this MFD.

		Note:
			If other_mfd is instance of :class:`EvenlyDiscretizedGRMFD`
			or if its min_mag is larger than max_mag of this MFD, an
			exception will be raised, prompting to convert to an instance
			of :class:`EvenlyDiscretizedGRMFD` first.
		"""
		if isinstance(other_mfd, TruncatedGRMFD) and other_mfd.b_val == self.bval and other_mfd.min_mag == self.max_mag:
			self.max_mag = other_mfd.max_mag
		else:
			raise Exception("MFD objects not compatible. Convert to EvenlyDiscretizedMFD")

	def plot(self, color='k', style="-", label="", discrete=False, cumul_or_inc="cumul", completeness=None, Mrange=(), Freq_range=(), title="", lang="en", fig_filespec=None, fig_width=0, dpi=300):
		"""
		Plot magnitude-frequency distribution

		:param color:
			matplotlib color specification (default: 'k')
		:param style:
			matplotlib symbol style or line style (default: '-')
		:param label:
			String, plot labels (default: "")
		:param discrete:
			Bool, whether or not to plot discrete MFD (default: False)
		:param cumul_or_inc:
			String, either "cumul", "inc" or "both", indicating
			whether to plot cumulative MFD, incremental MFD or both
			(default: "cumul")
		:param completeness:
			instance of :class:`Completeness`, used to plot completeness
			limits (default: None)
		:param Mrange:
			(Mmin, Mmax) tuple, minimum and maximum magnitude in X axis
			(default: ())
		:param Freq_range:
			(Freq_min, Freq_max) tuple, minimum and maximum values in frequency
			(Y) axis (default: ())
		:param title:
			String, plot title (default: "")
		:param lang:
			String, language of plot axis labels (default: "en")
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)
		"""
		plot_MFD([self], colors=[color], styles=[style], labels=[label], discrete=[discrete], cumul_or_inc=[cumul_or_inc], completeness=completeness, title=title)


def sum_MFDs(mfd_list, weights=[]):
	"""
	Sum two or more MFD's

	:param mfd_list:
		List containing instances of :class:`EvenlyDiscretizedMFD` or
		:class:`TruncatedGRMFD`

	:param weights:
		List or array containing weights of each MFD (default: [])

	:return:
		instance of :class:`TruncatedGRMFD` (if all MFD's in list are
		TruncatedGR, and have same min_mag, max_mag, and b_val) or else
		instance of :class:`EvenlyDiscretizedMFD`

	Note:
		Weights will be normalized!
	"""
	if weights in ([], None):
		weights = np.ones(len(mfd_list), 'd')
	total_weight = np.add.reduce(weights)
	weights = (np.array(weights) / total_weight) * len(mfd_list)
	bin_width = min([mfd.bin_width for mfd in mfd_list])
	Mtype = mfd_list[0].Mtype
	for mfd in mfd_list:
		if mfd.bin_width != bin_width:
			raise Exception("Bin widths not compatible!")
		if mfd.Mtype != Mtype:
			raise Exception("Magnitude types not compatible!")
	all_min_mags = set([mfd.get_min_mag_edge() for mfd in mfd_list])
	all_max_mags = set([mfd.max_mag for mfd in mfd_list])
	## If all MFD's are TruncatedGR, and have same min_mag, max_mag, and b_val
	## return TrucatedGR, else return EvenlyDiscretized
	is_truncated = np.array([isinstance(mfd, TruncatedGRMFD) for mfd in mfd_list])
	if is_truncated.all():
		all_bvals = set([mfd.b_val for mfd in mfd_list])
		all_Mtypes = set([mfd.Mtype for mfd in mfd_list])
		if len(all_min_mags) == len(all_max_mags) == len(all_bvals) == len(all_Mtypes) == 1:
			all_avals = np.array([mfd.a_val for mfd in mfd_list])
			a = np.log10(np.add.reduce(10**all_avals * weights))
			mfd = mfd_list[0]
			return TruncatedGRMFD(mfd.min_mag, mfd.max_mag, mfd.bin_width, a, mfd.b_val, mfd.b_sigma, mfd.Mtype)
	else:
		min_mag = min(all_min_mags)
		max_mag = max(all_max_mags)
		num_bins = int(round((max_mag - min_mag) / bin_width))
		occurrence_rates = np.zeros(num_bins, 'd')
		for i, mfd in enumerate(mfd_list):
			start_index = int(round((mfd.get_min_mag() - min_mag) / bin_width))
			end_index = start_index + len(mfd.occurrence_rates)
			occurrence_rates[start_index:end_index] += (mfd.occurrence_rates * weights[i])
		return EvenlyDiscretizedMFD(min_mag+bin_width/2, bin_width, list(occurrence_rates), Mtype)


def plot_MFD(mfd_list, colors=[], styles=[], labels=[], discrete=[], cumul_or_inc=[], completeness=None, Mrange=(), Freq_range=(), title="", lang="en", fig_filespec=None, fig_width=0, dpi=300):
	"""
	Plot one or more magnitude-frequency distributions

	:param mfd_list:
		List with instance of :class:`EvenlyDiscretizedMFD` or :class:`TruncatedGRMFD`
	:param colors:
		List with matplotlib color specifications, one for each mfd
		(default: [])
	:param styles:
		List with matplotlib symbol styles or line styles, one for each mfd
		(default: [])
	:param labels:
		List with plot labels, one for each mfd (default: [])
	:param discrete:
		List of bools, whether or not to plot discrete MFD's (default: [])
	:param cumul_or_inc:
		List of strings, either "cumul", "inc" or "both", indicating
		whether to plot cumulative MFD, incremental MFD or both
		(default: [])
	:param completeness:
		instance of :class:`Completeness`, used to plot completeness
		limits (default: None)
	:param Mrange:
		(Mmin, Mmax) tuple, minimum and maximum magnitude in X axis
		(default: ())
	:param Freq_range:
		(Freq_min, Freq_max) tuple, minimum and maximum values in frequency
		(Y) axis (default: ())
	:param title:
		String, plot title (default: "")
	:param lang:
		String, language of plot axis labels (default: "en")
	:param fig_filespec:
		String, full path to output image file, if None plot to screen
		(default: None)
	:param fig_width:
		Float, figure width in cm, used to recompute :param:`dpi` with
		respect to default figure width (default: 0)
	:param dpi:
		Int, image resolution in dots per inch (default: 300)
	"""
	if not colors:
		colors = ("r", "g", "b", "c", "m", "k")

	if not labels:
		labels = [""] * len(mfd_list)

	## Plot
	fig = pylab.figure()

	for i, mfd in enumerate(mfd_list):
		color = colors[i % len(colors)]

		try:
			want_discrete = discrete[i]
		except:
			if isinstance(mfd, EvenlyDiscretizedMFD):
				want_discrete = True
			elif isinstance(mfd, TruncatedGRMFD):
				want_discrete = False

		try:
			cumul_or_inc[i]
		except:
			if isinstance(mfd, EvenlyDiscretizedMFD):
				want_cumulative = True
				want_incremental = True
			elif isinstance(mfd, TruncatedGRMFD):
				want_cumulative = True
				want_incremental = False
		else:
			if cumul_or_inc[i] == "cumul":
				want_cumulative = True
				want_incremental = False
			elif cumul_or_inc[i] == "inc":
				want_cumulative = False
				want_incremental = True
			else:
				want_cumulative = True
				want_incremental = True

		## Discrete MFD
		if want_discrete:
			try:
				symbol = styles[i]
			except:
				symbol = 'o'
			else:
				if symbol in ("", None, "-", "--", ":", ":."):
					symbol = "o"

			## Cumulative
			if want_cumulative:
				label = labels[i]
				if want_incremental:
					label += " (cumul.)"
				ax = pylab.semilogy(mfd.get_magnitude_bin_edges(), mfd.get_cumulative_rates(), symbol, label=label)
				pylab.setp(ax, markersize=10.0, markeredgewidth=1.0, markeredgecolor='k', markerfacecolor=color)

			## Incremental
			if want_incremental:
				label = labels[i] + " (inc.)"
				ax = pylab.semilogy(mfd.get_magnitude_bin_centers(), mfd.occurrence_rates, symbol, label=label)
				pylab.setp(ax, markersize=10.0, markeredgewidth=1.0, markeredgecolor=color, markerfacecolor="None")

		## Continuous MFD
		else:
			try:
				linestyle = styles[i]
			except:
				linestyle = "-"
			else:
				if linestyle in ("", None) or not linestyle in ("-", "--", ":", ":."):
					linestyle = "-"

			## Cumulative
			if want_cumulative:
				label = labels[i]
				if want_incremental:
					label += " (cumul.)"
				ax = pylab.semilogy(mfd.get_magnitude_bin_edges(), mfd.get_cumulative_rates(), color, linestyle=linestyle, lw=3, label=label)

			## Incremental
			if want_incremental:
				label = labels[i] + " (inc.)"
				ax = pylab.semilogy(mfd.get_magnitude_bin_centers(), mfd.occurrence_rates, color, linestyle=linestyle, lw=1, label=label)

	if not Mrange:
		Mrange = pylab.axis()[:2]
	if not Freq_range:
		Freq_range = pylab.axis()[2:]

	## Plot limits of completeness
	if completeness:
		annoty = Freq_range[0] * 10**0.5
		bbox_props = dict(boxstyle="round,pad=0.4", fc="w", ec="k", lw=1)
		ax = pylab.gca()
		## Make sure min_mags is not sorted in place,
		## otherwise completeness object may misbehave
		min_mags = np.sort(completeness.min_mags)
		end_year = datetime.date.today().year
		for i in range(1, len(min_mags)):
			pylab.plot([min_mags[i], min_mags[i]], Freq_range, 'k--', lw=1, label="_nolegend_")
			ax.annotate("", xy=(min_mags[i-1], annoty), xycoords='data', xytext=(min_mags[i], annoty), textcoords='data', arrowprops=dict(arrowstyle="<->"),)
			label = "%s - %s" % (completeness.get_completeness_year(min_mags[i-1]), end_year)
			ax.text(np.mean([min_mags[i-1], min_mags[i]]), annoty*10**-0.25, label, ha="center", va="center", size=12, bbox=bbox_props)
		ax.annotate("", xy=(min_mags[i], annoty), xycoords='data', xytext=(min(mfd.max_mag, Mrange[1]), annoty), textcoords='data', arrowprops=dict(arrowstyle="<->"),)
		label = "%s - %s" % (completeness.get_completeness_year(min_mags[i]), end_year)
		ax.text(np.mean([min_mags[i], mfd.max_mag]), annoty*10**-0.25, label, ha="center", va="center", size=12, bbox=bbox_props)

	## Apply plot limits
	pylab.axis((Mrange[0], Mrange[1], Freq_range[0], Freq_range[1]))

	pylab.xlabel("Magnitude ($M_%s$)" % mfd.Mtype[1].upper(), fontsize="x-large")
	label = {"en": "Annual number of earthquakes", "nl": "Aantal aardbevingen per jaar", "fr": "Nombre de seismes par annee"}[lang.lower()]
	pylab.ylabel(label, fontsize="x-large")
	pylab.title(title, fontsize='x-large')
	pylab.grid(True)
	font = FontProperties(size='medium')
	pylab.legend(loc=1, prop=font)
	ax = pylab.gca()
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size('large')

	if fig_filespec:
		default_figsize = pylab.rcParams['figure.figsize']
		default_dpi = pylab.rcParams['figure.dpi']
		if fig_width:
			fig_width /= 2.54
			dpi = dpi * (fig_width / default_figsize[0])

		pylab.savefig(fig_filespec, dpi=dpi)
		pylab.clf()
	else:
		pylab.show()


def alphabetalambda(a, b, M=0):
	"""
	Calculate alpha, beta, lambda from a, b, and M0.

	:param a:
		Float, a value of Gutenberg-Richter relation
	:param b:
		Float, b value of Gutenberg-Richter relation
	:param M:
		Float, magnitude for which to compute lambda (default: 0)

	:return:
		(alpha, beta, lambda) tuple
	"""
	alpha = a * np.log(10)
	beta = b * np.log(10)
	lambda0 = np.exp(alpha - beta*M0)
	# This is identical
	# lambda0 = 10**(a - b*M0)
	return (alpha, beta, lambda0)

