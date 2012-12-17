"""
This module extends functionality of nhlib.mfd
"""

import datetime
import numpy as np
import pylab
from matplotlib.font_manager import FontProperties
import nhlib.mfd



class MFD:
	## Probably not necessary
	def __init__(self):
		pass

	def plot(self, completeness=None):
		print("Not yet implemented")


class EvenlyDiscretizedMFD(nhlib.mfd.EvenlyDiscretizedMFD, MFD):
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

	@property
	def max_mag(self):
		return self.min_mag + (len(self.occurrence_rates) + 1) * self.bin_width

	def get_magnitude_bin_edges(self):
		return np.array(zip(*self.get_annual_occurrence_rates())[0])

	def get_magnitude_bin_centers(self):
		return self.get_magnitude_bin_edges() + self.bin_width / 2

	def get_cumulative_rates(self):
		return np.add.accumulate(self.occurrence_rates[::-1])[::-1]

	def divide(self, weights):
		weights = np.array(weights, dtype='f')
		weights /= np.add.reduce(weights)
		mfd_list = []
		for w in weights:
			occurrence_rates = np.array(self.occurrence_rates) * w
			mfd = EvenlyDiscretizedMFD(self.min_mag, self.bin_width, list(occurrence_rates), self.Mtype)
			mfd_list.append(mfd)
		return mfd_list

	def split(self, M):
		# TODO: check that M is multiple of bin_width
		if self.min_mag < M < self.max_mag:
			index = int(round((M - self.min_mag) / self.bin_width))
			occurrence_rates1 = list(self.occurrence_rates[:index])
			occurrence_rates2 = list(self.occurrence_rates[index:])
			mfd1 = EvenlyDiscretizedMFD(self.min_mag, self.bin_width, occurrence_rates1, self.Mtype)
			mfd2 = EvenlyDiscretizedMFD(M, self.bin_width, occurrence_rates2, self.Mtype)
			return [mfd1, mfd2]
		else:
			raise Exception("Split magnitude not in valid range!")

	def plot(self, color='k', label="", completeness=None, title=""):
		plot_MFD([self], colors=[color], labels=[label], completeness=completeness, title=title)


class TruncatedGRMFD(nhlib.mfd.TruncatedGRMFD, MFD):
	def __init__(self, min_mag, max_mag, bin_width, a_val, b_val, b_sigma, Mtype="MW"):
		nhlib.mfd.TruncatedGRMFD.__init__(self, min_mag, max_mag, bin_width, a_val, b_val)
		self.b_sigma = b_sigma
		self.Mtype = Mtype

	def __div__(self, other):
		if isinstance(other, (int, float)):
			N = 10**self.a_val
			a_val = np.log10(N / float(other))
			return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, a_val, self.b_val, self.b_sigma, self.Mtype)
		else:
			raise TypeError("Divisor must be integer or float")

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			N = 10**self.a_val
			a_val = np.log10(N * float(other))
			return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, a_val, self.b_val, self.b_sigma, self.Mtype)
		else:
			raise TypeError("Multiplier must be integer or float")

	def __add__(self, other):
		if isinstance(other, (TruncatedGRMFD, EvenlyDiscretizedMFD)):
			return sum_MFDs([self, other])
		else:
			raise TypeError("Operand must be MFD")

	@property
	def occurrence_rates(self):
		return np.array(zip(*self.get_annual_occurrence_rates())[1])

	def get_magnitude_bin_centers(self):
		return np.array(zip(*self.get_annual_occurrence_rates())[0])

	def get_magnitude_bin_edges(self):
		return self.get_magnitude_bin_centers() - self.bin_width / 2

	def get_cumulative_rates(self, exponentially_tapered=True):
		if exponentially_tapered:
			#a, b = self.a_val, self.b_val
			#min_mag, max_mag = self.min_mag, self.max_mag
			#mags = self.get_magnitude_bin_edges()
			#return (10**(a-b*min_mag))*((10**(-1*b*mags)-10**(-1*b*max_mag))/(10**(-1*b*min_mag)-10**(-1*b*max_mag)))
			return np.add.accumulate(self.occurrence_rates[::-1])[::-1]
		else:
			return 10 ** (self.a_val - self.b_val * self.get_magnitude_bin_edges())

	def to_evenly_discretized_mfd(self):
		return EvenlyDiscretizedMFD(self.min_mag, self.bin_width, list(self.occurrence_rates))

	def divide(self, weights):
		weights = np.array(weights, dtype='f')
		weights /= np.add.reduce(weights)
		N = 10**self.a_val
		avalues = np.log10(weights * N)
		return [TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, aw, self.b_val, self.b_sigma, self.Mtype) for aw in avalues]

	def split(self, M):
		# TODO: check that M is multiple of bin_width
		if self.min_mag < M < self.max_mag:
			mfd1 = TruncatedGRMFD(self.min_mag, M, self.bin_width, self.a_val, self.b_val, self.b_sigma, self.Mtype)
			mfd2 = TruncatedGRMFD(M, self.max_mag, self.bin_width, self.a_val, self.b_val, self.b_sigma, self.Mtype)
			return [mfd1, mfd2]
		else:
			raise Exception("Split magnitude not in valid range!")

	def plot(self, color='k', label="", completeness=None, title=""):
		plot_MFD([self], colors=[color], labels=[label], completeness=completeness, title=title)


def sum_MFDs(mfd_list, weights=[]):
	## Note: take care with renormalized weights !
	if weights in ([], None):
		weights = np.ones(len(mfd_list), 'f')
	weights = np.array(weights) / len(mfd_list)
	bin_width = min([mfd.bin_width for mfd in mfd_list])
	Mtype = mfd_list[0].Mtype
	for mfd in mfd_list:
		if mfd.bin_width != bin_width:
			if isinstance(mfd, TruncatedGRMFD):
				mfd.bin_width = bin_width
			else:
				raise Exception("Bin widths not compatible!")
		if mfd.Mtype != Mtype:
			raise Exception("Magnitude types not compatible!")
	all_min_mags = set([mfd.min_mag for mfd in mfd_list])
	all_max_mags = set([mfd.max_mag for mfd in mfd_list])
	is_truncated = np.array([isinstance(mfd, TruncatedGRMFD) for mfd in mfd_list])
	## If all MFD's are TruncatedGR, and have same min_mag, max_mag, and b_val
	## return TrucatedGR, else return EvenlyDiscretized
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
			start_index = int(round((mfd.min_mag - min_mag) / bin_width))
			end_index = start_index + len(mfd.occurrence_rates)
			occurrence_rates[start_index:end_index] += (mfd.occurrence_rates * weights[i])
		return EvenlyDiscretizedMFD(min_mag, bin_width, list(occurrence_rates), Mtype)

def plot_MFD(mfd_list, colors=[], styles=[], labels=[], cumul=True, discrete=False, completeness=None, Mrange=(), Freq_range=(), want_exponential=False, title="", lang="en", fig_filespec=None, fig_width=0, dpi=300, verbose=False):
	"""
	Plot one or more magnitude-frequency distributions
	"""
	# TODO: add option to plot discrete / continuous and incremental / cumulative for each MFD
	if not colors:
		colors = ("r", "g", "b", "c", "m", "k")

	if not labels:
		labels = [""] * len(mfd_list)

	## Plot
	fig = pylab.figure()

	for i, mfd in enumerate(mfd_list):
		color = colors[i % len(colors)]

		if isinstance(mfd, EvenlyDiscretizedMFD):
			## Incremental
			label = labels[i] + " (incremental)"
			try:
				symbol = styles[i]
			except IndexError:
				symbol = 'o'
			ax = pylab.semilogy(mfd.get_magnitude_bin_centers(), mfd.occurrence_rates, symbol, label=label)
			pylab.setp(ax, markersize=10.0, markeredgewidth=1.0, markeredgecolor=color, markerfacecolor="None")

			## Cumulative
			label = labels[i] + " (cumulative)"
			ax = pylab.semilogy(mfd.get_magnitude_bin_edges(), mfd.get_cumulative_rates(), symbol, label=label)
			pylab.setp(ax, markersize=10.0, markeredgewidth=1.0, markeredgecolor='k', markerfacecolor=color)

		elif isinstance(mfd, TruncatedGRMFD):
			## Incremental
			label = labels[i]
			try:
				linestyle = styles[i]
			except:
				linestyle = "-"

			ax = pylab.semilogy(mfd.get_magnitude_bin_edges(), mfd.get_cumulative_rates(exponentially_tapered=True), color, linestyle=linestyle, lw=2, label=label)

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

