"""
Module containing declustering methods and windows
"""


import datetime
import numpy as np

from nhlib.geo.geodetic import geodetic_distance as distance

from eqcatalog import EQCatalog


class DeclusteringMethod():
	"""
	Class implementing declustering method
	"""
	def decluster(self):
		"""
		:returns:
		np array, declustering index
		"""
		pass


class WindowMethod(DeclusteringMethod):
	"""
	Class implementing window declustering method.
	"""
	
	def decluster(self, magnitudes, datetimes, lons, lats, window, fa_ratio):
		"""
		:param magnitudes:
			np array, List of magnitudes in chronological order
		:param datetimes:
			np array, List of datetimes in chronological order
		:param lons:
			np array, List of longitudes in chronological order
		:param lats:
			np array, List of latitudes in chronological order
		:param window:
			instance of :class:`DeclusteringWindow`
		:fa_ratio:
			float, ratio of foreshock window to aftershock window
		"""
		## get order of descending magnitudes
		order = np.argsort(magnitudes)[::-1]
		## create declustering index
		d_index = np.ones_like(order)
		## loop over magnitudes in descending order
		for i in order:
			## if earthquake is not marked as triggered
			if d_index[i] == 1:
				## get time and distance window
				t_window, d_window = window.get(magnitudes[i])
				## make datetime object from time window
				t_window = datetime.timedelta(days=t_window)
				## create time window index
				in_t_window = np.logical_and(
					datetimes >= (datetimes[i] - datetime.timedelta(seconds=fa_ratio*t_window.total_seconds())),
					datetimes <= (datetimes[i] + t_window),
				)
				## create distance window index
				in_d_window = distance(lons[i], lats[i], lons, lats) <= d_window
				## create window index
				in_window = np.logical_and(in_t_window, in_d_window)
				## remove earthquake from window index
				in_window[i] = 0
				## apply window to declustering index
				d_index[in_window] = 0
				
		return d_index


class ClusterMethod(DeclusteringMethod):
	"""
	Class implementing cluster declustering method.
	"""
	
	def decluster(self, magnitudes, datetimes, lons, lats, window, fa_ratio):
		"""
		:param magnitudes:
			np array, List of magnitudes in chronological order
		:param datetimes:
			np array, List of datetimes in chronological order
		:param lons:
			np array, List of longitudes in chronological order
		:param lats:
			np array, List of latitudes in chronological order
		:param window:
			instance of :class:`DeclusteringWindow`
		:fa_ratio:
			float, ratio of foreshock window to aftershock window
		"""
		## create declustering index
		declustering = np.ones_like(magnitudes)
		## loop over magnitudes
		for i, magnitude in enumerate(magnitudes):
			## get time and distance window
			t_window, d_window = window.get(magnitude)
			## make datetime object from time window
			t_window = datetime.timedelta(days=t_window)
			## create window index for equal or smaller earthquakes
			smaller_magnitude = magnitudes <= magnitude
			## create time window index
			in_t_window = np.logical_and(
				datetimes >= (datetimes[i] - datetime.timedelta(seconds=fa_ratio*t_window.total_seconds())),
				datetimes <= (datetimes[i] + t_window),
			)
			## create distance window index
			in_d_window = distance(lons[i], lats[i], lons, lats) <= d_window
			## create window index
			in_window = np.logical_and(in_t_window, in_d_window)
			## remove earthquake from window index if not marked as triggered
			if declustering[i] == 1:
				in_window[i] = 0
			## create window index for smaller earthquakes
			smaller_magnitude_window = np.logical_and(in_window, smaller_magnitude)
			## apply window to declustering index
			declustering[smaller_magnitude_window] = 0
		return declustering


class DeclusteringWindow():
	"""
	Class implementing declustering window
	"""
	def get(self, magnitude):
		"""
		:param magnitude:
			float, magnitude for wich to calculate window.
		
		:returns:
			(float, float), defining time window (in days) and distance window (in km)
		"""
		pass
	

class GardnerKnopoff1974Window(DeclusteringWindow):
	"""
	Class implementing Gardner-Knopoff (1974) declustering window
	"""
	def get(self, magnitude):
		if magnitude >= 6.5:
			t_window = 10**(0.0320*magnitude+2.7389)
		else:
			t_window = 10**(0.5409*magnitude-0.5470)
		s_window = 10**(0.1238*magnitude+0.983)
		return t_window, s_window


class Uhrhammer1986Window(DeclusteringWindow):
	"""
	Class implementing Uhrhammer (1986) declustering window
	"""
	def get(self, magnitude):
		t_window = np.exp(-2.870+1.235*magnitude)
		s_window = np.exp(-1.024+0.804*magnitude)
		return t_window, s_window