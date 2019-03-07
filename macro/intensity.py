# -*- coding: iso-Latin-1 -*-

from __future__ import absolute_import, division, print_function, unicode_literals


try:
	## Python 2
	basestring
	PY2 = True
except:
	## Python 3
	PY2 = False
	basestring = str

import numpy as np



__all__ = ["get_roman_intensity"]


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
		#if PY2:
		#	roman_intensity = roman_intensity.decode('ascii')
		roman_intensities.append(roman_intensity)
	if scalar:
		return roman_intensities[0]
	else:
		return roman_intensities
