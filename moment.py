
"""
Conversion of seismic moment to/from moment magnitude
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


__all__ = ["moment_to_mag", "mag_to_moment"]


def moment_to_mag(moment, unit='N.m'):
	"""
	Convert seismic moment to moment magnitude

	:param moment:
		array-like, seismic moment (in dyn.cm)
	:param unit:
		str, moment unit, either 'dyn.cm' or 'N.m'
		(default: 'N.m')

	:return:
		array-like, moment magnitude
	"""
	base_term = (2./3) * np.log10(moment)
	if unit == 'dyn.cm':
		return base_term - 10.73
	elif unit == 'N.m':
		return base_term - 6.06
	else:
		raise Exception("Moment unit %s not supported!" % unit)


def mag_to_moment(mag, unit='N.m'):
	"""
	Convert moment magnitude to seismic moment

	:param mag:
		array-like, moment magnitude
	:param unit:
		str, moment unit, either 'dyn.cm' or 'N.m'
		(default: 'N.m')

	:return:
		array-like, seismic moment
	"""
	mag = np.asarray(mag)

	if unit == 'dyn.cm':
		return 10**(1.5*mag + 16.095)
	elif unit == 'N.m':
		return 10**(1.5*mag + 9.09)
	else:
		raise Exception("Moment unit %s not supported!" % unit)
