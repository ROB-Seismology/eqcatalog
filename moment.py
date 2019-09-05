
"""
Conversion of seismic moment to/from moment magnitude
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


__all__ = ["moment_to_mag", "mag_to_moment"]


def log_moment_to_mag(log_moment, unit='N.m'):
	"""
	Convert seismic moment to moment magnitude

	:param log_moment:
		float or float array, log10 of seismic moment
	:param unit:
		str, moment unit, either 'dyn.cm' or 'N.m'
		(default: 'N.m')

	:return:
		float or float array, moment magnitude
	"""
	base_term = (2./3) * log_moment
	if unit == 'dyn.cm':
		return base_term - 10.73
	elif unit == 'N.m':
		## Note: additional decimals added to obtain exactly factor of 1E-7
		## difference with dyn.cm
		return base_term - 6.063333333333334
	else:
		raise Exception("Moment unit %s not supported!" % unit)


def moment_to_mag(moment, unit='N.m'):
	"""
	Convert seismic moment to moment magnitude

	:param moment:
		float or float array, seismic moment
	:param unit:
		str, moment unit, either 'dyn.cm' or 'N.m'
		(default: 'N.m')

	:return:
		float or float array, moment magnitude
	"""
	log_moment = np.log10(moment)
	return log_moment_to_mag(log_moment, unit=unit)


def mag_to_log_moment(mag, unit='N.m'):
	"""
	Convert moment magnitude to log10 of seismic moment

	:param mag:
		float or float array, moment magnitude
	:param unit:
		str, moment unit, either 'dyn.cm' or 'N.m'
		(default: 'N.m')

	:return:
		float or float array, log10 of seismic moment
	"""
	mag = np.asarray(mag)

	if unit == 'dyn.cm':
		return 1.5*mag + 16.095
	elif unit == 'N.m':
		return 1.5*mag + 9.095
	else:
		raise Exception("Moment unit %s not supported!" % unit)


def mag_to_moment(mag, unit='N.m'):
	"""
	Convert moment magnitude to seismic moment

	:param mag:
		float or float array, moment magnitude
	:param unit:
		str, moment unit, either 'dyn.cm' or 'N.m'
		(default: 'N.m')

	:return:
		float or float array, seismic moment
	"""
	log_moment = mag_to_log_moment(mag, unit=unit)
	return 10**log_moment


def calc_rupture_radius(mag, stress_drop=3E+6):
	"""
	Compute radius of circular rupture from magnitude (and stress drop)

	:param mag:
		float or float array, magnitude
	:param stress_drop:
		float, stress drop (in Pa)
		(default: 3E+6 = interplate average)

	:return:
		float or float array, rupture radius (in km)
	"""
	moment = mag_to_moment(mag)
	R = ((7. / 16) * (moment / stress_drop))**(1./3)
	return R / 1000.


def calc_moment(rupture_area, displacement, rigidity=3E+10):
	"""
	Compute seismic moment from rupture area and average displacement

	:param rupture_area:
		float or float array, rupture area (in km2)
	:param displacement:
		float or float array, average displacement over rupture area (in m)
	:param rigidity:
		float, rigidity of crust (in N/m2)
		(default: 3E+10)

	:return:
		float or float array, seismic moment (in N.m)
	"""
	## Convert rupture area from km2 to m2
	rupture_area *= 1E+6
	return rigidity * rupture_area * displacement


def estimate_fc_brune(moment, stress_drop=3E+6, VS=3500):
	"""
	Estimate corner frequency according to Brune (1970)

	:param moment:
		float, seismic moment (in N.m)
	:param stress_drop:
		float, stress drop (in Pa = bar * 1E+5)
	:param VS:
		float, shear-wave velocity in the crust near the source (in m/s)
		(default: 3500)

	:return:
		float, corner frequency in Hz
	"""
	fc = 0.37 * VS * ((16 * stress_drop) / (7 * moment))**(1./3)
	return fc
