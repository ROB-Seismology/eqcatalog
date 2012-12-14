import nhlib.mfd



class MFD:
	def __init__(self):
		pass

	def plot(self, completeness=None):
		print("Not yet implemented")


class EvenlyDiscretizedMFD(nhlib.mfd.EvenlyDiscretizedMFD, MFD):
	def __init__(self, min_mag, bin_width, occurrence_rates):
		nhlib.mfd.EvenlyDiscretizedMFD.__init__(self, min_mag, bin_width, occurrence_rates)


class TruncatedGRMFD(nhlib.mfd.TruncatedGRMFD, MFD):
	def __init__(self, min_mag, max_mag, bin_width, a_val, b_val, b_sigma):
		nhlib.mfd.TruncatedGRMFD.__init__(self, min_mag, max_mag, bin_width, a_val, b_val)
		self.b_sigma = b_sigma

	def to_evenly_discretized_mfd(self):
		pass


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

