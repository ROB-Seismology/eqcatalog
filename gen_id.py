# -*- coding: utf-8 -*-
"""
Generate event IDs similar to SeisComP
Created on Fri Apr 23 14:35:18 2021

@author: kris
"""


import numpy as np


NUM_MILLISECS_PER_YEAR = float(370 * 24 * 60 * 60 * 1000)


__all__ = ['generate_event_id', 'generate_unique_event_id',
			'generate_event_ids']


def encode(x, chars, len_code):
	"""
	Encode time in milliseconds to sequence of characters

	:param x:
		int, time in milliseconds
	:param chars:
		str, containing characters to be used for encoding
	:param len_code:
		int, number of characters to use for code

	:return:
		(code, resolution) tuple
		- code: generated code
		- resolution: resolution (in milliseconds) corresponding to code length
	"""
	len_code = len_code or 4
	num_chars = len(chars)
	X = NUM_MILLISECS_PER_YEAR

	num_combinations = np.power(num_chars, len_code)

	resolution = X / num_combinations
	resolution = resolution or 1

	if X >= num_combinations:
		x /= resolution
	else:
		x *= (num_combinations / X)

	code = ''
	for i in range(len_code):
		d, r = np.divmod(x , num_chars)
		code += chars[int(r)]
		x = d

	return code[::-1], np.ceil(resolution)


def encode_text(x, len_code):
	"""
	Encode time in milliseconds to sequence of characters of the alphabet

	:param x:
	:param len_code:
		see :func:`encode`

	:return:
		see :func:`encode`
	"""
	return encode(x, "abcdefghijklmnopqrstuvwxyz", len_code)


def encode_int(x, len_code):
	"""
	Encode time in milliseconds to sequence of integers

	:param x:
	:param len_code:
		see :func:`encode`

	:return:
		see :func:`encode`
	"""
	return encode(x, "0123456789", len_code)


def encode_hex(x, len_code):
	"""
	Encode time in milliseconds to sequence of hexadecimals

	:param x:
	:param len_code:
		see :func:`encode`

	:return:
		see :func:`encode`
	"""
	return encode(x, "0123456789abcdef", 16, len)


def generate_event_id(datetime, prefix, pattern, offset=0):
	"""
	Generate ID for single event

	:param datetime:
		datetime specification understood by eqcatalog.time
	:param prefix:
		str, ID prefix
	:param pattern:
		str, ID pattern
		(default: '%p%Y%4c')
	:param offset:
		int or float, offset (in milliseconds) to apply to datetime
		(to try finding an ID that doesn't exist yet)

	:return:
		(event_id, resolution) tuple
		- event_id (str)
		- resolution (float): resolution in milliseconds corresponding to
		  given pattern
	"""
	from . import time as timelib

	datetime = timelib.as_np_datetime(datetime)
	year = timelib.to_year(datetime)
	x = timelib.timespan('%d-01-01' % year, datetime, unit='ms') + offset

	event_id = ''
	iterator = iter(range(len(pattern)))
	for i in iterator:
		if pattern[i] != '%':
			event_id += pattern[i]
		else:
			i = next(iterator)
			len_code = 0
			while i < len(pattern):
				if '0' <= pattern[i] <= '9':
					len_code *= 10
					len_code += int(pattern[i])
					i = next(iterator)
					continue
				elif pattern[i] == '%':
					event_id += pattern[i]
				elif pattern[i] in ('c', 'C'):
					code, resolution = encode_text(x, len_code)
					if pattern[i] == 'C':
						code = code.upper()
					event_id += code
				elif pattern[i] == 'd':
					code, resolution = encode_int(x, len_code)
					event_id += code
				elif pattern[i] in ('x', 'X'):
					code, resolution = encode_hex(x, len_code)
					if pattern[i] == 'X':
						code = code.upper()
					event_id += code
				elif pattern[i] == 'Y':
					event_id += '%04d' % year
				elif pattern[i] == 'p':
					event_id += prefix
				else:
					return ''
				break

	return event_id, resolution


def generate_unique_event_id(datetime, prefix, pattern, blacklist, verbose=False):
	"""
	Generate unique ID for single event, taking into account previously
	generated event IDs

	:param datetime:
	:param prefix:
	:param pattern:
		see :func:`generate_event_id`
	:param blacklist:
		list containing IDs of other events that should be avoided
	:param verbose:
		bool, whether or not to print messages when duplicate IDs are encountered
		(default: False)

	:return:
		(event_id, resolution) tuple
		- event_id (str)
		- resolution (float): resolution in milliseconds corresponding to
		  given pattern
	"""
	event_id, resolution = generate_event_id(datetime, prefix, pattern)
	i = 0
	while event_id in blacklist:
		i += 1
		if verbose:
			print('Duplicate ID: %s, attempt #%d' % (event_id, i))
		offset = i * resolution + 1
		event_id, resolution = generate_event_id(datetime, prefix, pattern, offset=offset)

	return event_id


def generate_event_ids(datetimes, prefix, pattern='%p%Y%4c', verbose=False):
	"""
	Generate unique IDs for multiple events

	:param datetimes:
		list with datetime specifications understood by eqcatalog.time
	:param prefix:
		str, ID prefix
	:param pattern:
		str, ID pattern
		(default: '%p%Y%4c')
	:param verbose:
		bool, whether or not to print messages when duplicate IDs are encountered
		(default: False)

	:return:
		list of strings
	"""
	event_ids = []
	for dt in np.sort(datetimes):
		event_id = generate_unique_event_id(dt, prefix, pattern, event_ids,
													verbose=verbose)
		event_ids.append(event_id)

	return event_ids

