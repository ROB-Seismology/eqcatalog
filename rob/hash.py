# -*- coding: utf-8 -*-
"""
Convert between earthquake IDs and hashes

Created on Wed Dec 16 23:07:18 2020

@author: kris
"""

from __future__ import absolute_import, division, print_function, unicode_literals


__all__ = ['id2hash', 'hash2id']


SALT = "8dffaf6e-fb3a-11e5-86aa-5e5517507c66"
MIN_LENGTH = 9
ALPHABET = "abcdefghijklmnopqrstuvwxyz1234567890"


def id2hash(id_earth):
	"""
	Convert id_earth to hash

	:param id_earth:
		int: ROB earthquake ID

	:return:
		str
	"""
	import hashids

	hi = hashids.Hashids(salt=SALT, min_length=MIN_LENGTH, alphabet=ALPHABET)
	hash = hi.encode(id_earth)
	return hash


def hash2id(hash):
	"""
	Convert hash to id_earth

	:param hash:
		str, hashed earthquake ID

	:return:
		int
	"""
	import hashids

	hi = hashids.Hashids(salt=SALT, min_length=MIN_LENGTH, alphabet=ALPHABET)
	hash = hi.decode(hash)
	return hash


