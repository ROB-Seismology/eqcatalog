# -*- coding: iso-Latin-1 -*-

"""
Earthquake types
"""

from __future__ import absolute_import, division, print_function, unicode_literals


__all__ = ["EARTHQUAKE_TYPES", "get_earthquake_type"]


EARTHQUAKE_TYPES = {}

EARTHQUAKE_TYPES['NL'] = {
	"uk": "Onbepaald",
	"ke": "Aardbeving",
	"se": "Vermoedelijke aardbeving",
	#"kr": "Instorting van gesteentemassa",
	#"sr": "Vermoedelijke instorting van gesteentemassa",
	"kr": "Autoklaas",
	"sr": "Vermoedelijke autoklaas",
	"ki": "Geïnduceerde gebeurtenis",
	"si": "Vermoedelijke geïnduceerde gebeurtenis",
	"km": "Mijnexplosie",
	"sm": "Vermoedelijke mijnexplosie",
	"kx": "Experimentele explosie",
	"sx": "Vermoedelijke experimentele explosie",
	"kn": "Nucleaire explosie",
	"sn": "Vermoedelijke nucleaire explosie",
	"ls": "Aardverschuiving",
	"ex": "Oefening",
	"sb": "Supersone knal",
	"cb": "Gecontroleerde explosie",
	"qb": "Explosie in groeve",
	'scb': "Vermoedelijke gecontroleerde explosie",
	'sqb': "Vermoedelijke explosie in groeve"}

EARTHQUAKE_TYPES['FR'] = {
	"uk": "Inconnu",
	"ke": "Tremblement de terre",
	"se": "Tremblement de terre présumé",
	"kr": "Effondrement",
	"sr": "Effondrement présumé",
	"ki": "Événement induit",
	"si": "Événement induit présumé",
	"km": "Explosion minière",
	"sm": "Explosion minière présumée",
	"kx": "Explosion expérimentale",
	"sx": "Explosion expérimentale présumée",
	"kn": "Explosion nucléaire",
	"sn": "Explosion nucléaire présumée",
	"ls": "Éboulement",
	"ex": "Exercice",
	"sb": "Bang supersonique",
	"cb": "Explosion contrôlée",
	"qb": "Tir de carrière",
	'scb': "Explosion contrôlée présumé",
	'sqb': "Tir de carrière présumé"}

EARTHQUAKE_TYPES['EN'] = {
    "uk": "Unknown",
    "ke": "Earthquake",
    "se": "Suspected earthquake",
    "kr": "Rockburst",
    "sr": "Suspected rockburst",
    "ki": "Induced event",
    "si": "Suspected induced event",
    "km": "Mine explosion",
    "sm": "Suspected mine explosion",
    "kx": "Experimental explosion",
    "sx": "Suspected experimental explosion",
    "kn": "Nuclear explosion",
    "sn": "Suspected nuclear explosion",
    "ls": "Landslide",
    "ex": "Exercise",
    'sb': "Sonic Boom",
    'cb': "Controlled explosion",
    'qb': "Quarry blast",
    'scb': "Suspected controlled explosion",
    'sqb': "Suspected quarry blast"}

EARTHQUAKE_TYPES['DE'] = {
    "uk": "Unbekannt",
    "ke": "Erdbeben",
    "se": "Vermutliches Erdbeben",
    "kr": "Gebirgsschlag",
    "sr": "Vermutlicher Gebirgsschlag",
    "ki": "Induziertes Ereignis",
    "si": "Vermutliches induziertes Ereignis",
    "km": "Bergbauexplosion",
    "sm": "Vermutliche Bergbauexplosion",
    "kx": "Versuchsexplosion",
    "sx": "Vermutliche Versuchsexplosion",
    "kn": "Atomexplosion",
    "sn": "Vermutliche Atomexplosion",
    "ls": "Bergrutsch",
    "ex": "Übung",
    "sb": "Überschallknall",
    "cb": "Kontrollierte Explosion",
    "qb": "Sprengung im Steinbruch",
    "scb": "Vermutliche kontrollierte Explosion",
    "sqb": "Vermutliche Sprengung im Steinbruch"}


def get_earthquake_type(code, lang='EN'):
	"""
	Get full earthquake type name in given language

	:param code:
		str, earthquake type code
	:param lang:
		str, language, one of 'EN', 'NL', 'FR', 'DE'

	:return:
		str
	"""
	return EARTHQUAKE_TYPES[lang.upper()].get(code, '')



if __name__ == "__main__":
	## Write translated earthquake type names to database
	import db.simpledb as simpledb
	from secrets.seismodb import (host, database, user_rw, passwd_rw)

	seismodb = simpledb.MySQLDB(database, host, user_rw, passwd_rw)
	table_name = 'earthquake_types'
	id_col_name = 'code'

	for lang in ['NL', 'FR', 'DE'][:1]:
		col_name = 'name_%s' % lang.lower()
		eq_type_dict = EARTHQUAKE_TYPES[lang]
		codes, col_values = eq_type_dict.keys(), eq_type_dict.values()
		#seismodb.update_column(table_name, col_name, col_values, id_col_name, codes)
