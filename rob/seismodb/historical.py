 #-*- coding: iso-Latin-1 -*-

"""
seismodb
Python module to retrieve information from the ROB seismology database
======================================================================
Author: Kris Vanneste, Royal Observatory of Belgium.
Date: Apr 2008.

Required modules:
	Third-party:
		MySQLdb
	ROB:
		db_secrets (python file containing host, user, password and database name)
"""


from __future__ import absolute_import, division, print_function, unicode_literals



__all__ = ["query_historical_texts"]


def query_historical_texts(id_earth, id_com, include_doubtful=False, verbose=False):
	"""
	Query historical database for texts corresponding to paritucular
	earthquake and commune

	:param id_earth:
		int, earthquake ID
	:param id_com:
		int, commune ID
	:param include_doubtful:
		bool, whether or not to include doubtful records
	:param verbose:
		bool, whether or not to print SQL query

	:return:
		list of dicts
	"""
	from .base import query_seismodb_table

	table_clause = 'historical_text'

	column_clause = ['id_commune AS id_com',
					'id_earth',
					'historical_text.id_text',
					'historical_text.id_source',
					'historical_text.id_historian',
					'historical_tp.doubtful',
					'historical_text.origin',
					'historical_text.text',
					'historical_text.translation',
					'historical_text.remark',
					'historical_place.id_place',
					'historical_place.name',
					'historical_source.name',
					'historical_source.date',
					'historical_source.edition',
					'historical_source.redaction_place',
					'historical_source.remark',
					'id_bibliography']

	join_clause = [('LEFT JOIN', 'historical_tp',
					'historical_text.id_text=historical_tp.id_text'),
					('JOIN', 'historical_place',
					'historical_tp.id_place=historical_place.id_place'),
					('JOIN', 'historical_source',
					'historical_text.id_source = historical_source.id_source')]

	where_clause = 'id_earth = %d AND id_commune = %d'
	where_clause %= (id_earth, id_com)
	if not include_doubtful:
		where_clause += ' AND doubtful = 0'

	return query_seismodb_table(table_clause, column_clause=column_clause,
								join_clause=join_clause, where_clause=where_clause,
								verbose=verbose)



# TODO: historical texts
"""
SELECT	c2.id_com,
       c2.commune_name,
       COUNT(distinct ht.id_text) AS cpt,
       GROUP_CONCAT(distinct ht.id_text) AS id_text
FROM (
   SELECT 	c.id 			AS id_com,
           c.name 			AS commune_name,
           c.country,
           c.id_main 		AS cmain
     FROM 	macro_detail 	AS md,
           communes 		AS c
    WHERE 	md.id_com 	= c.id
      AND 	md.id_earth = 89
) AS c2
LEFT JOIN 	communes 			AS c3	ON (c2.id_com 	= c3.id 		OR c2.id_com 	= c3.id_main)
LEFT JOIN 	historical_place	AS hp 	ON c3.id 		= hp.id_commune
LEFT JOIN 	historical_tp 		AS htp 	ON hp.id_place 	= htp.id_place
LEFT JOIN 	historical_text 	AS ht 	ON htp.id_text 	= ht.id_text
   WHERE 	c3.id IS NOT NULL
     AND 	ht.id_earth 	= 89
     AND 	ht.validation 	= 1
     AND 	ht.public_data 	= 1
GROUP BY 	c2.id_com
ORDER BY 	c2.country , c2.commune_name
"""
