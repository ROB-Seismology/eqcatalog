"""
Demonstrate some database inconsistencies related to macroseismic data
"""

import db.simpledb as simpledb
from seismodb_secrets import host, database, user, passwd


## Print macro_detail records where intensity_max = 13
"""
table_clause = 'macro_detail'
join_clause = [('LEFT JOIN', 'earthquakes',
				'macro_detail.id_earth = earthquakes.id_earth')]
column_clause = ['macro_detail.id_earth',
				'count(*)',
				'earthquakes.date',
				'earthquakes.name',
				# 'macro_detail.intensity_min',
				'macro_detail.intensity_max',
				'MIN(macro_detail.fiability)']
where_clause = 'macro_detail.intensity_max = 13'
group_clause = 'macro_detail.id_earth'
"""


## Print macro_detail records with non-existing commune ID
table_clause = 'macro_detail'
join_clause = [('LEFT JOIN', 'communes', 'macro_detail.id_com = communes.id')]
column_clause = ['macro_detail.id_com',
				'GROUP_CONCAT(macro_detail.id_earth SEPARATOR ",") AS id_earth']
where_clause = 'communes.id IS NULL'
group_clause = 'macro_detail.id_com'


seismodb = simpledb.MySQLDB(database, host, user, passwd)
seismodb.query(table_clause, column_clause, join_clause=join_clause,
				where_clause=where_clause, group_clause=group_clause,
				print_table=True)

