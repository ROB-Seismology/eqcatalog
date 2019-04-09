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
"""
table_clause = 'macro_detail'
#table_clause = 'macro_official_inquires'
join_clause = [('LEFT JOIN', 'communes', '%s.id_com = communes.id' % table_clause)]
column_clause = ['%s.id_com' % table_clause,
				'GROUP_CONCAT(%s.id_earth SEPARATOR ",") AS id_earth' % table_clause]
where_clause = 'communes.id IS NULL'
group_clause = '%s.id_com' % table_clause
"""


## Print macro_official_inquires records that are not in macro_detail
"""
table_clause = 'macro_official_inquires'
column_clause = ['id_earth',
				'GROUP_CONCAT(id_com SEPARATOR ",") AS id_com2']
where_clause = ('CONCAT(id_earth, id_com) NOT IN'
				' (SELECT CONCAT(id_earth, id_com)'
				' FROM macro_detail)')
group_clause = 'id_earth'
join_clause = ''
"""


## Print communes that are in macro_detail, but should also be in macro_official_inquires
## for a particular earthquake
id_earth = 509
table_clause = 'macro_detail'
column_clause = [#'ROW_NUMBER() OVER (ORDER BY id_com) row_num',
				'id_macro_detail',
				'id_com']
where_clause = ('id_earth = %d AND country="BE" AND id_com NOT IN'
				' (SELECT id_com FROM macro_official_inquires WHERE id_earth = %d)')
where_clause %= (id_earth, id_earth)
join_clause = [('LEFT JOIN', 'communes',
				'macro_detail.id_com = communes.id')]
group_clause = ''



## Find macroseismic forms that are not in macro_official_inquires
## There are only 2: # 5608 (OITB), # 2178 (OITC)
"""
form_type = 'B'
table_clause = 'official_inquires_type%c' % form_type.lower()
column_clause = 'id_form'
where_clause = ('id_form NOT IN (SELECT id_source FROM macro_official_inquires'
				' WHERE source = "OIT%c")')
where_clause %= form_type.upper()
join_clause = ''
group_clause = ''
"""


## Execute query
seismodb = simpledb.MySQLDB(database, host, user, passwd)
recs = seismodb.query(table_clause, column_clause, join_clause=join_clause,
				where_clause=where_clause, group_clause=group_clause,
				print_table=True, verbose=True)


## Compare number of official inquiries in macro_detail / macro_official_inquires
"""
from prettytable import PrettyTable
import eqcatalog

cat = eqcatalog.rob.get_earthquakes_with_official_enquiries()
eq_ids = cat.get_ids()
#eq_ids = [509]
tab = PrettyTable(['id_earth', 'date', 'macro_detail', 'macro_detail (BE)',
					'macro_official_inquires',
					'macro_detail + macro_official_inquires'])

table_clause1 = 'macro_detail'
table_clause2 = 'macro_official_inquires'
table_clause3 = table_clause1
column_clause = ['id_earth', 'Count(*) as count']
where_clause = 'id_earth IN (%s) AND country="BE"' % ','.join(map(str, eq_ids))
join_clause1 = [('LEFT JOIN', 'communes', '%s.id_com = communes.id' % table_clause1)]
join_clause2 = [('LEFT JOIN', 'communes', '%s.id_com = communes.id' % table_clause2)]
join_clause3 = join_clause1 + [('RIGHT JOIN', table_clause2,
								('CONCAT(%s.id_earth, %s.id_com) = CONCAT(%s.id_earth, %s.id_com)'
								% (table_clause1, table_clause1, table_clause2, table_clause2)))]
group_clause = 'id_earth'

db_recs1a = eqcatalog.rob.query_seismodb_table(table_clause1, column_clause,
						where_clause=where_clause.split(' AND')[0], group_clause=group_clause,
						join_clause='', verbose=True)
counts1a = {rec['id_earth']: rec['count'] for rec in db_recs1a}
db_recs1b = eqcatalog.rob.query_seismodb_table(table_clause1, column_clause,
						where_clause=where_clause, group_clause=group_clause,
						join_clause=join_clause1, verbose=True)
counts1b = {rec['id_earth']: rec['count'] for rec in db_recs1b}

db_recs2 = eqcatalog.rob.query_seismodb_table(table_clause2, column_clause,
						where_clause=where_clause, group_clause=group_clause,
						join_clause=join_clause2, verbose=True)
counts2 = {rec['id_earth']: rec['count'] for rec in db_recs2}

column_clause[0] = '%s.%s' % (table_clause3, column_clause[0])
where_clause = '%s.%s' % (table_clause3, where_clause)
group_clause = '%s.%s' % (table_clause3, group_clause)
db_recs3 = eqcatalog.rob.query_seismodb_table(table_clause3, column_clause,
						where_clause=where_clause, group_clause=group_clause,
						join_clause=join_clause3, verbose=True)
counts3 = {rec['id_earth']: rec['count'] for rec in db_recs3}

for id_earth in eq_ids:
	eq = cat.get_event_by_id(id_earth)
	row = [id_earth, eq.date, counts1a.get(id_earth, ''),
			counts1b.get(id_earth, ''), counts2.get(id_earth, ''),
			counts3.get(id_earth, '')]
	tab.add_row(row)
print(tab)
"""