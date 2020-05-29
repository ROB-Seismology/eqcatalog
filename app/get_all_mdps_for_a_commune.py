"""
Extract all macroseismic data from the database that were recorded
in a particular commune
"""

import os
from prettytable import PrettyTable
from thirdparty.recipes.ptable_to_csv import ptable_to_csv
import eqcatalog
from eqcatalog.macro import (get_eq_intensities_for_commune_traditional,
							get_eq_intensities_for_commune_online)

OUT_FOLDER = r"C:\Temp"

zip_code = 1000
id_comm = eqcatalog.rob.seismodb.zip2ID(zip_code)

## Traditional data
eq_id_intensity_dict = {}
for as_main_commune in (False, True):
	for Imin_or_max in ('min', 'max'):
		eid = get_eq_intensities_for_commune_traditional(id_comm,
												as_main_commune=as_main_commune,
												min_or_max=Imin_or_max)
		for id_earth, intensities in eid.items():
			if not id_earth in eq_id_intensity_dict:
				eq_id_intensity_dict[id_earth] = {}
			comm_type = {True: 'main', False: 'comm'}[as_main_commune]
			if not comm_type in eq_id_intensity_dict[id_earth]:
				eq_id_intensity_dict[id_earth][comm_type] = {}
			if not Imin_or_max in eq_id_intensity_dict[id_earth][comm_type]:
				eq_id_intensity_dict[id_earth][comm_type][Imin_or_max] = intensities

columns = ['ID', 'Date', 'MS', 'ML', 'Imin (comm)', 'Imax (comm)',
			'Imin (main)', 'Imax (main)']
tab = PrettyTable(columns)
for id_earth in sorted(eq_id_intensity_dict.keys()):
	[eq] = eqcatalog.rob.query_local_eq_catalog_by_id(id_earth)
	Imin_comm = eq_id_intensity_dict[id_earth]['comm']['min']
	Imin_comm = ' '.join(map(str, Imin_comm))
	Imax_comm = eq_id_intensity_dict[id_earth]['comm']['max']
	Imax_comm = ' '.join(map(str, Imax_comm))
	Imin_main = eq_id_intensity_dict[id_earth]['main']['min']
	Imin_main = ' '.join(map(str, Imin_main))
	if ' ' in Imin_main:
		Imin_main = '('+Imin_main+')'
	Imax_main = eq_id_intensity_dict[id_earth]['main']['max']
	Imax_main = ' '.join(map(str, Imax_main))
	if ' ' in Imax_main:
		Imax_main = '('+Imax_main+')'
	row = [id_earth, str(eq.date), eq.MS, eq.ML, Imin_comm, Imax_comm,
			Imin_main, Imax_main]
	tab.add_row(row)

csv_file = os.path.join(OUT_FOLDER, 'Bxl_macro_traditional.csv')
#ptable_to_csv(tab, csv_file)
print(tab)


## DYFI
eq_id_intensity_dict = {}
for as_main_commune in (False, True):
	for agg_method in ('mean', 'dyfi'):
		eid = get_eq_intensities_for_commune_online(id_comm,
											as_main_commune=as_main_commune,
											agg_method=agg_method)
		for id_earth, intensities in eid.items():
			if not id_earth in eq_id_intensity_dict:
				eq_id_intensity_dict[id_earth] = {}
			comm_type = {True: 'main', False: 'comm'}[as_main_commune]
			if not comm_type in eq_id_intensity_dict[id_earth]:
				eq_id_intensity_dict[id_earth][comm_type] = {}
			if not agg_method in eq_id_intensity_dict[id_earth][comm_type]:
				eq_id_intensity_dict[id_earth][comm_type][agg_method] = round(intensities)

columns = ['ID', 'Date', 'MS', 'ML', 'CII mean (comm)', 'CII dyfi (comm)',
			'CII mean (main)', 'CII dyfi (main)']
tab = PrettyTable(columns)
for id_earth in sorted(eq_id_intensity_dict.keys()):
	[eq] = eqcatalog.rob.query_local_eq_catalog_by_id(id_earth)
	Imean_comm = eq_id_intensity_dict[id_earth]['comm']['mean']
	Idyfi_comm = eq_id_intensity_dict[id_earth]['comm']['dyfi']
	Imean_main = eq_id_intensity_dict[id_earth]['main']['mean']
	Idyfi_main = eq_id_intensity_dict[id_earth]['main']['dyfi']
	row = [id_earth, str(eq.date), eq.MS, eq.ML, Imean_comm, Idyfi_comm,
			Imean_main, Idyfi_main]
	tab.add_row(row)

csv_file = os.path.join(OUT_FOLDER, 'Bxl_macro_dyfi.csv')
#ptable_to_csv(tab, csv_file)
print(tab)
