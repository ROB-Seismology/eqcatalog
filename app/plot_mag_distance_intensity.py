"""
Plot mag vs distance and intensity
"""

import os
import numpy as np
import mapping.layeredbasemap as lbm
import plotting.generic_mpl as generic_mpl
import eqcatalog


OUT_FOLDER = r'E:\Home\_kris\Project Proposals\2020 - ROB internal PhD grants'

cmap = lbm.cm.get_cmap('macroseismic', 'usgs')
norm = lbm.cm.get_norm('macroseismic', 'usgs')

aggregate_by = 'id_com'

data = {}

## Official / Historical
catalog = eqcatalog.rob.get_earthquakes_with_traditional_enquiries()

for data_type in ('historical', 'official'):
	print('%s macro data' % data_type.title())
	data[data_type] = {}
	intensities, distances, ML = [], [], []
	for eq in catalog:
		print(eq.ID)
		ML = eq.get_or_convert_mag('ML', Mrelation={'MS': 'IdentityMSC'})
		if not np.isnan(ML):
			ML_key = '%.1f' % ML
			if not ML_key in data[data_type]:
				data[data_type][ML_key] = {'intensities': [], 'distances': []}
			if data_type == 'official':
				agg_macro_info = eq.get_aggregated_official_macro_info(aggregate_by=aggregate_by)
			else:
				agg_macro_info = eq.get_aggregated_historical_macro_info(aggregate_by=aggregate_by)
			if agg_macro_info and len(agg_macro_info):
				d = agg_macro_info.get_epicentral_distances()
				idxs = agg_macro_info.intensities > 1
				data[data_type][ML_key]['intensities'].extend(list(agg_macro_info.intensities[idxs]))
				data[data_type][ML_key]['distances'].extend(list(d[idxs]))
			else:
				print('Skipping eq #%s' % eq.ID)


## DYFI
print('Online macro data')
data['dyfi'] = {}
catalog = eqcatalog.rob.get_earthquakes_with_online_enquiries()

intensities, distances, ML = [], [], []
for eq in catalog:
	print(eq.ID)
	ML_key = '%.1f' % eq.ML
	if not ML_key in data['dyfi']:
		data['dyfi'][ML_key] = {'intensities': [], 'distances': []}
	agg_macro_info = eq.get_aggregated_online_macro_info(aggregate_by=aggregate_by)
	if agg_macro_info and len(agg_macro_info):
		d = agg_macro_info.get_epicentral_distances()
		idxs = agg_macro_info.intensities > 1
		data['dyfi'][ML_key]['intensities'].extend(list(agg_macro_info.intensities))
		data['dyfi'][ML_key]['distances'].extend(list(d))
	else:
		print('Skipping eq #%s' % eq.ID)


## Plot
print('Plotting')
labels = ['Historical', 'Official', 'DYFI']
linestyles = ['']
markers = ['s', 'o', 'd']


## All intensities combined
plot_sets = []
marker_fill_colors = []
plot_labels = []
for key in labels:
	key = key.lower()
	distances, magnitudes, intensities = [], [], []
	for ML_key in data[key]:
		ML = float(ML_key)
		d = data[key][ML_key]['distances']
		int = data[key][ML_key]['intensities']
		distances.extend(d)
		intensities.extend(int)
		magnitudes.extend([ML] * len(d))
	plot_sets.append((distances, magnitudes))
	marker_fill_colors.append(cmap(norm(intensities)))
	plot_labels.append('%s (n=%d)' % (key.title(), len(distances)))

title = 'I = all'
print(title)
fig_filespec = os.path.join(OUT_FOLDER, 'mag_distance_I=all.png')
#fig_filespec = None
generic_mpl.plot_xy(plot_sets, linestyles=linestyles, labels=plot_labels,
					markers=markers, marker_fill_colors=marker_fill_colors,
					xlabel='Epicentral distance (km)', ylabel='Magnitude (ML)',
					xmin=0, xmax=300, ymin=0, ymax=7,
					title=title, fig_filespec=fig_filespec)

#exit()


## Individual intensities
for intensity in range(2, 8):
	plot_sets = []
	marker_fill_colors = []
	color = cmap(norm(intensity))
	plot_labels = []
	for key in labels:
		key = key.lower()
		distances, magnitudes = [], []
		for ML_key in data[key]:
			ML = float(ML_key)
			d = np.array(data[key][ML_key]['distances'])
			int = np.array(data[key][ML_key]['intensities'])
			idxs = np.isclose(np.round(int), intensity)
			d = list(d[idxs])
			distances.extend(d)
			magnitudes.extend([ML] * len(d))
		plot_sets.append((distances, magnitudes))
		marker_fill_colors.append([color])
		plot_labels.append('%s (n=%d)' % (key.title(), len(distances)))

	title = 'I = %d' % intensity
	print(title)
	fig_filespec = os.path.join(OUT_FOLDER, 'mag_distance_I=%d.png' % intensity)
	#fig_filespec = None
	generic_mpl.plot_xy(plot_sets, linestyles=linestyles, labels=plot_labels,
						markers=markers, marker_fill_colors=marker_fill_colors,
						xlabel='Epicentral distance (km)', ylabel='Magnitude (ML)',
						xmin=0, xmax=300, ymin=0, ymax=7,
						title=title, fig_filespec=fig_filespec)
