"""
Test declustering
"""

import numpy as np
import eqcatalog

dc_window = eqcatalog.declustering.Gruenthal2009Window()

"""
Event 0: isolated M=3 foreshock
Event 1: M=3 foreshock
Event 2: M=2 foreshock in time window of event 1
Event 3: M=5 mainshock in time window of event 2
Event 4: M=3 aftershock in time window of mainshock
Event 5: M=4 aftershock in time window of mainshock, outside of time window of event 4
Event 6: M=3 aftershock in time window of event 5, but outside of time window of mainshock
Event 7: isolated M=2 aftershock
"""
lon, lat, depth = 5., 51., 0.
mags = np.array([3., 3., 2., 5., 3., 4., 3., 2.])
start_date = np.datetime64('2018-01-01')
time_delta_days = np.array([0, 30, 25, 5, 30, 180, 75, 35])
print(np.cumsum(time_delta_days) - np.cumsum(time_delta_days)[3])
time_deltas = np.cumsum(time_delta_days).astype('m8[D]')
datetimes = start_date + time_deltas

print(datetimes)
print(eqcatalog.time.fractional_time_delta(np.diff(datetimes), 'D'))
print(eqcatalog.time.fractional_time_delta(dc_window.get_time_window(mags), 'D'))


eq_list = []
for i in range(len(mags)):
	mag = mags[i]
	datetime = datetimes[i]
	date = eqcatalog.time.as_np_date(datetime)
	time = eqcatalog.time.to_py_time(datetime)
	eq = eqcatalog.LocalEarthquake(i, date, time, lon, lat, depth, mag={'MW': mag})
	eq_list.append(eq)
catalog = eqcatalog.EQCatalog(eq_list)

Mrelation = {}
fa_ratio = 0

print("Window Method")
## Expected result: [1,2], [3,4,5]
dc_method = eqcatalog.declustering.WindowMethod(fa_ratio)
dc_result = dc_method.analyze_clusters(catalog, dc_window, Mrelation, verbose=True)
for clust in dc_result.get_clusters():
	clust.to_catalog().print_list()
#dc_cat = dc_method.decluster_catalog(catalog, dc_window, fa_ratio, Mrelation)
#dc_cat.print_list()


print("Cluster Method")
## Expected result: [1,2,3,4,5,6]
dc_method = eqcatalog.declustering.ClusterMethod()
dc_result = dc_method.analyze_clusters(catalog, dc_window, Mrelation, verbose=True)
for clust in dc_result.get_clusters():
	clust.to_catalog().print_list()
