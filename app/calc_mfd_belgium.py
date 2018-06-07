"""
Compute MFD specifically for Belgium,
taking into account increasingly larger area around Belgium in function
of magnitude
"""

import os
import datetime
from collections import OrderedDict
import numpy as np
import mapping.layeredbasemap as lbm
import eqcatalog
import hazard.rshalib as rshalib


GIS_ROOT = r"D:\seismo-gis\collections"


def get_declustering_distance(mag, dc_window_name):
	"""
	Determine declustering distance for a given magnitude.

	:param mag:
		float, magnitude
	:param dc_window_name:
		str, name of declustering window definition, one of
		"GardnerKnopoff1974", "Uhrhammer1986", "Gruenthal2009"

	:return:
		float, distance in km
	"""
	dc_window = getattr(eqcatalog.declustering, dc_window_name+"Window")()
	t_window, s_window = dc_window.get(mag)
	return s_window


def create_buffer_polygon(gis_file, buffer_distance, show_plot=False):
	"""
	Create buffer polygon for a country or region.

	:param gis_file:
		str, full path to GIS file containing country border
		Note: the main polygon from the first record will be selected
	:param buffer_distance:
		float, buffer distance (in km)
	:param show_plot:
		bool, whether or not a plot of the country border and buffer
		should be shown
		(default: False)

	:return:
		OGR Geometry object
	"""
	import pylab
	import mapping.geotools.coordtrans as ct
	from mapping.geotools.readGIS import read_GIS_file

	## Read country polygon
	#gis_data = lbm.GisData(gis_file)
	#_, _, polygon_data = gis_data.get_data()
	recs = read_GIS_file(gis_file, out_srs=ct.lambert1972, verbose=False)
	geom = recs[0]['obj']
	polygon_data = lbm.MultiPolygonData.from_ogr(geom)

	## Select main polygon
	country_pg, pg_len = None, 0
	for p, pg in enumerate(polygon_data):
		if len(pg.lons) > pg_len:
			country_pg = pg
			pg_len = len(pg.lons)

	## Create buffer polygon
	if buffer_distance:
		buffer_pg = country_pg.create_buffer(buffer_distance * 1000)
	else:
		buffer_pg = country_pg

	if show_plot:
		pylab.plot(country_pg.lons, country_pg.lats)
		pylab.plot(buffer_pg.lons, buffer_pg.lats)
		pylab.show()

	## Reproject buffer polygon to lon, lat
	lons, lats = ct.transform_array_coordinates(ct.lambert1972, ct.wgs84,
												buffer_pg.lons, buffer_pg.lats)
	buffer_pg.lons = lons
	buffer_pg.lats = lats

	#pylab.plot(buffer_pg.lons, buffer_pg.lats)
	#pylab.show()

	return buffer_pg.to_ogr_geom()


## Print magnitudes / distances
#for mag in range(3, 7):
#	print mag, get_declustering_distance(mag, "GardnerKnopoff1974")



## Selection parameters
region = (0, 8, 49, 52)
start_date = datetime.date(1350, 1, 1)
end_date = datetime.date(2017, 12, 31)

## Magnitude scaling
Mtype = "MW"
#Mrelation = {"ML": "Ahorner1983", "MS": "Utsu2002"}
Mrelation = OrderedDict([("MS", "Utsu2002"), ("ML", "Ahorner1983")])

## Completeness
completeness = eqcatalog.completeness.Completeness_MW_201303a

## Mmax = mean of CEUS_COMP prior
Mmax = 6.5

## Declustering parameters
dc_method = "Cluster"
dc_window = "Uhrhammer1986"
dc_fa_ratio = 0.5

## MFD parameters
Mmin_mfd = completeness.min_mag
mfd_bin_width = 0.1

## GIS file containing country border
## Note: does not contain Belgian continental platform !
gis_filename = "Bel_border.TAB"
gis_file = os.path.join(GIS_ROOT, "Bel_administrative_ROB", "TAB", gis_filename)


## Read catalog
raw_catalog = eqcatalog.read_catalogSQL(region, start_date, end_date)
# TODO: find out why Verviers disappears after declustering...!
#dc_catalog = raw_catalog.subselect_declustering(method=dc_method, window=dc_window,
#							fa_ratio=dc_fa_ratio, Mtype=Mtype, Mrelation=Mrelation)
cc_catalog = raw_catalog.subselect_completeness(completeness=completeness,
								Mtype=Mtype, Mrelation=Mrelation, verbose=True)
cc_catalog.name = "ROB Catalog 1350-2014 (declustered)"
#cc_catalog = cc_catalog.sort()
print("Catalog: n=%d, Mmax=%.1f" % (len(raw_catalog), raw_catalog.get_Mmax(Mtype=Mtype,
										Mrelation=Mrelation)))

## Construct MFD
mag_bins = np.arange(Mmin_mfd, Mmax, mfd_bin_width)
inc_numbers = np.zeros_like(mag_bins)
elapsed_years = np.zeros_like(mag_bins)

dist_window = "GardnerKnopoff1974"
#dist_window = "Uhrhammer1986"
for m, mag in enumerate(mag_bins):
	subcatalog = cc_catalog.subselect(Mmin=mag, Mmax=mag+mfd_bin_width-1E-5,
									Mtype=Mtype, Mrelation=Mrelation)
	buffer_distance = get_declustering_distance(mag, dist_window)
	buffer_pg = create_buffer_polygon(gis_file, buffer_distance)
	subcatalog = subcatalog.subselect_polygon(buffer_pg)
	subcatalog = subcatalog.sort()
	inc_numbers[m] = len(subcatalog)
	if len(subcatalog):
		et = end_date.year + 1 - subcatalog.get_fractional_years()[-1]
	else:
		et = np.nan
	elapsed_years[m] = et
	print("M=%.1f: n=%d, ET=%.3f yr" % (mag, len(subcatalog), et))
print("Total number of earthquakes counted: %d" % np.sum(inc_numbers))

bins_timespans = completeness.get_completeness_timespans(mag_bins, end_date)
inc_freqs = inc_numbers / bins_timespans

observed_mfd = rshalib.mfd.EvenlyDiscretizedMFD(Mmin_mfd+mfd_bin_width/2,
											mfd_bin_width, inc_freqs, Mtype)
## wLSQc fit is better for lower magnitudes
#fit_method = "Weichert"
fit_method = "wLSQc"
fitted_mfd = observed_mfd.to_truncated_GR_mfd(completeness, end_date, method=fit_method)
fitted_mfd.print_report()

mfd_list = [observed_mfd, fitted_mfd]
labels = ["Observed", "%s fit (a=%.3f, b=%.3f)" % (fit_method, fitted_mfd.a_val,
															fitted_mfd.b_val)]
title = "MFD Belgium (dist. window: %s)" % dist_window
fig_filespec = r"C:\Temp\MFD_Bel_%s_%s.PNG" % (dist_window, fit_method)
#fig_filespec = None
rshalib.mfd.plot_MFD(mfd_list, labels=labels, title=title, fig_filespec=fig_filespec)


## Compute Poisson and time-dependent probabilities
from prettytable import PrettyTable
import scipy.stats
from hazard.rshalib.utils.poisson import PoissonTau
t = 5.
print("Scenario probabilities (t=%d yr)" % t)

## Weichert fit is better for large magnitudes
fitted_mfd = observed_mfd.to_truncated_GR_mfd(completeness, end_date, method="Weichert")
sigma_mfd1 = fitted_mfd.get_mfd_from_b_val(fitted_mfd.b_val + fitted_mfd.b_sigma)
sigma_mfd2 = fitted_mfd.get_mfd_from_b_val(fitted_mfd.b_val - fitted_mfd.b_sigma)

col_names = ["Magnitude", "Return period (yr)", "Elapsed time (yr)", "P (Poisson)", "P (time-dep)"]
tab_inc = PrettyTable(col_names)
tab_cumul = PrettyTable(col_names)

scenario_mags = [4.6, 5.3, 6.0]
for scen_mag in scenario_mags:
	[idx] = np.argwhere(np.isclose(mag_bins, scen_mag))
	## Using incremental rates
	Tinc = 1. / np.sum(fitted_mfd.occurrence_rates[idx:idx+4])
	T1inc = 1. / np.sum(sigma_mfd1.occurrence_rates[idx:idx+4])
	T2inc = 1. / np.sum(sigma_mfd2.occurrence_rates[idx:idx+4])
	Tinc_sigma = np.mean([np.abs(Tinc - T1inc), np.abs(Tinc - T2inc)])
	## Using cumulative rates
	Tcumul = 1. / fitted_mfd.get_cumulative_rates()[idx]
	T1cumul = 1. / sigma_mfd1.get_cumulative_rates()[idx]
	T2cumul = 1. / sigma_mfd2.get_cumulative_rates()[idx]
	Tcumul_sigma = np.mean([np.abs(Tcumul - T1cumul), np.abs(Tcumul - T2cumul)])
	## Take elapsed time as minimum elapsed time in window -2/+3 bins
	ET = np.nanmin(elapsed_years[idx-2:idx+4]) + 0.5
	PTinc = PoissonTau(Tinc)
	PTcumul = PoissonTau(Tcumul)
	Ppois_inc = PTinc.get_prob_one_or_more(t)
	Ppois_cumul = PTcumul.get_prob_one_or_more(t)

	## Time-dependent probability
	t0 = 2018.5
	t1 = 2018 + t
	et0 = ET
	et1 = et0 + t
	Z0inc = (et0 - Tinc) / Tinc_sigma
	Z1inc = (et1 - Tinc) / Tinc_sigma
	Z0cumul = (et0 - Tcumul) / Tcumul_sigma
	Z1cumul = (et1 - Tcumul) / Tcumul_sigma
	N = scipy.stats.norm()
	P0inc = N.cdf(Z0inc)
	P1inc = N.cdf(Z1inc)
	P0cumul = N.cdf(Z0cumul)
	P1cumul = N.cdf(Z1cumul)
	Pcond_inc = (P1inc - P0inc) / (1 - P0inc)
	Pcond_cumul = (P1cumul - P0cumul) / (1 - P0cumul)

	#print("M=%.1f, T=%.1f +/- %.1 yr, P=%.3f, ET=%.1f yr" % (scen_mag, T, T_sigma, Ppois, ET))
	row = ["%.1f" % scen_mag,
			"%.1f +/- %.1f" % (Tinc, Tinc_sigma),
			"%.1f" % ET,
			"%.5f" % Ppois_inc,
			"%.5f" % Pcond_inc]
	tab_inc.add_row(row)

	row = ["%.1f" % scen_mag,
			"%.1f +/- %.1f" % (Tcumul, Tcumul_sigma),
			"%.1f" % ET,
			"%.5f" % Ppois_cumul,
			"%.5f" % Pcond_cumul]
	tab_cumul.add_row(row)


print("Using incremental rates:")
print(tab_inc)
print()
print("Using cumulative rates:")
print(tab_cumul)
