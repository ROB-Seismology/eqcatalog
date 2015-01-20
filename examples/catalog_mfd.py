"""
"""

import datetime
import eqcatalog


## Selection parameters
region = (0, 8, 49.15, 52)
start_date = datetime.date(1350, 1, 1)
end_date = datetime.date(2014, 12, 31)

## Magnitude scaling
Mtype = "MW"
Mrelation = {"ML": "Ahorner1983", "MS": "Utsu2002"}

## Completeness
completeness = eqcatalog.completeness.Completeness_MW_201303a

## Mmax = mean of CEUS_COMP prior
Mmax = 7.2

## Declustering parameters
dc_method = "Cluster"
dc_window = "Uhrhammer1986"
dc_fa_ratio = 0.5

## MFD parameters
Mmin_mfd = completeness.min_mag
mfd_bin_width = 0.1


raw_catalog = eqcatalog.read_catalogSQL(region, start_date, end_date)
dc_catalog = raw_catalog.subselect_declustering(method=dc_method, window=dc_window,
							fa_ratio=dc_fa_ratio, Mtype=Mtype, Mrelation=Mrelation)
cc_catalog = dc_catalog.subselect_completeness(completeness=completeness,
								Mtype=Mtype, Mrelation=Mrelation, verbose=True)
cc_catalog.name = "ROB Catalog 1350-2014 (declustered)"

catalog_observed_mfd = cc_catalog.get_incremental_MFD(Mmin_mfd, Mmax, mfd_bin_width)
catalog_weichert_mfd = cc_catalog.get_estimated_MFD(Mmin_mfd, Mmax, mfd_bin_width,
				method="Weichert", b_val=None, Mtype=Mtype, Mrelation=Mrelation,
				completeness=completeness, verbose=True)

catalog_weichert_mfd.print_report()

#fig_filespec = None
fig_filespec = r"C:\Temp\Catalog_MFD.png"
cc_catalog.plot_MFD(Mmin_mfd, Mmax, dM=mfd_bin_width, method="Weichert", Mtype=Mtype,
					Mrelation=Mrelation, completeness=completeness, verbose=False,
					fig_filespec=fig_filespec)
