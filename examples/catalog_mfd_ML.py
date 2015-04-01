"""
Catalog MFD based on ML
"""

import eqcatalog


region = (-1.25, 8.75, 49.15, 53.30)
start_year = 1983
end_year = 2014
Mtype = "ML"
Mrelation = {}
Mmin = 2.0
Mmax = 6.5
dM = 0.2

raw_catalog = eqcatalog.read_catalogSQL(region, start_year, end_year, Mmin, Mmax)
completeness = eqcatalog.Completeness([start_year], [Mmin], Mtype)

## Declustering parameters
dc_method = "Cluster"
dc_window = "Uhrhammer1986"
dc_fa_ratio = 0.5

dc_catalog = raw_catalog.subselect_declustering(method=dc_method, window=dc_window,
							fa_ratio=dc_fa_ratio, Mtype=Mtype, Mrelation=Mrelation)

dc_catalog.plot_MFD(Mmin, Mmax, dM=dM, method="Weichert", Mtype=Mtype, Mrelation=Mrelation,
                 completeness=completeness, plot_completeness_limits=False)
