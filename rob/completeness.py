"""
Completeness definitions used at ROB
"""

from __future__ import absolute_import, division, print_function, unicode_literals


from ..completeness import Completeness


## NOTE: I think threshold magnitudes should be a multiple of dM (or dM/2)!
#Completeness_Leynaud = Completeness([1350, 1911, 1985], [4.75, 3.25, 1.75], "MS")
Completeness_Leynaud = Completeness([1350, 1911, 1985],
									[4.7, 3.3, 1.8], "MS")

Completeness_Rosset = Completeness([1350, 1926, 1960, 1985],
									[5.0, 4.0, 3.0, 1.8], "MS")

Completeness_MS_2020 = Completeness([1350, 1750, 1900, 1960, 1985],
									[4.9, 4.5, 4.0, 3.0, 1.9], "MS")

## Following relation is for MW based on conversion from ML using Ahorner (1983)
Completeness_MW_201303a = Completeness([1350, 1750, 1860, 1905, 1960, 1985],
										[5.2, 4.9, 4.5, 4.0, 3.0, 2.2], "MW")
Completeness_MW_201303a_ext = Completeness([1350, 1750, 1860, 1905, 1960, 1985, 1996],
										[5.2, 4.9, 4.5, 4.0, 3.0, 2.2, 1.8], "MW")

## Following relation is for MW based on conversion from ML using Reamer and Hinzen (2004)
Completeness_MW_201303b = Completeness([1350, 1750, 1860, 1905, 1960, 1985],
										[5.2, 4.9, 4.5, 3.9, 2.9, 2.0], "MW")
Completeness_MW_201303b_ext = Completeness([1350, 1750, 1860, 1905, 1960, 1985, 1996],
										[5.2, 4.9, 4.5, 3.9, 2.9, 2.0, 1.7], "MW")

DEFAULT_COMPLETENESS = Completeness_MW_201303a
