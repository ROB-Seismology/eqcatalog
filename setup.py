#!/usr/bin/env python

from setuptools import setup
import datetime


today = datetime.date.today()

setup(name = 'eqcatalog',
	version = '%d.%d.%d' % today.timetuple()[:3],
	description = 'ROB Earthquake Catalog python library',
	author = 'Kris Vanneste, Bart Vleminckx',
	author_email = 'kris.vanneste@oma.be',
	url = 'https://svn.seismo.oma.be/svn/seismo/eqcatalog/trunk',
	package_dir = {'eqcatalog': ''},
	packages = ['eqcatalog'],
	install_requires=['numpy', 'scipy', 'matplotlib', 'egenix-mx-base', 'gdal']
	)
