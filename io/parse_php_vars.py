# -*- coding: iso-Latin-1 -*-

"""
Parse variables from PHP file.
"""

from __future__ import absolute_import, division, print_function, unicode_literals


def parse_php_vars(php_file):
	"""
	Parse PHP file to extract variables and their values

	:param php_file:
		str, full path to file containing PHP code

	:return:
		dict, mapping variable names (with leading '$') to values
	"""
	with open(php_file, 'rb') as fp:
		php = fp.read()
	php = php.decode('Latin-1')

	php_lines = [line for line in php.split(';')]
	php_vars= {}
	for php_line in php_lines:
		if '=' in php_line:
			var, val = php_line.split('=')
			var = var.strip()
			val = val.strip()
			val = " ".join(val.split())
			if 'array' in val:
				val = val.replace('array', 'list').replace('(', '([').replace(')', '])')
				val = eval(val)
			else:
				val = val.replace('"', '')
			php_vars[var] = val

	return php_vars



if __name__ == "__main__":
	php_file = "E:\\Home\\_kris\\Python\\seismo\\eqcatalog\\webenq\\const_inqFR.php"
	php_vars = parse_php_vars(php_file)
	#for var, val in php_vars.items():
	#	print("%s: %s" % (var, val))
	#print(php_vars['$form23'])
	for item in sorted(php_vars.items()):
		print(item)
