"""
Parse variables from PHP file.

"""

def parse_php_vars(php_file):
	with open(php_file) as fp:
		php = fp.read()

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
	php_file = r"E:\Home\_kris\Python\seismo\eqcatalog\webenq\const_inqEN.php"
	php_vars = parse_php_vars(php_file)
	#for var, val in php_vars.items():
	#	print("%s: %s" % (var, val))
	#print php_vars['$form23']
	for item in sorted(php_vars.items()):
		print(item)
