"""
	Bonus task: load all the available coffee recipes from the folder 'recipes/'
	File format:
		first line: coffee name
		next lines: resource=percentage

	info and examples for handling files:
		http://cs.curs.pub.ro/wiki/asc/asc:lab1:index#operatii_cu_fisiere
		https://docs.python.org/3/library/io.html
		https://docs.python.org/3/library/os.path.html
"""

import glob;

RECIPES_FOLDER = "recipes"

def read_recipes():
	recipes = {}
	filenames = glob.glob(RECIPES_FOLDER + "/*.txt")
	for filename in filenames:
		f = open(filename, 'r')
		coffee_name = f.readline().rstrip()
		for line in f:
			l = line.split('=')
			ingredient = (l[0], int(l[1].rstrip()))
			if coffee_name in recipes:
				recipes[coffee_name].append(ingredient)
			else:
				recipes[coffee_name] = [ingredient]
		f.close()
	return recipes
	
if __name__ == "__main__":
	print(read_recipes())
