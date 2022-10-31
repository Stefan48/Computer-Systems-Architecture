"""
A command-line controlled coffee maker.
"""

import sys

import load_recipes

"""
Implement the coffee maker's commands. Interact with the user via stdin and print to stdout.

Requirements:
    - use functions
    - use __main__ code block
    - access and modify dicts and/or lists
    - use at least once some string formatting (e.g. functions such as strip(), lower(),
    format()) and types of printing (e.g. "%s %s" % tuple(["a", "b"]) prints "a b"
    - BONUS: read the coffee recipes from a file, put the file-handling code in another module
    and import it (see the recipes/ folder)

There's a section in the lab with syntax and examples for each requirement.

Feel free to define more commands, other coffee types, more resources if you'd like and have time.
"""

"""
Tips:
*  Start by showing a message to the user to enter a command, remove our initial messages
*  Keep types of available coffees in a data structure such as a list or dict
e.g. a dict with coffee name as a key and another dict with resource mappings (resource:percent)
as value
"""

# Commands
EXIT = "exit"
LIST_COFFEES = "list"
MAKE_COFFEE = "make"  #!!! when making coffee you must first check that you have enough resources!
HELP = "help"
REFILL = "refill"
RESOURCE_STATUS = "status"
commands = [EXIT, LIST_COFFEES, MAKE_COFFEE, REFILL, RESOURCE_STATUS, HELP]

# Coffee examples
ESPRESSO = "espresso"
AMERICANO = "americano"
CAPPUCCINO = "cappuccino"

# Resources examples
WATER = "water"
COFFEE = "coffee"
MILK = "milk"

# Coffee maker's resources - the values represent the fill percents
RESOURCES = {WATER: 100, COFFEE: 100, MILK: 100}

"""
Example result/interactions:

I'm a smart coffee maker
Enter command:
list
americano, cappuccino, espresso
Enter command:
status
water: 100%
coffee: 100%
milk: 100%
Enter command:
make
Which coffee?
espresso
Here's your espresso!
Enter command:
refill
Which resource? Type 'all' for refilling everything
water
water: 100%
coffee: 90%
milk: 100%
Enter command:
exit
"""


def main():

	""" Read recipes """
	recipes = load_recipes.read_recipes()
	#print(recipes)

	ingredients = {"water": 100, "coffee":100, "milk":100}
	#print(ingredients)

	print("I'm a simple coffee maker")

	while True:
		print("Enter command:")
		cmd = sys.stdin.readline().rstrip().upper()
		if cmd == "EXIT":
			break
		if cmd == "LIST":
			i = 0
			for key in recipes.keys():
				if i > 0:
					print(", " + key, end = '')
				else:
					print(key, end = '')
				i += 1
			print()
		if cmd == "STATUS":
			for key in ingredients.keys():
				print(key + ":" + str(ingredients[key]) + "%")
		elif cmd == "REFILL":
			print("Which resource? Type 'all' for refilling everything")
			ingredient = sys.stdin.readline().rstrip()
			if ingredient == "all":
				for key in ingredients.keys():
					ingredients[key] = 100
			elif not ingredient in ingredients.keys():
				print("Ingredient unknown")
			else:
				ingredients[ingredient] = 100
		elif cmd == "MAKE":
			print("Which coffee?")
			coffee = sys.stdin.readline().rstrip()
			if not coffee in recipes.keys():
				print("Sorry, can't do that")
			else:
				to_consume = recipes[coffee]
				#print(to_consume)
				ok = True
				for ingredient in to_consume:
					if ingredients[ingredient[0]] < ingredient[1]:
						ok = False
						break
				if ok:
					for ingredient in to_consume:
						ingredients[ingredient[0]] -= ingredient[1]
					print("Here's your " + coffee)
				else:
					print("Sorry, that's not available at the moment")
		elif cmd == "HELP":
			print("Enter one of the following commands:\nlist, status, refill, make, exit")
		else:
			print("Command unknown")

if __name__ == "__main__":
	main()



