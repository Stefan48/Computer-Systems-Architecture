"""
    Basic thread handling exercise:

    Use the Thread class to create and run more than 10 threads which print their name and a random
    number they receive as argument. The number of threads must be received from the command line.

    e.g. Hello, I'm Thread-96 and I received the number 42

"""

import sys
from random import randint
from threading import Thread

class PrintThread(Thread):
	""" Thread class to print its name and a number received as parameter """
	def __init__(self, name, number):
		Thread.__init__(self)
		self.name = name
		self.number = number
		
	def run(self):
		print("Thread %s has number %d" % (self.name, self.number))
		
def main():

	if(len(sys.argv) < 2):
		print("Run: python3 task2.py <nr_threads>")
		return


	nr_threads = int(sys.argv[1])

	threads = []

	for i in range(nr_threads):
		threads.append(PrintThread("t-" + str(i), randint(0, 100)))

	for thread in threads:
		thread.start()
		
	for thread in threads:
		thread.join()
		
if __name__ == '__main__':
	main()
	
	


