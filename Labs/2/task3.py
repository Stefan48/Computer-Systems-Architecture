"""
Coffee Factory: A multiple producer - multiple consumer approach

Generate a base class Coffee which knows only the coffee name
Create the Espresso, Americano and Cappuccino classes which inherit the base class knowing that
each coffee type has a predetermined size.
Each of these classes have a get message method

Create 3 additional classes as following:
    * Distributor - A shared space where the producers puts coffees and the consumers takes them
    * CoffeeFactory - An infinite loop, which always sends coffees to the distributor
    * User - Another infinite loop, which always takes coffees from the distributor

The scope of this exercise is to correctly use threads, classes and synchronization objects.
The size of the coffee (ex. small, medium, large) is chosen randomly everytime.
The coffee type is chosen randomly everytime.

Example of output:

Consumer 65 consumed espresso
Factory 7 produced a nice small espresso
Consumer 87 consumed cappuccino
Factory 9 produced an italian medium cappuccino
Consumer 90 consumed americano
Consumer 84 consumed espresso
Factory 8 produced a strong medium americano
Consumer 135 consumed cappuccino
Consumer 94 consumed americano
"""

"""
class ExampleCoffee:
    # Espresso implementation
    def __init__(self, size):
        pass

    def get_message(self):
        # Output message
        raise NotImplementedError
"""

import sys
from threading import Thread, Lock, Semaphore
from random import randint


class Coffee:
    """ Base class """
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def get_name(self):
        """ Returns the coffee name """
        return self.name

    def get_size(self):
        """ Returns the coffee size """
        return self.size
        
        
class Espresso(Coffee):
	def __init__(self, size):
		Coffee.__init__(self, "espresso", size)

class Americano(Coffee):
	def __init__(self, size):
		Coffee.__init__(self, "americano", size)
		
class Cappuccino(Coffee):
	def __init__(self, size):
		Coffee.__init__(self, "cappuccino", size)
		
class Distributor:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		
	def insert(self, coffee):
		""" Insert in buffer """
		if len(self.buffer) >= self.capacity:
			return False
		self.buffer.append(coffee)
		return True
			
	def extract(self):
		""" Extract from buffer """
		if len(self.buffer) == 0:
			return None
		return self.buffer.pop()

class CoffeeFactory(Thread):
	def __init__(self, index, iterations, distributor, factory_mutex, sem_full, sem_empty):
		Thread.__init__(self)
		self.index = index
		self.iterations = iterations
		self.distributor = distributor
		self.factory_mutex = factory_mutex
		self.sem_full = sem_full
		self.sem_empty = sem_empty
		
	def insert(self, coffee_name, coffee_size):
		if coffee_name == "espresso":
			self.distributor.insert(Espresso(coffee_size))
		elif coffee_name == "americano":
			self.distributor.insert(Americano(coffee_size))
		elif coffee_name == "cappuccino":
			self.distributor.insert(Cappuccino(coffee_size))
	
	def run(self):
		""" Inserts a coffee on each iteration """
		for i in range(self.iterations):
			coffee_name = ["espresso", "americano", "cappuccino"][randint(0, 2)]
			coffee_size = ["small", "medium", "large"][randint(0, 2)]
			self.sem_empty.acquire()
			self.factory_mutex.acquire()
			self.insert(coffee_name, coffee_size)
			print("Factory %d produced a %s %s" % (self.index, coffee_size, coffee_name))
			self.factory_mutex.release()
			self.sem_full.release()
			
class Consumer(Thread):
	remaining = 0 # static variable - number of total remaining coffees to consume
	
	def __init__(self, index, distributor, consumer_mutex, sem_full, sem_empty):
		Thread.__init__(self)
		self.index = index
		self.distributor = distributor
		self.consumer_mutex = consumer_mutex
		self.sem_full = sem_full
		self.sem_empty = sem_empty
			
	def run(self):
		""" Consumes a coffee on each iteration, until all coffees have been consumed """
		while True:
			self.consumer_mutex.acquire()
			if Consumer.remaining <= 0:
				self.consumer_mutex.release()
				break
			Consumer.remaining -= 1
			self.consumer_mutex.release()
			self.sem_full.acquire()
			self.consumer_mutex.acquire()
			coffee = self.distributor.extract()
			print("Consumer %d consumed a %s %s" % (self.index, coffee.get_size(), coffee.get_name()))
			self.consumer_mutex.release()
			self.sem_empty.release()
		
	
def main():
	if(len(sys.argv) < 4):
		print("Run: python3 task3.py <nr_threads> <nr_factories> <nr_consumers>")
		return
	iterations = 5
	capacity = 10
	
	distributor = Distributor(capacity)
	""" initialize synchronization variables """
	factory_mutex = Lock()
	consumer_mutex = Lock()
	sem_full = Semaphore(value = 0)
	sem_empty = Semaphore(value = capacity)
	
	nr_threads = int(sys.argv[1])
	nr_factories = int(sys.argv[2])
	nr_consumers = int(sys.argv[3])
	Consumer.remaining = iterations * nr_factories
	
	""" initialize factories """
	factories = []
	for i in range(nr_factories):
		factories.append(CoffeeFactory(i, iterations, distributor, factory_mutex, sem_full, sem_empty))
	
	""" initialize consumers """
	consumers = []
	for i in range(nr_consumers):
		consumers.append(Consumer(i, distributor, consumer_mutex, sem_full, sem_empty))
	
	""" start threads """
	for factory in factories:
		factory.start()
		
	for consumer in consumers:
		consumer.start()
		
	""" wait threads """
	for factory in factories:
		factory.join()
		
	for consumer in consumers:
		consumer.join()
	

if __name__ == '__main__':
    main()
    
    
