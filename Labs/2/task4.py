# Philosophers problem

import sys
from threading import Thread, Lock

class PrintThread(Thread):
	def __init__(self, index, left_lock, right_lock):
		Thread.__init__(self)
		self.index = index
		self.left_lock = left_lock
		self.right_lock = right_lock
		
	def run(self):
		if self.index == 0:
			self.left_lock.acquire()
			self.right_lock.acquire()
			print("Philosopher 0 is eating")
			self.right_lock.release()
			self.left_lock.release()
		else:
			self.right_lock.acquire()
			self.left_lock.acquire()
			print("Philosopher %d is eating" % (self.index,))
			self.left_lock.release()
			self.right_lock.release()

def main():
	if(len(sys.argv) < 2):
		print("Run: python3 task4.py <nr_threads>")
		return
	N = int(sys.argv[1])
	
	locks = []
	for i in range(N):
		locks.append(Lock())
		
	threads = []
	for i in range(N):
		threads.append(PrintThread(i, locks[i], locks[(i+1)%N]))
		
	for thread in threads:
		thread.start()
		
	for thread in threads:
		thread.join()

	
if __name__ == '__main__':
	main()
	
	
