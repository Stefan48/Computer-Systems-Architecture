from threading import enumerate, Event, Thread, Condition
from time import sleep

class Master(Thread):
    def __init__(self, max_work, available):
        Thread.__init__(self, name = "Master")
        self.max_work = max_work
        self.available = available
    
    def set_worker(self, worker):
        self.worker = worker
    
    def run(self):
        # Worker has to call wait() before Master calls notify()
        sleep(1)
        for i in range(self.max_work):
        	with self.available:
		        # generate work
		        self.work = i
		        print("Master generated work")
		        # notify worker
		        self.available.notify()
		        # get result
		        self.available.wait()
		        if self.get_work() + 1 != self.worker.get_result():
		            print ("oops")
		        print ("%d -> %d" % (self.work, self.worker.get_result()))
    
    def get_work(self):
        return self.work

class Worker(Thread):
    def __init__(self, terminate, available):
        Thread.__init__(self, name = "Worker")
        self.terminate = terminate
        self.available = available

    def set_master(self, master):
        self.master = master
    
    def run(self):
        while(True):
        	with self.available:
		        # wait work
		        self.available.wait()#
		        if(terminate.is_set()):
		        	#self.available.release()
		        	break
		        # generate result
		        self.result = self.master.get_work() + 1
		        print("Worker generated result")
		        # notify master
		        self.available.notify()
    
    def get_result(self):
        return self.result

if __name__ ==  "__main__":
    # create shared objects
    terminate = Event()
    available = Condition()
    
    # start worker and master
    m = Master(10, available)
    w = Worker(terminate, available)
    m.set_worker(w)
    w.set_master(m)
    m.start()
    w.start()

    # wait for master
    m.join()

    # wait for worker
    terminate.set()
    # notify worker to exit endless loop
    with available:
    	available.notifyAll()
    w.join()

    # print running threads for verification
    print(enumerate())

