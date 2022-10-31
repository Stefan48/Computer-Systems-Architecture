from concurrent.futures import ThreadPoolExecutor
import random

def searchPattern(pattern, sample, index):
	if pattern in sample:
		print("Patern found in sample %d" % (index,))
	

if __name__ == "__main__":
    samples = [''.join([random.choice("ATCG") for i in range(10000)]) for j in range(100)]
    pattern = "CGATGCTA"
    results = []
    #for sample in samples:
    	#print(sample)
    with ThreadPoolExecutor(max_workers = 32) as executor:
    	for i in range(len(samples)):
     		results.append(executor.submit(searchPattern, pattern, samples[i], i))
     		#print(results[-1])
    for result in results:
      	print(result)
