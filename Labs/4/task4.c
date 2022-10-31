#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

/* Local specs:
LEVEL1_ICACHE_SIZE                 32768
LEVEL1_ICACHE_ASSOC                8
LEVEL1_ICACHE_LINESIZE             64
LEVEL1_DCACHE_SIZE                 32768
LEVEL1_DCACHE_ASSOC                8
LEVEL1_DCACHE_LINESIZE             64
LEVEL2_CACHE_SIZE                  262144
LEVEL2_CACHE_ASSOC                 8
LEVEL2_CACHE_LINESIZE              64
LEVEL3_CACHE_SIZE                  8388608
LEVEL3_CACHE_ASSOC                 16
LEVEL3_CACHE_LINESIZE              64
LEVEL4_CACHE_SIZE                  0
LEVEL4_CACHE_ASSOC                 0
LEVEL4_CACHE_LINESIZE              0
*/


int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        printf("run with %s <line_size> <vector_size> <iterations>\n", argv[0]);
        return -1;
    }

    int64_t l = atoi(argv[1]);  // dimensiunea liniei de cache
    int64_t n = atoi(argv[2]);  // dimensiunea vectorului
    int64_t c = atoi(argv[3]);  // numarul de iteratii

    // TODO alocari si initializari
    char *a = malloc(n * sizeof(char));

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // TODO bucla de test
    // in variabila ops calculati numarul de operatii efectuate
    int64_t ops = 0;
    
    for(int step = 0; step < c; ++step)
    {
    	for(int i = 0; i < n; i += l)
    	{
    		a[i]++;
    		ops++;
    	}
    }
    

    gettimeofday(&end, NULL);

    float elapsed = ((end.tv_sec - start.tv_sec)*1000000.0f + end.tv_usec - start.tv_usec)/1000000.0f;
    printf("%12ld, %12ld, %12f, %12g\n", n, ops, elapsed, ops/elapsed);

    free(a);

    return 0;
}

