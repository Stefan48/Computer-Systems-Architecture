#include <stdio.h>
#define N 1000

void printMatrix(double m[][N])
{
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			printf("%lf ", m[i][j]);
		}
		printf("\n");
	}
}

int main()
{
	double a[N][N], b[N][N], c[N][N];
	int i, j, k;
	int bi, bj, bk;
	
	// initialize matrices
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			a[i][j] = 0.5;
			b[i][j] = 0.5;
			c[i][j] = 0.0;
		}
	}
	
    // TODO: set block dimension blockSize less or equal to sqrt(cache_size / 3)
    // cache size = 32768 => block size <= 104
    // cache size = 8388608 => block size <= 1672
    
    int blockSize = 100; 
 
 	// multiply
    for(bi = 0; bi < N; bi += blockSize)
    {
        for(bj = 0; bj < N; bj += blockSize)
        {
            for(bk = 0; bk < N; bk += blockSize)
            {
                for(i = 0; i < blockSize; ++i)
                {
                    for(j = 0; j < blockSize; ++j)
                    {
                        for(k = 0; k < blockSize; ++k)
                        {
                            c[bi+i][bj+j] += a[bi+i][bk+k] * b[bk+k][bj+j];
                        }
                    }
                }
            }
        }
    }
	
	
	//printMatrix(c);
	return 0;
}
