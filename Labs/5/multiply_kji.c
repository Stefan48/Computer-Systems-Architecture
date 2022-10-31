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
	
	// multiply
	for (k = 0; k < N; ++k)
	{
		for (j = 0; j < N; ++j)
		{
			for (i = 0; i < N; ++i)
			{
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	
	//printMatrix(c);
	return 0;
}
