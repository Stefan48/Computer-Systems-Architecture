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
	
	// initialize matrices a, b
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			a[i][j] = 0.5;
			b[i][j] = 0.5;
		}
	}
	
	// multiply
	double *orig_pa, *pa, *pb;
	register double suma;
	for(i = 0; i < N; ++i)
	{
		orig_pa = &a[i][0];
	  	for(j = 0; j < N; ++j)
	  	{
			pa = orig_pa;
			pb = &b[0][j];
			suma = 0;
			for(k = 0; k < N; ++k)
			{
				suma += *pa * *pb;
			  	pa++;
			  	pb += N;
			}
			c[i][j] = suma;
	  	}
	}
	
	//printMatrix(c);
	return 0;
}
