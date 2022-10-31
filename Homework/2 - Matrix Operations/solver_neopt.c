/*
 * Tema 2 ASC
 * 2021 Spring
 */
#include "utils.h"

/*
 * Add your unoptimized implementation here
 */
double* my_solver(int N, double *A, double* B) {
    printf("NEOPT SOLVER\n");
    
    int i, j, k, stop;
    double *C, *aux;

    C = calloc(N * N, sizeof(double));
    aux = calloc(N * N, sizeof(double));
    if (C == NULL || aux == NULL)
    {
        return NULL;
    }
    /* aux = A x B */
    for (i = 0; i < N; ++i)
    {
         for (j = 0; j < N; ++j)
         {
            for (k = i; k < N; ++k)
            {
                aux[N * i + j] += A[N * i + k] * B[N * k + j];
            }
         }
    }
    /* C = aux x Bt */
    for (i = 0; i < N; ++i)
    {
         for (j = 0; j < N; ++j)
         {
              for (k = 0; k < N; ++k)
              {
                  C[N * i + j] += aux[N * i + k] * B[N * j + k];
              }
         }
    }
    /* C = C + At x A */
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            stop = i < j ? i : j;
            for (k = 0; k <= stop; ++k)
            {
                C[N * i + j] += A[N * k + i] * A[N * k + j];
            }
        }
    }
    
    free(aux);
    return C;
}
