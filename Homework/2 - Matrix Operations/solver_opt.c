/*
 * Tema 2 ASC
 * 2021 Spring
 */
#include "utils.h"

/*
 * Add your optimized implementation here
 */
double* my_solver(int N, double *A, double* B) {
    printf("OPT SOLVER\n");
    
    int i, j, k;
    double *C, *aux;
    double *p, *p1, *p2;
    register double sum;

    C = calloc(N * N, sizeof(double));
    aux = calloc(N * N, sizeof(double));
    if (C == NULL || aux == NULL)
    {
        return NULL;
    }
    /* aux = B x Bt */
    for (i = 0; i < N; ++i)
    {
        p = &B[N * i];
        for (j = i; j < N; ++j)
        {
            p1 = p;
            p2 = &B[N * j];
            sum = 0.0;
            for (k = 0; k < N; ++k)
            {
                sum += *p1 * *p2;
                p1++;
                p2++;
            }
            aux[N * i + j] = sum;
            if (i != j)
            {
                aux[N * j + i] = aux[N * i + j];
            }
        }
    }
    /* C = A x aux */
    for (i = 0; i < N; ++i)
    {
        p = &A[(N + 1) * i];
        for (j = 0; j < N; ++j)
        {
            p1 = p;
            p2 = &aux[N * i + j];
            sum = 0.0;
            for (k = i; k < N; ++k)
            {
                sum += *p1 * *p2;
                p1++;
                p2 += N;
            }
            C[N * i + j] = sum;
        }
    }
    /* aux = At x A */
    for (i = 0; i < N; ++i)
    {
        p = &A[i];
        for (j = i; j < N; ++j)
        {
            p1 = p;
            p2 = &A[j];
            sum = 0.0;
            for (k = 0; k <= i; ++k)
            {
                sum += *p1 * *p2;
                p1 += N;
                p2 += N;
            }
            aux[N * i + j] = sum;
            if (i != j)
            {
                aux[N * j + i] = aux[N * i + j];
            }
        }
    }
    /* C = C + aux */
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            C[N * i + j] += aux[N * i + j];
        }
    }
    
    free(aux);
    return C;
}
