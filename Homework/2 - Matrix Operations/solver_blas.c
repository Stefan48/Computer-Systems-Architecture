/*
 * Tema 2 ASC
 * 2021 Spring
 */
#include "utils.h"
#include "cblas.h"
#include <string.h>

/* 
 * Add your BLAS implementation here
 */
double* my_solver(int N, double *A, double *B) {
    printf("BLAS SOLVER\n");
    
    double *AB = calloc(N * N, sizeof(double));
    double *AtA = calloc(N * N, sizeof(double));
    if (AB == NULL || AtA == NULL)
    {
        return NULL;
    }
    /* AB = A x B */
    memcpy(AB, B, N * N * sizeof(double));
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, AB, N);
    /* AtA = At x A */
    memcpy(AtA, A, N * N * sizeof(double));
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, AtA, N, AtA, N);
    /* AtA = AB x Bt + AtA */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, AtA, N);
    
    free(AB);
    return AtA;
}
