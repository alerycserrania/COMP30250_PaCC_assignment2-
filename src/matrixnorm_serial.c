#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <sys/time.h>

typedef struct {
    double* a;
    double* b;
    double* c;
    int m_size;
} matrix_prod_data;

typedef struct {
    double* c;
    double* global_norm;
    int m_size;
} matrix_norm_data;

void print_matrix(int N, double* M)
{
    int i, j, k;

    for (i = 0; i < N; i++) 
    {
        for (j = 0; j < N; j++) 
        {
            printf("%f\t", M[i * N + j]);
        }
        printf("\n");
    }
}

void* matrix_product(matrix_prod_data* m_data) 
{
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        m_data->m_size, m_data->m_size, m_data->m_size, 
        1.0, m_data->a, m_data->m_size, m_data->b, m_data->m_size, 
        1.0, m_data->c, m_data->m_size
    );

}

void* matrix_norm(matrix_norm_data* m_data)
{
    int i, j;
    double max_col_sum, col_sum;
    double val;

    for (j = 0; j < m_data->m_size; j++)
    {
        col_sum = 0.;
        for (i = 0; i < m_data->m_size; i++)
        {
            val = m_data->c[i * m_data->m_size + j];
            col_sum += val > 0. ? val : -val;
        }
        max_col_sum = (max_col_sum < col_sum) ? col_sum : max_col_sum;
    }

    *(m_data->global_norm) = max_col_sum;
    
}

int main(int argc, char *argv[])
{
    int m_size, nb_thrds, slice_size;
    double *a, *b, *c;
    double norm;
    int i;

    matrix_prod_data matrix_prod_serial_data;
    matrix_norm_data matrix_norm_serial_data;
    
    struct timeval tv1, tv2;
    struct timezone tz;
    double elapsed;

    char PRINT_RESULT, PRINT_TIME;

    if(argc != 4)
    {
       printf("Please, use: %s N PRINT_RESULT PRINT_TIME:\n", argv[0]);
       printf("\t- N: matrix size\n");
       printf("\t- PRINT_RESULT (y/n): print result to stdout\n");
       printf("\t- PRINT_TIME (y/n): print time elasped to stdout\n");
       exit(EXIT_FAILURE);
    }

    m_size = atoi(argv[1]);
    PRINT_RESULT = argv[2][0];
    PRINT_TIME = argv[3][0];

    a = malloc(m_size * m_size * sizeof(double));
    b = malloc(m_size * m_size * sizeof(double));
    c = malloc(m_size * m_size * sizeof(double));

    for (i = 0; i < m_size * m_size; i++)
    {
        a[i] = rand() % 10 - 5;
        b[i] = rand() % 10 - 5;
        c[i] = 0.;
    }

    gettimeofday(&tv1, &tz);
    
    matrix_prod_serial_data.a = a;
    matrix_prod_serial_data.b = b;
    matrix_prod_serial_data.c = c;
    matrix_prod_serial_data.m_size = m_size;
    matrix_product(&matrix_prod_serial_data);
    
    norm = 0;
    matrix_norm_serial_data.c = c;
    matrix_norm_serial_data.global_norm = &norm;
    matrix_norm_serial_data.m_size = m_size;
    matrix_norm(&matrix_norm_serial_data);

    gettimeofday(&tv2, &tz);


    if (PRINT_RESULT == 'y')
    {
        printf("A =\n");
        print_matrix(m_size, a);
        printf("B =\n");
        print_matrix(m_size, b);
        printf("C =\n");
        print_matrix(m_size, c);
        printf("Norm: %f\n", norm);
    }

    if (PRINT_TIME == 'y')
    {
        elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec)*1e-6;
        printf("elapsed: %fs\n", elapsed);
    }

    free(a);
    free(b);
    free(c);

    return 0;
}

