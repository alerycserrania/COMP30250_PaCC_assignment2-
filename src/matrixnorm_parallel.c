#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <cblas.h>
#include <sys/time.h>

typedef struct {
    double* a;
    double* b;
    double* c;
    int m_size;
    int slice_size;
} matrix_prod_data;

typedef struct {
    double* c;
    double* global_norm;
    int m_size;
    int starting_slice;
    int ending_slice;
    pthread_mutex_t *mutex;
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

void* partial_matrix_product(void* arg) 
{
    matrix_prod_data* m_data;
    
    m_data = arg;
    
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        m_data->slice_size, m_data->m_size, m_data->m_size, 
        1.0, m_data->a, m_data->m_size, m_data->b, m_data->m_size, 
        1.0, m_data->c, m_data->m_size
    );

}

void* partial_matrix_norm(void* arg)
{
    matrix_norm_data* m_data;
    int i, j;
    double slice_max_col_sum, col_sum;
    double global_norm;
    double val;

    m_data = arg;
    for (j = m_data->starting_slice; j < m_data->ending_slice; j++)
    {
        col_sum = 0.;
        for (i = 0; i < m_data->m_size; i++)
        {
            val = m_data->c[i * m_data->m_size + j];
            col_sum += val > 0. ? val : -val;
        }
        slice_max_col_sum = (slice_max_col_sum < col_sum) ? col_sum : slice_max_col_sum;
    }

    pthread_mutex_lock(m_data->mutex);
    global_norm = *(m_data->global_norm);
    *(m_data->global_norm) = (global_norm < slice_max_col_sum) ? slice_max_col_sum : global_norm;
    pthread_mutex_unlock(m_data->mutex);
    
}

int main(int argc, char *argv[])
{
    int m_size, nb_thrds, slice_size;
    double *a, *b, *c;
    double norm;
    int i;
    void *status;

    pthread_t *matrix_prod_workers, *matrix_norm_workers;
    matrix_prod_data *matrix_prod_thread_data;
    matrix_norm_data *matrix_norm_thread_data;
    pthread_mutex_t *matrix_norm_mutex;
    
    struct timeval tv1, tv2;
    struct timezone tz;
    double elapsed;

    char PRINT_RESULT, PRINT_TIME;

    if(argc != 5)
    {
       printf("Please, use: %s N P PRINT_RESULT PRINT_TIME:\n", argv[0]);
       printf("\t- N: matrix size\n");
       printf("\t- P: number of threads\n");
       printf("\t- PRINT_RESULT (y/n): print result to stdout\n");
       printf("\t- PRINT_TIME (y/n): print time elasped to stdout\n");
       exit(EXIT_FAILURE);
    }

    m_size = atoi(argv[1]);
    nb_thrds = atoi(argv[2]);
    PRINT_RESULT = argv[3][0];
    PRINT_TIME = argv[4][0];

    if (m_size < nb_thrds) {
        printf("Number of threads must be less than the size");
        exit(EXIT_FAILURE);
    }

    slice_size = m_size/nb_thrds;

    a = malloc(m_size * m_size * sizeof(double));
    b = malloc(m_size * m_size * sizeof(double));
    c = malloc(m_size * m_size * sizeof(double));

    for (i = 0; i < m_size * m_size; i++)
    {
        a[i] = rand() % 10 - 5;
        b[i] = rand() % 10 - 5;
        c[i] = 0.;
    }


    matrix_prod_workers = malloc(nb_thrds * sizeof(pthread_t));
    matrix_prod_thread_data = malloc(nb_thrds * sizeof(matrix_prod_data));
    matrix_norm_workers = malloc(nb_thrds * sizeof(pthread_t));
    matrix_norm_thread_data = malloc(nb_thrds * sizeof(matrix_norm_data));
    matrix_norm_mutex = malloc(sizeof(pthread_mutex_t));

    gettimeofday(&tv1, &tz);

    // create thread ro parallelize matrix product
    for (i = 0; i < nb_thrds; i++) 
    {
        matrix_prod_thread_data[i].a = a + (i * slice_size * m_size);
        matrix_prod_thread_data[i].b = b;
        matrix_prod_thread_data[i].c = c + (i * slice_size * m_size);
        matrix_prod_thread_data[i].m_size = m_size;
        matrix_prod_thread_data[i].slice_size = 
            (i == nb_thrds - 1) ? (m_size - i * slice_size): slice_size;
        pthread_create(&matrix_prod_workers[i], NULL, partial_matrix_product,(void*)&matrix_prod_thread_data[i]);
    }

    // join matrix prod threads 
    for (i = 0; i < nb_thrds; i++)
    {
        pthread_join(matrix_prod_workers[i], &status);
    }

    // create new threads to parallelize the norm computation
    pthread_mutex_init(matrix_norm_mutex, NULL);
    norm = 0.;
    for (i = 0; i < nb_thrds; i++) 
    {
        matrix_norm_thread_data[i].c = c;
        matrix_norm_thread_data[i].global_norm = &norm;
        matrix_norm_thread_data[i].m_size = m_size;
        matrix_norm_thread_data[i].starting_slice = i*slice_size;
        matrix_norm_thread_data[i].ending_slice = (i == nb_thrds - 1) ? m_size: (i+1)*slice_size;
        matrix_norm_thread_data[i].mutex = matrix_norm_mutex;
        pthread_create(&matrix_norm_workers[i], NULL, partial_matrix_norm,(void*)&matrix_norm_thread_data[i]);
    }

    // join matrix prod threads 
    for (i = 0; i < nb_thrds; i++)
    {
        pthread_join(matrix_norm_workers[i], &status);
    }

    gettimeofday(&tv2, &tz);

    // release resources of these threads
    free(matrix_prod_workers);
    free(matrix_prod_thread_data);
    free(matrix_norm_workers);
    free(matrix_norm_thread_data);
    free(matrix_norm_mutex);

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

