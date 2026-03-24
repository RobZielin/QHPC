// monte carlo method for approximating sqrt(2) using openMPI and openMP
// uses long datatypes for better precision
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// seed (currently my student number)
#define SEED 20335307
#define EPS 1e-8L     // small epsilon to avoid x=0
#define X_MAX 2.0L    // integration domain [EPS, 2]

int main(int argc, char** argv) {
    long long total_samples, local_samples;
    long double local_sum = 0.0L, global_sum = 0.0L;

    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 2) {
        if (world_rank == 0) {
            printf("Usage: %s <total_samples>\n", argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    total_samples = atoll(argv[1]);
    local_samples = total_samples / world_size;

    // seed (hardcoded for reproducibility) + world_rank
    unsigned int seed = SEED + world_rank * 1234567;

    #pragma omp parallel reduction(+:local_sum)
    {
        // unique seed per mpi thread
        unsigned int thread_seed = seed ^ omp_get_thread_num();

        #pragma omp for
        for (long long i = 0; i < local_samples; i++) {
            // sample x uniformly in [EPS, X_MAX]
            long double x = EPS + ((rand_r(&thread_seed) / (long double)RAND_MAX) * (X_MAX - EPS));

            // Monte Carlo integration: f(x) = 1/sqrt(x)
            local_sum += 1.0L / sqrtl(x);
        }
    }

    // collect results
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // estimate is average value of f(x)
        long double sqrt2_est = global_sum / (long double)total_samples;

        printf("Estimated sqrt(2) = %.10Lf\n", sqrt2_est);
        printf("Actual sqrt(2)    = %.10Lf\n", sqrtl(2.0L));
        printf("Absolute error    = %.10Le\n", fabsl(sqrt2_est - sqrtl(2.0L)));
    }

    MPI_Finalize();
    return 0;
}