// monte carlo method for approximating sqrt(2) using openMPI and openMP
// uses long datatypes for better precision
// updated to a stable, low-variance estimator using min(U,V)

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// seed (currently my student number)
#define SEED 20335307
#define TWO 2.0L

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
            // sample U, V ~ Uniform(0,1]
            long double U = (rand_r(&thread_seed) + 1.0L) /
                            ((long double)RAND_MAX + 1.0L);

            long double V = (rand_r(&thread_seed) + 1.0L) /
                            ((long double)RAND_MAX + 1.0L);

            // sum the minimum of U and V
            long double m = (U < V ? U : V);

            local_sum += m;
        }
    }

    // collect results
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // low-variance estimator: sqrt(2) = 2 * mean(min(U,V))
        long double mean = global_sum / (long double)total_samples;
        long double sqrt2_est = TWO * mean;

        printf("Estimated sqrt(2) = %.10Lf\n", sqrt2_est);
        printf("Actual sqrt(2)    = %.10Lf\n", sqrtl(2.0L));
        printf("Absolute error    = %.10Le\n", fabsl(sqrt2_est - sqrtl(2.0L)));
    }

    MPI_Finalize();
    return 0;
}