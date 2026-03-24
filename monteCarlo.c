// monte carlo method for approximating sqrt(2) using openMPI and openMP
// uses long datatypes for better precision
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// seed (currently my student number)
#define SEED 20335307

int main(int argc, char** argv) {
    long long total_samples, local_samples;
    long long local_hits = 0, global_hits = 0;

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

    long double sqrt2 = sqrtl(2.0);

    // seed (hardcoded for reproducibility) + world_rank
    unsigned int seed = SEED + world_rank;

    #pragma omp parallel reduction(+:local_hits)
    {
        // unique seed per mpi thread
        unsigned int thread_seed = seed ^ omp_get_thread_num();

        #pragma omp for
        for (long long i = 0; i < local_samples; i++) {
            long double x = rand_r(&thread_seed) / (long double)RAND_MAX; 
            long double y = (rand_r(&thread_seed) / (long double)RAND_MAX) * sqrt2;

            // check boundary condition
            if (y <= sqrt2 * (1.0 - x))
                local_hits++;
        }
    }

    // collect results
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        long double probability_hit = (long double)global_hits / (long double)total_samples;
        long double sqrt2_est = 2.0 * probability_hit;

        printf("Estimated sqrt(2) = %.10Lf\n", sqrt2_est);
        printf("Actual sqrt(2)    = %.10Lf\n", (long double)sqrt(2.0));
        printf("Absolute error    = %.10Le\n", fabsl(sqrt2_est - (long double)sqrt(2.0)));
    }

    MPI_Finalize();
    return 0;
}