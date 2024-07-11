/**
 * @file main.f08
 * @brief This file provides you with the original implementation of pagerank.
 * Your challenge is to optimise it using OpenMP and/or MPI.
 * @author Ludovic Capelli (l.capelli@epcc.ed.ac.uk)
 **/

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/// The number of vertices in the graph.
#define GRAPH_ORDER 1000
/// Parameters used in pagerank convergence, do not change.
#define DAMPING_FACTOR 0.85
/// The number of seconds to not exceed for the calculation loop.
#define MAX_TIME 10.0

/**
 * @brief Indicates which vertices are connected.
 * @details If an edge links vertex A to vertex B, then adjacency_matrix[A][B]
 * will be 1.0. The absence of edge is represented with value 0.0.
 * Redundant edges are still represented with value 1.0.
 */
double adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER];
double max_diff = 0.0;
double min_diff = 1.0;
double total_diff = 0.0;

void initialize_graph(void) {
  for (int i = 0; i < GRAPH_ORDER; i++) {
    for (int j = 0; j < GRAPH_ORDER; j++) {
      adjacency_matrix[i][j] = 0.0;
    }
  }
}

/**
 * @brief Calculates the pagerank of all vertices in the graph.
 * @param pagerank The array in which store the final pageranks.
 */
void calculate_pagerank(double pagerank[], const int rank, const int n_proc) {
  double initial_rank = 1.0 / GRAPH_ORDER;

  // Initialise all vertices to 1/n.
  for (int i = 0; i < GRAPH_ORDER; i++)
    pagerank[i] = initial_rank;

  double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;
  double diff = 1.0;
  size_t iteration = 0;
  double start = omp_get_wtime();
  double elapsed = omp_get_wtime() - start;
  double time_per_iteration = 0;

  double new_pagerank[GRAPH_ORDER];
  double pr_divided_by_od[GRAPH_ORDER];

  int local_size = GRAPH_ORDER / n_proc;
  int local_begin = rank * local_size;
  double* local_pagerank=(double*)malloc(local_size*sizeof(double));

  int* counts=(int*)malloc(n_proc*sizeof(int));
  int* displs=(int*)malloc(n_proc*sizeof(int));
  for(int i = 0; i<n_proc;++i){
    counts[i] = local_size;
    displs[i] = local_size*i;
  }

  for (int i = 0; i < GRAPH_ORDER; i++)
    new_pagerank[i] = 0.0;

  int outdegrees[GRAPH_ORDER] = {0};
#pragma omp parallel for
  for (int j = 0; j < GRAPH_ORDER; j++)
    for (int k = 0; k < GRAPH_ORDER; k++)
      outdegrees[j] += adjacency_matrix[j][k];

      // Tranpose the adjacency matrix
#pragma omp parallel for schedule(auto)
  for (int i = 0; i < GRAPH_ORDER; ++i)
    for (int j = i + 1; j < GRAPH_ORDER; j++) {
      double temp = adjacency_matrix[i][j];
      adjacency_matrix[i][j] = adjacency_matrix[j][i];
      adjacency_matrix[j][i] = temp;
    }

  bool rank_is_in=true;
  // If we exceeded the MAX_TIME seconds, we stop. If we typically spend X
  // seconds on an iteration, and we are less than X seconds away from
  // MAX_TIME, we stop.
  while (rank_is_in) {

#pragma omp parallel
    {
#pragma omp for
      for (int i = 0; i < local_size; ++i)
        local_pagerank[i] = 0.0;

#pragma omp for
      for (int i = 0; i < GRAPH_ORDER; ++i)
        pr_divided_by_od[i] =
            pagerank[i] / (double)outdegrees[i]; // It is ok to divide floating
                                                 // number by zero? (IEEE 754)
#pragma omp barrier

// PArallel MPI this one
// int local_i=0;
#pragma omp for
      for (int i = local_begin; i < local_begin+local_size; ++i){
        for (int j = 0; j < GRAPH_ORDER; j++){
          local_pagerank[i%local_size] += adjacency_matrix[i][j] * pr_divided_by_od[j];
        }
        // ++local_i;
      }

#pragma omp for
      for (int i = 0; i < local_size; ++i)
        local_pagerank[i] = DAMPING_FACTOR * local_pagerank[i] + damping_value;
    }

//if true
MPI_Allgatherv(local_pagerank,
               local_size,
               MPI_DOUBLE,
               new_pagerank,
               counts,
               displs,
               MPI_DOUBLE,
               MPI_COMM_WORLD);

    diff = 0.0;
#pragma omp parallel for reduction(+ : diff)
    for (int i = 0; i < GRAPH_ORDER; i++)
      diff += fabs(new_pagerank[i] - pagerank[i]);
    max_diff = (max_diff < diff) ? diff : max_diff;
    total_diff += diff;
    min_diff = (min_diff > diff) ? diff : min_diff;

#pragma omp parallel for
    for (int i = 0; i < GRAPH_ORDER; i++)
      pagerank[i] = new_pagerank[i];

    double pagerank_total = 0.0;
    // double pagerank_local = 0.0;
#pragma omp parallel for reduction(+ : pagerank_total)
    for (int i = 0; i < GRAPH_ORDER; ++i)
      pagerank_total += pagerank[i];
      // pagerank_local += pagerank[i];

    if(rank==0){
      if (fabs(pagerank_total - 1.0) >= 1E-12)
        printf("[ERROR] Iteration %zu: sum of all pageranks is not 1 but "
              "%.12f.\n",
              iteration, pagerank_total);
    }

    elapsed = omp_get_wtime() - start;
    iteration++;
    time_per_iteration = elapsed / iteration;
    
    if(elapsed < MAX_TIME && (elapsed + time_per_iteration) < MAX_TIME){
      rank_is_in=true;
    }else{
      rank_is_in=false;
    }

   MPI_Allreduce(MPI_IN_PLACE,
                  &rank_is_in,
                  1,
                  MPI_C_BOOL,
                  MPI_LAND,
                  MPI_COMM_WORLD);

  if(!rank_is_in){
    break;
  }
  }//end while


  printf("%zu iterations achieved in %.2f seconds\n", iteration, elapsed);

  free(local_pagerank);
  free(counts);
  free(displs);
}

/**
 * @brief Populates the edges in the graph for testing.
 **/
void generate_nice_graph(void) {
  printf("Generate a graph for testing purposes (i.e.: a nice and conveniently "
         "designed graph :) )\n");
  double start = omp_get_wtime();
  initialize_graph();
  for (int i = 0; i < GRAPH_ORDER; i++) {
    for (int j = 0; j < GRAPH_ORDER; j++) {
      int source = i;
      int destination = j;
      if (i != j) {
        adjacency_matrix[source][destination] = 1.0;
      }
    }
  }
  printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

/**
 * @brief Populates the edges in the graph for the challenge.
 **/
void generate_sneaky_graph(void) {
  printf("Generate a graph for the challenge (i.e.: a sneaky graph :P )\n");
  double start = omp_get_wtime();
  initialize_graph();
  for (int i = 0; i < GRAPH_ORDER; i++) {
    for (int j = 0; j < GRAPH_ORDER - i; j++) {
      int source = i;
      int destination = j;
      if (i != j) {
        adjacency_matrix[source][destination] = 1.0;
      }
    }
  }
  printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

int main(int argc, char *argv[]) {
  // We do not need argc, this line silences potential compilation warnings.
  (void)argc;
  // We do not need argv, this line silences potential compilation warnings.
  (void)argv;
  MPI_Init(&argc, &argv);

  int rank, n_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

  if (rank == 0)
    printf("This program has two graph generators: generate_nice_graph and "
           "generate_sneaky_graph. If you intend to submit, your code will be "
           "timed on the sneaky graph, remember to try both.\n");

  // Get the time at the very start.
  double start = omp_get_wtime();

  // generate_nice_graph();
  generate_sneaky_graph(); // the real task!

  /// The array in which each vertex pagerank is stored.
  double pagerank[GRAPH_ORDER];
  calculate_pagerank(pagerank, rank, n_proc);

  // Calculates the sum of all pageranks. It should be 1.0, so it can be used
  // as a quick verification.
  if (rank == 0) {
    double sum_ranks = 0.0;
    for (int i = 0; i < GRAPH_ORDER; i++) {
      if (i % 100 == 0) {
        printf("PageRank of vertex %d: %.6f\n", i, pagerank[i]);
      }
      sum_ranks += pagerank[i];
    }
    printf("Sum of all pageranks = %.12f, total diff = %.12f, max diff = %.12f "
           "and min diff = %.12f.\n",
           sum_ranks, total_diff, max_diff, min_diff);
    double end = omp_get_wtime();

    printf("Total time taken: %.2f seconds.\n", end - start);
  }

  MPI_Finalize();
  return 0;
}
