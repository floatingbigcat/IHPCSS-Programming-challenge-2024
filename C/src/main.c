/**
 * @file main.f08
 * @brief This file provides you with the original implementation of pagerank.
 * Your challenge is to optimise it using OpenMP and/or MPI.
 * @author Ludovic Capelli (l.capelli@epcc.ed.ac.uk)
 **/

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/// The number of vertices in the graph.
#define GRAPH_ORDER 1000
/// Parameters used in pagerank convergence, do not change.
#define DAMPING_FACTOR 0.85
/// The number of seconds to not exceed forthe calculation loop.
#define MAX_TIME 10

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


// =============================================================================
// =============================================================================
// =============================================================================
// =============================================================================
// =============================================================================
// HOST INIT FUNCTIONS
void initialize_graph(void) {
  for (int i = 0; i < GRAPH_ORDER; i++) {
    for (int j = 0; j < GRAPH_ORDER; j++) {
      adjacency_matrix[i][j] = 0.0;
    }
  }
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
// =============================================================================
// =============================================================================

// =============================================================================
// =============================================================================
// =============================================================================
// =============================================================================
// MAIN
int main(int argc, char *argv[]) {
  bool sneaky = false;
  if (argc > 1) {
    sneaky = true;
  }

  // We do not need argv, this line silences potential compilation warnings.
  (void)argv;

  printf("This program has two graph generators: generate_nice_graph and "
         "generate_sneaky_graph. If you intend to submit, your code will be "
         "timed on the sneaky graph, remember to try both.\n");

  // Get the time at the very start.
  double start = omp_get_wtime();

  // #pragma omp target enter data map(alloc:adjacency_matrix)

  if (sneaky) {
    generate_sneaky_graph();
  } else {
    generate_nice_graph();
  }

// =============================================================================
// Main algorithm
// calculate_pagerank(pagerank);
// =============================================================================
  /// The array in which each vertex pagerank is stored.
  double pagerank[GRAPH_ORDER];
  // #pragma omp target enter data map(alloc:pagerank)
  double initial_rank = 1.0 / GRAPH_ORDER;

  // Initialise all vertices to 1/n.
  // #pragma omp target parallel for map(to:initial_rank) shared(pagerank)
  // schedule(static)
  #pragma omp parallel for shared(pagerank)
  for (int i = 0; i < GRAPH_ORDER; i++) {
    pagerank[i] = initial_rank;
  }

  double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;
  double diff = 1.0;
  size_t iteration = 0;
  double sstart = omp_get_wtime();
  double elapsed = omp_get_wtime() - sstart;
  double time_per_iteration = 0;
  double new_pagerank[GRAPH_ORDER];


  // #pragma omp target enter data map(alloc:adjacency_matrix[0:GRAPH_ORDER*GRAPH_ORDER], new_pagerank[0:GRAPH_ORDER], pagerank[0:GRAPH_ORDER])
  // #pragma omp target update to(pagerank[0:GRAPH_ORDER], adjacency_matrix[0:GRAPH_ORDER*GRAPH_ORDER])

  // #pragma omp target enter data map(alloc:adjacency_matrix, new_pagerank, pagerank)
  // #pragma omp target update to(pagerank, adjacency_matrix)
#pragma omp target data map(tofrom:new_pagerank, adjacency_matrix, pagerank, diff)
{
  while (elapsed < MAX_TIME && (elapsed + time_per_iteration) < MAX_TIME) {
    double iteration_start = omp_get_wtime();

    // #pragma omp target parallel for shared(adjacency_matrix) schedule(static)
    
    //=========
    // On DEVICE
    #pragma omp target teams distribute parallel for shared(new_pagerank) //map(tofrom:new_pagerank)
    for (int i = 0; i < GRAPH_ORDER; i++) {
      new_pagerank[i] = 0.0;
    }

    #pragma omp target teams distribute parallel for //map(tofrom:adjacency_matrix, new_pagerank, pagerank)
    for (int i = 0; i < GRAPH_ORDER; i++) {
      for (int j = 0; j < GRAPH_ORDER; j++) {
        if (adjacency_matrix[j][i] == 1.0) {
          int outdegree = 0;

          for (int k = 0; k < GRAPH_ORDER; k++) {
            if (adjacency_matrix[j][k] == 1.0) {
              outdegree++;
            }
          }
          new_pagerank[i] += pagerank[j] / (double)outdegree;
        }
      }
    }

    // #pragma omp target update from(new_pagerank[0:GRAPH_ORDER], pagerank[0:GRAPH_ORDER])

    // #pragma omp target teams distribute parallel for shared(new_pagerank)
    #pragma omp target teams distribute parallel for shared(new_pagerank) //map(tofrom:new_pagerank)
    for (int i = 0; i < GRAPH_ORDER; i++) {
      new_pagerank[i] = DAMPING_FACTOR * new_pagerank[i] + damping_value;
    }

    // ===========
    // ON HOST
    diff = 0.0;
    #pragma omp target teams distribute parallel for shared(new_pagerank, pagerank) reduction(+:diff) // map(tofrom:diff)
    for (int i = 0; i < GRAPH_ORDER; i++) {
      diff += fabs(new_pagerank[i] - pagerank[i]);
    }

    max_diff = (max_diff < diff) ? diff : max_diff;
    total_diff += diff;
    min_diff = (min_diff > diff) ? diff : min_diff;

    // ===========
    // ON HOST or DEVICE??
    #pragma omp target teams distribute parallel for shared(new_pagerank, pagerank)
    for (int i = 0; i < GRAPH_ORDER; i++) {
      pagerank[i] = new_pagerank[i];
    }

    // ===========
    // ON HOST
    double pagerank_total = 0.0;

    #pragma omp target teams distribute parallel for shared(pagerank) reduction(+:pagerank_total) //map(tofrom:pagerank_total)
    for (int i = 0; i < GRAPH_ORDER; i++) {
      pagerank_total += pagerank[i];
    }
    if (fabs(pagerank_total - 1.0) >= 1E-12) {
      printf(
          "[ERROR] Iteration %zu: sum of all pageranks is not 1 but %.12f.\n",
          iteration, pagerank_total);
          // return 1;
    }

    double iteration_end = omp_get_wtime();
    elapsed = omp_get_wtime() - sstart;
    iteration++;
    time_per_iteration = elapsed / iteration;
  }
}


  printf("%zu iterations achieved in %.2f seconds\n", iteration, elapsed);
// =============================================================================
// =============================================================================


  /* Need UPDATE FROM device to host before verification*/



  // Calculates the sum of all pageranks. It should be 1.0, so it can be used as
  // a quick verification.

  // #pragma omp target parallel for shared(pagerank) reduction(+:sum_ranks)

  // #pragma omp target update from(pagerank[0:GRAPH_ORDER])

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

  // #pragma omp target exit data map(delete:adjacency_matrix, pagerank)


  // #pragma omp target exit data map(delete:adjacency_matrix, new_pagerank, pagerank)

  return 0;
}
