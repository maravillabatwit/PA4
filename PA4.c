#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

extern int* imageToMat(char* name, int* dims);
extern void matToImage(char* name, int* mat, int* dims);

void applyConvolution(int* matrix, int* temp, int height, int width, int k) {
    int kHalf = k / 2;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            int sum = 0;
            int counter = 0;
            for (int u = -kHalf; u <= kHalf; u++) {
                for (int v = -kHalf; v <= kHalf; v++) {
                    int ci = i + u;
                    int cj = j + v;
                    int cindex = ci * width + cj;
                    if (ci >= 0 && ci < height && cj >= 0 && cj < width) {
                        sum += matrix[cindex];
                        counter++;
                    }
                }
            }
            temp[index] = sum / counter;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* matrix = NULL;
    int* temp = NULL;
    int* dims = (int*)malloc(2 * sizeof(int));
    double total_time = 0.0, comm_time = 0.0, calc_time = 0.0;

    if (rank == 0) {
        matrix = imageToMat("image.jpg", dims);
        if (!matrix || !dims) {
            printf("Rank 0: Failed to load image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        temp = (int*)malloc(dims[0] * dims[1] * sizeof(int));
    }

    MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
    int height = dims[0];
    int width = dims[1];

    if (rank != 0) {
        matrix = (int*)malloc(height * width * sizeof(int));
        temp = (int*)malloc(height * width * sizeof(int));
    }

    double start_total = MPI_Wtime();
    MPI_Bcast(matrix, height * width, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: matrix[0]=%d, matrix[1]=%d\n", rank, matrix[0], matrix[1]);

    int rows_per_rank = height / size;
    int remainder = height % size;
    int start_row = rank * rows_per_rank + (rank < remainder ? rank : remainder);
    int end_row = start_row + rows_per_rank + (rank < remainder ? 1 : 0);
    int local_height = end_row - start_row;

    int* local_matrix = matrix + start_row * width;
    int* local_temp = temp + start_row * width;

    // Copy matrix to temp to ensure full array is initialized
    memcpy(temp, matrix, height * width * sizeof(int));
    double start_calc = MPI_Wtime();
    applyConvolution(local_matrix, local_temp, local_height, width, 10);
    double end_calc = MPI_Wtime();
    calc_time = end_calc - start_calc;
    printf("Rank %d: local_temp[0]=%d\n", rank, local_temp[0]);

    int* recv_counts = NULL;
    int* displacements = NULL;
    if (rank == 0) {
        recv_counts = (int*)malloc(size * sizeof(int));
        displacements = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int local_rows = rows_per_rank + (i < remainder ? 1 : 0);
            recv_counts[i] = local_rows * width;
            displacements[i] = i * rows_per_rank * width + (i < remainder ? i : remainder) * width;
        }
    }

    double start_gather = MPI_Wtime();
    MPI_Gatherv(local_temp, local_height * width, MPI_INT,
                temp, recv_counts, displacements, MPI_INT,
                0, MPI_COMM_WORLD);
    double end_gather = MPI_Wtime();
    comm_time = end_gather - start_gather;

    double end_total = MPI_Wtime();
    total_time = end_total - start_total;

    if (rank == 0) {
        printf("Rank 0: temp[0]=%d, temp[%d]=%d\n", temp[0], width, temp[width]);
        matToImage("output2.jpg", temp, dims);
        free(recv_counts);
        free(displacements);
    }
    free(matrix);
    free(temp);
    free(dims);

    MPI_Finalize();
    return 0;
}
