#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern int* imageToMat(char* name, int* dims);
extern void matToImage(char* name, int* mat, int* dims);

void applyConvolution(int* matrix, int* temp, int height, int width, int k) {
    int kHalf = k / 2;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            int sum = 0;
            int counter = 0;

            // Convolution sum
            for (int u = -kHalf; u <= kHalf; u++) {
                for (int v = -kHalf; v <= kHalf; v++) {
                    int ci = i + u;
                    int cj = j + v;
                    int cindex = ci * width + cj;
                    if (ci >= 0 && ci < height && cj >= 0 && cj < width) {
                        sum += matrix[cindex]; // Simple average convolution (no kernel weights)
                        counter++;
                    }
                }
            }
            temp[index] = sum / counter; // Average the sum
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
    int* dims = NULL;

    if (rank == 0) {
        // 1. Read image on rank 0
        dims = (int*)malloc(2 * sizeof(int));
        matrix = imageToMat("image.jpg", dims);
        if (!matrix || !dims) {
            printf("Error loading image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        temp = (int*)malloc(dims[0] * dims[1] * sizeof(int));
    }

    // 2. Broadcast dimensions
    int local_dims[2];
    MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
    int height = dims[0];
    int width = dims[1];

    // Allocate memory on all ranks
    if (rank != 0) {
        matrix = (int*)malloc(height * width * sizeof(int));
        temp = (int*)malloc(height * width * sizeof(int));
    }

    // 3. Broadcast image data
    MPI_Bcast(matrix, height * width, MPI_INT, 0, MPI_COMM_WORLD);

    // 4. Compute local portion
    int rows_per_rank = height / size;
    int start_row = rank * rows_per_rank;
    int end_row = (rank == size - 1) ? height : start_row + rows_per_rank;
    int local_height = end_row - start_row;

    int* local_matrix = matrix + start_row * width;
    int* local_temp = temp + start_row * width;

    int k = 10; // Kernel size
    applyConvolution(local_matrix, local_temp, local_height, width, k);

    // 5. Gather results to rank 0
    MPI_Gather(local_temp, local_height * width, MPI_INT,
               temp, local_height * width, MPI_INT,
               0, MPI_COMM_WORLD);

    // 6. Save image on rank 0
    if (rank == 0) {
        matToImage("processedImage.jpg", temp, dims);
        free(matrix);
        free(temp);
        free(dims);
    } else {
        free(matrix);
        free(temp);
    }

    MPI_Finalize();
    return 0;
}
