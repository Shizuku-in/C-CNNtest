#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"

/* Big-Endian -> Little-Endian */
int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

/* read MNIST file */
float** read_mnist_images(char* filename, int* number_of_images) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Cannot open: %s\n", filename);
        exit(1);
    }

    int magic_number = 0;
    int n_rows = 0;
    int n_cols = 0;

    fread(&magic_number, sizeof(int), 1, file);
    magic_number = reverse_int(magic_number);

    fread(number_of_images, sizeof(int), 1, file);
    *number_of_images = reverse_int(*number_of_images);

    fread(&n_rows, sizeof(int), 1, file);
    n_rows = reverse_int(n_rows);

    fread(&n_cols, sizeof(int), 1, file);
    n_cols = reverse_int(n_cols);

    int image_size = n_rows * n_cols; // 28 * 28 = 784

    float** images = (float**)malloc(*number_of_images * sizeof(float*));

    for (int i = 0; i < *number_of_images; i++) {
        images[i] = (float*)malloc(image_size * sizeof(float));
        unsigned char temp = 0;
        for (int j = 0; j < image_size; j++) {
            fread(&temp, sizeof(unsigned char), 1, file);
            images[i][j] = (float)temp / 255.0; // normalize pixel values to [0, 1]
        }
    }

    fclose(file);
    return images;
}

/* read MNIST labels */
int* read_mnist_labels(char* filename, int* number_of_labels) {
    FILE* file = fopen(filename, "rb");
    if (!file) exit(1);

    int magic_number;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = reverse_int(magic_number);

    fread(number_of_labels, sizeof(int), 1, file);
    *number_of_labels = reverse_int(*number_of_labels);

    int* labels = (int*)malloc(*number_of_labels * sizeof(int));

    for (int i = 0; i < *number_of_labels; i++) {
        unsigned char temp = 0;
        fread(&temp, sizeof(unsigned char), 1, file);
        labels[i] = (int)temp;
    }
    fclose(file);
    return labels;
}

void free_mnist_images(float** images, int number_of_images) {
    for (int i = 0; i < number_of_images; i++) {
		free(images[i]);
    }
	free(images);
}

/* debug 

void print_mnist_image(float* image) {
    for (int i = 0; i < MNIST_IMG_HEIGHT; i++) {
        for (int j = 0; j < MNIST_IMG_WIDTH; j++) {
            float pixel = image[i * MNIST_IMG_WIDTH + j];
			if (pixel > 0.5) printf("# ");      // dark pixel
			else if (pixel > 0.2) printf(". "); // light pixel
			else printf("  ");                  // background
        }
        printf("\n");
    }
    printf("----------------------------------\n");
}
*/