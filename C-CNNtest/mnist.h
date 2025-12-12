#ifndef MNIST_H
#define MNIST_H

#define MNIST_IMG_WIDTH 28
#define MNIST_IMG_HEIGHT 28
#define MNIST_IMG_SIZE 784  // 28 * 28

float** read_mnist_images(char* filename, int* number_of_images);

int* read_mnist_labels(char* filename, int* number_of_labels);

void free_mnist_images(float** images, int number_of_images);

// void print_mnist_image(float* image);

#endif