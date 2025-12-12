#ifndef LAYERS_H
#define LAYERS_H

#include "matrix.h"

typedef struct {
    int input_w, input_h;
    int input_depth;

    int kernel_size;
    int num_kernels;

    int output_w, output_h;

    Matrix** filters;
    Matrix* biases;
} ConvLayer;

typedef struct {
    int input_w, input_h, input_depth;
    int pool_size;
    int output_w, output_h;

    int* mask;
} PoolLayer;


ConvLayer* conv_create(int in_w, int in_h, int in_depth, int k_size, int num_k);
void conv_free(ConvLayer* l);

Matrix* conv_forward(ConvLayer* l, Matrix* input);

Matrix* conv_backward(ConvLayer* l, Matrix* d_out, Matrix* input, float lr);

PoolLayer* pool_create(int in_w, int in_h, int in_depth, int pool_size);
void pool_free(PoolLayer* l);
Matrix* pool_forward(PoolLayer* l, Matrix* input);
Matrix* pool_backward(PoolLayer* l, Matrix* d_out);

#endif