#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "matrix.h"
#include "layer_base.h"

// Fully Connected Layer
typedef struct {
    int input_size;
    int output_size;
    
    Matrix* weights;
    Matrix* biases;
} FCLayer;

// ReLU Activation Layer
typedef struct {
    int size;
} ReLULayer;

// Softmax Activation Layer
typedef struct {
    int size;
} SoftmaxLayer;

// Flatten Layer
typedef struct {
    int input_depth;
    int input_h;
    int input_w;
    int output_size;
} FlattenLayer;

// FC Layer functions
Layer* fc_layer_create(int input_size, int output_size);
FCLayer* fc_create(int input_size, int output_size);
void fc_free(FCLayer* layer);
Matrix* fc_forward(FCLayer* layer, Matrix* input);
Matrix* fc_backward(FCLayer* layer, Matrix* input, Matrix* grad_output, float learning_rate);

// ReLU Layer functions
Layer* relu_layer_create(int size);

// Softmax Layer functions
Layer* softmax_layer_create(int size);

// Flatten Layer functions
Layer* flatten_layer_create(int depth, int h, int w);

#endif
