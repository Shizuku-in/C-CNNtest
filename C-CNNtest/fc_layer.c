#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "fc_layer.h"
#include "activations.h"

// ===== Fully Connected Layer =====

FCLayer* fc_create(int input_size, int output_size) {
    FCLayer* layer = (FCLayer*)malloc(sizeof(FCLayer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    layer->weights = matrix_create(output_size, input_size);
    matrix_randomize(layer->weights, input_size);
    
    layer->biases = matrix_create(output_size, 1);
    matrix_fill(layer->biases, 0.0f);
    
    return layer;
}

void fc_free(FCLayer* layer) {
    if (layer) {
        matrix_free(layer->weights);
        matrix_free(layer->biases);
        free(layer);
    }
}

Matrix* fc_forward(FCLayer* layer, Matrix* input) {
    // Z = W * X + b
    Matrix* output = matrix_multiply(layer->weights, input);
    matrix_add(output, layer->biases);
    return output;
}

Matrix* fc_backward(FCLayer* layer, Matrix* input, Matrix* grad_output, float learning_rate) {
    // Compute gradient w.r.t. weights: dW = grad_output * input^T
    Matrix* input_t = matrix_transpose(input);
    Matrix* grad_weights = matrix_multiply(grad_output, input_t);
    matrix_free(input_t);
    
    // Update weights: W = W - lr * dW
    for (int i = 0; i < layer->weights->rows * layer->weights->cols; i++) {
        layer->weights->data[i] -= learning_rate * grad_weights->data[i];
    }
    
    // Update biases: b = b - lr * grad_output
    for (int i = 0; i < layer->biases->rows; i++) {
        layer->biases->data[i] -= learning_rate * grad_output->data[i];
    }
    
    matrix_free(grad_weights);
    
    // Compute gradient w.r.t. input: grad_input = W^T * grad_output
    Matrix* weights_t = matrix_transpose(layer->weights);
    Matrix* grad_input = matrix_multiply(weights_t, grad_output);
    matrix_free(weights_t);
    
    return grad_input;
}

// Layer Interface for FC
static Matrix* fc_layer_forward_impl(Layer* self, Matrix* input) {
    FCLayer* fc = (FCLayer*)self->impl;
    return fc_forward(fc, input);
}

static Matrix* fc_layer_backward_impl(Layer* self, Matrix* grad_output, float learning_rate) {
    FCLayer* fc = (FCLayer*)self->impl;
    return fc_backward(fc, self->cached_input, grad_output, learning_rate);
}

static void fc_layer_free_impl(Layer* self) {
    FCLayer* fc = (FCLayer*)self->impl;
    fc_free(fc);
}

static const char* fc_layer_get_name_impl(Layer* self) {
    return "FCLayer";
}

static LayerVTable fc_vtable = {
    .forward = fc_layer_forward_impl,
    .backward = fc_layer_backward_impl,
    .free = fc_layer_free_impl,
    .get_name = fc_layer_get_name_impl
};

Layer* fc_layer_create(int input_size, int output_size) {
    FCLayer* fc = fc_create(input_size, output_size);
    return layer_create(fc, &fc_vtable);
}

// ===== ReLU Activation Layer =====

static Matrix* relu_layer_forward_impl(Layer* self, Matrix* input) {
    Matrix* output = matrix_copy(input);
    apply_relu(output);
    return output;
}

static Matrix* relu_layer_backward_impl(Layer* self, Matrix* grad_output, float learning_rate) {
    // Apply ReLU derivative: gradient passes through if input > 0
    Matrix* grad_input = matrix_copy(grad_output);
    Matrix* cached_output = self->cached_output;
    
    for (int i = 0; i < grad_input->rows * grad_input->cols; i++) {
        if (cached_output->data[i] <= 0.0f) {
            grad_input->data[i] = 0.0f;
        }
    }
    
    return grad_input;
}

static void relu_layer_free_impl(Layer* self) {
    ReLULayer* relu = (ReLULayer*)self->impl;
    free(relu);
}

static const char* relu_layer_get_name_impl(Layer* self) {
    return "ReLULayer";
}

static LayerVTable relu_vtable = {
    .forward = relu_layer_forward_impl,
    .backward = relu_layer_backward_impl,
    .free = relu_layer_free_impl,
    .get_name = relu_layer_get_name_impl
};

Layer* relu_layer_create(int size) {
    ReLULayer* relu = (ReLULayer*)malloc(sizeof(ReLULayer));
    relu->size = size;
    return layer_create(relu, &relu_vtable);
}

// ===== Softmax Activation Layer =====

static Matrix* softmax_layer_forward_impl(Layer* self, Matrix* input) {
    Matrix* output = matrix_copy(input);
    apply_softmax(output);
    return output;
}

static Matrix* softmax_layer_backward_impl(Layer* self, Matrix* grad_output, float learning_rate) {
    // For Softmax with CrossEntropy, gradient is already computed (y_pred - y_true)
    // Just pass through
    return matrix_copy(grad_output);
}

static void softmax_layer_free_impl(Layer* self) {
    SoftmaxLayer* softmax = (SoftmaxLayer*)self->impl;
    free(softmax);
}

static const char* softmax_layer_get_name_impl(Layer* self) {
    return "SoftmaxLayer";
}

static LayerVTable softmax_vtable = {
    .forward = softmax_layer_forward_impl,
    .backward = softmax_layer_backward_impl,
    .free = softmax_layer_free_impl,
    .get_name = softmax_layer_get_name_impl
};

Layer* softmax_layer_create(int size) {
    SoftmaxLayer* softmax = (SoftmaxLayer*)malloc(sizeof(SoftmaxLayer));
    softmax->size = size;
    return layer_create(softmax, &softmax_vtable);
}

// ===== Flatten Layer =====

static Matrix* flatten_layer_forward_impl(Layer* self, Matrix* input) {
    // Convert multi-dimensional input to 1D vector
    FlattenLayer* flatten = (FlattenLayer*)self->impl;
    Matrix* output = matrix_create(flatten->output_size, 1);
    memcpy(output->data, input->data, flatten->output_size * sizeof(float));
    return output;
}

static Matrix* flatten_layer_backward_impl(Layer* self, Matrix* grad_output, float learning_rate) {
    // Unflatten gradient back to original shape
    FlattenLayer* flatten = (FlattenLayer*)self->impl;
    Matrix* grad_input = matrix_create(flatten->input_depth, flatten->input_h * flatten->input_w);
    memcpy(grad_input->data, grad_output->data, flatten->output_size * sizeof(float));
    return grad_input;
}

static void flatten_layer_free_impl(Layer* self) {
    FlattenLayer* flatten = (FlattenLayer*)self->impl;
    free(flatten);
}

static const char* flatten_layer_get_name_impl(Layer* self) {
    return "FlattenLayer";
}

static LayerVTable flatten_vtable = {
    .forward = flatten_layer_forward_impl,
    .backward = flatten_layer_backward_impl,
    .free = flatten_layer_free_impl,
    .get_name = flatten_layer_get_name_impl
};

Layer* flatten_layer_create(int depth, int h, int w) {
    FlattenLayer* flatten = (FlattenLayer*)malloc(sizeof(FlattenLayer));
    flatten->input_depth = depth;
    flatten->input_h = h;
    flatten->input_w = w;
    flatten->output_size = depth * h * w;
    return layer_create(flatten, &flatten_vtable);
}
