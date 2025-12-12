#include <stdlib.h>
#include <stdio.h>
#include "layer_base.h"

Layer* layer_create(void* impl, LayerVTable* vtable) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->impl = impl;
    layer->vtable = vtable;
    layer->cached_input = NULL;
    layer->cached_output = NULL;
    return layer;
}

void layer_free(Layer* layer) {
    if (layer == NULL) return;
    
    // Free cached matrices
    if (layer->cached_input) matrix_free(layer->cached_input);
    if (layer->cached_output) matrix_free(layer->cached_output);
    
    // Call implementation-specific free
    if (layer->vtable && layer->vtable->free) {
        layer->vtable->free(layer);
    }
    
    free(layer);
}

Matrix* layer_forward(Layer* layer, Matrix* input) {
    if (layer == NULL || layer->vtable == NULL || layer->vtable->forward == NULL) {
        printf("Error: Invalid layer or forward function\n");
        return NULL;
    }
    
    // Cache input for backpropagation
    if (layer->cached_input) matrix_free(layer->cached_input);
    layer->cached_input = matrix_copy(input);
    
    // Call implementation-specific forward
    Matrix* output = layer->vtable->forward(layer, input);
    
    // Cache output
    if (layer->cached_output) matrix_free(layer->cached_output);
    layer->cached_output = matrix_copy(output);
    
    return output;
}

Matrix* layer_backward(Layer* layer, Matrix* grad_output, float learning_rate) {
    if (layer == NULL || layer->vtable == NULL || layer->vtable->backward == NULL) {
        printf("Error: Invalid layer or backward function\n");
        return NULL;
    }
    
    return layer->vtable->backward(layer, grad_output, learning_rate);
}

const char* layer_get_name(Layer* layer) {
    if (layer == NULL || layer->vtable == NULL || layer->vtable->get_name == NULL) {
        return "Unknown";
    }
    return layer->vtable->get_name(layer);
}
