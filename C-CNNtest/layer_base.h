#ifndef LAYER_BASE_H
#define LAYER_BASE_H

#include "matrix.h"

// Forward declaration
typedef struct Layer Layer;

// Virtual function table for polymorphism
typedef struct {
    Matrix* (*forward)(Layer* self, Matrix* input);
    Matrix* (*backward)(Layer* self, Matrix* grad_output, float learning_rate);
    void (*free)(Layer* self);
    const char* (*get_name)(Layer* self);
} LayerVTable;

// Base Layer structure (common for all layers)
typedef struct Layer {
    LayerVTable* vtable;
    void* impl;  // Pointer to actual implementation (ConvLayer, PoolLayer, etc.)
    
    // Cache for backpropagation
    Matrix* cached_input;
    Matrix* cached_output;
} Layer;

// Layer constructor and destructor
Layer* layer_create(void* impl, LayerVTable* vtable);
void layer_free(Layer* layer);

// Common interface functions
Matrix* layer_forward(Layer* layer, Matrix* input);
Matrix* layer_backward(Layer* layer, Matrix* grad_output, float learning_rate);
const char* layer_get_name(Layer* layer);

#endif
