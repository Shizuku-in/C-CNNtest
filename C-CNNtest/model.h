#ifndef MODEL_H
#define MODEL_H

#include "layer_base.h"
#include "matrix.h"

#define MAX_LAYERS 32

typedef struct {
    Layer* layers[MAX_LAYERS];
    int num_layers;
    
    float learning_rate;
} Model;

// Model management
Model* model_create(float learning_rate);
void model_free(Model* model);

// Layer management
void model_add_layer(Model* model, Layer* layer);

// Forward and backward propagation
Matrix* model_forward(Model* model, Matrix* input);
void model_backward(Model* model, Matrix* grad_output);

// Training
void model_train_step(Model* model, Matrix* input, Matrix* target);

// Inference
Matrix* model_predict(Model* model, Matrix* input);

// Utilities
void model_summary(Model* model);
int model_argmax(Matrix* output);

#endif
