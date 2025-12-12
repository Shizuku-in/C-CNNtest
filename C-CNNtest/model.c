#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "model.h"

Model* model_create(float learning_rate) {
    Model* model = (Model*)malloc(sizeof(Model));
    model->num_layers = 0;
    model->learning_rate = learning_rate;
    memset(model->layers, 0, sizeof(model->layers));
    return model;
}

void model_free(Model* model) {
    if (model == NULL) return;
    
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) {
            layer_free(model->layers[i]);
        }
    }
    free(model);
}

void model_add_layer(Model* model, Layer* layer) {
    if (model->num_layers >= MAX_LAYERS) {
        printf("Error: Maximum number of layers (%d) reached\n", MAX_LAYERS);
        return;
    }
    model->layers[model->num_layers++] = layer;
}

Matrix* model_forward(Model* model, Matrix* input) {
    Matrix* output = input;
    
    for (int i = 0; i < model->num_layers; i++) {
        Matrix* next_output = layer_forward(model->layers[i], output);
        
        // Free intermediate outputs (except the original input)
        if (output != input) {
            matrix_free(output);
        }
        
        output = next_output;
    }
    
    return output;
}

void model_backward(Model* model, Matrix* grad_output) {
    Matrix* grad = grad_output;
    
    // Backward pass through all layers in reverse order
    for (int i = model->num_layers - 1; i >= 0; i--) {
        Matrix* next_grad = layer_backward(model->layers[i], grad, model->learning_rate);
        
        // Free intermediate gradients (except the original grad_output)
        if (grad != grad_output) {
            matrix_free(grad);
        }
        
        grad = next_grad;
    }
    
    // Free the final gradient
    if (grad != grad_output) {
        matrix_free(grad);
    }
}

void model_train_step(Model* model, Matrix* input, Matrix* target) {
    // Forward pass
    Matrix* output = model_forward(model, input);
    
    // Compute gradient (for Softmax + CrossEntropy: grad = output - target)
    Matrix* grad_output = matrix_create(target->rows, target->cols);
    for (int i = 0; i < target->rows * target->cols; i++) {
        grad_output->data[i] = output->data[i] - target->data[i];
    }
    
    // Backward pass
    model_backward(model, grad_output);
    
    // Free memory
    matrix_free(output);
    matrix_free(grad_output);
}

Matrix* model_predict(Model* model, Matrix* input) {
    return model_forward(model, input);
}

void model_summary(Model* model) {
    printf("\n========== Model Summary ==========\n");
    printf("Total layers: %d\n", model->num_layers);
    printf("Learning rate: %.4f\n", model->learning_rate);
    printf("-----------------------------------\n");
    
    for (int i = 0; i < model->num_layers; i++) {
        printf("Layer %d: %s\n", i + 1, layer_get_name(model->layers[i]));
    }
    
    printf("===================================\n\n");
}

int model_argmax(Matrix* output) {
    int max_idx = 0;
    float max_val = output->data[0];
    
    for (int i = 1; i < output->rows * output->cols; i++) {
        if (output->data[i] > max_val) {
            max_val = output->data[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}
