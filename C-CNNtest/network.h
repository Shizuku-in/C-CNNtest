#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;

    Matrix* w1;
    Matrix* b1;

    Matrix* w2;
    Matrix* b2;
} NeuralNetwork;

NeuralNetwork* network_create(int input, int hidden, int output);
void network_train(NeuralNetwork* net, Matrix* input, Matrix* target, float learning_rate);
Matrix* network_predict(NeuralNetwork* net, Matrix* input);
void network_free(NeuralNetwork* net);

#endif