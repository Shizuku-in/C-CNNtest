#include <stdlib.h>
#include <stdio.h>
#include "network.h"
#include "activations.h"

NeuralNetwork* network_create(int input, int hidden, int output) {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->input_size = input;
    net->hidden_size = hidden;
    net->output_size = output;

	// initialize
    net->w1 = matrix_create(hidden, input);
    matrix_randomize(net->w1, input);
    net->b1 = matrix_create(hidden, 1);
    matrix_fill(net->b1, 0.0f);

    net->w2 = matrix_create(output, hidden);
    matrix_randomize(net->w2, hidden);
    net->b2 = matrix_create(output, 1);
    matrix_fill(net->b2, 0.0f);

    return net;
}

void network_free(NeuralNetwork* net) {
    matrix_free(net->w1); matrix_free(net->b1);
    matrix_free(net->w2); matrix_free(net->b2);
    free(net);
}

// input: (784 x 1), target: (10 x 1)
void network_train(NeuralNetwork* net, Matrix* input, Matrix* target, float learning_rate) {
    /* Forward Pass */
    
    // Hidden Layer: Z1 = W1 * X + B1
    Matrix* z1 = matrix_multiply(net->w1, input);
    matrix_add(z1, net->b1);

    // Activation: A1 = ReLU(Z1)
    Matrix* a1 = matrix_copy(z1);
    apply_relu(a1);

    // Output Layer: Z2 = W2 * A1 + B2
    Matrix* z2 = matrix_multiply(net->w2, a1);
    matrix_add(z2, net->b2);

    // Activation: Output = Softmax(Z2)
    Matrix* output = matrix_copy(z2);
    apply_softmax(output);

    /* Backpropagation */

    // Error = Output - Target
    Matrix* output_errors = matrix_create(net->output_size, 1);
    for (int i = 0; i < net->output_size; i++) {
        output_errors->data[i] = output->data[i] - target->data[i];
    }

    // dW2 = Error * A1^T
    Matrix* a1_t = matrix_transpose(a1);
    Matrix* dw2 = matrix_multiply(output_errors, a1_t);

    // Hidden Error = W2^T * Output Error
    Matrix* w2_t = matrix_transpose(net->w2);
    Matrix* hidden_errors = matrix_multiply(w2_t, output_errors);

    // d_Hidden = Hidden Error * ReLU'(Z1)
    Matrix* z1_prime = matrix_copy(z1);
    apply_relu_derivative(z1_prime);

    // Hadamard product
    for (int i = 0; i < hidden_errors->rows * hidden_errors->cols; i++) {
        hidden_errors->data[i] *= z1_prime->data[i];
    }

    // dW1 = Hidden Error * Input^T
    Matrix* input_t = matrix_transpose(input);
    Matrix* dw1 = matrix_multiply(hidden_errors, input_t);

    /* SGD */

    // W = W - lr * dW
    for (int i = 0; i < net->w2->rows * net->w2->cols; i++)
        net->w2->data[i] -= learning_rate * dw2->data[i];
    for (int i = 0; i < net->b2->rows * net->b2->cols; i++)
        net->b2->data[i] -= learning_rate * output_errors->data[i];

    for (int i = 0; i < net->w1->rows * net->w1->cols; i++)
        net->w1->data[i] -= learning_rate * dw1->data[i];
    for (int i = 0; i < net->b1->rows * net->b1->cols; i++)
        net->b1->data[i] -= learning_rate * hidden_errors->data[i];


    matrix_free(z1); matrix_free(a1); matrix_free(z2); matrix_free(output);
    matrix_free(output_errors); matrix_free(dw2); matrix_free(a1_t);
    matrix_free(hidden_errors); matrix_free(w2_t); matrix_free(z1_prime);
    matrix_free(dw1); matrix_free(input_t);
}

Matrix* network_predict(NeuralNetwork* net, Matrix* input) {
    Matrix* z1 = matrix_multiply(net->w1, input);
    matrix_add(z1, net->b1);
    apply_relu(z1);

    Matrix* z2 = matrix_multiply(net->w2, z1);
    matrix_add(z2, net->b2);
    apply_softmax(z2);

    matrix_free(z1);
    return z2;
}