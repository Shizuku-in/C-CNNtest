#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "matrix.h"

// x > 0 ? x : 0
void apply_relu(Matrix* m);

// x > 0 ? 1 : 0
void apply_relu_derivative(Matrix* m);

void apply_softmax(Matrix* m);

#endif