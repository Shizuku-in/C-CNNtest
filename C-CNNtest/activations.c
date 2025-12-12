#include <math.h>
#include "activations.h"

void apply_relu(Matrix* m) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        if (m->data[i] < 0) {
            m->data[i] = 0;
        }
    }
}

void apply_relu_derivative(Matrix* m) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        if (m->data[i] > 0) {
            m->data[i] = 1;
        }
        else {
            m->data[i] = 0;
        }
    }
}

void apply_softmax(Matrix* m) {
    float total = 0.0f;
    float max_val = -999999.0f;

	// find max value
    for (int i = 0; i < m->rows * m->cols; i++) {
        if (m->data[i] > max_val) max_val = m->data[i];
    }

	// calculate exp and total
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = expf(m->data[i] - max_val);
        total += m->data[i];
    }

	// normalize
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] /= total;
    }
}