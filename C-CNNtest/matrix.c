#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

Matrix* matrix_create(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
	m->data = (float*)calloc(rows * cols, sizeof(float)); // initialize to 0
    return m;
}

void matrix_free(Matrix* m) {
    if (m != NULL) {
        if (m->data != NULL) free(m->data);
        free(m);
    }
}

/* random to [-1.0, 1.0] */
void matrix_randomize(Matrix* m, int n) {
    float scale = sqrtf(2.0f / n);

    for (int i = 0; i < m->rows * m->cols; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        m->data[i] = (r * 2.0f - 1.0f) * scale;
    }
}

void matrix_fill(Matrix* m, float val) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = val;
    }
}

Matrix* matrix_multiply(Matrix* a, Matrix* b) {
    if (a->cols != b->rows) {
        printf("dim error (multiply): %dx%d * %dx%d\n", a->rows, a->cols, b->rows, b->cols);
        exit(1);
    }

    Matrix* c = matrix_create(a->rows, b->cols);

    for (int i = 0; i < c->rows; i++) {
        for (int j = 0; j < c->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            c->data[i * c->cols + j] = sum;
        }
    }
    return c;
}

Matrix* matrix_transpose(Matrix* m) {
    Matrix* t = matrix_create(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            t->data[j * t->cols + i] = m->data[i * m->cols + j];
        }
    }
    return t;
}

void matrix_add(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("dim error (plus)\n");
        exit(1);
    }
    for (int i = 0; i < a->rows * a->cols; i++) {
        a->data[i] += b->data[i];
    }
}

Matrix* matrix_copy(Matrix* m) {
    Matrix* copy = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows * m->cols; i++) {
        copy->data[i] = m->data[i];
    }
    return copy;
}

/* debug 

void matrix_print(Matrix* m) {
    printf("Matrix (%dx%d):\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%6.3f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}
*/