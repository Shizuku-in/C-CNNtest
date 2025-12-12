#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int rows;
    int cols;
    float* data;
} Matrix;

Matrix* matrix_create(int rows, int cols);

void matrix_free(Matrix* m);

void matrix_randomize(Matrix* m, int n);

void matrix_fill(Matrix* m, float val);

Matrix* matrix_multiply(Matrix* a, Matrix* b);

void matrix_add(Matrix* a, Matrix* b);

Matrix* matrix_transpose(Matrix* m);

Matrix* matrix_copy(Matrix* m);

// void matrix_print(Matrix* m);

#endif