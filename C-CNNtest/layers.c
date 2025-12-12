#include <stdio.h>
#include <stdlib.h>
#include "layers.h"

float get_val(Matrix* m, int ch, int r, int c, int w, int h) {
    if (r < 0 || r >= h || c < 0 || c >= w) return 0.0f;
    return m->data[ch * (w * h) + r * w + c];
}

ConvLayer* conv_create(int in_w, int in_h, int in_depth, int k_size, int num_k) {
    ConvLayer* l = (ConvLayer*)malloc(sizeof(ConvLayer));
    l->input_w = in_w; l->input_h = in_h; l->input_depth = in_depth;
    l->kernel_size = k_size; l->num_kernels = num_k;

    l->output_w = in_w - k_size + 1;
    l->output_h = in_h - k_size + 1;

    l->filters = (Matrix**)malloc(sizeof(Matrix*) * num_k);
    for (int i = 0; i < num_k; i++) {
        l->filters[i] = matrix_create(in_depth, k_size * k_size);
        matrix_randomize(l->filters[i], in_depth * k_size * k_size);
    }

    l->biases = matrix_create(num_k, 1);
    matrix_fill(l->biases, 0.0f);

    return l;
}

void conv_free(ConvLayer* l) {
    for (int i = 0; i < l->num_kernels; i++) matrix_free(l->filters[i]);
    free(l->filters);
    matrix_free(l->biases);
    free(l);
}

Matrix* conv_forward(ConvLayer* l, Matrix* input) {
    // Output shape: (num_kernels, output_h * output_w)
    Matrix* output = matrix_create(l->num_kernels, l->output_h * l->output_w);

    for (int k = 0; k < l->num_kernels; k++) {
        for (int i = 0; i < l->output_h; i++) {
            for (int j = 0; j < l->output_w; j++) {

                float sum = 0.0f;
                for (int d = 0; d < l->input_depth; d++) {
                    for (int ki = 0; ki < l->kernel_size; ki++) {
                        for (int kj = 0; kj < l->kernel_size; kj++) {
                            float val_in = get_val(input, d, i + ki, j + kj, l->input_w, l->input_h);
                            float val_w = l->filters[k]->data[d * (l->kernel_size * l->kernel_size) + ki * l->kernel_size + kj];
                            sum += val_in * val_w;
                        }
                    }
                }
                sum += l->biases->data[k];
                output->data[k * (l->output_h * l->output_w) + i * l->output_w + j] = sum;
            }
        }
    }
    return output;
}

Matrix* conv_backward(ConvLayer* l, Matrix* d_out, Matrix* input, float lr) {
    Matrix* d_input = matrix_create(l->input_depth, l->input_w * l->input_h);
    matrix_fill(d_input, 0.0f);

    for (int k = 0; k < l->num_kernels; k++) {
        for (int i = 0; i < l->output_h; i++) {
            for (int j = 0; j < l->output_w; j++) {
                float gradient = d_out->data[k * (l->output_w * l->output_h) + i * l->output_w + j];

                // dL/dW = Input * Gradient
                for (int d = 0; d < l->input_depth; d++) {
                    for (int ki = 0; ki < l->kernel_size; ki++) {
                        for (int kj = 0; kj < l->kernel_size; kj++) {
                            float val_in = get_val(input, d, i + ki, j + kj, l->input_w, l->input_h);

                            int w_idx = d * (l->kernel_size * l->kernel_size) + ki * l->kernel_size + kj;

                            // SGD: W = W - lr * (Input * Gradient)
                            l->filters[k]->data[w_idx] -= lr * (val_in * gradient);

                            // dL/dX += W * Gradient
                            float val_w = l->filters[k]->data[w_idx];

                            int in_idx = d * (l->input_w * l->input_h) + (i + ki) * l->input_w + (j + kj);
                            d_input->data[in_idx] += val_w * gradient;
                        }
                    }
                }
                l->biases->data[k] -= lr * gradient;
            }
        }
    }
    return d_input;
}

PoolLayer* pool_create(int in_w, int in_h, int in_depth, int pool_size) {
    PoolLayer* l = (PoolLayer*)malloc(sizeof(PoolLayer));
    l->input_w = in_w; l->input_h = in_h; l->input_depth = in_depth;
    l->pool_size = pool_size;
    l->output_w = in_w / pool_size;
    l->output_h = in_h / pool_size;
    l->mask = (int*)calloc(in_depth * l->output_w * l->output_h, sizeof(int));
    return l;
}

void pool_free(PoolLayer* l) {
    free(l->mask);
    free(l);
}

Matrix* pool_forward(PoolLayer* l, Matrix* input) {
    Matrix* output = matrix_create(l->input_depth, l->output_w * l->output_h);

    for (int d = 0; d < l->input_depth; d++) {
        for (int i = 0; i < l->output_h; i++) {
            for (int j = 0; j < l->output_w; j++) {

                float max_val = -999999.0f;
                int max_idx = -1;

                for (int pi = 0; pi < l->pool_size; pi++) {
                    for (int pj = 0; pj < l->pool_size; pj++) {
                        int r = i * l->pool_size + pi;
                        int c = j * l->pool_size + pj;
                        int idx = d * (l->input_w * l->input_h) + r * l->input_w + c;

                        if (input->data[idx] > max_val) {
                            max_val = input->data[idx];
                            max_idx = idx;
                        }
                    }
                }

                int out_idx = d * (l->output_w * l->output_h) + i * l->output_w + j;
                output->data[out_idx] = max_val;
                l->mask[out_idx] = max_idx;
            }
        }
    }
    return output;
}

Matrix* pool_backward(PoolLayer* l, Matrix* d_out) {
    // upsample
    Matrix* d_input = matrix_create(l->input_depth, l->input_w * l->input_h);
    matrix_fill(d_input, 0.0f);

    int num_outputs = l->input_depth * l->output_w * l->output_h;
    for (int i = 0; i < num_outputs; i++) {
        int input_idx = l->mask[i];
        d_input->data[input_idx] = d_out->data[i];
    }

    return d_input;
}