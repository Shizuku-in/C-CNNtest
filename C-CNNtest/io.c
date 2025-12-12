#include <stdio.h>
#include <stdlib.h>
#include "layers.h"
#include "network.h"

void save_model(const char* filename, ConvLayer* conv, NeuralNetwork* fc) {
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("Cannot open model.\n"); return; }

    printf("Saving to %s ...\n", filename);

    // write filters: num_kernels * input_depth * k * k
    int conv_w_size = conv->num_kernels * conv->input_depth * conv->kernel_size * conv->kernel_size;
    for (int k = 0; k < conv->num_kernels; k++) {
        fwrite(conv->filters[k]->data, sizeof(float), conv->filters[k]->rows * conv->filters[k]->cols, f);
    }
    // write biases
    fwrite(conv->biases->data, sizeof(float), conv->num_kernels, f);

    fwrite(fc->w1->data, sizeof(float), fc->w1->rows * fc->w1->cols, f);
    fwrite(fc->b1->data, sizeof(float), fc->b1->rows * fc->b1->cols, f);
    fwrite(fc->w2->data, sizeof(float), fc->w2->rows * fc->w2->cols, f);
    fwrite(fc->b2->data, sizeof(float), fc->b2->rows * fc->b2->cols, f);

    fclose(f);
    printf("Succeed to save model!\n");
}

void load_model(const char* filename, ConvLayer* conv, NeuralNetwork* fc) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("Cannot find model.\n"); exit(1); }

    printf("Loading model...\n");

    for (int k = 0; k < conv->num_kernels; k++) {
        fread(conv->filters[k]->data, sizeof(float), conv->filters[k]->rows * conv->filters[k]->cols, f);
    }
    fread(conv->biases->data, sizeof(float), conv->num_kernels, f);

    fread(fc->w1->data, sizeof(float), fc->w1->rows * fc->w1->cols, f);
    fread(fc->b1->data, sizeof(float), fc->b1->rows * fc->b1->cols, f);
    fread(fc->w2->data, sizeof(float), fc->w2->rows * fc->w2->cols, f);
    fread(fc->b2->data, sizeof(float), fc->b2->rows * fc->b2->cols, f);

    fclose(f);
    printf("Succeed to load model!\n");
}