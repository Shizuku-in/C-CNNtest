#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mnist.h"
#include "matrix.h"
#include "layers.h"
#include "activations.h"
#include "network.h"
#include "io.h" 
#include "image_util.h"

#define CONV_K_SIZE 3
#define CONV_K_NUM 8
#define POOL_SIZE 2
#define FC_HIDDEN 128
#define FC_OUT 10

// edit this >>
#define IMAGEPATH "data/train-images.idx3-ubyte"
#define LABELPATH "data/train-labels.idx1-ubyte"

void build_architecture(ConvLayer** conv, PoolLayer** pool, NeuralNetwork** fc);
int argmax(Matrix* m);
void run_training_mode(int epochs);
void run_inference_mode(char* model_path, char* img_path);

int main() {
    srand((unsigned int)time(NULL));
    int choice;
	printf("1: train\n2: infer\n");

    if (scanf("%d", &choice) != 1) {
        printf("invalid...\n");
        return 1;
    }

    if (choice == 1) {
        int epochs;
        printf("Epochs: ");
        scanf("%d", &epochs);
        run_training_mode(epochs);
    }
    else if (choice == 2) {
        char model_path[256];
        char img_path[256];

        printf("Model: ");
        scanf("%s", model_path);

        printf("Image: ");
        scanf("%s", img_path);

        run_inference_mode(model_path, img_path);
    }
    else {
        printf("invalid...\n");
    }

    return 0;
}

void build_architecture(ConvLayer** conv, PoolLayer** pool, NeuralNetwork** fc) {
    *conv = conv_create(28, 28, 1, CONV_K_SIZE, CONV_K_NUM);
    // Conv out: 26x26x8
    *pool = pool_create(26, 26, 8, POOL_SIZE);
    // Pool out: 13x13x8 = 1352
    *fc = network_create(13 * 13 * 8, FC_HIDDEN, FC_OUT);
}

int argmax(Matrix* m) {
    int max_idx = 0;
    float max_val = m->data[0];
    for (int i = 1; i < m->rows * m->cols; i++) {
        if (m->data[i] > max_val) { max_val = m->data[i]; max_idx = i; }
    }
    return max_idx;
}

void run_training_mode(int epochs) {
    printf("--- Training (epochs = %d) ---\n", epochs);

    int num_imgs, num_lbls;
    float** images = read_mnist_images(IMAGEPATH, &num_imgs);
    int* labels = read_mnist_labels(LABELPATH, &num_lbls);

    ConvLayer* conv; PoolLayer* pool; NeuralNetwork* fc;
    build_architecture(&conv, &pool, &fc);

    float lr = 0.01f;

    for (int e = 0; e < epochs; e++) {
        int correct = 0;
        for (int i = 0; i < num_imgs; i++) {
            Matrix* input = matrix_create(1, 784);
            memcpy(input->data, images[i], 784 * sizeof(float));

            Matrix* target = matrix_create(10, 1);
            matrix_fill(target, 0.0f);
            target->data[labels[i]] = 1.0f;

            Matrix* conv_out = conv_forward(conv, input);
            apply_relu(conv_out);
            Matrix* pool_out = pool_forward(pool, conv_out);

            Matrix* flattened = matrix_create(pool_out->rows * pool_out->cols, 1);

            memcpy(flattened->data, pool_out->data, flattened->rows * sizeof(float));

            Matrix* z1 = matrix_multiply(fc->w1, flattened);
            matrix_add(z1, fc->b1);
            Matrix* a1 = matrix_copy(z1); apply_relu(a1);

            Matrix* z2 = matrix_multiply(fc->w2, a1);
            matrix_add(z2, fc->b2);
            Matrix* output = matrix_copy(z2); apply_softmax(output);

            if (argmax(output) == labels[i]) correct++;

            Matrix* output_err = matrix_create(10, 1);
            for (int k = 0; k < 10; k++) output_err->data[k] = output->data[k] - target->data[k];

            Matrix* dw2 = matrix_multiply(output_err, matrix_transpose(a1));
            Matrix* hidden_err = matrix_multiply(matrix_transpose(fc->w2), output_err);
            apply_relu_derivative(z1);
            for (int k = 0; k < hidden_err->rows; k++) hidden_err->data[k] *= z1->data[k];

            Matrix* dw1 = matrix_multiply(hidden_err, matrix_transpose(flattened));
            Matrix* d_flattened = matrix_multiply(matrix_transpose(fc->w1), hidden_err);

            Matrix* d_pool_out = matrix_create(pool->input_depth, pool->output_w * pool->output_h);
            memcpy(d_pool_out->data, d_flattened->data, d_pool_out->rows * d_pool_out->cols * sizeof(float));

            Matrix* d_conv_out = pool_backward(pool, d_pool_out);
            apply_relu_derivative(conv_out);
            for (int k = 0; k < d_conv_out->rows * d_conv_out->cols; k++) d_conv_out->data[k] *= conv_out->data[k];

            Matrix* d_input = conv_backward(conv, d_conv_out, input, lr);

            for (int k = 0; k < fc->w2->rows * fc->w2->cols; k++) fc->w2->data[k] -= lr * dw2->data[k];
            for (int k = 0; k < fc->b2->rows; k++) fc->b2->data[k] -= lr * output_err->data[k];
            for (int k = 0; k < fc->w1->rows * fc->w1->cols; k++) fc->w1->data[k] -= lr * dw1->data[k];
            for (int k = 0; k < fc->b1->rows; k++) fc->b1->data[k] -= lr * hidden_err->data[k];

            matrix_free(input); matrix_free(target); matrix_free(conv_out); matrix_free(pool_out);
            matrix_free(flattened); matrix_free(z1); matrix_free(a1); matrix_free(z2); matrix_free(output);
            matrix_free(output_err); matrix_free(dw2); matrix_free(hidden_err); matrix_free(dw1);
            matrix_free(d_flattened); matrix_free(d_pool_out); matrix_free(d_conv_out); matrix_free(d_input);

            if (i % 100 == 0) printf("Epoch %d: %d/%d (Acc: %.2f%%)\r", e + 1, i, num_imgs, (float)correct / (i + 1) * 100);
        }
        printf("\nEpoch %d is over.\n", e + 1);
    }

    save_model("cnn_model.bin", conv, fc);

    conv_free(conv); pool_free(pool); network_free(fc);
    free_mnist_images(images, num_imgs); free(labels);
}

void run_inference_mode(char* model_path, char* img_path) {
    printf("--- Infering ---\n");
    printf("Load model: %s\n", model_path);
    printf("Load sample: % s\n", img_path);

    ConvLayer* conv; PoolLayer* pool; NeuralNetwork* fc;
    build_architecture(&conv, &pool, &fc);

    load_model(model_path, conv, fc);

    float* img_data = load_image_from_file(img_path);
    if (!img_data) return;

    Matrix* input = matrix_create(1, 784);
    memcpy(input->data, img_data, 784 * sizeof(float));

    Matrix* conv_out = conv_forward(conv, input);
    apply_relu(conv_out);

    Matrix* pool_out = pool_forward(pool, conv_out);

    Matrix* flattened = matrix_create(pool_out->rows * pool_out->cols, 1);
    memcpy(flattened->data, pool_out->data, flattened->rows * sizeof(float));

    Matrix* z1 = matrix_multiply(fc->w1, flattened);
    matrix_add(z1, fc->b1);
    apply_relu(z1);

    Matrix* z2 = matrix_multiply(fc->w2, z1);
    matrix_add(z2, fc->b2);
    apply_softmax(z2);

    int result = argmax(z2);
    printf("\n-------------------------\n");
    printf(">> Result: [ %d ] <<\n", result);
    printf(">> Acc: %.2f%%\n", z2->data[result] * 100);

    matrix_free(input); matrix_free(conv_out); matrix_free(pool_out);
    matrix_free(flattened); matrix_free(z1); matrix_free(z2);
    free(img_data);
    conv_free(conv); pool_free(pool); network_free(fc);
}