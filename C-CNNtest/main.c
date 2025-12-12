#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mnist.h"
#include "matrix.h"
#include "layers.h"
#include "fc_layer.h"
#include "activations.h"
#include "model.h"
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

Model* build_model(float learning_rate);
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

Model* build_model(float learning_rate) {
    Model* model = model_create(learning_rate);
    
    // Conv Layer: 28x28x1 -> 26x26x8
    model_add_layer(model, conv_layer_create(28, 28, 1, CONV_K_SIZE, CONV_K_NUM));
    
    // ReLU Activation
    model_add_layer(model, relu_layer_create(26 * 26 * 8));
    
    // Max Pooling: 26x26x8 -> 13x13x8
    model_add_layer(model, pool_layer_create(26, 26, 8, POOL_SIZE));
    
    // Flatten: 13x13x8 = 1352
    model_add_layer(model, flatten_layer_create(8, 13, 13));
    
    // Fully Connected Hidden: 1352 -> 128
    model_add_layer(model, fc_layer_create(1352, FC_HIDDEN));
    
    // ReLU Activation
    model_add_layer(model, relu_layer_create(FC_HIDDEN));
    
    // Fully Connected Output: 128 -> 10
    model_add_layer(model, fc_layer_create(FC_HIDDEN, FC_OUT));
    
    // Softmax Activation
    model_add_layer(model, softmax_layer_create(FC_OUT));
    
    return model;
}

void run_training_mode(int epochs) {
    printf("--- Training (epochs = %d) ---\n", epochs);

    int num_imgs, num_lbls;
    float** images = read_mnist_images(IMAGEPATH, &num_imgs);
    int* labels = read_mnist_labels(LABELPATH, &num_lbls);

    Model* model = build_model(0.01f);
    model_summary(model);

    for (int e = 0; e < epochs; e++) {
        int correct = 0;
        
        for (int i = 0; i < num_imgs; i++) {
            // Prepare input
            Matrix* input = matrix_create(1, 784);
            memcpy(input->data, images[i], 784 * sizeof(float));

            // Prepare target (one-hot encoding)
            Matrix* target = matrix_create(10, 1);
            matrix_fill(target, 0.0f);
            target->data[labels[i]] = 1.0f;

            // Forward pass
            Matrix* output = model_predict(model, input);
            
            // Check accuracy
            if (model_argmax(output) == labels[i]) correct++;

            // Compute loss gradient (output - target)
            Matrix* grad_output = matrix_create(10, 1);
            for (int k = 0; k < 10; k++) {
                grad_output->data[k] = output->data[k] - target->data[k];
            }

            // Backward pass
            model_backward(model, grad_output);

            // Free memory
            matrix_free(input);
            matrix_free(target);
            matrix_free(output);
            matrix_free(grad_output);

            if (i % 100 == 0) {
                printf("Epoch %d: %d/%d (Acc: %.2f%%)\r", 
                       e + 1, i, num_imgs, (float)correct / (i + 1) * 100);
            }
        }
        
        printf("\nEpoch %d finished. Accuracy: %.2f%%\n", 
               e + 1, (float)correct / num_imgs * 100);
    }

    printf("\nTraining complete! Saving model...\n");
    // TODO: Implement save_model for new architecture
    // save_model("cnn_model.bin", model);

    model_free(model);
    free_mnist_images(images, num_imgs);
    free(labels);
}

void run_inference_mode(char* model_path, char* img_path) {
    printf("--- Inference Mode ---\n");
    printf("Load model: %s\n", model_path);
    printf("Load image: %s\n", img_path);

    Model* model = build_model(0.01f);
    
    // TODO: Implement load_model for new architecture
    // load_model(model_path, model);
    printf("Warning: Model loading not yet implemented for new architecture\n");

    float* img_data = load_image_from_file(img_path);
    if (!img_data) {
        model_free(model);
        return;
    }

    Matrix* input = matrix_create(1, 784);
    memcpy(input->data, img_data, 784 * sizeof(float));

    Matrix* output = model_predict(model, input);

    int result = model_argmax(output);
    printf("\n-------------------------\n");
    printf(">> Predicted: [ %d ] <<\n", result);
    printf(">> Confidence: %.2f%%\n", output->data[result] * 100);
    printf("-------------------------\n");

    matrix_free(input);
    matrix_free(output);
    free(img_data);
    model_free(model);
}