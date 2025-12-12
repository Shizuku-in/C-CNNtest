#ifndef IO_H
#define IO_H

#include "network.h"
#include "layers.h"

void save_model(const char* filename, ConvLayer* conv, NeuralNetwork* fc);

void load_model(const char* filename, ConvLayer* conv, NeuralNetwork* fc);

#endif