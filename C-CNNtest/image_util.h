#ifndef IMAGE_UTIL_H
#define IMAGE_UTIL_H

unsigned char* resize_image(unsigned char* src, int src_w, int src_h, int channels);

float* load_image_from_file(const char* filename);

#endif