#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

unsigned char* resize_image(unsigned char* src, int src_w, int src_h, int channels) {
    int dst_w = 28;
    int dst_h = 28;
    unsigned char* dst = (unsigned char*)malloc(dst_w * dst_h * sizeof(unsigned char));

    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;

    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            int src_y_start = (int)(y * scale_y);
            int src_y_end = (int)((y + 1) * scale_y);
            int src_x_start = (int)(x * scale_x);
            int src_x_end = (int)((x + 1) * scale_x);

            if (src_y_end > src_h) src_y_end = src_h;
            if (src_x_end > src_w) src_x_end = src_w;

            long sum = 0;
            int count = 0;

            for (int sy = src_y_start; sy < src_y_end; sy++) {
                for (int sx = src_x_start; sx < src_x_end; sx++) {
                    int pixel_idx = (sy * src_w + sx) * channels;
                    sum += src[pixel_idx];
                    count++;
                }
            }

            if (count > 0) {
                dst[y * dst_w + x] = (unsigned char)(sum / count);
            }
            else {
                int sy = (int)(y * scale_y);
                int sx = (int)(x * scale_x);
                dst[y * dst_w + x] = src[(sy * src_w + sx) * channels];
            }
        }
    }
    return dst;
}

float* load_image_from_file(const char* filename) {
    int w, h, channels;
    unsigned char* img_raw = stbi_load(filename, &w, &h, &channels, 1);

    if (!img_raw) {
        printf("Cannot open: %s\n", filename);
        return NULL;
    }

    printf("Original: %dx%d. Resampling to 28x28...\n", w, h);

    unsigned char* img_resized = resize_image(img_raw, w, h, 1);

    stbi_image_free(img_raw);

    float* data = (float*)malloc(28 * 28 * sizeof(float));

    int corner_sum = img_resized[0] + img_resized[27] + img_resized[27 * 28] + img_resized[27 * 28 + 27];
    int need_invert = (corner_sum > 255 * 2);

    for (int i = 0; i < 28 * 28; i++) {
        float pixel = (float)img_resized[i];

        if (need_invert) {
            data[i] = (255.0f - pixel) / 255.0f;
        }
        else {
            data[i] = pixel / 255.0f;
        }

        if (data[i] < 0.2f) data[i] = 0.0f;
        if (data[i] > 0.8f) data[i] = 1.0f;
    }

    free(img_resized);
    return data;
}