#ifndef QOI_H_
#define QOI_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdbool.h>
#include <unistd.h>

#define QOI_OP_RGB   0xfe
#define QOI_OP_RGBA  0xff
#define QOI_OP_INDEX 0x00
#define QOI_OP_DIFF  0x40
#define QOI_OP_LUMA  0x80
#define QOI_OP_RUN   0xc0
#define QOI_FOUNDATION_HEADER  0x716F6966 // Avoid macro name collision keywords
#define QOI_MAX_WIDTH 4294967295
#define QOI_MAX_HEIGHT 4294967295

#define QOI_RGBA 4

inline uint64_t round_to_page_size(uint64_t length) {
    long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) pageSize = 4096;
    return (length + (pageSize - 1)) & ~(pageSize - 1);
}

struct rgba8_t {
    uint8_t r, g, b, a;
};

struct QOI_header_info {
    uint32_t width;
    uint32_t height;
    uint8_t channels;
    uint8_t colorspace;
};

typedef struct rgba8_t rgba8_t;
typedef struct QOI_header_info QOI_header_info;

inline bool pixel_compare(rgba8_t *a, rgba8_t *b) {
    return (a->r == b->r && a->g == b->g && a->b == b->b && a->a == b->a);
}

inline uint32_t read_uint32(FILE *file) {
    uint8_t bytes[4];
    if (fread(bytes, 1, 4, file) != 4) return 0;
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8)  |  (uint32_t)bytes[3];
}

inline void write_uint32(FILE *file, uint32_t value) {
    uint8_t bytes[4] = {
        (uint8_t)((value >> 24) & 0xFF),
        (uint8_t)((value >> 16) & 0xFF),
        (uint8_t)((value >> 8)  & 0xFF),
        (uint8_t)(value         & 0xFF)
    };
    fwrite(bytes, 1, 4, file);
}

// FIXED: Hardcoded to parse incoming streams using 4-byte RGBA 
inline bool qoi_encoder(uint8_t *pixels, QOI_header_info info, const char* file_name) {
    if (info.width <= 0 || info.width >= QOI_MAX_WIDTH || info.height <= 0 || info.height >= QOI_MAX_HEIGHT)
        return false;
    
    if (info.colorspace < 0 || info.colorspace > 1 || info.channels < 3 || info.channels > 4) return false;
        
        
    FILE *file = fopen(file_name, "wb");
    if (file == NULL) {
        perror("fopen failed.");
        printf("file name: %s\n", file_name);
        return false;
    }

    write_uint32(file, QOI_FOUNDATION_HEADER);
    write_uint32(file, info.width);
    write_uint32(file, info.height);
    fputc(info.channels, file); // Keeps original specs metadata trace safe
    fputc(info.colorspace, file);

    rgba8_t local_lookuptable[64] = {0}; // Avoid indexing name collisions
    uint8_t run_length = 0;
    rgba8_t ppixel = {0, 0, 0, 255};
    rgba8_t cpixel = ppixel;
    
    // Explicitly calculate layout based on standard 4-channel layout constraints
    uint64_t size = (uint64_t)info.width * info.height * QOI_RGBA;

    for (uint64_t i = 0; i < size; i += QOI_RGBA) {
        cpixel.r = pixels[i];
        cpixel.g = pixels[i+1];
        cpixel.b = pixels[i+2];
        cpixel.a = pixels[i+3]; // Assumes raw output buffer from GPU is strictly RGBA

        if (pixel_compare(&cpixel, &ppixel)) {
            run_length++;
            if (run_length == 62) {
                fputc(QOI_OP_RUN | (run_length - 1), file);
                run_length = 0;
            }
        } else {
            if (run_length > 0) {
                fputc(QOI_OP_RUN | (run_length - 1), file);
                run_length = 0;
            }
            
            int index_pos = (cpixel.r * 3 + cpixel.g * 5 + cpixel.b * 7 + cpixel.a * 11) % 64;

            if (pixel_compare(&cpixel, &local_lookuptable[index_pos])) {
                fputc(QOI_OP_INDEX | index_pos, file);
            } else {
                local_lookuptable[index_pos] = cpixel;

                if (cpixel.a == ppixel.a) {
                    int8_t dr = cpixel.r - ppixel.r;
                    int8_t dg = cpixel.g - ppixel.g;
                    int8_t db = cpixel.b - ppixel.b;

                    int8_t dr_g = dr - dg;
                    int8_t db_g = db - dg;

                    if (dr >= -2 && dr <= 1 && dg >= -2 && dg <= 1 && db >= -2 && db <= 1) {
                        fputc(QOI_OP_DIFF | ((dr + 2) << 4) | ((dg + 2) << 2) | (db + 2), file);
                    } else if (dg >= -32 && dg <= 31 && dr_g >= -8 && dr_g <= 7 && db_g >= -8 && db_g <= 7) {
                        fputc(QOI_OP_LUMA | (dg + 32), file);
                        fputc(((dr_g + 8) << 4) | (db_g + 8), file);
                    } else {
                        fputc(QOI_OP_RGB, file);
                        fputc(cpixel.r, file);
                        fputc(cpixel.g, file);
                        fputc(cpixel.b, file);
                    }
                } else {
                    fputc(QOI_OP_RGBA, file);
                    fputc(cpixel.r, file);
                    fputc(cpixel.g, file);
                    fputc(cpixel.b, file);
                    fputc(cpixel.a, file);
                }
            }
        }
        ppixel = cpixel;
    }
    
    if (run_length > 0) {
        fputc(QOI_OP_RUN | (run_length - 1), file);
    }

    // Stream termination marker
    for (int i = 0; i < 7; i++) fputc(0, file);
    fputc(1, file);

    const bool write_success = (ferror(file) == 0);
    const bool close_success = (fclose(file) == 0);

    return write_success && close_success;
}

inline void *decoder_fail_condition(void* buffer) {
    if (buffer) free(buffer);
    
    return NULL;
}

inline bool qoi_read_byte(FILE* file, uint8_t* value) {
    int byte = fgetc(file);

    if (byte == EOF)
        return false;

    *value = (uint8_t)byte;
    return true;
}
    
inline void *qoi_decoder(FILE *file, QOI_header_info *info) {
    if (file == NULL || info == NULL) return NULL;
    if (read_uint32(file) != QOI_FOUNDATION_HEADER) return NULL;
        
    info->width = read_uint32(file);
    info->height = read_uint32(file);

    const int fileChannels = fgetc(file);
    const int fileColorspace = fgetc(file);

    if ((fileChannels != 3 && fileChannels != 4) ||
        (fileColorspace != 0 && fileColorspace != 1))
        return nullptr;

    constexpr uint64_t MAX_PIXELS = 100'000'000ULL;

    const uint64_t pixels =
        uint64_t(info->width) * info->height;

    if (pixels == 0 || pixels > MAX_PIXELS)
        return nullptr;

    const uint64_t size = pixels * 4;
    const uint64_t buffer_size = round_to_page_size(size);

    if (buffer_size < size)
        return nullptr;

    // The decoder always returns RGBA8.
    info->channels = 4;
    info->colorspace = static_cast<uint8_t>(fileColorspace);
    
    if (info->width <= 0 || info->width >= QOI_MAX_WIDTH || info->height <= 0 || info->height >= QOI_MAX_HEIGHT) return NULL;
        
        
    void *main_buffer = NULL;

    if (posix_memalign(&main_buffer, sysconf(_SC_PAGESIZE), buffer_size) != 0)
        return NULL;
    
    uint8_t *buffer = (uint8_t*)main_buffer;
    uint64_t i = 0;
    rgba8_t local_lookuptable[64] = {0};
    rgba8_t pixel = {0, 0, 0, 255};

    while (i < size) {
        int c = fgetc(file);
        if (c == EOF) return decoder_fail_condition(main_buffer);

        if (c == QOI_OP_RGB) {
            if (!qoi_read_byte(file, &pixel.r) || !qoi_read_byte(file, &pixel.g) || !qoi_read_byte(file, &pixel.b))
                return decoder_fail_condition(main_buffer);
        } else if (c == QOI_OP_RGBA) {
            if (!qoi_read_byte(file, &pixel.r) || !qoi_read_byte(file, &pixel.g) || !qoi_read_byte(file, &pixel.b) || !qoi_read_byte(file, &pixel.a))
                return decoder_fail_condition(main_buffer);
        } else {
            switch (c & 0xc0) {
                case QOI_OP_INDEX:
                    pixel = local_lookuptable[c & 0x3f];
                    break;
                case QOI_OP_DIFF:
                    pixel.r += ((c >> 4) & 0x03) - 2;
                    pixel.g += ((c >> 2) & 0x03) - 2;
                    pixel.b += (c & 0x03) - 2;
                    break;
                case QOI_OP_LUMA: {
                    int vg = (c & 0x3f) - 32;
                    int c2 = fgetc(file);
                    if (c2 == EOF) return decoder_fail_condition(main_buffer);
                    int dr_g = ((c2 >> 4) & 0x0f) - 8;
                    int db_g = (c2 & 0x0f) - 8;
                    pixel.r += vg + dr_g;
                    pixel.g += vg;
                    pixel.b += vg + db_g;
                    break;
                }
                case QOI_OP_RUN: {
                    uint8_t run_len = (c & 0x3f) + 1;

                    if (run_len > (size - i) / QOI_RGBA)
                        return decoder_fail_condition(main_buffer);

                    for (uint8_t r = 0; r < run_len; ++r) {
                        buffer[i]   = pixel.r;
                        buffer[i+1] = pixel.g;
                        buffer[i+2] = pixel.b;
                        buffer[i+3] = pixel.a;

                        i += QOI_RGBA;
                    }

                    int r_index = (
                        pixel.r * 3 + pixel.g * 5 +
                        pixel.b * 7 + pixel.a * 11
                    ) % 64;

                    local_lookuptable[r_index] = pixel;

                    continue;
                }
            }
        }

        if (i < size) {
            buffer[i]   = pixel.r;
            buffer[i+1] = pixel.g;
            buffer[i+2] = pixel.b;
            buffer[i+3] = pixel.a;

            int r_index = (pixel.r * 3 + pixel.g * 5 + pixel.b * 7 + pixel.a * 11) % 64;
            local_lookuptable[r_index] = pixel;
            i += QOI_RGBA;
        }
    }
    
    
    const uint8_t expected_end[8] = {0, 0, 0, 0, 0, 0, 0, 1};

    uint8_t actual_end[8];

    if (fread(actual_end, 1, 8, file) != 8 || memcmp(actual_end, expected_end, 8) != 0)
        return decoder_fail_condition(main_buffer);
    
    return main_buffer;
}

#endif // !QOI_H_
