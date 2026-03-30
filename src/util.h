#pragma once

#include <stddef.h>

double util_wall_seconds(void);
void util_sleep_ms(int ms);
unsigned int util_seed_from_time(void);

int util_mkdir_p(const char* path);

int util_read_file(const char* path, unsigned char** out_buf, size_t* out_len);
int util_write_file(const char* path, const unsigned char* data, size_t len);

void util_trim_inplace(char* s);

// Very small PRNG (xorshift32). Good enough for training randomness.
unsigned int util_rand_u32(unsigned int* state);
float util_rand_f32(unsigned int* state); // in [0,1)

