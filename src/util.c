#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#  include <windows.h>
#  include <direct.h>
#else
#  include <errno.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

double util_wall_seconds(void) {
#ifdef _WIN32
  static LARGE_INTEGER freq;
  static int init = 0;
  if (!init) {
    QueryPerformanceFrequency(&freq);
    init = 1;
  }
  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);
  return (double)counter.QuadPart / (double)freq.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

void util_sleep_ms(int ms) {
#ifdef _WIN32
  Sleep((DWORD)ms);
#else
  usleep(ms * 1000);
#endif
}

unsigned int util_seed_from_time(void) {
  return (unsigned int)time(NULL) ^ 0xA5A5A5A5u;
}

int util_mkdir_p(const char* path) {
#ifdef _WIN32
  // Naive mkdir - fine for this project (no concurrent runs).
  if (path == NULL || path[0] == '\0') return -1;
  int rc = 0;
  char* tmp = (char*)malloc(strlen(path) + 1);
  if (!tmp) return -1;
  strcpy(tmp, path);
  size_t len = strlen(tmp);
  if (len == 0) {
    free(tmp);
    return -1;
  }
  if (tmp[len - 1] == '\\' || tmp[len - 1] == '/') tmp[len - 1] = '\0';

  for (char* p = tmp + 1; *p; p++) {
    if (*p == '\\' || *p == '/') {
      *p = '\0';
      if (_mkdir(tmp) != 0) rc = -1;
      *p = '\\';
    }
  }
  if (_mkdir(tmp) != 0) rc = -1;
  free(tmp);
  return rc;
#else
  if (path == NULL || path[0] == '\0') return -1;
  // Create parent dirs incrementally.
  char* tmp = strdup(path);
  if (!tmp) return -1;
  size_t len = strlen(tmp);
  if (len == 0) {
    free(tmp);
    return -1;
  }
  if (tmp[len - 1] == '/') tmp[len - 1] = '\0';
  for (char* p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = '\0';
      if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        free(tmp);
        return -1;
      }
      *p = '/';
    }
  }
  int rc = mkdir(tmp, 0755);
  if (rc != 0 && errno != EEXIST) {
    free(tmp);
    return -1;
  }
  free(tmp);
  return 0;
#endif
}

int util_read_file(const char* path, unsigned char** out_buf, size_t* out_len) {
  if (!path || !out_buf || !out_len) return -1;
  FILE* f = fopen(path, "rb");
  if (!f) return -1;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return -1;
  }
  long sz = ftell(f);
  if (sz < 0) {
    fclose(f);
    return -1;
  }
  rewind(f);
  unsigned char* buf = (unsigned char*)malloc((size_t)sz + 1);
  if (!buf) {
    fclose(f);
    return -1;
  }
  size_t read_sz = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  buf[read_sz] = 0;
  *out_buf = buf;
  *out_len = read_sz;
  return 0;
}

int util_write_file(const char* path, const unsigned char* data, size_t len) {
  if (!path || !data) return -1;
  FILE* f = fopen(path, "wb");
  if (!f) return -1;
  size_t w = fwrite(data, 1, len, f);
  fclose(f);
  return (w == len) ? 0 : -1;
}

void util_trim_inplace(char* s) {
  if (!s) return;
  size_t n = strlen(s);
  while (n > 0 && (s[n - 1] == '\r' || s[n - 1] == '\n' || s[n - 1] == ' ' || s[n - 1] == '\t')) {
    s[n - 1] = '\0';
    n--;
  }
  size_t i = 0;
  while (s[i] == ' ' || s[i] == '\t') i++;
  if (i > 0) memmove(s, s + i, strlen(s + i) + 1);
}

unsigned int util_rand_u32(unsigned int* state) {
  if (!state) return 0;
  unsigned int x = *state;
  // xorshift32
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

float util_rand_f32(unsigned int* state) {
  unsigned int r = util_rand_u32(state);
  return (r & 0x00FFFFFFu) / (float)0x01000000u;
}

