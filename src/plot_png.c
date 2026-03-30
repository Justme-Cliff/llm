#include "plot_png.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// Minimal PNG writer for 24-bit RGB images.
// It uses zlib "stored blocks" (no compression) to keep the code small and dependency-free.

static uint32_t crc32_table[256];
static int crc32_table_ready = 0;

static void crc32_init(void) {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int k = 0; k < 8; k++) {
      if (c & 1) c = 0xEDB88320u ^ (c >> 1);
      else c = (c >> 1);
    }
    crc32_table[i] = c;
  }
  crc32_table_ready = 1;
}

static uint32_t crc32(const uint8_t* data, size_t len) {
  if (!crc32_table_ready) crc32_init();
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; i++) {
    c = crc32_table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
  }
  return c ^ 0xFFFFFFFFu;
}

static void u32le_write(uint8_t* out, uint32_t v) {
  out[0] = (uint8_t)(v & 0xFFu);
  out[1] = (uint8_t)((v >> 8) & 0xFFu);
  out[2] = (uint8_t)((v >> 16) & 0xFFu);
  out[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static void write_chunk(FILE* f, const char type[4], const uint8_t* data, uint32_t len) {
  // Chunk format: length(4) + type(4) + data + crc(4)
  uint8_t len_buf[4];
  u32le_write(len_buf, len);
  fwrite(len_buf, 1, 4, f);
  fwrite(type, 1, 4, f);

  if (len > 0) fwrite(data, 1, len, f);

  // CRC is computed over type + data.
  uint8_t* crc_buf = NULL;
  if (len > 0) {
    crc_buf = (uint8_t*)malloc((size_t)len + 4);
    memcpy(crc_buf, type, 4);
    memcpy(crc_buf + 4, data, (size_t)len);
  } else {
    crc_buf = (uint8_t*)malloc(4);
    memcpy(crc_buf, type, 4);
  }
  uint32_t c = crc32(crc_buf, (size_t)len + 4);
  free(crc_buf);
  uint8_t crc_buf2[4];
  u32le_write(crc_buf2, c);
  fwrite(crc_buf2, 1, 4, f);
}

typedef struct {
  float* ys;
  size_t n;
  size_t cap;
} FloatList;

static void floatlist_free(FloatList* l) {
  if (!l) return;
  free(l->ys);
  l->ys = NULL;
  l->n = 0;
  l->cap = 0;
}

static int floatlist_push(FloatList* l, float y) {
  if (l->n + 1 > l->cap) {
    size_t new_cap = l->cap == 0 ? 256 : l->cap * 2;
    float* ny = (float*)realloc(l->ys, new_cap * sizeof(float));
    if (!ny) return -1;
    l->ys = ny;
    l->cap = new_cap;
  }
  l->ys[l->n++] = y;
  return 0;
}

static int render_loss_csv_to_points(
  const char* loss_csv,
  FloatList* out_ys
) {
  if (!loss_csv || !out_ys) return -1;
  memset(out_ys, 0, sizeof(*out_ys));

  FILE* f = fopen(loss_csv, "rb");
  if (!f) return -1;

  char line[512];
  int first = 1;
  while (fgets(line, (int)sizeof(line), f)) {
    if (line[0] == '\0' || line[0] == '\n' || line[0] == '\r') continue;
    if (first) {
      first = 0;
      if (strstr(line, "loss") != NULL) continue;
    }

    int step = 0;
    float loss = 0.0f;
    float elapsed = 0.0f;
    int rc = sscanf(line, "%d,%f,%f", &step, &loss, &elapsed);
    if (rc >= 2) {
      if (floatlist_push(out_ys, loss) != 0) break;
    }
  }

  fclose(f);
  return out_ys->n == 0 ? -1 : 0;
}

static void draw_line_rgb(uint8_t* rgb, int w, int h, int x0, int y0, int x1, int y1,
                           uint8_t r, uint8_t g, uint8_t b) {
  // Bresenham line.
  int dx = abs(x1 - x0);
  int sx = x0 < x1 ? 1 : -1;
  int dy = -abs(y1 - y0);
  int sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;

  for (;;) {
    if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h) {
      size_t idx = ((size_t)y0 * (size_t)w + (size_t)x0) * 3u;
      rgb[idx + 0] = r;
      rgb[idx + 1] = g;
      rgb[idx + 2] = b;
    }
    if (x0 == x1 && y0 == y1) break;
    int e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
}

static int write_png_rgb_no_compress(const char* path, const uint8_t* rgb, int w, int h) {
  if (!path || !rgb || w <= 0 || h <= 0) return -1;

  FILE* f = fopen(path, "wb");
  if (!f) return -1;

  // PNG signature
  const uint8_t sig[8] = { 137,80,78,71,13,10,26,10 };
  fwrite(sig, 1, 8, f);

  // IHDR
  uint8_t ihdr[13];
  u32le_write(ihdr + 0, (uint32_t)w);
  u32le_write(ihdr + 4, (uint32_t)h);
  // Convert to big-endian for PNG fields:
  // We'll write big-endian directly.
  ihdr[0] = (uint8_t)((w >> 24) & 0xFFu);
  ihdr[1] = (uint8_t)((w >> 16) & 0xFFu);
  ihdr[2] = (uint8_t)((w >> 8) & 0xFFu);
  ihdr[3] = (uint8_t)(w & 0xFFu);
  ihdr[4] = (uint8_t)((h >> 24) & 0xFFu);
  ihdr[5] = (uint8_t)((h >> 16) & 0xFFu);
  ihdr[6] = (uint8_t)((h >> 8) & 0xFFu);
  ihdr[7] = (uint8_t)(h & 0xFFu);

  ihdr[8] = 8;    // bit depth
  ihdr[9] = 2;    // color type: RGB
  ihdr[10] = 0;   // compression: deflate
  ihdr[11] = 0;   // filter: adaptive
  ihdr[12] = 0;   // interlace: none

  write_chunk(f, "IHDR", ihdr, 13);

  // IDAT data: zlib header + DEFLATE stored blocks + adler32.
  // We'll store raw image bytes with no filtering (we add filter byte 0 per row).
  const int bpp = 3;
  size_t row_bytes = (size_t)w * (size_t)bpp;
  size_t raw_stride = row_bytes + 1; // +1 filter byte
  size_t raw_size = raw_stride * (size_t)h;

  uint8_t* raw = (uint8_t*)malloc(raw_size);
  if (!raw) { fclose(f); return -1; }

  for (int y = 0; y < h; y++) {
    raw[(size_t)y * raw_stride + 0] = 0; // filter type 0
    memcpy(raw + (size_t)y * raw_stride + 1, rgb + (size_t)y * (size_t)w * (size_t)bpp, row_bytes);
  }

  // Build zlib stream.
  // zlib header: CMF/FLG, then stored blocks, then adler32.
  // We'll write into a dynamic buffer.
  size_t max_blocks = (raw_size + 65535 - 1) / 65535;
  size_t deflate_overhead = max_blocks * 5 + max_blocks * 4; // rough
  size_t zlib_header = 2;
  size_t adler = 4;
  size_t idat_cap = zlib_header + raw_size + deflate_overhead + adler + 64;
  uint8_t* zbuf = (uint8_t*)malloc(idat_cap);
  if (!zbuf) { free(raw); fclose(f); return -1; }
  size_t zp = 0;

  // zlib header (0x78 0x01)
  zbuf[zp++] = 0x78;
  zbuf[zp++] = 0x01;

  // deflate stored blocks:
  size_t offset = 0;
  while (offset < raw_size) {
    size_t block_len = raw_size - offset;
    if (block_len > 65535) block_len = 65535;

    // Write 3 bits: BFINAL (0/1) + BTYPE=00
    // We'll keep it simple by writing a full byte where bit0=BFINAL and others 0.
    int is_last = (offset + block_len >= raw_size) ? 1 : 0;
    // Because stored blocks require byte alignment, we ensure we are at byte boundary anyway.
    zbuf[zp++] = (uint8_t)(is_last ? 1 : 0); // 00000001 sets BFINAL=1.
    // Align to next byte boundary already satisfied since we used whole byte.

    uint16_t len16 = (uint16_t)block_len;
    uint16_t nlen16 = (uint16_t)(~len16);
    zbuf[zp++] = (uint8_t)(len16 & 0xFFu);
    zbuf[zp++] = (uint8_t)((len16 >> 8) & 0xFFu);
    zbuf[zp++] = (uint8_t)(nlen16 & 0xFFu);
    zbuf[zp++] = (uint8_t)((nlen16 >> 8) & 0xFFu);

    memcpy(zbuf + zp, raw + offset, block_len);
    zp += block_len;
    offset += block_len;
  }

  // adler32
  uint32_t a = 1, b = 0;
  for (size_t i = 0; i < raw_size; i++) {
    a = (a + raw[i]) % 65521u;
    b = (b + a) % 65521u;
  }
  uint32_t ad = (b << 16) | a;
  zbuf[zp++] = (uint8_t)((ad >> 24) & 0xFFu);
  zbuf[zp++] = (uint8_t)((ad >> 16) & 0xFFu);
  zbuf[zp++] = (uint8_t)((ad >> 8) & 0xFFu);
  zbuf[zp++] = (uint8_t)(ad & 0xFFu);

  // IDAT chunk
  write_chunk(f, "IDAT", zbuf, (uint32_t)zp);
  // IEND
  write_chunk(f, "IEND", NULL, 0);

  fclose(f);
  free(raw);
  free(zbuf);
  return 0;
}

int plot_png_loss_from_csv(const char* loss_csv, const char* out_png, int width, int height) {
  if (!loss_csv || !out_png || width <= 0 || height <= 0) return -1;

  FloatList ys;
  if (render_loss_csv_to_points(loss_csv, &ys) != 0) return -1;

  float miny = ys.ys[0];
  float maxy = ys.ys[0];
  for (size_t i = 1; i < ys.n; i++) {
    if (ys.ys[i] < miny) miny = ys.ys[i];
    if (ys.ys[i] > maxy) maxy = ys.ys[i];
  }
  if (maxy - miny < 1e-8f) maxy = miny + 1.0f;

  // RGB buffer
  uint8_t* rgb = (uint8_t*)malloc((size_t)width * (size_t)height * 3u);
  if (!rgb) {
    floatlist_free(&ys);
    return -1;
  }

  // Background
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      size_t idx = ((size_t)y * (size_t)width + (size_t)x) * 3u;
      rgb[idx + 0] = 0x0B;
      rgb[idx + 1] = 0x0B;
      rgb[idx + 2] = 0x0B;
    }
  }

  // Plot area
  int pad_left = 48;
  int pad_right = 16;
  int pad_top = 16;
  int pad_bottom = 32;
  int plot_w = width - pad_left - pad_right;
  int plot_h = height - pad_top - pad_bottom;
  if (plot_w <= 10 || plot_h <= 10) {
    free(rgb);
    floatlist_free(&ys);
    return -1;
  }

  // Border rectangle (simple)
  uint8_t br = 0x44, bg = 0x44, bb = 0x44;
  draw_line_rgb(rgb, width, height, pad_left, pad_top, pad_left + plot_w, pad_top, br, bg, bb);
  draw_line_rgb(rgb, width, height, pad_left, pad_top + plot_h, pad_left + plot_w, pad_top + plot_h, br, bg, bb);
  draw_line_rgb(rgb, width, height, pad_left, pad_top, pad_left, pad_top + plot_h, br, bg, bb);
  draw_line_rgb(rgb, width, height, pad_left + plot_w, pad_top, pad_left + plot_w, pad_top + plot_h, br, bg, bb);

  // Loss polyline
  uint8_t lr = 0x33, lg = 0xAA, lb = 0xFF;
  for (size_t i = 1; i < ys.n; i++) {
    float t0 = (ys.n == 1) ? 0.0f : (float)(i - 1) / (float)(ys.n - 1);
    float t1 = (ys.n == 1) ? 0.0f : (float)i / (float)(ys.n - 1);
    int x0 = pad_left + (int)lroundf((float)plot_w * t0);
    int x1 = pad_left + (int)lroundf((float)plot_w * t1);
    float y0t = (ys.ys[i - 1] - miny) / (maxy - miny);
    float y1t = (ys.ys[i] - miny) / (maxy - miny);
    int y0 = pad_top + plot_h - (int)lroundf((float)plot_h * y0t);
    int y1 = pad_top + plot_h - (int)lroundf((float)plot_h * y1t);

    draw_line_rgb(rgb, width, height, x0, y0, x1, y1, lr, lg, lb);
  }

  int rc = write_png_rgb_no_compress(out_png, rgb, width, height);
  free(rgb);
  floatlist_free(&ys);
  return rc;
}

