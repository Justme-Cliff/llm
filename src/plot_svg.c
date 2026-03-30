#include "plot_svg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  float* ys;
  size_t n;
  size_t cap;
} FloatList;

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

int plot_svg_loss_from_csv(const char* loss_csv, const char* out_svg, int width, int height) {
  if (!loss_csv || !out_svg) return -1;
  FILE* f = fopen(loss_csv, "rb");
  if (!f) return -1;
  // loss log format: step,loss,elapsed_seconds
  // We'll read loss values from each line after header (if present).

  FloatList xs;
  memset(&xs, 0, sizeof(xs));

  char line[512];
  int first = 1;
  while (fgets(line, (int)sizeof(line), f)) {
    // Skip empty lines.
    if (line[0] == '\0' || line[0] == '\n' || line[0] == '\r') continue;
    if (first) {
      first = 0;
      // Try to detect header (contains non-numeric).
      if (strstr(line, "loss") != NULL) continue;
    }
    int step = 0;
    float loss = 0.0f;
    float elapsed = 0.0f;
    int rc = sscanf(line, "%d,%f,%f", &step, &loss, &elapsed);
    if (rc >= 2) {
      if (floatlist_push(&xs, loss) != 0) break;
    }
  }
  fclose(f);
  if (xs.n == 0) {
    free(xs.ys);
    return -1;
  }

  float miny = xs.ys[0], maxy = xs.ys[0];
  for (size_t i = 1; i < xs.n; i++) {
    if (xs.ys[i] < miny) miny = xs.ys[i];
    if (xs.ys[i] > maxy) maxy = xs.ys[i];
  }
  if (maxy - miny < 1e-8f) maxy = miny + 1.0f;

  // Layout
  const int pad_left = 48;
  const int pad_right = 16;
  const int pad_top = 16;
  const int pad_bottom = 32;
  int plot_w = width - pad_left - pad_right;
  int plot_h = height - pad_top - pad_bottom;
  if (plot_w <= 10 || plot_h <= 10) {
    free(xs.ys);
    return -1;
  }

  FILE* out = fopen(out_svg, "wb");
  if (!out) {
    free(xs.ys);
    return -1;
  }

  fprintf(out, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
  fprintf(out, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n",
          width, height, width, height);
  fprintf(out, "<rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"#0b0b0b\"/>\n", width, height);
  fprintf(out, "<g stroke=\"#cccccc\" stroke-width=\"1\" fill=\"none\">\n");
  fprintf(out, "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" stroke=\"#444\" fill=\"none\"/>\n",
          pad_left, pad_top, plot_w, plot_h);
  fprintf(out, "</g>\n");

  // Polyline points
  fprintf(out, "<polyline fill=\"none\" stroke=\"#33aaff\" stroke-width=\"2\" points=\"");
  for (size_t i = 0; i < xs.n; i++) {
    float x = pad_left + (plot_w * (xs.n == 1 ? 0.0f : (float)i / (float)(xs.n - 1)));
    float t = (xs.ys[i] - miny) / (maxy - miny);
    float y = pad_top + plot_h - (plot_h * t);
    fprintf(out, "%0.2f,%0.2f ", x, y);
  }
  fprintf(out, "\"/>\n");

  // Labels
  fprintf(out, "<text x=\"%d\" y=\"%d\" fill=\"#ddd\" font-size=\"14\">loss</text>\n", pad_left, pad_top - 4);
  free(xs.ys);
  fprintf(out, "</svg>\n");
  fclose(out);
  return 0;
}

