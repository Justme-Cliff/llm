#include "model_gru.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "util.h"

#ifdef _WIN32
#  include <malloc.h> // _alloca/alloca on MSVC
#endif

static float act_sigmoid(float x) {
  // Avoid overflow in expf for large magnitude.
  if (x >= 0.0f) {
    float z = expf(-x);
    return 1.0f / (1.0f + z);
  } else {
    float z = expf(x);
    return z / (1.0f + z);
  }
}

static void matvec_mul_rowmajor(const float* A, int rows, int cols, const float* x, float* y) {
  // y[i] = sum_j A[i*cols + j] * x[j]
  for (int i = 0; i < rows; i++) {
    float s = 0.0f;
    const float* row = A + (size_t)i * (size_t)cols;
    for (int j = 0; j < cols; j++) {
      s += row[j] * x[j];
    }
    y[i] = s;
  }
}


void model_gru_init(GRUModel* m, int vocab_size, int hidden_size, unsigned int seed) {
  if (!m) return;
  memset(m, 0, sizeof(*m));
  m->vocab_size = vocab_size;
  m->hidden_size = hidden_size;

  size_t V = (size_t)vocab_size;
  size_t H = (size_t)hidden_size;

  // Allocate weights.
  m->embeddings = (float*)calloc(V * H, sizeof(float));
  m->W_z = (float*)calloc(H * H, sizeof(float));
  m->U_z = (float*)calloc(H * H, sizeof(float));
  m->b_z = (float*)calloc(H, sizeof(float));

  m->W_r = (float*)calloc(H * H, sizeof(float));
  m->U_r = (float*)calloc(H * H, sizeof(float));
  m->b_r = (float*)calloc(H, sizeof(float));

  m->W_n = (float*)calloc(H * H, sizeof(float));
  m->U_n = (float*)calloc(H * H, sizeof(float));
  m->b_n = (float*)calloc(H, sizeof(float));

  m->out_W = (float*)calloc(V * H, sizeof(float));
  m->out_b = (float*)calloc(V, sizeof(float));

  if (!m->embeddings || !m->W_z || !m->U_z || !m->b_z || !m->W_r || !m->U_r || !m->b_r ||
      !m->W_n || !m->U_n || !m->b_n || !m->out_W || !m->out_b) {
    model_gru_free(m);
    return;
  }

  unsigned int rng = seed ? seed : 1u;
  float scale = 0.02f;

  // Small init around 0. Embeddings also small.
  for (size_t i = 0; i < V * H; i++) m->embeddings[i] = (util_rand_f32(&rng) * 2.0f - 1.0f) * scale;
  for (size_t i = 0; i < H * H; i++) {
    m->W_z[i] = (util_rand_f32(&rng) * 2.0f - 1.0f) * scale;
    m->U_z[i] = (util_rand_f32(&rng) * 2.0f - 1.0f) * scale;
    m->W_r[i] = (util_rand_f32(&rng) * 2.0f - 1.0f) * scale;
    m->U_r[i] = (util_rand_f32(&rng) * 2.0f - 1.0f) * scale;
    m->W_n[i] = (util_rand_f32(&rng) * 2.0f - 1.0f) * scale;
    m->U_n[i] = (util_rand_f32(&rng) * 2.0f - 1.0f) * scale;
  }
  for (size_t i = 0; i < H; i++) {
    m->b_z[i] = 0.0f;
    m->b_r[i] = 0.0f;
    m->b_n[i] = 0.0f;
  }
  for (size_t i = 0; i < V * H; i++) m->out_W[i] = (util_rand_f32(&rng) * 2.0f - 1.0f) * scale;
  for (size_t i = 0; i < V; i++) m->out_b[i] = 0.0f;
}

void model_gru_free(GRUModel* m) {
  if (!m) return;
  free(m->embeddings);
  free(m->W_z);
  free(m->U_z);
  free(m->b_z);
  free(m->W_r);
  free(m->U_r);
  free(m->b_r);
  free(m->W_n);
  free(m->U_n);
  free(m->b_n);
  free(m->out_W);
  free(m->out_b);
  memset(m, 0, sizeof(*m));
}

int model_gru_save(const GRUModel* m, const char* path) {
  if (!m || !path) return -1;
  FILE* f = fopen(path, "wb");
  if (!f) return -1;
  const char magic[4] = {'M','L','G','R'};
  if (fwrite(magic, 1, 4, f) != 4) { fclose(f); return -1; }
  int32_t vocab = (int32_t)m->vocab_size;
  int32_t hidden = (int32_t)m->hidden_size;
  if (fwrite(&vocab, sizeof(vocab), 1, f) != 1) { fclose(f); return -1; }
  if (fwrite(&hidden, sizeof(hidden), 1, f) != 1) { fclose(f); return -1; }

  size_t V = (size_t)m->vocab_size;
  size_t H = (size_t)m->hidden_size;

  // Write raw float arrays.
  if (fwrite(m->embeddings, sizeof(float), V * H, f) != V * H) { fclose(f); return -1; }
  if (fwrite(m->W_z, sizeof(float), H * H, f) != H * H) { fclose(f); return -1; }
  if (fwrite(m->U_z, sizeof(float), H * H, f) != H * H) { fclose(f); return -1; }
  if (fwrite(m->b_z, sizeof(float), H, f) != H) { fclose(f); return -1; }
  if (fwrite(m->W_r, sizeof(float), H * H, f) != H * H) { fclose(f); return -1; }
  if (fwrite(m->U_r, sizeof(float), H * H, f) != H * H) { fclose(f); return -1; }
  if (fwrite(m->b_r, sizeof(float), H, f) != H) { fclose(f); return -1; }
  if (fwrite(m->W_n, sizeof(float), H * H, f) != H * H) { fclose(f); return -1; }
  if (fwrite(m->U_n, sizeof(float), H * H, f) != H * H) { fclose(f); return -1; }
  if (fwrite(m->b_n, sizeof(float), H, f) != H) { fclose(f); return -1; }
  if (fwrite(m->out_W, sizeof(float), V * H, f) != V * H) { fclose(f); return -1; }
  if (fwrite(m->out_b, sizeof(float), V, f) != V) { fclose(f); return -1; }

  fclose(f);
  return 0;
}

int model_gru_load(GRUModel* m, const char* path) {
  if (!m || !path) return -1;
  memset(m, 0, sizeof(*m));
  FILE* f = fopen(path, "rb");
  if (!f) return -1;
  char magic[4];
  if (fread(magic, 1, 4, f) != 4) { fclose(f); return -1; }
  if (!(magic[0] == 'M' && magic[1] == 'L' && magic[2] == 'G' && magic[3] == 'R')) { fclose(f); return -1; }

  int32_t vocab = 0, hidden = 0;
  if (fread(&vocab, sizeof(vocab), 1, f) != 1) { fclose(f); return -1; }
  if (fread(&hidden, sizeof(hidden), 1, f) != 1) { fclose(f); return -1; }

  model_gru_init(m, vocab, hidden, 1u);
  size_t V = (size_t)m->vocab_size;
  size_t H = (size_t)m->hidden_size;

  if (fread(m->embeddings, sizeof(float), V * H, f) != V * H) { fclose(f); model_gru_free(m); return -1; }
  if (fread(m->W_z, sizeof(float), H * H, f) != H * H) { fclose(f); model_gru_free(m); return -1; }
  if (fread(m->U_z, sizeof(float), H * H, f) != H * H) { fclose(f); model_gru_free(m); return -1; }
  if (fread(m->b_z, sizeof(float), H, f) != H) { fclose(f); model_gru_free(m); return -1; }
  if (fread(m->W_r, sizeof(float), H * H, f) != H * H) { fclose(f); model_gru_free(m); return -1; }
  if (fread(m->U_r, sizeof(float), H * H, f) != H * H) { fclose(f); model_gru_free(m); return -1; }
  if (fread(m->b_r, sizeof(float), H, f) != H) { fclose(f); model_gru_free(m); return -1; }
  if (fread(m->W_n, sizeof(float), H * H, f) != H * H) { fclose(f); model_gru_free(m); return -1; }
  if (fread(m->U_n, sizeof(float), H * H, f) != H * H) { fclose(f); model_gru_free(m); return -1; }
  if (fread(m->b_n, sizeof(float), H, f) != H) { fclose(f); model_gru_free(m); return -1; }
  if (fread(m->out_W, sizeof(float), V * H, f) != V * H) { fclose(f); model_gru_free(m); return -1; }
  if (fread(m->out_b, sizeof(float), V, f) != V) { fclose(f); model_gru_free(m); return -1; }

  fclose(f);
  return 0;
}

void gru_cell_forward(
  const GRUModel* m,
  const float* x,
  const float* h_prev,
  float* h_next,
  float* z_out,
  float* r_out,
  float* n_out
) {
  const int H = m->hidden_size;
  // Temporary buffers.
  float* a_z = (float*)alloca((size_t)H * sizeof(float));
  float* a_r = (float*)alloca((size_t)H * sizeof(float));
  float* a_n = (float*)alloca((size_t)H * sizeof(float));
  float* rh = (float*)alloca((size_t)H * sizeof(float));
  float* tmp = (float*)alloca((size_t)H * sizeof(float));

  // z = sigmoid(W_z*x + U_z*h_prev + b_z)
  matvec_mul_rowmajor(m->W_z, H, H, x, a_z);
  {
    matvec_mul_rowmajor(m->U_z, H, H, h_prev, tmp);
    for (int i = 0; i < H; i++) a_z[i] = a_z[i] + tmp[i] + m->b_z[i];
  }
  for (int i = 0; i < H; i++) z_out[i] = act_sigmoid(a_z[i]);

  // r = sigmoid(W_r*x + U_r*h_prev + b_r)
  matvec_mul_rowmajor(m->W_r, H, H, x, a_r);
  {
    matvec_mul_rowmajor(m->U_r, H, H, h_prev, tmp);
    for (int i = 0; i < H; i++) a_r[i] = a_r[i] + tmp[i] + m->b_r[i];
  }
  for (int i = 0; i < H; i++) r_out[i] = act_sigmoid(a_r[i]);

  // candidate
  for (int i = 0; i < H; i++) rh[i] = r_out[i] * h_prev[i];
  matvec_mul_rowmajor(m->W_n, H, H, x, a_n);
  {
    matvec_mul_rowmajor(m->U_n, H, H, rh, tmp);
    for (int i = 0; i < H; i++) a_n[i] = a_n[i] + tmp[i] + m->b_n[i];
  }
  for (int i = 0; i < H; i++) n_out[i] = tanhf(a_n[i]);

  // h_next = (1-z)*n + z*h_prev
  for (int i = 0; i < H; i++) h_next[i] = (1.0f - z_out[i]) * n_out[i] + z_out[i] * h_prev[i];
}

void model_logits_from_hidden(
  const GRUModel* m,
  const float* h,
  float* logits
) {
  const int V = m->vocab_size;
  const int H = m->hidden_size;
  // logits[v] = out_W[v*H + j]*h[j] + out_b[v]
  for (int v = 0; v < V; v++) {
    float s = 0.0f;
    const float* row = m->out_W + (size_t)v * (size_t)H;
    for (int j = 0; j < H; j++) s += row[j] * h[j];
    logits[v] = s + m->out_b[v];
  }
}

