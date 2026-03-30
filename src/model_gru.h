#pragma once

#include <stdint.h>

typedef struct GRUModel {
  int vocab_size;
  int hidden_size;

  // Token -> embedding vector
  float* embeddings; // [vocab_size * hidden_size]

  // Update gate
  float* W_z; // [hidden_size * hidden_size] row-major
  float* U_z; // [hidden_size * hidden_size] row-major
  float* b_z; // [hidden_size]

  // Reset gate
  float* W_r;
  float* U_r;
  float* b_r;

  // Candidate
  float* W_n;
  float* U_n;
  float* b_n;

  // Output projection: logits = out_W * h + out_b
  float* out_W; // [vocab_size * hidden_size] row-major
  float* out_b; // [vocab_size]
} GRUModel;

void model_gru_init(GRUModel* m, int vocab_size, int hidden_size, unsigned int seed);
void model_gru_free(GRUModel* m);

int model_gru_save(const GRUModel* m, const char* path);
int model_gru_load(GRUModel* m, const char* path);

// Computes GRU cell: h_next given token embedding (x) and previous hidden state.
// Also returns z/r/n for training backprop.
void gru_cell_forward(
  const GRUModel* m,
  const float* x,      // [hidden_size]
  const float* h_prev,// [hidden_size]
  float* h_next,       // [hidden_size]
  float* z_out,        // [hidden_size]
  float* r_out,        // [hidden_size]
  float* n_out         // [hidden_size]
);

void model_logits_from_hidden(
  const GRUModel* m,
  const float* h, // [hidden_size]
  float* logits   // [vocab_size]
);

