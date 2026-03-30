#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "util.h"
#include "tokenizer_byte.h"
#include "dataset_csv.h"
#include "model_gru.h"
#include "plot_png.h"

#ifdef _WIN32
#  include <windows.h>
#endif

typedef struct {
  const char* data_path;
  const char* out_dir;
  int seq_len;
  int hidden_size;
  int pretrain_seconds;
  int finetune_seconds;
  float lr_pretrain;
  float lr_finetune;
  int batch_size;        // currently supported (updates accumulate grads across batch)
  int log_every_updates; // log frequency
  float grad_clip_abs;  // clamp gradients to +/- this
  float assistant_bias; // finetune: probability to sample assistant-biased windows
} TrainArgs;

static int str_eq(const char* a, const char* b) {
  return (a && b) ? (strcmp(a, b) == 0) : 0;
}

static const char* argv_get_str(int argc, char** argv, const char* name, const char* def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (str_eq(argv[i], name)) return argv[i + 1];
  }
  return def;
}

static int argv_get_int(int argc, char** argv, const char* name, int def) {
  const char* s = argv_get_str(argc, argv, name, NULL);
  return s ? atoi(s) : def;
}

static float argv_get_float(int argc, char** argv, const char* name, float def) {
  const char* s = argv_get_str(argc, argv, name, NULL);
  return s ? (float)atof(s) : def;
}

static void generate_default_data_csv(const char* path, int rows) {
  FILE* f = fopen(path, "wb");
  if (!f) return;
  fprintf(f, "text\n");

  // Dataset rows are kept CSV-safe (no quotes, no commas).
  for (int i = 0; i < rows; i++) {
    const char* prompt = NULL;
    const char* answer = NULL;
    int kind = i % 12;
    if (kind == 0) { prompt = "What is 2+2"; answer = "4"; }
    if (kind == 1) { prompt = "Name a color of the sky"; answer = "Blue"; }
    if (kind == 2) { prompt = "Say hello"; answer = "Hello"; }
    if (kind == 3) { prompt = "How do I boil water"; answer = "Heat it until it simmers"; }
    if (kind == 4) { prompt = "What is a CPU"; answer = "A processor that runs instructions"; }
    if (kind == 5) { prompt = "Explain RAM"; answer = "Temporary fast memory for active work"; }
    if (kind == 6) { prompt = "Give a short greeting"; answer = "Hi there"; }
    if (kind == 7) { prompt = "How to write in C"; answer = "Use functions and compile then run"; }
    if (kind == 8) { prompt = "What is 10-3"; answer = "7"; }
    if (kind == 9) { prompt = "Solve 3*3"; answer = "9"; }
    if (kind == 10) { prompt = "What is an LLM"; answer = "A model that predicts text"; }
    if (kind == 11) { prompt = "Tell me a fact about computers"; answer = "They follow instructions"; }

    // Use literal backslash-n so the CSV parser stays line-based.
    fprintf(f, "\"User: %s?\\\\nAssistant: %s\"\n", prompt, answer);
  }
  fclose(f);
}

static void softmax(const float* logits, int n, float* probs) {
  float maxv = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    probs[i] = expf(logits[i] - maxv);
    sum += probs[i];
  }
  if (sum < 1e-12f) sum = 1e-12f;
  for (int i = 0; i < n; i++) probs[i] /= sum;
}

static inline float clamp_f(float x, float a) {
  if (x > a) return a;
  if (x < -a) return -a;
  return x;
}

static void print_progress_bar(const char* stage_name, double elapsed, int total_seconds, double avg_loss, int step) {
  if (total_seconds <= 0) total_seconds = 1;
  double p = elapsed / (double)total_seconds;
  if (p < 0.0) p = 0.0;
  if (p > 1.0) p = 1.0;
  int bar_w = 32;
  int fill = (int)(p * (double)bar_w);
  if (fill > bar_w) fill = bar_w;

  char bar[40];
  for (int i = 0; i < bar_w; i++) bar[i] = (i < fill) ? '#' : '-';
  bar[bar_w] = '\0';

  printf("\r%s [%s] %3d%% | step %d | avg loss %.4f | %5.1fs/%ds",
         stage_name, bar, (int)(p * 100.0), step, (float)avg_loss, (float)elapsed, total_seconds);
  fflush(stdout);
}

static void compute_assistant_candidate_starts(
  const uint16_t* tokens,
  size_t token_count,
  int seq_len,
  const uint8_t* marker_bytes,
  size_t marker_len,
  uint16_t** out_candidates,
  size_t* out_count
) {
  *out_candidates = NULL;
  *out_count = 0;
  if (!tokens || token_count < (size_t)seq_len + 2) return;

  size_t max_start = token_count - (size_t)seq_len - 2;

  // Worst-case candidate count could be large, but with our small vocab and marker it's manageable.
  size_t cap = 1024;
  uint16_t* cand = (uint16_t*)malloc(cap * sizeof(uint16_t));
  if (!cand) return;

  for (size_t p = 0; p + marker_len + (size_t)seq_len + 1 <= token_count; p++) {
    int ok = 1;
    for (size_t k = 0; k < marker_len; k++) {
      uint16_t t = tokens[p + k];
      if (t >= 256 || (uint8_t)t != marker_bytes[k]) { ok = 0; break; }
    }
    if (!ok) continue;

    // We want the input window start to be right before the bytes after "Assistant:".
    // If marker ends at p + marker_len - 1, then first target byte is at p + marker_len.
    size_t start = p + marker_len - 1;
    if (start > max_start) continue;

    if (*out_count >= cap) {
      size_t ncap = cap * 2;
      uint16_t* ndata = (uint16_t*)realloc(cand, ncap * sizeof(uint16_t));
      if (!ndata) break;
      cand = ndata;
      cap = ncap;
    }
    cand[*out_count] = (uint16_t)start;
    (*out_count)++;
  }

  *out_candidates = cand;
}

static float train_stage(
  const TrainArgs* args,
  const TokenizerByte* tok,
  const DatasetTokens* ds,
  GRUModel* model,
  const char* loss_csv_path,
  const char* ckpt_path,
  float lr,
  const uint16_t* assistant_candidates,
  size_t assistant_candidates_count,
  int use_assistant_bias,
  const uint8_t* assistant_marker_bytes,
  size_t assistant_marker_len
) {
  (void)assistant_marker_bytes;
  (void)assistant_marker_len;
  (void)tok;

  const int B = (args->batch_size <= 0) ? 1 : args->batch_size;
  const int H = model->hidden_size;
  const int V = model->vocab_size;
  const int T = args->seq_len;

  if (!ds || !ds->tokens || ds->token_count < (size_t)T + 2) return 0.0f;

  size_t max_start = ds->token_count - (size_t)T - 2; // ensure t+T+1 exists
  if (max_start < 2) return 0.0f;

  util_mkdir_p(args->out_dir);

  FILE* log_file = fopen(loss_csv_path, "wb");
  if (!log_file) {
    fprintf(stderr, "Failed to open loss log: %s\n", loss_csv_path);
    return 0.0f;
  }
  fprintf(log_file, "step,loss,elapsed_seconds\n");
  fflush(log_file);

  unsigned int rng = util_seed_from_time();

  // Token ids for the batch window.
  uint16_t* tok_in = (uint16_t*)malloc((size_t)B * (size_t)T * sizeof(uint16_t));
  uint16_t* tok_target = (uint16_t*)malloc((size_t)B * (size_t)T * sizeof(uint16_t));
  if (!tok_in || !tok_target) {
    fclose(log_file);
    return 0.0f;
  }

  // Activations.
  // h[b,0] = 0, h[b,t+1] = GRU output after consuming input at time t.
  float* x = (float*)malloc((size_t)B * (size_t)T * (size_t)H * sizeof(float));
  float* h = (float*)malloc((size_t)B * (size_t)(T + 1) * (size_t)H * sizeof(float));
  float* z = (float*)malloc((size_t)B * (size_t)T * (size_t)H * sizeof(float));
  float* r = (float*)malloc((size_t)B * (size_t)T * (size_t)H * sizeof(float));
  float* n = (float*)malloc((size_t)B * (size_t)T * (size_t)H * sizeof(float));
  float* dh_from_output = (float*)malloc((size_t)B * (size_t)T * (size_t)H * sizeof(float));

  // Temporary buffers.
  float* logits = (float*)malloc((size_t)V * sizeof(float));
  float* probs = (float*)malloc((size_t)V * sizeof(float));
  float* dh_next = (float*)malloc((size_t)B * (size_t)H * sizeof(float));
  float* dh = (float*)malloc((size_t)H * sizeof(float));
  float* da_z = (float*)malloc((size_t)H * sizeof(float));
  float* da_r = (float*)malloc((size_t)H * sizeof(float));
  float* da_n = (float*)malloc((size_t)H * sizeof(float));
  float* rh = (float*)malloc((size_t)H * sizeof(float));
  float* g = (float*)malloc((size_t)H * sizeof(float));
  float* dx_from_z = (float*)malloc((size_t)H * sizeof(float));
  float* dx_from_r = (float*)malloc((size_t)H * sizeof(float));
  float* dx_from_n = (float*)malloc((size_t)H * sizeof(float));
  float* dhprev_from_r = (float*)malloc((size_t)H * sizeof(float));
  float* tmpH = (float*)malloc((size_t)H * sizeof(float));

  if (!x || !h || !z || !r || !n || !dh_from_output ||
      !logits || !probs || !dh_next || !dh || !da_z || !da_r || !da_n || !rh ||
      !g || !dx_from_z || !dx_from_r || !dx_from_n || !dhprev_from_r || !tmpH) {
    fclose(log_file);
    return 0.0f;
  }

  // Grad buffers for all parameters.
  float* g_emb = (float*)calloc((size_t)V * (size_t)H, sizeof(float));
  float* g_Wz = (float*)calloc((size_t)H * (size_t)H, sizeof(float));
  float* g_Uz = (float*)calloc((size_t)H * (size_t)H, sizeof(float));
  float* g_bz = (float*)calloc((size_t)H, sizeof(float));

  float* g_Wr = (float*)calloc((size_t)H * (size_t)H, sizeof(float));
  float* g_Ur = (float*)calloc((size_t)H * (size_t)H, sizeof(float));
  float* g_br = (float*)calloc((size_t)H, sizeof(float));

  float* g_Wn = (float*)calloc((size_t)H * (size_t)H, sizeof(float));
  float* g_Un = (float*)calloc((size_t)H * (size_t)H, sizeof(float));
  float* g_bn = (float*)calloc((size_t)H, sizeof(float));

  float* g_out_W = (float*)calloc((size_t)V * (size_t)H, sizeof(float));
  float* g_out_b = (float*)calloc((size_t)V, sizeof(float));

  if (!g_emb || !g_Wz || !g_Uz || !g_bz || !g_Wr || !g_Ur || !g_br || !g_Wn || !g_Un || !g_bn || !g_out_W || !g_out_b) {
    fclose(log_file);
    return 0.0f;
  }

  int log_update_every = args->log_every_updates;
  if (log_update_every <= 0) log_update_every = 20;

  int step = 0;
  int updates = 0;
  double stage_start = util_wall_seconds();
  double last_progress_print = 0.0;

  double total_loss = 0.0;
  int loss_samples = 0;
  double latest_avg_loss = 0.0;

  int stage_seconds = use_assistant_bias ? args->finetune_seconds : args->pretrain_seconds;

  while (util_wall_seconds() - stage_start < (double)stage_seconds) {
    updates++;

    // Clear grads.
    memset(g_emb, 0, (size_t)V * (size_t)H * sizeof(float));
    memset(g_Wz, 0, (size_t)H * (size_t)H * sizeof(float));
    memset(g_Uz, 0, (size_t)H * (size_t)H * sizeof(float));
    memset(g_bz, 0, (size_t)H * sizeof(float));

    memset(g_Wr, 0, (size_t)H * (size_t)H * sizeof(float));
    memset(g_Ur, 0, (size_t)H * (size_t)H * sizeof(float));
    memset(g_br, 0, (size_t)H * sizeof(float));

    memset(g_Wn, 0, (size_t)H * (size_t)H * sizeof(float));
    memset(g_Un, 0, (size_t)H * (size_t)H * sizeof(float));
    memset(g_bn, 0, (size_t)H * sizeof(float));

    memset(g_out_W, 0, (size_t)V * (size_t)H * sizeof(float));
    memset(g_out_b, 0, (size_t)V * sizeof(float));

    // Sample windows for each batch element.
    for (int b = 0; b < B; b++) {
      size_t start = 0;
      if (use_assistant_bias && assistant_candidates_count > 0) {
        float p = util_rand_f32(&rng);
        if (p < args->assistant_bias) {
          size_t idx = (size_t)(util_rand_u32(&rng) % (unsigned int)assistant_candidates_count);
          start = assistant_candidates[idx];
        } else {
          start = (size_t)(util_rand_u32(&rng) % (unsigned int)max_start);
        }
      } else {
        start = (size_t)(util_rand_u32(&rng) % (unsigned int)max_start);
      }
      if (start + (size_t)T + 1 >= ds->token_count) start = 0;

      for (int t = 0; t < T; t++) {
        uint16_t in_id = ds->tokens[start + (size_t)t];
        uint16_t target_id = ds->tokens[start + (size_t)t + 1];
        tok_in[(size_t)b * (size_t)T + (size_t)t] = in_id;
        tok_target[(size_t)b * (size_t)T + (size_t)t] = target_id;

        // x[b,t,:] = embedding(tok_in)
        const float* emb = model->embeddings + (size_t)in_id * (size_t)H;
        float* x_bt = x + (((size_t)b * (size_t)T + (size_t)t) * (size_t)H);
        memcpy(x_bt, emb, (size_t)H * sizeof(float));
      }
    }

    // h[b,0] = 0
    for (int b = 0; b < B; b++) {
      float* h0 = h + (((size_t)b * (size_t)(T + 1) + 0) * (size_t)H);
      memset(h0, 0, (size_t)H * sizeof(float));
    }

    // Forward
    double batch_loss = 0.0;
    for (int t = 0; t < T; t++) {
      for (int b = 0; b < B; b++) {
        float* h_next = h + (((size_t)b * (size_t)(T + 1) + (size_t)(t + 1)) * (size_t)H);
        const float* h_prev = h + (((size_t)b * (size_t)(T + 1) + (size_t)t) * (size_t)H);
        const float* x_bt = x + (((size_t)b * (size_t)T + (size_t)t) * (size_t)H);
        float* z_bt = z + (((size_t)b * (size_t)T + (size_t)t) * (size_t)H);
        float* r_bt = r + (((size_t)b * (size_t)T + (size_t)t) * (size_t)H);
        float* n_bt = n + (((size_t)b * (size_t)T + (size_t)t) * (size_t)H);

        gru_cell_forward(model, x_bt, h_prev, h_next, z_bt, r_bt, n_bt);

        // logits = out_W * h_next + out_b
        model_logits_from_hidden(model, h_next, logits);
        softmax(logits, V, probs);

        uint16_t target_id = tok_target[(size_t)b * (size_t)T + (size_t)t];
        float prob_t = probs[target_id];
        if (prob_t < 1e-12f) prob_t = 1e-12f;
        batch_loss += -logf(prob_t);

        // dlogits = probs; dlogits[target] -= 1
        // Compute dh_from_output = out_W^T * dlogits
        // and accumulate out layer grads.
        for (int v = 0; v < V; v++) {
          float dlogit = probs[v];
          if (v == (int)target_id) dlogit -= 1.0f;
          g_out_b[v] += dlogit;
          const float* hvec = h_next;
          float* gw = g_out_W + (size_t)v * (size_t)H;
          for (int j = 0; j < H; j++) {
            gw[j] += dlogit * hvec[j];
          }
        }

        // dh_from_output[b,t,:]
        // dh_j = sum_v out_W[v*H + j] * dlogits[v]
        float* dh_t = dh_from_output + (((size_t)b * (size_t)T + (size_t)t) * (size_t)H);
        for (int j = 0; j < H; j++) dh_t[j] = 0.0f;
        for (int v = 0; v < V; v++) {
          float dlogit = probs[v];
          if (v == (int)target_id) dlogit -= 1.0f;
          const float* row = model->out_W + (size_t)v * (size_t)H;
          for (int j = 0; j < H; j++) {
            dh_t[j] += row[j] * dlogit;
          }
        }
      }
    }

    // Backward through time
    // dh_next[b,:] starts at 0 at the end.
    for (int b = 0; b < B; b++) {
      float* dhn = dh_next + ((size_t)b * (size_t)H);
      memset(dhn, 0, (size_t)H * sizeof(float));
    }

    for (int t = T - 1; t >= 0; t--) {
      for (int b = 0; b < B; b++) {
        float* dhn = dh_next + ((size_t)b * (size_t)H);
        const float* dh_out = dh_from_output + (((size_t)b * (size_t)T + (size_t)t) * (size_t)H);
        const float* h_prev = h + (((size_t)b * (size_t)(T + 1) + (size_t)t) * (size_t)H);
        const float* x_bt = x + (((size_t)b * (size_t)T + (size_t)t) * (size_t)H);
        const float* z_bt = z + (((size_t)b * (size_t)T + (size_t)t) * (size_t)H);
        const float* r_bt = r + (((size_t)b * (size_t)T + (size_t)t) * (size_t)H);
        const float* n_bt = n + (((size_t)b * (size_t)T + (size_t)t) * (size_t)H);

        // dh = dh_from_output + dh_next
        for (int j = 0; j < H; j++) dh[j] = dh_out[j] + dhn[j];

        // h_curr is h_next in naming; candidate and gates use n,z,r.
        // dh_part_n = dh * (1 - z)
        for (int j = 0; j < H; j++) {
          float dh_part_n = dh[j] * (1.0f - z_bt[j]);
          float dh_part_z = dh[j] * (h_prev[j] - n_bt[j]);
          float dh_part_hprev = dh[j] * z_bt[j];

          // da_n = dh_part_n * (1 - n^2)
          da_n[j] = dh_part_n * (1.0f - n_bt[j] * n_bt[j]);
          // da_z = dh_part_z * z * (1 - z)
          da_z[j] = dh_part_z * z_bt[j] * (1.0f - z_bt[j]);
          // We'll compute da_r after we compute d_r (needs g).
          // Temporarily store dh_part_hprev in dhprev_from_r buffer.
          tmpH[j] = dh_part_hprev;
        }

        // Candidate weight gradients:
        // g = U_n^T * da_n
        // g[i] = sum_j U_n[j*H + i] * da_n[j]
        for (int i = 0; i < H; i++) g[i] = 0.0f;
        for (int j = 0; j < H; j++) {
          float aj = da_n[j];
          const float* row = model->U_n + (size_t)j * (size_t)H; // U_n[j,:]
          for (int i = 0; i < H; i++) {
            g[i] += row[i] * aj;
          }
        }

        // dr = g * h_prev
        for (int j = 0; j < H; j++) {
          float dr = g[j] * h_prev[j];
          // da_r = dr * r*(1-r)
          da_r[j] = dr * r_bt[j] * (1.0f - r_bt[j]);
        }

        // Gradients for gate parameters:
        // z gate:
        // dW_z += outer(da_z, x)
        // dU_z += outer(da_z, h_prev)
        // db_z += da_z
        for (int i = 0; i < H; i++) {
          float dai = da_z[i];
          g_bz[i] += dai;
          const float* xt = x_bt;
          const float* hpt = h_prev;
          float* rowW = g_Wz + (size_t)i * (size_t)H;
          float* rowU = g_Uz + (size_t)i * (size_t)H;
          for (int j = 0; j < H; j++) {
            rowW[j] += dai * xt[j];
            rowU[j] += dai * hpt[j];
          }
        }

        // r gate:
        for (int i = 0; i < H; i++) {
          float dai = da_r[i];
          g_br[i] += dai;
          float* rowW = g_Wr + (size_t)i * (size_t)H;
          float* rowU = g_Ur + (size_t)i * (size_t)H;
          for (int j = 0; j < H; j++) {
            rowW[j] += dai * x_bt[j];
            rowU[j] += dai * h_prev[j];
          }
        }

        // Candidate:
        // Candidate uses m = r * h_prev
        for (int j = 0; j < H; j++) rh[j] = r_bt[j] * h_prev[j];

        for (int i = 0; i < H; i++) {
          float dai = da_n[i];
          g_bn[i] += dai;
          float* rowW = g_Wn + (size_t)i * (size_t)H;
          float* rowU = g_Un + (size_t)i * (size_t)H;
          for (int j = 0; j < H; j++) {
            rowW[j] += dai * x_bt[j];
            rowU[j] += dai * rh[j];
          }
        }

        // Embedding gradient accumulation:
        // dx = W_z^T * da_z + W_r^T * da_r + W_n^T * da_n
        // We also need dhprev contributions to previous time step:
        // dhprev_from_z = U_z^T * da_z
        // dhprev_from_r = U_r^T * da_r
        // dhprev_from_candidate = g * r (since gradient to h_prev via (r*h_prev))

        // dx_from_z:
        for (int j = 0; j < H; j++) dx_from_z[j] = 0.0f;
        for (int i = 0; i < H; i++) {
          float dai = da_z[i];
          const float* colW = model->W_z + (size_t)i * (size_t)H;
          for (int j = 0; j < H; j++) dx_from_z[j] += colW[j] * dai;
        }
        // dx_from_r:
        for (int j = 0; j < H; j++) dx_from_r[j] = 0.0f;
        for (int i = 0; i < H; i++) {
          float dai = da_r[i];
          const float* colW = model->W_r + (size_t)i * (size_t)H;
          for (int j = 0; j < H; j++) dx_from_r[j] += colW[j] * dai;
        }
        // dx_from_n:
        for (int j = 0; j < H; j++) dx_from_n[j] = 0.0f;
        for (int i = 0; i < H; i++) {
          float dai = da_n[i];
          const float* colW = model->W_n + (size_t)i * (size_t)H;
          for (int j = 0; j < H; j++) dx_from_n[j] += colW[j] * dai;
        }

        for (int j = 0; j < H; j++) {
          tmpH[j] = dx_from_z[j] + dx_from_r[j] + dx_from_n[j];
        }

        // Update embedding gradients:
        uint16_t in_id = tok_in[(size_t)b * (size_t)T + (size_t)t];
        float* gemb_row = g_emb + (size_t)in_id * (size_t)H;
        for (int j = 0; j < H; j++) gemb_row[j] += tmpH[j];

        // dhprev contributions:
        // dhprev_from_z = U_z^T * da_z
        for (int j = 0; j < H; j++) dhprev_from_r[j] = 0.0f; // reuse buffer temporarily
        for (int j = 0; j < H; j++) {
          // dhprev_from_z stored into dhprev_from_r buffer for now then later add other parts.
          float s = 0.0f;
          for (int i = 0; i < H; i++) {
            // U_z is row-major [i][j], so U_z[i*H + j] is element (i,j).
            // dhprev_from_z[j] = sum_i U_z[i,j] * da_z[i]
            s += model->U_z[(size_t)i * (size_t)H + (size_t)j] * da_z[i];
          }
          dhprev_from_r[j] = s;
        }

        // dhprev_from_r = U_r^T * da_r
        float* dhprev_vec = dh_next + ((size_t)b * (size_t)H); // writing for next time step (t)
        for (int j = 0; j < H; j++) {
          float s = 0.0f;
          for (int i = 0; i < H; i++) {
            s += model->U_r[(size_t)i * (size_t)H + (size_t)j] * da_r[i];
          }
          // Start with dh_part_hprev from z term that we stored in tmpH earlier? careful:
          // We stored dh_part_hprev in tmpH at each j, but overwrote tmpH with dx sum.
          // We'll recompute dh_part_hprev_from_h = dh * z and add it later.
          // For now set dhprev_vec to 0; we will add correct components below.
          dhprev_vec[j] = s;
        }

        // Add dh_part_hprev_from_h (this is dh * z)
        for (int j = 0; j < H; j++) {
          float dh_part_hprev_from_h = dh[j] * z_bt[j];
          dhprev_vec[j] += dh_part_hprev_from_h;
        }

        // Add dhprev_from_candidate = g * r
        for (int j = 0; j < H; j++) {
          dhprev_vec[j] += g[j] * r_bt[j];
        }

        // Add dhprev_from_z (from U_z^T * da_z) computed earlier.
        for (int j = 0; j < H; j++) {
          dhprev_vec[j] += dhprev_from_r[j];
        }
      }
    }

    // Gradient clipping and parameter update (SGD).
    float clip = args->grad_clip_abs;
    if (clip <= 0.0f) clip = 1.0f;

    size_t Vsz = (size_t)V;
    size_t Hsz = (size_t)H;

    size_t n_emb = Vsz * Hsz;
    for (size_t i = 0; i < n_emb; i++) {
      float grad = g_emb[i];
      grad = clamp_f(grad, clip);
      model->embeddings[i] -= lr * grad;
    }
    for (size_t i = 0; i < Hsz * Hsz; i++) {
      model->W_z[i] -= lr * clamp_f(g_Wz[i], clip);
      model->U_z[i] -= lr * clamp_f(g_Uz[i], clip);
      model->W_r[i] -= lr * clamp_f(g_Wr[i], clip);
      model->U_r[i] -= lr * clamp_f(g_Ur[i], clip);
      model->W_n[i] -= lr * clamp_f(g_Wn[i], clip);
      model->U_n[i] -= lr * clamp_f(g_Un[i], clip);
    }
    for (int i = 0; i < H; i++) {
      model->b_z[i] -= lr * clamp_f(g_bz[i], clip);
      model->b_r[i] -= lr * clamp_f(g_br[i], clip);
      model->b_n[i] -= lr * clamp_f(g_bn[i], clip);
    }
    for (size_t v = 0; v < Vsz; v++) {
      model->out_b[v] -= lr * clamp_f(g_out_b[v], clip);
      float* outWrow = model->out_W + v * Hsz;
      float* gWrow = g_out_W + v * Hsz;
      for (int j = 0; j < H; j++) {
        outWrow[j] -= lr * clamp_f(gWrow[j], clip);
      }
    }

    // Log average loss across tokens (we used batch_loss across B*T tokens).
    double avg_loss = batch_loss / (double)(B * T);
    latest_avg_loss = avg_loss;
    total_loss += avg_loss;
    loss_samples++;
    step++;

    if (step % log_update_every == 0) {
      double elapsed = util_wall_seconds() - stage_start;
      double avg = total_loss / (double)loss_samples;
      fprintf(log_file, "%d,%.8f,%.3f\n", step, avg, elapsed);
      fflush(log_file);
      total_loss = 0.0;
      loss_samples = 0;
    }

    // Live progress bar, independent from CSV log cadence.
    {
      double elapsed = util_wall_seconds() - stage_start;
      if ((elapsed - last_progress_print) >= 0.2 || elapsed >= (double)stage_seconds) {
        double current_avg = (loss_samples > 0) ? (total_loss / (double)loss_samples) : latest_avg_loss;
        print_progress_bar(use_assistant_bias ? "finetune" : "pretrain", elapsed, stage_seconds, current_avg, step);
        last_progress_print = elapsed;
      }
    }
  }
  // Finish bar line
  {
    double elapsed = util_wall_seconds() - stage_start;
    double current_avg = (loss_samples > 0) ? (total_loss / (double)loss_samples) : latest_avg_loss;
    print_progress_bar(use_assistant_bias ? "finetune" : "pretrain", elapsed, stage_seconds, current_avg, step);
    printf("\n");
  }

  fclose(log_file);

  // Save checkpoint.
  if (ckpt_path && ckpt_path[0] != '\0') {
    if (model_gru_save(model, ckpt_path) != 0) {
      fprintf(stderr, "Warning: failed to save checkpoint: %s\n", ckpt_path);
    }
  }

  return 0.0f;
}

int main(int argc, char** argv) {
  TrainArgs args;
  memset(&args, 0, sizeof(args));
  args.data_path = argv_get_str(argc, argv, "--data", "data.csv");
  args.out_dir = argv_get_str(argc, argv, "--out", "out");
  args.seq_len = argv_get_int(argc, argv, "--seq-len", 96);
  args.hidden_size = argv_get_int(argc, argv, "--hidden", 128);
  args.pretrain_seconds = argv_get_int(argc, argv, "--pretrain-seconds", 600);
  args.finetune_seconds = argv_get_int(argc, argv, "--finetune-seconds", 600);
  args.lr_pretrain = argv_get_float(argc, argv, "--lr-pretrain", 3e-4f);
  args.lr_finetune = argv_get_float(argc, argv, "--lr-finetune", 1e-4f);
  args.batch_size = argv_get_int(argc, argv, "--batch", 1);
  args.log_every_updates = argv_get_int(argc, argv, "--log-every", 20);
  args.grad_clip_abs = argv_get_float(argc, argv, "--grad-clip", 1.0f);
  args.assistant_bias = argv_get_float(argc, argv, "--assistant-bias", 0.7f);

  util_mkdir_p(args.out_dir);

  // Ensure dataset exists; generate default if missing.
  {
    FILE* f = fopen(args.data_path, "rb");
    if (!f) {
      fprintf(stderr, "Dataset not found at '%s'. Generating a default dataset.\n", args.data_path);
      generate_default_data_csv(args.data_path, 2000);
    } else {
      fclose(f);
    }
  }

  TokenizerByte tok;
  tokenizer_byte_init(&tok);

  DatasetTokens ds;
  if (dataset_load_csv_first_col_as_text(args.data_path, &tok, &ds) != 0) {
    fprintf(stderr, "Failed to load dataset: %s\n", args.data_path);
    return 1;
  }

  GRUModel model;
  model_gru_init(&model, tok.vocab_size, args.hidden_size, util_seed_from_time());

  // Assistant candidates for finetune.
  const uint8_t assistant_marker_bytes[] = { 'A','s','s','i','s','t','a','n','t',':'}; // "Assistant:"
  size_t assistant_marker_len = sizeof(assistant_marker_bytes) / sizeof(assistant_marker_bytes[0]);

  uint16_t* assistant_candidates = NULL;
  size_t assistant_candidates_count = 0;
  compute_assistant_candidate_starts(ds.tokens, ds.token_count, args.seq_len,
                                      assistant_marker_bytes, assistant_marker_len,
                                      &assistant_candidates, &assistant_candidates_count);

  char pre_loss_csv[512];
  char ft_loss_csv[512];
  char pre_ckpt[512];
  char ft_ckpt[512];
  char pre_png[512];
  char ft_png[512];
  snprintf(pre_loss_csv, sizeof(pre_loss_csv), "%s/pretrain_loss.csv", args.out_dir);
  snprintf(ft_loss_csv, sizeof(ft_loss_csv), "%s/finetune_loss.csv", args.out_dir);
  snprintf(pre_ckpt, sizeof(pre_ckpt), "%s/pretrain.ckpt", args.out_dir);
  snprintf(ft_ckpt, sizeof(ft_ckpt), "%s/finetune.ckpt", args.out_dir);
  snprintf(pre_png, sizeof(pre_png), "%s/pretrain_loss.png", args.out_dir);
  snprintf(ft_png, sizeof(ft_png), "%s/finetune_loss.png", args.out_dir);

  fprintf(stdout, "Stage 1/2: pretraining for %d seconds...\n", args.pretrain_seconds);
  train_stage(&args, &tok, &ds, &model, pre_loss_csv, pre_ckpt, args.lr_pretrain,
              assistant_candidates, assistant_candidates_count, 0,
              assistant_marker_bytes, assistant_marker_len);

  fprintf(stdout, "Pretraining complete. Generating pretrain chart...\n");
  plot_png_loss_from_csv(pre_loss_csv, pre_png, 900, 480);

  fprintf(stdout, "Stage 2/2: finetuning for %d seconds...\n", args.finetune_seconds);
  train_stage(&args, &tok, &ds, &model, ft_loss_csv, ft_ckpt, args.lr_finetune,
              assistant_candidates, assistant_candidates_count, 1,
              assistant_marker_bytes, assistant_marker_len);

  fprintf(stdout, "Finetuning complete. Generating finetune chart...\n");
  plot_png_loss_from_csv(ft_loss_csv, ft_png, 900, 480);

  fprintf(stdout, "Done. Checkpoints: %s and %s\n", pre_ckpt, ft_ckpt);

  free(assistant_candidates);
  dataset_free(&ds);
  model_gru_free(&model);
  return 0;
}

