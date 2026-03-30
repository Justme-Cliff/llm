#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

#include "util.h"
#include "tokenizer_byte.h"
#include "model_gru.h"

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <math.h>
#endif

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

static void json_escape(const char* in, char* out, size_t out_sz) {
  size_t o = 0;
  for (size_t i = 0; in && in[i] && o + 2 < out_sz; i++) {
    unsigned char c = (unsigned char)in[i];
    if (c == '\\') {
      out[o++] = '\\';
      out[o++] = '\\';
    } else if (c == '"') {
      out[o++] = '\\';
      out[o++] = '"';
    } else if (c == '\n') {
      out[o++] = '\\';
      out[o++] = 'n';
    } else if (c == '\r') {
      // drop
    } else {
      out[o++] = (char)c;
    }
  }
  out[o] = '\0';
}

static int json_get_string_value(const char* body, const char* key, char* out, size_t out_sz) {
  if (!body || !key || !out || out_sz == 0) return -1;
  const char* p = strstr(body, key);
  if (!p) return -1;
  // Find colon
  p = strchr(p, ':');
  if (!p) return -1;
  p++;
  while (*p && isspace((unsigned char)*p)) p++;
  if (*p != '"') return -1;
  p++; // skip opening quote

  size_t o = 0;
  while (*p && *p != '"') {
    if (o + 1 >= out_sz) break;
    char c = *p++;
    if (c == '\\') {
      char esc = *p++;
      if (esc == 'n') c = '\n';
      else if (esc == 'r') c = '\r';
      else if (esc == 't') c = '\t';
      else if (esc == '"' || esc == '\\' || esc == '/') c = esc;
      else c = esc;
    }
    out[o++] = c;
  }
  out[o] = '\0';
  return 0;
}

static int json_get_int_value(const char* body, const char* key, int* out_value) {
  if (!body || !key || !out_value) return -1;
  const char* p = strstr(body, key);
  if (!p) return -1;
  p = strchr(p, ':');
  if (!p) return -1;
  p++;
  while (*p && isspace((unsigned char)*p)) p++;
  int v = 0;
  int sign = 1;
  if (*p == '-') { sign = -1; p++; }
  while (*p && isdigit((unsigned char)*p)) {
    v = v * 10 + (*p - '0');
    p++;
  }
  *out_value = v * sign;
  return 0;
}

static void normalize_prompt_newlines(const char* in, char* out, size_t out_sz) {
  // Convert actual newlines into literal "\n" so the model sees the same pattern as training data.
  size_t o = 0;
  for (size_t i = 0; in && in[i] && o + 2 < out_sz; i++) {
    char c = in[i];
    if (c == '\r') continue;
    if (c == '\n') {
      out[o++] = '\\';
      out[o++] = 'n';
      continue;
    }
    out[o++] = c;
  }
  out[o] = '\0';
}

static void softmax_probs_temp(const float* logits, int n, float* probs, float temperature) {
  float maxv = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
  float sum = 0.0f;
  if (temperature <= 1e-8f) temperature = 1e-8f;
  for (int i = 0; i < n; i++) {
    float v = (logits[i] - maxv) / temperature;
    probs[i] = expf(v);
    sum += probs[i];
  }
  if (sum < 1e-12f) sum = 1e-12f;
  for (int i = 0; i < n; i++) probs[i] /= sum;
}

static uint16_t sample_probs(const float* probs, int n, unsigned int* rng) {
  float r = util_rand_f32(rng);
  float c = 0.0f;
  for (int i = 0; i < n; i++) {
    c += probs[i];
    if (r <= c) return (uint16_t)i;
  }
  return (uint16_t)(n - 1);
}

static int ends_with_user_marker(const uint16_t* toks, size_t n) {
  // Detect literal "\nUser:" in byte tokens.
  // Pattern: '\\' 'n' 'U' 's' 'e' 'r' ':'
  const uint8_t pat[] = { '\\', 'n', 'U', 's', 'e', 'r', ':' };
  const size_t m = sizeof(pat) / sizeof(pat[0]);
  if (!toks || n < m) return 0;
  for (size_t i = 0; i < m; i++) {
    uint16_t t = toks[n - m + i];
    if (t >= 256) return 0;
    if ((uint8_t)t != pat[i]) return 0;
  }
  return 1;
}

#ifdef _WIN32
static int send_all(SOCKET s, const char* buf, size_t len) {
  size_t sent = 0;
  while (sent < len) {
    int rc = send(s, buf + sent, (int)(len - sent), 0);
    if (rc <= 0) return -1;
    sent += (size_t)rc;
  }
  return 0;
}
#endif

static void http_send_text(SOCKET client, int status, const char* content_type, const char* body) {
#ifdef _WIN32
  const char* status_text = (status == 200) ? "OK" : (status == 400 ? "Bad Request" : "Not Found");
  size_t body_len = body ? strlen(body) : 0;

  char header[512];
  snprintf(header, sizeof(header),
           "HTTP/1.1 %d %s\r\n"
           "Content-Type: %s\r\n"
           "Content-Length: %lu\r\n"
           "Connection: close\r\n"
           "\r\n",
           status, status_text, content_type, (unsigned long)body_len);
  send_all(client, header, strlen(header));
  if (body_len > 0) send_all(client, body, body_len);
#else
  (void)client; (void)status; (void)content_type; (void)body;
#endif
}

static void http_send_bytes(SOCKET client, int status, const char* content_type, const unsigned char* data, size_t len) {
#ifdef _WIN32
  const char* status_text = (status == 200) ? "OK" : (status == 400 ? "Bad Request" : "Not Found");
  char header[512];
  snprintf(header, sizeof(header),
           "HTTP/1.1 %d %s\r\n"
           "Content-Type: %s\r\n"
           "Content-Length: %lu\r\n"
           "Connection: close\r\n"
           "\r\n",
           status, status_text, content_type, (unsigned long)len);
  send_all(client, header, strlen(header));
  if (len > 0) send_all(client, (const char*)data, len);
#else
  (void)client; (void)status; (void)content_type; (void)data; (void)len;
#endif
}

#ifdef _WIN32
static int http_handle_request(SOCKET client, GRUModel* model, const TokenizerByte* tok, const char* ui_dir, float temperature) {
  enum { RECV_CAP = 1024 * 1024 };
  char* buf = (char*)malloc(RECV_CAP);
  if (!buf) return -1;

  int received = recv(client, buf, RECV_CAP - 1, 0);
  if (received <= 0) {
    free(buf);
    return -1;
  }
  buf[received] = '\0';

  // Basic header parsing.
  char method[16] = {0};
  char path[256] = {0};
  char version[16] = {0};
  if (sscanf(buf, "%15s %255s %15s", method, path, version) < 2) {
    http_send_text(client, 400, "text/plain", "Bad Request");
    free(buf);
    return -1;
  }

  const char* header_end = strstr(buf, "\r\n\r\n");
  if (!header_end) {
    http_send_text(client, 400, "text/plain", "Bad Request");
    free(buf);
    return -1;
  }
  size_t header_len = (size_t)(header_end - buf) + 4;
  size_t body_len = (size_t)received - header_len;
  char* body = buf + header_len;

  // For POST we might not have full body; attempt a second recv if Content-Length is larger.
  if (strcmp(method, "POST") == 0) {
    int content_length = 0;
    const char* cl = strstr(buf, "Content-Length:");
    if (cl) {
      cl += strlen("Content-Length:");
      while (*cl && isspace((unsigned char)*cl)) cl++;
      content_length = atoi(cl);
    }
    if (content_length > 0) {
      while ((int)body_len < content_length) {
        int more = recv(client, buf + received, (int)(RECV_CAP - 1 - received), 0);
        if (more <= 0) break;
        received += more;
        buf[received] = '\0';
        body_len = (size_t)received - header_len;
        body = buf + header_len;
      }
    }
  }

  // Route: GET /
  if (strcmp(method, "GET") == 0) {
    char file_path[512];
    if (strcmp(path, "/") == 0) {
      snprintf(file_path, sizeof(file_path), "%s/index.html", ui_dir);
    } else if (strcmp(path, "/app.js") == 0) {
      snprintf(file_path, sizeof(file_path), "%s/app.js", ui_dir);
    } else {
      http_send_text(client, 404, "text/plain", "Not Found");
      free(buf);
      return 0;
    }

    unsigned char* data = NULL;
    size_t len = 0;
    if (util_read_file(file_path, &data, &len) != 0) {
      http_send_text(client, 404, "text/plain", "Not Found");
      free(buf);
      return 0;
    }

    const char* ct = "text/plain";
    if (strstr(file_path, ".html")) ct = "text/html";
    if (strstr(file_path, ".js")) ct = "application/javascript";

    http_send_bytes(client, 200, ct, data, len);
    free(data);
    free(buf);
    return 0;
  }

  // Route: POST /api/chat
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/chat") == 0) {
    // Ensure body is treated as string.
    char* body_copy = (char*)malloc(body_len + 1);
    if (!body_copy) { free(buf); return -1; }
    memcpy(body_copy, body, body_len);
    body_copy[body_len] = '\0';

    char prompt[2048];
    memset(prompt, 0, sizeof(prompt));
    int max_new_tokens = 120;

    // keys include quotes; body_search patterns are like "\"prompt\""
    if (json_get_string_value(body_copy, "\"prompt\"", prompt, sizeof(prompt)) != 0) {
      http_send_text(client, 400, "text/plain", "Missing prompt");
      free(body_copy);
      free(buf);
      return 0;
    }
    json_get_int_value(body_copy, "\"max_new_tokens\"", &max_new_tokens);
    if (max_new_tokens < 1) max_new_tokens = 1;
    if (max_new_tokens > 512) max_new_tokens = 512;

    // Normalize newlines to literal backslash-n to match training data.
    char prompt_norm[2048];
    normalize_prompt_newlines(prompt, prompt_norm, sizeof(prompt_norm));

    // Prompt template: "User: ...\nAssistant:"
    char tmpl[4096];
    snprintf(tmpl, sizeof(tmpl), "User: %s?\\nAssistant: ", prompt_norm);

    // Encode template without EOS at the end.
    uint16_t prompt_tokens[8192];
    size_t max_enc = 8192;
    // Use tokenizer encode then drop the last EOS.
    size_t enc = tokenizer_byte_encode_text(tok, tmpl, prompt_tokens, max_enc);
    if (enc == 0) {
      http_send_text(client, 500, "text/plain", "Tokenization failed");
      free(body_copy);
      free(buf);
      return 0;
    }
    // Drop trailing EOS if present (encode_text always appends EOS).
    if (enc >= 1 && prompt_tokens[enc - 1] == tok->eos_id) enc--;

    const int H = model->hidden_size;
    float* h = (float*)malloc((size_t)H * sizeof(float));
    float* h_next = (float*)malloc((size_t)H * sizeof(float));
    float* ztmp = (float*)malloc((size_t)H * sizeof(float));
    float* rtmp = (float*)malloc((size_t)H * sizeof(float));
    float* ntmp = (float*)malloc((size_t)H * sizeof(float));
    float* logits = (float*)malloc((size_t)model->vocab_size * sizeof(float));
    float* probs = (float*)malloc((size_t)model->vocab_size * sizeof(float));
    if (!h || !h_next || !ztmp || !rtmp || !ntmp || !logits || !probs) {
      http_send_text(client, 500, "text/plain", "Allocation failed");
      free(body_copy);
      free(buf);
      free(h); free(h_next); free(ztmp); free(rtmp); free(ntmp); free(logits); free(probs);
      return 0;
    }

    memset(h, 0, (size_t)H * sizeof(float));
    // Feed prompt.
    for (size_t i = 0; i < enc; i++) {
      uint16_t tid = prompt_tokens[i];
      const float* x = model->embeddings + (size_t)tid * (size_t)H;
      gru_cell_forward(model, x, h, h_next, ztmp, rtmp, ntmp);
      // Copy back
      memcpy(h, h_next, (size_t)H * sizeof(float));
    }

    // Generate.
    uint16_t out_tokens[2048];
    size_t out_n = 0;
    unsigned int rng = util_seed_from_time();
    if (temperature < 0.05f) temperature = 0.05f;
    if (temperature > 2.0f) temperature = 2.0f;

    for (int step = 0; step < max_new_tokens; step++) {
      model_logits_from_hidden(model, h, logits);
      softmax_probs_temp(logits, model->vocab_size, probs, temperature);
      uint16_t next = sample_probs(probs, model->vocab_size, &rng);

      if (next == tok->eos_id) break;
      if (out_n < sizeof(out_tokens) / sizeof(out_tokens[0])) {
        out_tokens[out_n++] = next;
      }
      // If it starts producing a new "User:" block, stop.
      if (ends_with_user_marker(out_tokens, out_n)) {
        out_n -= 7; // remove the marker itself
        break;
      }

      // Update hidden with generated token.
      const float* x = model->embeddings + (size_t)next * (size_t)H;
      gru_cell_forward(model, x, h, h_next, ztmp, rtmp, ntmp);
      memcpy(h, h_next, (size_t)H * sizeof(float));
    }

    // Decode bytes-only tokens into string.
    char response[4096];
    tokenizer_byte_decode_text(tok, out_tokens, out_n, response, sizeof(response));

    // JSON response
    char response_esc[8192];
    json_escape(response, response_esc, sizeof(response_esc));

    char resp_body[9000];
    snprintf(resp_body, sizeof(resp_body), "{\"response\":\"%s\"}", response_esc);
    http_send_text(client, 200, "application/json", resp_body);

    free(body_copy);
    free(buf);
    free(h); free(h_next); free(ztmp); free(rtmp); free(ntmp); free(logits); free(probs);
    return 0;
  }

  http_send_text(client, 404, "text/plain", "Not Found");
  free(buf);
  return 0;
}

#endif // _WIN32

int main(int argc, char** argv) {
  const char* ckpt_path = argv_get_str(argc, argv, "--ckpt", "out/finetune.ckpt");
  const char* ui_dir = argv_get_str(argc, argv, "--ui", "ui");
  int port = argv_get_int(argc, argv, "--port", 8080);
  float temperature = argv_get_float(argc, argv, "--temp", 0.7f);

  TokenizerByte tok;
  tokenizer_byte_init(&tok);

  GRUModel model;
  if (model_gru_load(&model, ckpt_path) != 0) {
    fprintf(stderr, "Failed to load checkpoint: %s\n", ckpt_path);
    return 1;
  }

#ifdef _WIN32
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    fprintf(stderr, "WSAStartup failed\n");
    model_gru_free(&model);
    return 1;
  }

  SOCKET server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (server == INVALID_SOCKET) {
    fprintf(stderr, "socket failed\n");
    WSACleanup();
    model_gru_free(&model);
    return 1;
  }

  // Bind
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons((u_short)port);

  if (bind(server, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
    fprintf(stderr, "bind failed (port %d)\n", port);
    closesocket(server);
    WSACleanup();
    model_gru_free(&model);
    return 1;
  }

  if (listen(server, 8) != 0) {
    fprintf(stderr, "listen failed\n");
    closesocket(server);
    WSACleanup();
    model_gru_free(&model);
    return 1;
  }

  printf("Chat server running on http://localhost:%d\n", port);

  for (;;) {
    struct sockaddr_in client_addr;
    int client_len = sizeof(client_addr);
    SOCKET client = accept(server, (struct sockaddr*)&client_addr, &client_len);
    if (client == INVALID_SOCKET) continue;
    http_handle_request(client, &model, &tok, ui_dir, temperature);
    closesocket(client);
  }

  closesocket(server);
  WSACleanup();
#else
  fprintf(stderr, "This build currently supports Windows only.\n");
#endif

  model_gru_free(&model);
  return 0;
}

